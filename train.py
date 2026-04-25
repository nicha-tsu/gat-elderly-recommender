"""
Training Pipeline for Bi-Level GAT Recommender
================================================
Covers all 3 stages from the proposal:
  Stage 1 — Data & Feature Engineering
  Stage 2 — Bi-Level GAT Training with Adaptive Feedback Loop
  Stage 3 — Evaluation (RMSE, NDCG@K) + XAI

Usage:
    python train.py
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path

from config import Config
from data.generate_data import save_data
from data.preprocessing import load_and_build
from models.bilevel_gat import BiLevelGAT, AdaptiveFeedbackLoop
from evaluation.metrics import evaluate_all
from evaluation.xai import (
    extract_attention_weights,
    explain_recommendation,
    plot_attention_heatmap,
    plot_top_influential_friends,
)
from user_study.qol_analysis import simulate_user_study, run_statistical_tests, \
    plot_pre_post_comparison, print_report

torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)


# ── helpers ──────────────────────────────────────────────────────────────────

def get_device():
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    print("No GPU found — using CPU.")
    return torch.device("cpu")


def save_results(metrics: dict, path: str):
    Path(path).parent.mkdir(exist_ok=True, parents=True)

    def _convert(obj):
        if isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_convert(metrics), f, indent=2, ensure_ascii=False)
    print(f"Results saved → {path}")


# ── training loop ─────────────────────────────────────────────────────────────

def train_epoch(
    model: BiLevelGAT,
    graph,
    train_df: pd.DataFrame,
    optimizer,
    criterion,
    feedback_loop: AdaptiveFeedbackLoop,
    device,
    batch_size: int = Config.BATCH_SIZE,
):
    model.train()
    total_loss = 0.0
    n_batches = 0

    # Shuffle interactions
    idx = np.random.permutation(len(train_df))
    users_arr = train_df["user_id"].values[idx]
    items_arr = train_df["item_id"].values[idx]
    ratings_arr = train_df["rating"].values[idx].astype(np.float32)
    qol_arr = train_df[["qol_delta"]].values[idx].astype(np.float32)

    x_dict = {k: v.to(device) for k, v in graph.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}

    for start in range(0, len(users_arr), batch_size):
        end = start + batch_size
        u_batch = torch.tensor(users_arr[start:end], dtype=torch.long, device=device)
        i_batch = torch.tensor(items_arr[start:end], dtype=torch.long, device=device)
        r_batch = torch.tensor(ratings_arr[start:end], device=device)

        # QoL context — ต้องส่งครบทุก user (num_users, qol_dim) ไม่ใช่แค่ batch
        all_user_ids = torch.arange(graph["user"].num_nodes, device="cpu")
        qol_context_full = feedback_loop.get_context(all_user_ids).to(device)

        optimizer.zero_grad()
        user_emb, item_emb = model(x_dict, edge_index_dict, qol_context_full)
        preds = model.predict(user_emb, item_emb, u_batch, i_batch)
        loss = criterion(preds, r_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Update QoL state (broadcast single delta to 4 domains)
        delta_1d = torch.tensor(qol_arr[start:end], device="cpu").reshape(-1)
        qol_delta_4d = delta_1d.unsqueeze(1).expand(-1, Config.QOL_DIM).contiguous()
        feedback_loop.update(u_batch.cpu(), qol_delta_4d)

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def predict_all(
    model: BiLevelGAT,
    graph,
    test_df: pd.DataFrame,
    feedback_loop: AdaptiveFeedbackLoop,
    device,
):
    model.eval()
    x_dict = {k: v.to(device) for k, v in graph.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}

    all_users = torch.arange(graph["user"].num_nodes, device=device)
    qol_context = feedback_loop.get_context(all_users).to(device)
    user_emb, item_emb = model(x_dict, edge_index_dict, qol_context)

    u = torch.tensor(test_df["user_id"].values, dtype=torch.long, device=device)
    i = torch.tensor(test_df["item_id"].values, dtype=torch.long, device=device)
    preds = model.predict(user_emb, item_emb, u, i)
    return preds.cpu().numpy()


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    device = get_device()
    Path(Config.MODEL_DIR).mkdir(exist_ok=True)
    Path(Config.RESULTS_DIR).mkdir(exist_ok=True)

    # ── Stage 1: Data & Feature Engineering ──────────────────────────────────
    print("\n" + "=" * 55)
    print("  Stage 1: Data Collection & Feature Engineering")
    print("=" * 55)

    data_dir = Config.DATA_DIR
    if not Path(f"{data_dir}/train.csv").exists():
        save_data(data_dir)

    train_graph, full_graph, users_df, items_df, train_df, test_df, content_emb = \
        load_and_build(data_dir)

    train_graph = train_graph.to(device)

    print(f"\nUser features: {train_graph['user'].x.shape}")
    print(f"Item features: {train_graph['item'].x.shape}")
    print(f"Social edges:  {train_graph['user', 'social', 'user'].edge_index.shape[1]}")
    print(f"Train interactions: {len(train_df)}")
    print(f"Test  interactions: {len(test_df)}")

    # ── Stage 2: Model Development ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Stage 2: Bi-Level GAT Training")
    print("=" * 55)

    user_in_dim = train_graph["user"].x.shape[1]
    item_in_dim = train_graph["item"].x.shape[1]

    model = BiLevelGAT(
        user_in_dim=user_in_dim,
        item_in_dim=item_in_dim,
        hidden_dim=Config.GAT_HIDDEN_DIM,
        out_dim=Config.GAT_OUTPUT_DIM,
        heads=Config.GAT_HEADS,
        dropout=Config.DROPOUT,
        qol_dim=Config.QOL_DIM,
        qol_alpha=Config.QOL_ALPHA,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    criterion = nn.MSELoss()

    feedback_loop = AdaptiveFeedbackLoop(
        num_users=len(users_df), qol_dim=Config.QOL_DIM
    )

    best_rmse = float("inf")
    history = []

    for epoch in range(1, Config.EPOCHS + 1):
        train_loss = train_epoch(
            model, train_graph, train_df, optimizer, criterion, feedback_loop, device
        )
        scheduler.step()

        if epoch % 10 == 0 or epoch == 1:
            preds = predict_all(model, train_graph, test_df, feedback_loop, device)
            y_true = test_df["rating"].values
            metrics = evaluate_all(
                test_df["user_id"].values, test_df["item_id"].values,
                y_true, preds, ks=Config.TOP_K,
            )
            history.append({"epoch": epoch, "loss": train_loss, **metrics})
            print(f"Epoch {epoch:3d} | Loss: {train_loss:.4f} | "
                  f"RMSE: {metrics['RMSE']:.4f} | "
                  f"NDCG@10: {metrics['NDCG@10']:.4f} | "
                  f"HR@10: {metrics['HR@10']:.4f}")

            if metrics["RMSE"] < best_rmse:
                best_rmse = metrics["RMSE"]
                torch.save(model.state_dict(), f"{Config.MODEL_DIR}/best_model.pt")

    save_results(history, f"{Config.RESULTS_DIR}/training_history.json")

    # Final evaluation
    model.load_state_dict(torch.load(f"{Config.MODEL_DIR}/best_model.pt",
                                      map_location=device))
    preds = predict_all(model, train_graph, test_df, feedback_loop, device)
    y_true = test_df["rating"].values
    final_metrics = evaluate_all(
        test_df["user_id"].values, test_df["item_id"].values,
        y_true, preds, ks=Config.TOP_K,
    )
    save_results(final_metrics, f"{Config.RESULTS_DIR}/final_metrics.json")

    print("\n── Final Test Metrics ──────────────────────────────")
    for k, v in final_metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── Stage 3: XAI ─────────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  Stage 3: XAI — Attention Weight Analysis")
    print("=" * 55)

    attn_data = extract_attention_weights(model, train_graph, device=str(device))
    plot_attention_heatmap(attn_data, save_path=f"{Config.RESULTS_DIR}/attention_heatmap.png")
    plot_top_influential_friends(
        user_id=0, attention_data=attn_data, users_df=users_df,
        save_path=f"{Config.RESULTS_DIR}/top_friends_user0.png",
    )

    # Sample explanation
    explanation = explain_recommendation(
        user_id=0, item_id=5, attention_data=attn_data,
        users_df=users_df, items_df=items_df,
    )
    print("\nSample XAI Explanation:")
    for reason in explanation["reasons"]:
        print(f"  [{reason['type']}] {reason['message']}")

    # ── Stage 3: User Study & QoL Analysis ───────────────────────────────────
    print("\n" + "=" * 55)
    print("  Stage 3: User Study — QoL Impact Analysis")
    print("=" * 55)

    study_df = simulate_user_study(
        n_participants=Config.NUM_PARTICIPANTS, weeks=Config.STUDY_WEEKS
    )
    study_df.to_csv(f"{Config.RESULTS_DIR}/user_study_data.csv", index=False)
    stats_results = run_statistical_tests(study_df)
    print_report(stats_results)
    save_results(stats_results, f"{Config.RESULTS_DIR}/qol_stats.json")
    plot_pre_post_comparison(study_df, f"{Config.RESULTS_DIR}/qol_pre_post.png")

    print("\n✓ Pipeline complete. Results saved to:", Config.RESULTS_DIR)


if __name__ == "__main__":
    main()
