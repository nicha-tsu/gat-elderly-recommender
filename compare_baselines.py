"""
Baseline Comparison Experiment
================================
เปรียบเทียบ 3 โมเดล ใน task เดียวกัน:
  1. MF (Matrix Factorization)
  2. GCN (Graph Convolutional Network)
  3. Bi-Level GAT (เรา)

Output:
  - results/baseline_comparison.json — ตัวเลขทั้งหมด
  - results/baseline_comparison.png — กราฟเปรียบเทียบ
"""

import sys
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")

sys.path.insert(0, str(Path(__file__).parent))

from config import Config
from data.preprocessing import load_and_build
from models.bilevel_gat import BiLevelGAT, AdaptiveFeedbackLoop
from baselines.mf import MatrixFactorization
from baselines.gcn import GCNRecommender
from evaluation.metrics import evaluate_all


torch.manual_seed(Config.SEED)
np.random.seed(Config.SEED)

EPOCHS = 50
BATCH_SIZE = 512


# ── Trainers ─────────────────────────────────────────────────────────────────

def train_mf(train_df, test_df, num_users, num_items, device):
    print("\n[1/3] 🧮 Training Matrix Factorization...")
    model = MatrixFactorization(num_users, num_items, latent_dim=32).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-4)
    criterion = nn.MSELoss()

    u = torch.tensor(train_df["user_id"].values, dtype=torch.long, device=device)
    i = torch.tensor(train_df["item_id"].values, dtype=torch.long, device=device)
    r = torch.tensor(train_df["rating"].values.astype(np.float32), device=device)

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(u), device=device)
        total_loss = 0
        for s in range(0, len(u), BATCH_SIZE):
            b = idx[s:s + BATCH_SIZE]
            opt.zero_grad()
            pred = model(u[b], i[b])
            loss = criterion(pred, r[b])
            loss.backward()
            opt.step()
            total_loss += loss.item()
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1:3d}/{EPOCHS} | Loss: {total_loss / (len(u)//BATCH_SIZE):.4f}")
    train_time = time.time() - t0

    # Evaluate
    model.eval()
    with torch.no_grad():
        ut = torch.tensor(test_df["user_id"].values, dtype=torch.long, device=device)
        it = torch.tensor(test_df["item_id"].values, dtype=torch.long, device=device)
        preds = model(ut, it).cpu().numpy()
    metrics = evaluate_all(
        test_df["user_id"].values, test_df["item_id"].values,
        test_df["rating"].values, preds, ks=Config.TOP_K,
    )
    metrics["train_time_sec"] = round(train_time, 2)
    metrics["params"] = sum(p.numel() for p in model.parameters())
    return metrics


def train_gcn(graph, train_df, test_df, device):
    print("\n[2/3] 🕸 Training GCN...")
    model = GCNRecommender(
        user_in_dim=graph["user"].x.shape[1],
        item_in_dim=graph["item"].x.shape[1],
        hidden_dim=Config.GAT_HIDDEN_DIM,
        out_dim=Config.GAT_OUTPUT_DIM,
        dropout=Config.DROPOUT,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=Config.LR,
                           weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.MSELoss()

    x_dict = {k: v.to(device) for k, v in graph.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}

    u_train = torch.tensor(train_df["user_id"].values, dtype=torch.long, device=device)
    i_train = torch.tensor(train_df["item_id"].values, dtype=torch.long, device=device)
    r_train = torch.tensor(train_df["rating"].values.astype(np.float32), device=device)

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(u_train), device=device)
        total_loss = 0
        n_batches = 0
        for s in range(0, len(u_train), BATCH_SIZE):
            b = idx[s:s + BATCH_SIZE]
            opt.zero_grad()
            user_emb, item_emb = model(x_dict, edge_index_dict)
            pred = model.predict(user_emb, item_emb, u_train[b], i_train[b])
            loss = criterion(pred, r_train[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()
            n_batches += 1
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1:3d}/{EPOCHS} | Loss: {total_loss / max(n_batches,1):.4f}")
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        user_emb, item_emb = model(x_dict, edge_index_dict)
        ut = torch.tensor(test_df["user_id"].values, dtype=torch.long, device=device)
        it = torch.tensor(test_df["item_id"].values, dtype=torch.long, device=device)
        preds = model.predict(user_emb, item_emb, ut, it).cpu().numpy()
    metrics = evaluate_all(
        test_df["user_id"].values, test_df["item_id"].values,
        test_df["rating"].values, preds, ks=Config.TOP_K,
    )
    metrics["train_time_sec"] = round(train_time, 2)
    metrics["params"] = sum(p.numel() for p in model.parameters())
    return metrics


def train_bilevel_gat(graph, train_df, test_df, num_users, device):
    print("\n[3/3] 🌟 Training Bi-Level GAT (Ours)...")
    model = BiLevelGAT(
        user_in_dim=graph["user"].x.shape[1],
        item_in_dim=graph["item"].x.shape[1],
        hidden_dim=Config.GAT_HIDDEN_DIM,
        out_dim=Config.GAT_OUTPUT_DIM,
        heads=Config.GAT_HEADS,
        dropout=Config.DROPOUT,
        qol_dim=Config.QOL_DIM,
        qol_alpha=Config.QOL_ALPHA,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=Config.LR,
                           weight_decay=Config.WEIGHT_DECAY)
    criterion = nn.MSELoss()
    feedback_loop = AdaptiveFeedbackLoop(num_users=num_users, qol_dim=Config.QOL_DIM)

    x_dict = {k: v.to(device) for k, v in graph.x_dict.items()}
    edge_index_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}

    u_train = torch.tensor(train_df["user_id"].values, dtype=torch.long, device=device)
    i_train = torch.tensor(train_df["item_id"].values, dtype=torch.long, device=device)
    r_train = torch.tensor(train_df["rating"].values.astype(np.float32), device=device)
    qol_train = torch.tensor(train_df["qol_delta"].values.astype(np.float32), device=device)

    t0 = time.time()
    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(u_train), device=device)
        total_loss = 0
        n_batches = 0
        for s in range(0, len(u_train), BATCH_SIZE):
            b = idx[s:s + BATCH_SIZE]
            all_uids = torch.arange(num_users)
            qol_full = feedback_loop.get_context(all_uids).to(device)

            opt.zero_grad()
            user_emb, item_emb = model(x_dict, edge_index_dict, qol_full)
            pred = model.predict(user_emb, item_emb, u_train[b], i_train[b])
            loss = criterion(pred, r_train[b])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            delta_4d = qol_train[b].unsqueeze(1).expand(-1, Config.QOL_DIM).contiguous()
            feedback_loop.update(u_train[b].cpu(), delta_4d.cpu())
            total_loss += loss.item()
            n_batches += 1
        if (ep + 1) % 10 == 0:
            print(f"  Epoch {ep+1:3d}/{EPOCHS} | Loss: {total_loss / max(n_batches,1):.4f}")
    train_time = time.time() - t0

    model.eval()
    with torch.no_grad():
        all_uids = torch.arange(num_users)
        qol_full = feedback_loop.get_context(all_uids).to(device)
        user_emb, item_emb = model(x_dict, edge_index_dict, qol_full)
        ut = torch.tensor(test_df["user_id"].values, dtype=torch.long, device=device)
        it = torch.tensor(test_df["item_id"].values, dtype=torch.long, device=device)
        preds = model.predict(user_emb, item_emb, ut, it).cpu().numpy()
    metrics = evaluate_all(
        test_df["user_id"].values, test_df["item_id"].values,
        test_df["rating"].values, preds, ks=Config.TOP_K,
    )
    metrics["train_time_sec"] = round(train_time, 2)
    metrics["params"] = sum(p.numel() for p in model.parameters())
    return metrics


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_comparison(results: dict, save_path: str):
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)
    models = list(results.keys())
    metric_keys = ["RMSE", "NDCG@5", "NDCG@10", "NDCG@20", "HR@10"]
    colors = {"MF": "#94a3b8", "GCN": "#60a5fa", "Bi-Level GAT": "#10b981"}

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for idx, mk in enumerate(metric_keys):
        ax = axes[idx]
        values = [results[m][mk] for m in models]
        bars = ax.bar(models, values, color=[colors[m] for m in models], edgecolor="black", linewidth=0.8)
        ax.set_title(mk, fontsize=14, fontweight="bold")
        ax.set_ylabel("Score" if "NDCG" in mk or "HR" in mk else "Error")
        if "NDCG" in mk or "HR" in mk:
            ax.set_ylim(0, max(values) * 1.15)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    # Last subplot — params + train time
    ax = axes[5]
    params = [results[m]["params"] / 1000 for m in models]
    times = [results[m]["train_time_sec"] for m in models]
    x = np.arange(len(models))
    w = 0.35
    ax.bar(x - w/2, params, w, label="Params (K)", color="#fbbf24", edgecolor="black")
    ax.bar(x + w/2, times, w, label="Train time (s)", color="#ef4444", edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(models)
    ax.set_title("Model Complexity & Training Time", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.suptitle("Baseline Comparison — Bi-Level GAT vs MF vs GCN",
                 fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📊 Saved comparison chart → {save_path}")


def print_summary_table(results: dict):
    print("\n" + "=" * 78)
    print(f"{'Model':<18} {'RMSE↓':>8} {'NDCG@5↑':>9} {'NDCG@10↑':>10} "
          f"{'NDCG@20↑':>10} {'HR@10↑':>8} {'Params':>10}")
    print("-" * 78)
    for m, r in results.items():
        print(f"{m:<18} {r['RMSE']:>8.4f} {r['NDCG@5']:>9.4f} "
              f"{r['NDCG@10']:>10.4f} {r['NDCG@20']:>10.4f} "
              f"{r['HR@10']:>8.4f} {r['params']:>10,}")
    print("=" * 78)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading data...")
    graph, _, users_df, items_df, train_df, test_df, _ = load_and_build(Config.DATA_DIR)
    graph = graph.to(device)
    n_users, n_items = len(users_df), len(items_df)

    results = {}
    results["MF"] = train_mf(train_df, test_df, n_users, n_items, device)
    results["GCN"] = train_gcn(graph, train_df, test_df, device)
    results["Bi-Level GAT"] = train_bilevel_gat(graph, train_df, test_df, n_users, device)

    Path(Config.RESULTS_DIR).mkdir(exist_ok=True)
    with open(f"{Config.RESULTS_DIR}/baseline_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    print_summary_table(results)
    plot_comparison(results, f"{Config.RESULTS_DIR}/baseline_comparison.png")

    # Improvement analysis
    print("\n📈 Improvement of Bi-Level GAT over baselines:")
    gat = results["Bi-Level GAT"]
    for baseline in ["MF", "GCN"]:
        b = results[baseline]
        rmse_imp = (b["RMSE"] - gat["RMSE"]) / b["RMSE"] * 100
        ndcg_imp = (gat["NDCG@10"] - b["NDCG@10"]) / b["NDCG@10"] * 100
        print(f"  vs {baseline}: RMSE ↓{rmse_imp:+.1f}%  |  NDCG@10 ↑{ndcg_imp:+.1f}%")

    print(f"\n✓ Results saved to: {Config.RESULTS_DIR}/baseline_comparison.json")
    print(f"✓ Chart saved to:   {Config.RESULTS_DIR}/baseline_comparison.png")


if __name__ == "__main__":
    main()
