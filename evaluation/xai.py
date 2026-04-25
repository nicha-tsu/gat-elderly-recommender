"""
Explainability via GAT Attention Weights (XAI — Stage 3, Objective 2)

Extracts and visualises attention scores to answer:
  "Why did the system recommend this activity?"
  e.g. "Because friend B (high QoL) engages with similar content."
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from pathlib import Path


def extract_attention_weights(model, graph, device="cpu"):
    """
    Run a forward pass with return_attention_weights=True on each GAT layer
    and collect (edge_index, alpha) tuples per relation type.

    Returns dict: relation_type -> (edge_index, attention_weights)
    """
    import torch
    model.eval()

    attention_data = {}

    with torch.no_grad():
        x_dict = {k: v.to(device) for k, v in graph.x_dict.items()}
        edge_index_dict = {k: v.to(device) for k, v in graph.edge_index_dict.items()}

        h_user = model.user_proj(x_dict["user"])
        h_item = model.item_proj(x_dict["item"])
        h = {"user": h_user, "item": h_item}

        # Extract from Layer 1
        for rel, conv in model.conv1.convs.items():
            src_type, edge_type, dst_type = rel
            if src_type == dst_type:
                x_pair = h[src_type]
            else:
                x_pair = (h[src_type], h[dst_type])

            edge_index = edge_index_dict.get(rel)
            if edge_index is None:
                continue

            try:
                _, (ei, alpha) = conv(x_pair, edge_index,
                                      return_attention_weights=True)
                attention_data[edge_type] = (
                    ei.cpu().numpy(),
                    alpha.cpu().numpy().mean(axis=-1),  # average over heads
                )
            except Exception:
                pass

    return attention_data


def explain_recommendation(
    user_id: int,
    item_id: int,
    attention_data: dict,
    users_df: pd.DataFrame,
    items_df: pd.DataFrame,
    top_n: int = 5,
) -> dict:
    """
    For a given (user, item) pair, return:
      - Top social friends influencing the recommendation
      - Top item features driving the score
    """
    explanation = {"user_id": user_id, "item_id": item_id, "reasons": []}

    # Social Attention — which friends influenced this user?
    if "social" in attention_data:
        ei, alpha = attention_data["social"]
        mask = ei[1] == user_id              # edges targeting this user
        if mask.any():
            src_ids = ei[0][mask]
            weights = alpha[mask]
            top_idx = np.argsort(weights)[::-1][:top_n]
            for rank, idx in enumerate(top_idx):
                friend_id = int(src_ids[idx])
                w = float(weights[idx])
                friend_row = users_df[users_df["user_id"] == friend_id].iloc[0]
                explanation["reasons"].append({
                    "type": "social_influence",
                    "rank": rank + 1,
                    "friend_id": friend_id,
                    "attention_weight": round(w, 4),
                    "friend_age": int(friend_row["age"]),
                    "friend_qol": round(
                        float(friend_row[["qol_physical", "qol_psychological",
                                          "qol_social", "qol_environment"]].mean()), 1
                    ),
                    "message": (
                        f"เพื่อนผู้ใช้ #{friend_id} (อายุ {friend_row['age']} ปี, "
                        f"QoL={round(float(friend_row[['qol_physical','qol_psychological','qol_social','qol_environment']].mean()),1)}) "
                        f"มีอิทธิพลสูง (w={w:.3f})"
                    ),
                })

    # Item Attention — why this specific activity?
    if "interacts" in attention_data:
        ei, alpha = attention_data["interacts"]
        mask = (ei[0] == user_id)
        if mask.any():
            item_ids = ei[1][mask]
            weights = alpha[mask]
            # Find the target item
            target_mask = item_ids == item_id
            if target_mask.any():
                w = float(weights[target_mask].mean())
                item_row = items_df[items_df["item_id"] == item_id].iloc[0]
                explanation["reasons"].append({
                    "type": "item_content",
                    "item_id": item_id,
                    "attention_weight": round(w, 4),
                    "category": item_row["category"],
                    "cognitive_benefit": round(float(item_row["cognitive_benefit"]), 2),
                    "message": (
                        f"กิจกรรม '{item_row['title']}' (ประเภท: {item_row['category']}, "
                        f"ประโยชน์สมอง={round(float(item_row['cognitive_benefit']),2)}) "
                        f"ตรงกับความสนใจของผู้ใช้ (w={w:.3f})"
                    ),
                })

    return explanation


def plot_attention_heatmap(
    attention_data: dict,
    n_users: int = 20,
    save_path: str = "results/attention_heatmap.png",
):
    """Visualise social attention weights as a heatmap (first N users)."""
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)

    if "social" not in attention_data:
        print("No social attention data available.")
        return

    ei, alpha = attention_data["social"]
    mask = (ei[0] < n_users) & (ei[1] < n_users)
    matrix = np.zeros((n_users, n_users))
    for s, d, w in zip(ei[0][mask], ei[1][mask], alpha[mask]):
        matrix[s, d] = w

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="Blues", aspect="auto")
    ax.set_xlabel("Target User ID", fontsize=12)
    ax.set_ylabel("Source User ID", fontsize=12)
    ax.set_title("Social Attention Weights\n(ค่าความสนใจเชิงสังคมระหว่างผู้ใช้)", fontsize=13)
    plt.colorbar(im, ax=ax, label="Attention Weight")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved attention heatmap → {save_path}")


def plot_top_influential_friends(
    user_id: int,
    attention_data: dict,
    users_df: pd.DataFrame,
    top_n: int = 10,
    save_path: str = "results/top_friends.png",
):
    Path(save_path).parent.mkdir(exist_ok=True, parents=True)

    if "social" not in attention_data:
        return

    ei, alpha = attention_data["social"]
    mask = ei[1] == user_id
    if not mask.any():
        print(f"No social edges found for user {user_id}")
        return

    src_ids = ei[0][mask]
    weights = alpha[mask]
    top_idx = np.argsort(weights)[::-1][:top_n]

    friend_ids = src_ids[top_idx]
    friend_weights = weights[top_idx]
    labels = [f"User #{fid}" for fid in friend_ids]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(labels[::-1], friend_weights[::-1], color="steelblue")
    ax.set_xlabel("Attention Weight", fontsize=12)
    ax.set_title(
        f"Top {top_n} Influential Friends for User #{user_id}\n"
        "(เพื่อนที่มีอิทธิพลต่อคำแนะนำสูงสุด)",
        fontsize=12,
    )
    for bar, w in zip(bars[::-1], friend_weights[::-1]):
        ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{w:.3f}", va="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved top friends chart → {save_path}")
