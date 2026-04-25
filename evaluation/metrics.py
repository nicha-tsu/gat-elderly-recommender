"""
Evaluation Metrics — RMSE and NDCG@K
(Section 2, Stage 3 of proposal)
"""

import numpy as np
import torch


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def ndcg_at_k(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compute mean NDCG@K across all users.
    Treats ratings >= 4 as relevant items.
    """
    ndcgs = []
    for uid in np.unique(user_ids):
        mask = user_ids == uid
        true_u = y_true[mask]
        pred_u = y_pred[mask]

        if len(true_u) == 0:
            continue

        # Rank by predicted score
        order = np.argsort(pred_u)[::-1][:k]
        relevance = (true_u[order] >= 4).astype(float)

        # Ideal ranking
        ideal = np.sort(true_u)[::-1][:k]
        ideal_rel = (ideal >= 4).astype(float)

        # DCG
        positions = np.arange(1, len(relevance) + 1)
        dcg = np.sum(relevance / np.log2(positions + 1))
        idcg = np.sum(ideal_rel / np.log2(np.arange(1, len(ideal_rel) + 1) + 1))

        ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

    return float(np.mean(ndcgs)) if ndcgs else 0.0


def hit_rate_at_k(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: int = 10,
) -> float:
    hits = []
    for uid in np.unique(user_ids):
        mask = user_ids == uid
        true_u = y_true[mask]
        pred_u = y_pred[mask]
        top_k = np.argsort(pred_u)[::-1][:k]
        hits.append(int(np.any(true_u[top_k] >= 4)))
    return float(np.mean(hits)) if hits else 0.0


def evaluate_all(
    user_ids: np.ndarray,
    item_ids: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ks=(5, 10, 20),
) -> dict:
    results = {"RMSE": rmse(y_true, y_pred)}
    for k in ks:
        results[f"NDCG@{k}"] = ndcg_at_k(user_ids, item_ids, y_true, y_pred, k)
        results[f"HR@{k}"] = hit_rate_at_k(user_ids, item_ids, y_true, y_pred, k)
    return results
