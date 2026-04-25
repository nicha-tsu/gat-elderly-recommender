"""
Matrix Factorization (Collaborative Filtering) Baseline
=========================================================
Classic latent-factor model — ไม่ใช้ social network และไม่ใช้ content features
ใช้เฉพาะ User-Item interaction matrix
"""

import torch
import torch.nn as nn


class MatrixFactorization(nn.Module):
    """
    Standard Matrix Factorization with bias terms.
    rating_ui ≈ p_u · q_i + b_u + b_i + μ
    """

    def __init__(self, num_users: int, num_items: int, latent_dim: int = 32):
        super().__init__()
        self.user_emb = nn.Embedding(num_users, latent_dim)
        self.item_emb = nn.Embedding(num_items, latent_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        self.item_bias = nn.Embedding(num_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))

        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.item_emb.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)

    def forward(self, user_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        u = self.user_emb(user_ids)
        i = self.item_emb(item_ids)
        bu = self.user_bias(user_ids).squeeze(-1)
        bi = self.item_bias(item_ids).squeeze(-1)
        return (u * i).sum(dim=-1) + bu + bi + self.global_bias
