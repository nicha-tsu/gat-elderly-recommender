"""
Bi-Level Graph Attention Network (Bi-Level GAT)
=================================================
Implements the core model from the proposal (Section 2, Stage 2):

  Level 1 — Social Attention:
      User ← User  (learn social influence weights among peers)

  Level 2 — Item Content Attention:
      User ← Item  (learn which activity features attract each user)

Both levels share the same GAT mechanism but operate on different
edge types of the heterogeneous graph.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, HeteroConv, Linear
from torch_geometric.data import HeteroData


class ContentProjection(nn.Module):
    """Project raw content embeddings (384-d SBERT) to GAT input dim."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim * 2),
            nn.LayerNorm(out_dim * 2),
            nn.GELU(),
            nn.Linear(out_dim * 2, out_dim),
        )

    def forward(self, x):
        return self.proj(x)


class UserProjection(nn.Module):
    """Project user demographic + QoL features to GAT input dim."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.proj(x)


class BiLevelGAT(nn.Module):
    """
    Two-layer heterogeneous GAT implementing Bi-Level Attention.

    Architecture:
        Input projections → Layer-1 GAT (social + item) →
        Layer-2 GAT (social + item) → User/Item embeddings
    """

    def __init__(
        self,
        user_in_dim: int,
        item_in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
        heads: int = 4,
        dropout: float = 0.3,
        qol_dim: int = 4,
        qol_alpha: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout
        self.qol_alpha = qol_alpha

        # Input projections
        self.user_proj = UserProjection(user_in_dim, hidden_dim)
        self.item_proj = ContentProjection(item_in_dim, hidden_dim)

        # Layer 1 — Bi-Level Attention
        self.conv1 = HeteroConv({
            ("user", "social", "user"): GATConv(
                hidden_dim, hidden_dim // heads, heads=heads,
                dropout=dropout, add_self_loops=False,
            ),
            ("user", "interacts", "item"): GATConv(
                (hidden_dim, hidden_dim), hidden_dim // heads,
                heads=heads, dropout=dropout, add_self_loops=False,
            ),
            ("item", "rev_interacts", "user"): GATConv(
                (hidden_dim, hidden_dim), hidden_dim // heads,
                heads=heads, dropout=dropout, add_self_loops=False,
            ),
        }, aggr="mean")

        # Layer 2 — deeper fusion
        self.conv2 = HeteroConv({
            ("user", "social", "user"): GATConv(
                hidden_dim, out_dim, heads=1,
                dropout=dropout, add_self_loops=False,
            ),
            ("user", "interacts", "item"): GATConv(
                (hidden_dim, hidden_dim), out_dim,
                heads=1, dropout=dropout, add_self_loops=False,
            ),
            ("item", "rev_interacts", "user"): GATConv(
                (hidden_dim, hidden_dim), out_dim,
                heads=1, dropout=dropout, add_self_loops=False,
            ),
        }, aggr="mean")

        # QoL Adaptive Feedback: inject QoL context into user embeddings
        self.qol_gate = nn.Sequential(
            nn.Linear(out_dim + qol_dim, out_dim),
            nn.Sigmoid(),
        )
        self.qol_proj = nn.Linear(qol_dim, out_dim)

        # Rating prediction head
        self.predictor = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )

        self._attention_weights = {}

    def forward(
        self,
        x_dict: dict,
        edge_index_dict: dict,
        qol_context: torch.Tensor | None = None,
    ):
        # Project inputs
        h = {
            "user": self.user_proj(x_dict["user"]),
            "item": self.item_proj(x_dict["item"]),
        }

        # Layer 1
        h1 = self.conv1(h, edge_index_dict)
        h1 = {k: F.elu(v) for k, v in h1.items()}
        h1 = {k: F.dropout(v, p=self.dropout, training=self.training)
              for k, v in h1.items()}

        # Residual skip
        h1["user"] = h1["user"] + F.linear(h["user"], torch.eye(
            h1["user"].shape[1], h["user"].shape[1],
            device=h["user"].device
        )) if h1["user"].shape[1] == h["user"].shape[1] else h1["user"]

        # Layer 2
        h2 = self.conv2(h1, edge_index_dict)
        h2 = {k: F.elu(v) for k, v in h2.items()}

        user_emb = h2["user"]
        item_emb = h2["item"]

        # QoL Adaptive Feedback Loop (Section 2, Stage 2)
        if qol_context is not None:
            qol_signal = self.qol_proj(qol_context)
            gate = self.qol_gate(torch.cat([user_emb, qol_context], dim=-1))
            user_emb = user_emb + self.qol_alpha * gate * qol_signal

        return user_emb, item_emb

    def predict(
        self,
        user_emb: torch.Tensor,
        item_emb: torch.Tensor,
        user_idx: torch.Tensor,
        item_idx: torch.Tensor,
    ) -> torch.Tensor:
        u = user_emb[user_idx]
        i = item_emb[item_idx]
        return self.predictor(torch.cat([u, i], dim=-1)).squeeze(-1)

    def get_attention_weights(self):
        """Return stored attention weights for XAI analysis."""
        return self._attention_weights


class AdaptiveFeedbackLoop:
    """
    Updates user QoL context after each interaction batch.
    The updated context is injected back into the next forward pass,
    implementing the Adaptive Feedback Loop from the proposal.
    """

    def __init__(self, num_users: int, qol_dim: int = 4, alpha: float = 0.1):
        self.qol_state = torch.zeros(num_users, qol_dim)
        self.alpha = alpha

    def update(self, user_ids: torch.Tensor, qol_deltas: torch.Tensor):
        """
        qol_deltas: (B, 4) — per-domain QoL change after interaction.
        Uses exponential moving average to smooth updates.
        """
        for uid, delta in zip(user_ids.cpu(), qol_deltas.cpu()):
            self.qol_state[uid] = (1 - self.alpha) * self.qol_state[uid] + self.alpha * delta

    def get_context(self, user_ids: torch.Tensor) -> torch.Tensor:
        return self.qol_state[user_ids]

    def to(self, device):
        self.qol_state = self.qol_state.to(device)
        return self
