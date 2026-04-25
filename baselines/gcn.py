"""
GCN Baseline — Graph Convolutional Network (using SAGEConv)
=============================================================
ใช้ GraphSAGE convolution (Hamilton et al., 2017) ซึ่งเป็น GCN-style
ที่รองรับ bipartite/heterogeneous graphs (GCNConv ไม่รองรับ tuple inputs)

ความแตกต่างจาก Bi-Level GAT คือ ไม่มี attention mechanism —
เพื่อนบ้านทุกคนมีน้ำหนักเท่ากัน (mean aggregation)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, HeteroConv


class GCNRecommender(nn.Module):
    """
    Heterogeneous GraphSAGE — ใช้ same edge types เหมือน Bi-Level GAT
    แต่ใช้ mean aggregation (ไม่มี attention)
    """

    def __init__(
        self,
        user_in_dim: int,
        item_in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 32,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.dropout = dropout
        self.user_proj = nn.Linear(user_in_dim, hidden_dim)
        self.item_proj = nn.Linear(item_in_dim, hidden_dim)

        self.conv1 = HeteroConv({
            ("user", "social", "user"): SAGEConv(hidden_dim, hidden_dim, aggr="mean"),
            ("user", "interacts", "item"): SAGEConv((hidden_dim, hidden_dim),
                                                     hidden_dim, aggr="mean"),
            ("item", "rev_interacts", "user"): SAGEConv((hidden_dim, hidden_dim),
                                                         hidden_dim, aggr="mean"),
        }, aggr="mean")

        self.conv2 = HeteroConv({
            ("user", "social", "user"): SAGEConv(hidden_dim, out_dim, aggr="mean"),
            ("user", "interacts", "item"): SAGEConv((hidden_dim, hidden_dim),
                                                     out_dim, aggr="mean"),
            ("item", "rev_interacts", "user"): SAGEConv((hidden_dim, hidden_dim),
                                                         out_dim, aggr="mean"),
        }, aggr="mean")

        self.predictor = nn.Sequential(
            nn.Linear(out_dim * 2, out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(out_dim, 1),
        )

    def forward(self, x_dict: dict, edge_index_dict: dict):
        h = {
            "user": self.user_proj(x_dict["user"]),
            "item": self.item_proj(x_dict["item"]),
        }
        h1 = self.conv1(h, edge_index_dict)
        h1 = {k: F.relu(v) for k, v in h1.items()}
        h1 = {k: F.dropout(v, p=self.dropout, training=self.training)
              for k, v in h1.items()}

        h2 = self.conv2(h1, edge_index_dict)
        h2 = {k: F.relu(v) for k, v in h2.items()}

        return h2["user"], h2["item"]

    def predict(self, user_emb, item_emb, user_idx, item_idx):
        u = user_emb[user_idx]
        i = item_emb[item_idx]
        return self.predictor(torch.cat([u, i], dim=-1)).squeeze(-1)
