"""
Feature Engineering & Graph Construction (Phase 1 in proposal)
"""

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData
from pathlib import Path


def get_content_embeddings(items: pd.DataFrame, embed_dim: int = 384) -> np.ndarray:
    # 1) Load pre-computed cache (ใช้ใน production บน Render)
    cache_path = Path("dataset") / "content_embeddings.npy"
    if cache_path.exists():
        embeddings = np.load(cache_path)
        if embeddings.shape[0] == len(items):
            print(f"[CACHE] Loaded SBERT embeddings: {embeddings.shape}")
            return embeddings.astype(np.float32)

    # 2) Try sentence-transformers (development)
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        texts = (items["title"] + " " + items["category"]).tolist()
        embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
        print(f"[SBERT] Content embeddings: {embeddings.shape}")
        return embeddings.astype(np.float32)
    except ImportError:
        print("[WARN] sentence-transformers not installed — using random projection surrogate.")
        rng = np.random.default_rng(42)
        raw = rng.standard_normal((len(items), embed_dim)).astype(np.float32)
        norms = np.linalg.norm(raw, axis=1, keepdims=True)
        return raw / norms


def build_user_features(users: pd.DataFrame) -> np.ndarray:
    skill_map = {"ต่ำ": 0, "ปานกลาง": 1, "สูง": 2}
    health_map = {"ดี": 2, "ปานกลาง": 1, "ต้องดูแล": 0}
    gender_map = {"ชาย": 0, "หญิง": 1}

    feats = np.column_stack([
        (users["age"].values - 60) / 25,
        users["gender"].map(gender_map).values,
        users["tech_skill"].map(skill_map).values / 2,
        users["health_status"].map(health_map).values / 2,
        users[["qol_physical", "qol_psychological",
               "qol_social", "qol_environment"]].values / 100,
    ]).astype(np.float32)
    return feats


def build_hetero_graph(users, items, interactions, social, content_embeddings) -> HeteroData:
    data = HeteroData()

    user_feats = build_user_features(users)
    data["user"].x = torch.tensor(user_feats, dtype=torch.float)
    data["item"].x = torch.tensor(content_embeddings, dtype=torch.float)
    data["user"].num_nodes = len(users)
    data["item"].num_nodes = len(items)

    src = torch.tensor(interactions["user_id"].values, dtype=torch.long)
    dst = torch.tensor(interactions["item_id"].values, dtype=torch.long)
    data["user", "interacts", "item"].edge_index = torch.stack([src, dst])
    data["user", "interacts", "item"].edge_attr = torch.tensor(
        interactions[["rating", "watch_ratio", "qol_delta"]].values, dtype=torch.float,
    )
    data["item", "rev_interacts", "user"].edge_index = torch.stack([dst, src])

    s = torch.tensor(social["src"].values, dtype=torch.long)
    t = torch.tensor(social["dst"].values, dtype=torch.long)
    data["user", "social", "user"].edge_index = torch.stack([torch.cat([s, t]), torch.cat([t, s])])

    return data


def load_and_build(data_dir: str = "dataset"):
    users = pd.read_csv(f"{data_dir}/users.csv")
    items = pd.read_csv(f"{data_dir}/items.csv")
    train = pd.read_csv(f"{data_dir}/train.csv")
    test = pd.read_csv(f"{data_dir}/test.csv")
    social = pd.read_csv(f"{data_dir}/social_graph.csv")

    content_emb = get_content_embeddings(items)
    train_graph = build_hetero_graph(users, items, train, social, content_emb)

    print(f"Train graph: {train_graph}")
    return train_graph, None, users, items, train, test, content_emb


if __name__ == "__main__":
    from generate_data import save_data
    save_data()
    load_and_build()
