"""
Synthetic data generation simulating TikTok elderly activity data
Mimics the dataset described in the proposal (Section 2, Phase 1)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import Config

ACTIVITY_CATEGORIES = [
    "ไทเก๊ก/โยคะ", "ทำอาหาร/ขนม", "งานฝีมือ/ศิลปะ", "ดนตรี/ร้องเพลง",
    "สวนครัว/ปลูกต้นไม้", "ทำบุญ/สมาธิ", "การเต้น/นาฏศิลป์", "ออกกำลังกาย",
    "เล่านิทาน/อ่านหนังสือ", "เทคโนโลยี/มือถือ",
]

ACTIVITY_NAMES = [
    "ท่าไทเก๊ก 24 ท่า", "โยคะผู้สูงวัยตอนเช้า", "สูตรขนมไทยโบราณ",
    "ทำแกงส้มปลาช่อน", "ปักผ้าลายดอกไม้", "งานจักสาน", "เล่นกีตาร์เพลงไทย",
    "ร้องเพลงลูกทุ่ง", "ปลูกผักสวนครัว", "ดูแลกล้วยไม้", "ทำบุญออนไลน์",
    "นั่งสมาธิ 15 นาที", "รำวงมาตรฐาน", "ลีลาศผู้สูงวัย", "เดินออกกำลังกาย",
    "ยืดเส้นยืดสาย", "อ่านนวนิยาย", "เล่านิทานหลาน", "ใช้ LINE วิดีโอคอล",
    "สอนใช้สมาร์ทโฟน", "ทำน้ำพริก", "ทอดกล้วยแขก", "วาดภาพสีน้ำ",
    "ประดิษฐ์ดอกไม้ผ้า", "เล่นขิม", "ตีกลองสงกรานต์", "ออกกำลังกายในน้ำ",
    "กายบริหารเก้าอี้", "อ่านธรรมะ", "ปฏิบัติธรรมวันสำคัญ",
]

np.random.seed(Config.SEED)


def generate_users(n=Config.NUM_USERS):
    ages = np.random.randint(60, 85, n)
    genders = np.random.choice(["ชาย", "หญิง"], n, p=[0.4, 0.6])
    tech_skills = np.random.choice(["ต่ำ", "ปานกลาง", "สูง"], n, p=[0.5, 0.35, 0.15])
    health_status = np.random.choice(["ดี", "ปานกลาง", "ต้องดูแล"], n, p=[0.4, 0.4, 0.2])

    qol_physical = np.random.uniform(40, 80, n)
    qol_psychological = np.random.uniform(40, 80, n)
    qol_social = np.random.uniform(30, 75, n)
    qol_environment = np.random.uniform(45, 85, n)

    return pd.DataFrame({
        "user_id": range(n),
        "age": ages,
        "gender": genders,
        "tech_skill": tech_skills,
        "health_status": health_status,
        "qol_physical": qol_physical,
        "qol_psychological": qol_psychological,
        "qol_social": qol_social,
        "qol_environment": qol_environment,
    })


def generate_items(n=Config.NUM_ITEMS):
    np.random.seed(Config.SEED + 1)
    items = []
    for i in range(n):
        cat = np.random.choice(ACTIVITY_CATEGORIES)
        name = np.random.choice(ACTIVITY_NAMES)
        items.append({
            "item_id": i,
            "title": f"{name} #{i}",
            "category": cat,
            "duration_min": np.random.choice([5, 10, 15, 20, 30]),
            "difficulty": np.random.choice(["ง่าย", "ปานกลาง", "ยาก"], p=[0.5, 0.35, 0.15]),
            "views": np.random.randint(100, 50000),
            "likes": np.random.randint(10, 5000),
            "cognitive_benefit": np.random.uniform(0.3, 1.0),
            "social_benefit": np.random.uniform(0.2, 1.0),
        })
    return pd.DataFrame(items)


def generate_interactions(users, items, n=Config.NUM_INTERACTIONS):
    np.random.seed(Config.SEED + 2)
    timestamps = np.sort(np.random.randint(0, 365 * 24 * 3600, n))

    skill_weights = users["tech_skill"].map({"ต่ำ": 1, "ปานกลาง": 2, "สูง": 3}).values
    user_probs = skill_weights / skill_weights.sum()
    user_ids = np.random.choice(users["user_id"], n, p=user_probs)

    item_weights = np.log1p(items["views"].values)
    item_probs = item_weights / item_weights.sum()
    item_ids = np.random.choice(items["item_id"], n, p=item_probs)

    ratings = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.1, 0.2, 0.35, 0.3])
    qol_delta = np.clip(np.random.normal(0.5, 0.8, n), -2.0, 5.0)

    return pd.DataFrame({
        "user_id": user_ids,
        "item_id": item_ids,
        "rating": ratings,
        "timestamp": timestamps,
        "watch_ratio": np.random.uniform(0.2, 1.0, n),
        "liked": (ratings >= 4).astype(int),
        "qol_delta": qol_delta,
    })


def generate_social_graph(users, n_edges=Config.NUM_SOCIAL_EDGES):
    np.random.seed(Config.SEED + 3)
    edges = set()
    n = len(users)
    ages = users["age"].values
    for _ in range(n_edges * 3):
        u = np.random.randint(0, n)
        candidates = np.where(np.abs(ages - ages[u]) <= 5)[0]
        candidates = candidates[candidates != u]
        if len(candidates) == 0:
            continue
        v = np.random.choice(candidates)
        if (u, v) not in edges and (v, u) not in edges:
            edges.add((u, v))
        if len(edges) >= n_edges:
            break
    src, dst = zip(*edges) if edges else ([], [])
    return pd.DataFrame({"src": list(src), "dst": list(dst)})


def temporal_split(interactions, train_ratio=Config.TRAIN_RATIO):
    interactions = interactions.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(interactions) * train_ratio)
    return interactions.iloc[:split_idx].copy(), interactions.iloc[split_idx:].copy()


def save_data(output_dir=Config.DATA_DIR):
    Path(output_dir).mkdir(exist_ok=True)
    users = generate_users()
    items = generate_items()
    interactions = generate_interactions(users, items)
    social = generate_social_graph(users)
    train, test = temporal_split(interactions)

    users.to_csv(f"{output_dir}/users.csv", index=False)
    items.to_csv(f"{output_dir}/items.csv", index=False)
    interactions.to_csv(f"{output_dir}/interactions.csv", index=False)
    social.to_csv(f"{output_dir}/social_graph.csv", index=False)
    train.to_csv(f"{output_dir}/train.csv", index=False)
    test.to_csv(f"{output_dir}/test.csv", index=False)

    print(f"Users:        {len(users)}")
    print(f"Items:        {len(items)}")
    print(f"Interactions: {len(interactions)}")
    print(f"Social edges: {len(social)}")
    print(f"Train/Test:   {len(train)}/{len(test)}")
    return users, items, interactions, social, train, test


if __name__ == "__main__":
    save_data()
