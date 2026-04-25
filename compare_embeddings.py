"""
เปรียบเทียบ Random vs Sentence-BERT embeddings
ดูว่ากิจกรรมที่ "ใกล้กันในความหมาย" มี embedding ใกล้กันจริงไหม
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data.preprocessing import get_content_embeddings
from config import Config


def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)


def find_similar(items, embeddings, query_idx, top_k=5):
    sims = []
    for i in range(len(items)):
        if i == query_idx:
            continue
        sim = cosine_similarity(embeddings[query_idx], embeddings[i])
        sims.append((i, sim))
    sims.sort(key=lambda x: x[1], reverse=True)
    return sims[:top_k]


def main():
    items = pd.read_csv(f"{Config.DATA_DIR}/items.csv")
    embeddings = get_content_embeddings(items)

    test_queries = [0, 5, 15, 25]

    print("\n" + "=" * 65)
    print("  ทดสอบ: หากิจกรรมที่ใกล้เคียงกันในเชิงความหมาย")
    print("=" * 65)

    for qi in test_queries:
        query = items.iloc[qi]
        print(f"\n🔍 Query: '{query['title']}' (ประเภท: {query['category']})")
        print("-" * 65)

        similar = find_similar(items, embeddings, qi, top_k=3)
        for rank, (idx, sim) in enumerate(similar, 1):
            target = items.iloc[idx]
            same_cat = "✓" if target["category"] == query["category"] else " "
            print(f"  [{rank}] {same_cat} sim={sim:.3f}  '{target['title']}' "
                  f"({target['category']})")

    print("\n" + "=" * 65)
    print("  ✓ หาก SBERT ทำงาน → กิจกรรมประเภทเดียวกันควรมี sim สูง")
    print("  ✗ หาก Random → ความใกล้เคียงไม่สัมพันธ์กับประเภท")
    print("=" * 65)


if __name__ == "__main__":
    main()
