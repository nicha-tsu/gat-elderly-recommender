"""
Pre-compute SBERT embeddings และบันทึกเป็นไฟล์
จะถูกโหลดใน production แทนการเรียก SBERT (ลด RAM)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from config import Config
from data.preprocessing import get_content_embeddings


def main():
    items_path = Path(Config.DATA_DIR) / "items.csv"
    if not items_path.exists():
        print(f"⚠ ไม่พบ {items_path} — รัน python train.py ก่อน")
        return

    items = pd.read_csv(items_path)
    embeddings = get_content_embeddings(items)

    out_path = Path(Config.DATA_DIR) / "content_embeddings.npy"
    np.save(out_path, embeddings)
    print(f"✓ บันทึก embeddings: {out_path}")
    print(f"  Shape: {embeddings.shape}, dtype: {embeddings.dtype}")
    print(f"  ขนาด: {out_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
