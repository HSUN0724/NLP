"""步驟 3：把文章摘要轉成向量，儲存成向量索引（index/vectors.npy + index/ids.json）。

向量已做 L2 正規化，查詢時用內積即為餘弦相似度。
用法：
    python build_index.py
"""
import json
import os

import numpy as np

from config import IDS_PATH, INDEX_DIR, VECTORS_PATH
from db import get_conn
from llm import embed

BATCH = 16


def main():
    conn = get_conn()
    rows = conn.execute(
        "SELECT id, summary FROM news WHERE summary IS NOT NULL AND summary != '' ORDER BY id"
    ).fetchall()
    if not rows:
        print("沒有任何摘要，請先執行 summarize.py")
        return

    ids, vectors = [], []
    for start in range(0, len(rows), BATCH):
        batch = rows[start:start + BATCH]
        vectors.extend(embed([r["summary"] for r in batch]))
        ids.extend(r["id"] for r in batch)
        print(f"已轉換 {len(ids)}/{len(rows)}")

    mat = np.array(vectors, dtype=np.float32)
    mat /= np.linalg.norm(mat, axis=1, keepdims=True)

    os.makedirs(INDEX_DIR, exist_ok=True)
    np.save(VECTORS_PATH, mat)
    with open(IDS_PATH, "w", encoding="utf-8") as f:
        json.dump(ids, f)
    print(f"向量索引完成：{mat.shape[0]} 筆，維度 {mat.shape[1]}")


if __name__ == "__main__":
    main()
