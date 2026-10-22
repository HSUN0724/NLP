"""SQLite 資料庫存取。"""
import os
import sqlite3

from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS news (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,  -- 流水號
    title        TEXT NOT NULL,                      -- 新聞標題
    url          TEXT NOT NULL UNIQUE,               -- 新聞連結
    category     TEXT,                               -- 新聞分類
    author       TEXT,                               -- 記者／來源
    content      TEXT NOT NULL,                      -- 文章全文
    summary      TEXT,                               -- 文章摘要（地端模型產生）
    published_at TEXT,                               -- 文章新增時間（網站發布時間）
    crawled_at   TEXT DEFAULT (datetime('now', 'localtime'))  -- 寫入資料庫時間
);
"""


def get_conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def get_news_by_ids(conn, ids):
    if not ids:
        return {}
    marks = ",".join("?" * len(ids))
    rows = conn.execute(f"SELECT * FROM news WHERE id IN ({marks})", ids).fetchall()
    return {r["id"]: dict(r) for r in rows}
