"""步驟 2：把「新聞標題 + 文章全文」交給地端模型產生文章摘要，寫回資料庫。

只處理 summary 為空的文章，可中斷後重跑。
用法：
    python summarize.py
    python summarize.py --limit 10   # 只先摘要 10 篇（測試用）
"""
import argparse

from db import get_conn
from llm import chat

MAX_CHARS = 4000  # 全文過長時截斷，避免超出模型上下文

PROMPT = (
    "你是專業的健康醫療新聞編輯。請閱讀以下新聞，"
    "用繁體中文寫出 150～250 字的摘要，需包含：主題、關鍵數據或研究發現、"
    "專家建議或民眾應注意事項。只輸出摘要本身，不要加標題或前言。\n\n"
    "{text}"
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=0, help="最多處理幾篇，0 表示全部")
    args = parser.parse_args()

    conn = get_conn()
    sql = "SELECT id, title, content FROM news WHERE summary IS NULL OR summary = '' ORDER BY id"
    if args.limit:
        sql += f" LIMIT {args.limit}"
    rows = conn.execute(sql).fetchall()
    print(f"待摘要文章：{len(rows)} 篇")

    for i, r in enumerate(rows, 1):
        text = (r["title"] + "\n" + r["content"])[:MAX_CHARS]  # 標題與全文字串合併
        summary = chat([{"role": "user", "content": PROMPT.format(text=text)}])
        conn.execute("UPDATE news SET summary = ? WHERE id = ?", (summary, r["id"]))
        conn.commit()
        print(f"[{i}/{len(rows)}] #{r['id']} {r['title']}\n    {summary[:60]}…")

    print("完成。")


if __name__ == "__main__":
    main()
