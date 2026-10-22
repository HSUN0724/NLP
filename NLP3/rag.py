"""RAG 核心：向量檢索 + 組合提示詞 + 呼叫地端模型。"""
import json

import numpy as np

from config import IDS_PATH, MIN_SCORE, TOP_K, VECTORS_PATH
from db import get_conn, get_news_by_ids
from llm import chat, embed

SYSTEM_PROMPT = (
    "你是「健康諮詢小幫手」，根據提供的健康醫療網新聞資料，用繁體中文回答使用者的健康問題。\n"
    "規則：\n"
    "1. 優先依據參考新聞內容回答，條理清楚，可用條列。\n"
    "2. 參考新聞中與問題無關的內容請直接忽略，不要硬套進回答。\n"
    "3. 參考資料沒有的資訊請明說，不要捏造數據。\n"
    "4. 涉及診斷或用藥時，提醒使用者諮詢專業醫師。\n"
    "5. 不需要在回答中列出參考新聞清單，系統會自動附上。\n"
    "6. 全程只能使用繁體中文（台灣用語），禁止使用簡體字。"
)


class Retriever:
    def __init__(self):
        self.vectors = np.load(VECTORS_PATH)
        with open(IDS_PATH, encoding="utf-8") as f:
            self.ids = json.load(f)

    def search(self, query, k=TOP_K, min_score=MIN_SCORE):
        q = np.array(embed(query)[0], dtype=np.float32)
        q /= np.linalg.norm(q)
        scores = self.vectors @ q
        top = np.argsort(-scores)[:k]
        # 過濾相關性太低的新聞，但至少保留第 1 名
        top = [i for n, i in enumerate(top) if n == 0 or scores[i] >= min_score]
        hit_ids = [self.ids[i] for i in top]
        conn = get_conn()
        news = get_news_by_ids(conn, hit_ids)
        conn.close()
        results = []
        for i in top:
            item = news[self.ids[i]]
            item["score"] = float(scores[i])
            results.append(item)
        return results


def build_context(docs):
    blocks = []
    for i, d in enumerate(docs, 1):
        blocks.append(
            f"【新聞{i}】{d['title']}（{d['published_at']}）\n"
            f"摘要：{d['summary']}\n"
            f"內文節錄：{d['content'][:800]}"
        )
    return "\n\n".join(blocks)


def answer(question, history, docs):
    """history: [{"role": "user"/"assistant", "content": ...}]（不含本次問題）"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    if docs:
        user_msg = f"參考新聞資料：\n{build_context(docs)}\n\n使用者問題：{question}"
    else:
        # 不檢索時依對話紀錄回答；列出先前所有問題，避免小模型只看最後一輪
        asked = [m["content"] for m in history if m["role"] == "user"]
        listing = "\n".join(f"{i}. {q}" for i, q in enumerate(asked, 1))
        user_msg = (
            f"（本輪不檢索，請依前面的對話內容，用繁體中文回答）\n"
            f"使用者先前問過的問題：\n{listing}\n\n使用者問題：{question}"
        ) if asked else question
    messages.append({"role": "user", "content": user_msg})
    return chat(messages)


def format_sources(docs):
    """回覆後面附上的檢索來源（條列）。"""
    return [
        {"title": d["title"], "url": d["url"], "published_at": d["published_at"]}
        for d in docs
    ]
