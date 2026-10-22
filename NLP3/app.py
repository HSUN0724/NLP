"""步驟 4：網頁問答系統（Flask）。

用法：
    python app.py
然後打開瀏覽器 http://127.0.0.1:5000
"""
from flask import Flask, jsonify, render_template, request

from config import CHAT_MODEL, EMBED_MODEL
from rag import Retriever, answer, format_sources

app = Flask(__name__)
retriever = Retriever()


@app.route("/")
def index():
    return render_template("index.html", chat_model=CHAT_MODEL, embed_model=EMBED_MODEL)


@app.post("/api/chat")
def api_chat():
    data = request.get_json()
    question = data["question"].strip()
    history = data.get("history", [])          # 前幾輪的問答（純文字）
    use_retrieval = data.get("use_retrieval", True)
    past_sources = data.get("past_sources", [])  # 先前各輪檢索過的新聞（已去重）

    # 第一輪一定檢索；之後依 checkbox 決定多輪對話是否重新檢索
    if not history or use_retrieval:
        docs = retriever.search(question)
        sources = format_sources(docs)
        retrieved = True
    else:
        docs = []
        sources = past_sources  # 沿用先前對話檢索到的新聞
        retrieved = False

    reply = answer(question, history, docs)
    return jsonify({"answer": reply, "sources": sources, "retrieved": retrieved})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
