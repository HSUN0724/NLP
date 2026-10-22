"""呼叫地端 Ollama 模型（對話與向量）。"""
import requests

from config import CHAT_MODEL, EMBED_MODEL, OLLAMA_URL


def chat(messages, model=CHAT_MODEL, temperature=0.3):
    resp = requests.post(
        f"{OLLAMA_URL}/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def embed(texts, model=EMBED_MODEL):
    """回傳 list[list[float]]，一次可送多筆文字。"""
    if isinstance(texts, str):
        texts = [texts]
    resp = requests.post(
        f"{OLLAMA_URL}/api/embed",
        json={"model": model, "input": texts},
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()["embeddings"]
