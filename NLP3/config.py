"""專案共用設定。可用環境變數覆寫。"""
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 資料庫與向量索引位置
DB_PATH = os.path.join(BASE_DIR, "data", "news.db")
INDEX_DIR = os.path.join(BASE_DIR, "index")
VECTORS_PATH = os.path.join(INDEX_DIR, "vectors.npy")
IDS_PATH = os.path.join(INDEX_DIR, "ids.json")

# 地端模型（Ollama）
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
CHAT_MODEL = os.getenv("CHAT_MODEL", "qwen2.5:3b")
EMBED_MODEL = os.getenv("EMBED_MODEL", "bge-m3")

# 檢索參數
TOP_K = int(os.getenv("TOP_K", "3"))
MIN_SCORE = float(os.getenv("MIN_SCORE", "0.58"))  # 相似度低於此值的新聞不採用（第 1 名一定保留）

# 爬蟲：健康醫療網「新聞分類」底下的各頻道
SITE = "https://www.healthnews.com.tw"
NEWS_CHANNELS = {
    "醫療疾病": "782fe66c-4593-806a-7995-0a9b0ecbc4ce",
    "生醫新知": "f231bdd3-3df1-1cb4-373e-e132629cdced",
    "健康保健": "ac24e543-bc5a-dbd5-0aab-a73a4e109c6d",
    "醫學美容": "d2385c7f-0c95-67f0-bf31-9ef9650a2e80",
    "中醫養生": "04558c00-7995-b05b-5c26-ef045abc9083",
    "運動復健": "ea5f19b7-827b-cb79-99e4-193e58c4a84c",
    "營養飲食": "f72c8a39-1105-2663-255a-b925ad108bac",
    "樂齡照顧": "f284edc2-9d31-30e2-e725-c1444baea353",
    "心靈樂活": "a0e95de0-b813-b2b0-cf1c-8b21a48efad4",
}
