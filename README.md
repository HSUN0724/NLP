# NLP 作業

> 每份作業也有獨立的 repository：
> [NLP-HW1](https://github.com/HSUN0724/NLP-HW1)｜[NLP-HW2](https://github.com/HSUN0724/NLP-HW2)｜[NLP-HW3](https://github.com/HSUN0724/NLP-HW3)

| 作業 | 主題 | 方法 | 結果 |
| --- | --- | --- | --- |
| [NLP1](NLP1/) | 中文多元情緒分類（8 種語調） | 微調 `bert-base-chinese` | 測試集準確率 89.18%（v1，3 epochs） |
| [NLP2](NLP2/) | 土味情話多輪對話專家 | `Qwen2.5-1.5B-Instruct` + LoRA | 微調後能穩定說土味情話，原始模型則完全不會（見 NLP2 README 6.3） |
| [NLP3](NLP3/) | RAG 健康諮詢小幫手 | 爬蟲 + SQLite + Ollama（`qwen2.5:3b`、`bge-m3`）+ Flask | 檢索測試集 22 題 Hit@1 100%；網頁可多輪問答並附上新聞來源 |

## NLP1：中文多元情緒分類

使用 Hugging Face 資料集 [Chinese_Multi-Emotion_Dialogue_Dataset](https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset)（4,159 筆，MIT License），
將句子分成平淡、開心、關切、憤怒、驚奇、悲傷、厭惡、疑問 8 種語調。
比較 3 個 epoch（v1）與 10 個 epoch（v2）：v2 過擬合、準確率沒有提升，v1 是較好的模型。

→ 詳細說明見 [NLP1/README.md](NLP1/README.md)

## NLP2：土味情話多輪對話專家

用自己整理的 57 段多輪對話語料，對 Qwen2.5-1.5B-Instruct 做 LoRA 微調（只訓練約 1% 參數），
讓模型能把日常話題轉成土味情話，並進行多輪對話。README 內附微調前後的實際對話比較。

→ 詳細說明見 [NLP2/README.md](NLP2/README.md)

## NLP3：RAG 健康諮詢小幫手

爬取[健康醫療網](https://www.healthnews.com.tw/)「新聞分類」的 175 篇新聞存入 SQLite，
用地端模型產生文章摘要並轉成向量索引，再以 Flask 網頁提供多輪問答。
網頁可用 checkbox 決定多輪對話是否重新檢索，每次回答後都會條列新聞標題、連結與文章新增時間。

→ 詳細說明見 [NLP3/README.md](NLP3/README.md)

## 環境

- Python 3.10+、支援 CUDA 的 PyTorch
- 訓練硬體：NVIDIA GeForce RTX 5050 Laptop GPU（8GB）
- 各作業的套件列在各自資料夾的 `requirements.txt`
- NLP3 另需安裝 [Ollama](https://ollama.com/download) 執行地端模型

> 訓練出的模型權重檔案較大，沒有上傳；依各資料夾 README 的步驟執行即可重現。
