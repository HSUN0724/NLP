# NLP 作業

| 作業 | 主題 | 方法 | 結果 |
| --- | --- | --- | --- |
| [NLP1](NLP1/) | 中文多元情緒分類（8 種語調） | 微調 `bert-base-chinese` | 測試集準確率 89.18%（v1，3 epochs） |
| [NLP2](NLP2/) | 土味情話多輪對話專家 | `Qwen2.5-1.5B-Instruct` + LoRA | 微調後能穩定說土味情話，原始模型則完全不會（見 NLP2 README 6.3） |

## NLP1：中文多元情緒分類

使用 Hugging Face 資料集 [Chinese_Multi-Emotion_Dialogue_Dataset](https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset)（4,159 筆，MIT License），
將句子分成平淡、開心、關切、憤怒、驚奇、悲傷、厭惡、疑問 8 種語調。
比較 3 個 epoch（v1）與 10 個 epoch（v2）：v2 過擬合、準確率沒有提升，v1 是較好的模型。

→ 詳細說明見 [NLP1/README.md](NLP1/README.md)

## NLP2：土味情話多輪對話專家

用自己整理的 57 段多輪對話語料，對 Qwen2.5-1.5B-Instruct 做 LoRA 微調（只訓練約 1% 參數），
讓模型能把日常話題轉成土味情話，並進行多輪對話。README 內附微調前後的實際對話比較。

→ 詳細說明見 [NLP2/README.md](NLP2/README.md)

## 環境

- Python 3.10+、支援 CUDA 的 PyTorch
- 訓練硬體：NVIDIA GeForce RTX 5050 Laptop GPU（8GB）
- 各作業的套件列在各自資料夾的 `requirements.txt`

> 訓練出的模型權重檔案較大，沒有上傳；依各資料夾 README 的步驟執行即可重現。
