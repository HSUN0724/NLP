# 微調 BERT：中文多元情緒分類（8 種語調）

> 使用 `bert-base-chinese` 在 **Chinese_Multi-Emotion_Dialogue_Dataset** 上進行多元分類微調。
> 程式：`bert_finetune_multiclass.py`（v1）、`bert_finetune_multiclass_v2.py`（v2）
> 輸出模型：`./bert_emotion_model`（v1）、`./bert_emotion_model_v2`（v2）

## 資料來源與授權

本專案使用的 `data.csv` 取自 Hugging Face 資料集：

| 項目 | 內容 |
|------|------|
| 資料集 | [Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset](https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset) |
| 作者 | Johnson8187 |
| 授權 | MIT License |

依 MIT 授權條款，轉載時應保留原作者的版權聲明與授權文字。資料集版權屬原作者所有。

---

## 專案結構

```text
NLP1/
├── data.csv                         # 資料集（Chinese_Multi-Emotion_Dialogue_Dataset）
├── bert_finetune_multiclass.py      # v1：3 epochs
├── bert_finetune_multiclass_v2.py   # v2：10 epochs
├── logs/
│   ├── trainer_state_v1.json        # v1 訓練紀錄
│   └── train_v2.log                 # v2 訓練紀錄
├── requirements.txt
└── README.md
```

## 執行方式

```bash
git clone https://github.com/HSUN0724/NLP.git
cd NLP/NLP1
pip install -r requirements.txt
python bert_finetune_multiclass.py      # v1，模型存到 bert_emotion_model/
python bert_finetune_multiclass_v2.py   # v2，模型存到 bert_emotion_model_v2/
```

> 需要支援 CUDA 的 PyTorch 才能用 GPU 訓練，請依顯卡到 [pytorch.org](https://pytorch.org/get-started/locally/) 選擇安裝指令。

---

## 目錄

1. [讀取資料集](#1-讀取資料集)
2. [標籤轉換](#2-標籤轉換)
3. [切分訓練／測試資料](#3-切分訓練測試資料)
4. [Tokenize](#4-tokenize)
5. [建立模型](#5-建立模型)
6. [訓練](#6-訓練)
7. [訓練結果](#7-訓練結果)
8. [預測](#8-預測)
9. [自訂句子測試](#9-自訂句子測試)
10. [v2：增加 epoch 至 10](#10-v2增加-epoch-至-10)

---

## 1. 讀取資料集

資料來源：[Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset](https://huggingface.co/datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset)

```python
# 方法一：直接從 Hugging Face 讀取
# df = pd.read_csv("hf://datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset/data.csv")
# 方法二：讀取已下載的 data.csv
df = pd.read_csv("data.csv")
df = df.dropna(subset=["text", "emotion"])
```

資料格式範例：

| text                                      | emotion  |
| ----------------------------------------- | -------- |
| 你要不要去吃午餐？                        | 平淡語氣 |
| 誒誒誒！我甄選上了！                      | 開心語調 |
| 我幾天身體好像有點不太舒服，肚子好痛      | 悲傷語調 |
| 我的小專題組員都不做事，幹!超後悔跟他一組 | 憤怒語調 |

各類別數量（共 **4,159** 筆）：

| 情緒     | 筆數 |
| -------- | ---: |
| 平淡語氣 |  705 |
| 開心語調 |  592 |
| 關切語調 |  560 |
| 憤怒語調 |  527 |
| 驚奇語調 |  499 |
| 悲傷語調 |  486 |
| 厭惡語調 |  404 |
| 疑問語調 |  386 |

---

## 2. 標籤轉換

將文字標籤排序後對應為整數 `0 ~ 7`：

```python
labels = sorted(df["emotion"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
df["label"] = df["emotion"].map(label2id)
```

| id | label    |
| -: | -------- |
|  0 | 厭惡語調 |
|  1 | 平淡語氣 |
|  2 | 悲傷語調 |
|  3 | 憤怒語調 |
|  4 | 疑問語調 |
|  5 | 開心語調 |
|  6 | 關切語調 |
|  7 | 驚奇語調 |

---

## 3. 切分訓練／測試資料

以 **8 : 2** 切分，並依類別比例分層抽樣（`stratify`），確保各類別在兩組中的比例一致。

```python
train_df, test_df = train_test_split(
    df[["text", "label"]], test_size=0.2, random_state=42, stratify=df["label"]
)
train_ds = Dataset.from_pandas(train_df, preserve_index=False)
test_ds = Dataset.from_pandas(test_df, preserve_index=False)
```

| 資料集 |  筆數 |
| ------ | ----: |
| 訓練集 | 3,327 |
| 測試集 |   832 |

---

## 4. Tokenize

使用 `bert-base-chinese` 的 tokenizer，統一截斷／補齊至長度 **128**。

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)
```

---

## 5. 建立模型

與二元分類的差異：`num_labels` 由 `2` 改為類別數 `8`，並寫入 `id2label` / `label2id`，讓預測時直接輸出中文標籤。

```python
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)
```

評估指標為 **Accuracy**：

```python
def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    return {"accuracy": (y_pred == y_true).mean()}
```

---

## 6. 訓練

### 超參數

| 參數                                  | 值                  |
| ------------------------------------- | ------------------- |
| `num_train_epochs`                  | 3                   |
| `per_device_train_batch_size`       | 16                  |
| `per_device_eval_batch_size`        | 32                  |
| `learning_rate`                     | 2e-5                |
| `weight_decay`                      | 0.01                |
| `eval_strategy` / `save_strategy` | epoch               |
| `load_best_model_at_end`            | True（依 accuracy） |
| `save_total_limit`                  | 1                   |
| `fp16`                              | 有 GPU 時啟用       |
| `logging_steps`                     | 50                  |

```python
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)
trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

每個 epoch 208 步，共 **624 步**。

---

## 7. 訓練結果

### 訓練 Loss（每 50 步）

| Step | Epoch |   Loss | Learning Rate |
| ---: | ----: | -----: | ------------: |
|   50 |  0.24 | 1.7506 |      1.84e-05 |
|  100 |  0.48 | 0.9461 |      1.68e-05 |
|  150 |  0.72 | 0.6889 |      1.52e-05 |
|  200 |  0.96 | 0.5907 |      1.36e-05 |
|  250 |  1.20 | 0.4293 |      1.20e-05 |
|  300 |  1.44 | 0.3479 |      1.04e-05 |
|  350 |  1.68 | 0.3424 |      8.81e-06 |
|  400 |  1.92 | 0.3323 |      7.21e-06 |
|  450 |  2.16 | 0.2257 |      5.61e-06 |
|  500 |  2.40 | 0.2080 |      4.01e-06 |
|  550 |  2.64 | 0.1950 |      2.40e-06 |
|  600 |  2.88 | 0.2005 |      8.01e-07 |

### 每個 Epoch 的驗證結果

| Epoch | Step | Eval Loss |    Eval Accuracy |
| ----: | ---: | --------: | ---------------: |
|     1 |  208 |    0.5535 |           82.21% |
|     2 |  416 |    0.4481 |           87.14% |
|     3 |  624 |    0.4330 | **89.18%** |

**最佳模型**：`checkpoint-624`（Epoch 3），測試集準確率 **89.18%**。

> 觀察：訓練 loss 由 1.75 穩定下降至約 0.20；驗證 loss 仍持續下降、準確率持續上升，尚無明顯過擬合跡象。

---

## 8. 預測

載入儲存的模型，以 `pipeline` 進行推論：

```python
classifier = pipeline(
    "text-classification",
    model=OUTPUT_DIR,
    tokenizer=OUTPUT_DIR,
    device=0 if torch.cuda.is_available() else -1,
)

texts = [
    "他們是不是吵架了？不會打起來吧？",
    "我剛剛看到有人的眉毛像是蠟筆小新！",
]

for text, result in zip(texts, classifier(texts)):
    print(f"{text} -> {result['label']} (score: {result['score']:.4f})")
```

---

## 9. 自訂句子測試

加上 `top_k=None`，輸出全部 8 個類別的分數，觀察模型對每種語氣的信心分布：

```python
import sys
import torch
from transformers import pipeline

sys.stdout.reconfigure(encoding="utf-8")
model_dir = "bert_emotion_model"
clf = pipeline("text-classification", model=model_dir, tokenizer=model_dir,
               device=0 if torch.cuda.is_available() else -1, top_k=None)
text = "我昨天夢到有一隻超大蜘蛛在追著我跑！"
for r in clf([text])[0]:
    print(f"{r['label']}\t{r['score']:.4f}")
```

> 在 PowerShell 用 `python -c` 直接傳入含中文標點的程式碼時，引號會被吃掉而出現 `SyntaxError: invalid character '！'`，因此改為寫成 `.py` 檔再執行。

### 測試一：「我昨天夢到有一隻超大蜘蛛在追著我跑！」

| 排名 | 情緒               |             分數 |
| ---: | ------------------ | ---------------: |
|    1 | **驚奇語調** | **0.6735** |
|    2 | 開心語調           |           0.1834 |
|    3 | 平淡語氣           |           0.0821 |
|    4 | 憤怒語調           |           0.0227 |
|    5 | 疑問語調           |           0.0109 |
|    6 | 關切語調           |           0.0095 |
|    7 | 悲傷語調           |           0.0095 |
|    8 | 厭惡語調           |           0.0084 |

觀察：

- 8 個類別中沒有「害怕」，「驚奇語調」是最接近的選項，結果合理。
- 「開心語調」排第二（0.18），推測與句尾「！」的興奮口氣有關（訓練資料中開心的句子常帶驚嘆號，例如「誒誒誒！我甄選上了！」），尚未驗證。
- 「厭惡語調」幾乎為 0，模型沒有把「蜘蛛」和厭惡連結起來。

### 測試二：「你覺得我今天哪裡不一樣？」

| 排名 | 情緒               |             分數 |
| ---: | ------------------ | ---------------: |
|    1 | **平淡語氣** | **0.7888** |
|    2 | 關切語調           |           0.0798 |
|    3 | 開心語調           |           0.0394 |
|    4 | 疑問語調           |           0.0384 |
|    5 | 驚奇語調           |           0.0261 |
|    6 | 悲傷語調           |           0.0175 |
|    7 | 憤怒語調           |           0.0055 |
|    8 | 厭惡語調           |           0.0043 |

觀察：

- 雖然是問句，「疑問語調」只有 0.0384。
- 資料集中不少問句被標為「平淡語氣」，例如「你要不要去吃午餐？」、「他們是不是吵架了？不會打起來吧？」。模型學到的「疑問語調」可能不是「有問號」，而是帶困惑、不解的口氣（推測，尚未逐筆檢查資料）。

### 測試三：「今天吃了麥當勞跟肯德基當晚餐,吃得好滿足.」

| 排名 | 情緒               |             分數 |
| ---: | ------------------ | ---------------: |
|    1 | **開心語調** | **0.7141** |
|    2 | 平淡語氣           |           0.1918 |
|    3 | 厭惡語調           |           0.0534 |
|    4 | 悲傷語調           |           0.0177 |
|    5 | 驚奇語調           |           0.0076 |
|    6 | 關切語調           |           0.0055 |
|    7 | 疑問語調           |           0.0052 |
|    8 | 憤怒語調           |           0.0047 |

觀察：

- 「好滿足」是明確的正面情緒，判為「開心語調」正確。
- 「平淡語氣」排第二（0.19），推測是前半句「今天吃了…當晚餐」屬於陳述事實，分走部分分數。
- 「厭惡語調」為 0.05，比前兩次測試高，但數值很小，原因不確定。
- 這句使用半形標點「,」「.」，訓練資料多為全形標點；標點是否影響分數，可改成全形再測一次比較（尚未測試）。

### 小結：如何提高信心分數

| 方法                         | 效果       | 說明                                                                             |
| ---------------------------- | ---------- | -------------------------------------------------------------------------------- |
| 增加 epoch（3 → 5）         | 真正改善   | 第 3 個 epoch 驗證 loss 仍在下降（0.5535 → 0.4481 → 0.4330），模型可能尚未學完（實測結果見第 10 節：未改善） |
| 整理標註不一致               | 真正改善   | 例如問句在「平淡語氣」與「疑問語調」間的標準不一，會讓分數分散                   |
| 換更大的模型                 | 真正改善   | 例如 `hfl/chinese-roberta-wwm-ext`                                             |
| Temperature Scaling（T < 1） | 僅改變呈現 | 分數變集中，但預測與準確率不變，判錯時也會「很有把握地錯」                       |

---

## 10. v2：增加 epoch 至 10

### 修改內容

| 項目 | v1 | v2 |
|------|----|----|
| 程式 | `bert_finetune_multiclass.py` | `bert_finetune_multiclass_v2.py` |
| 訓練紀錄 | `logs/trainer_state_v1.json` | `logs/train_v2.log` |
| `num_train_epochs` | 3 | 10 |
| `OUTPUT_DIR` | `./bert_emotion_model` | `./bert_emotion_model_v2` |

其餘超參數不變。`load_best_model_at_end=True` 會依驗證準確率保留最佳 checkpoint。訓練時間約 4 分鐘（RTX 5050 Laptop GPU，共 2,080 步）。

### v2 每個 Epoch 的驗證結果

| Epoch | Eval Loss | Eval Accuracy |
|------:|----------:|--------------:|
| 1 | 0.5636 | 81.61% |
| 2 | 0.5133 | 85.10% |
| 3 | **0.4640** | 87.86% |
| 4 | 0.5610 | 87.74% |
| 5 | 0.5916 | **89.06%** ← 保留的最佳模型（`checkpoint-1040`） |
| 6 | 0.6322 | 88.34% |
| 7 | 0.7094 | 87.86% |
| 8 | 0.6742 | 88.94% |
| 9 | 0.7050 | 89.06% |
| 10 | 0.7083 | 88.94% |

訓練 loss 從 1.67 一路降到約 0.001，但驗證 loss 在第 3 個 epoch 後由 0.46 回升至 0.71，屬於典型的**過擬合**；準確率則停在 88%–89% 之間。

### v1 與 v2 比較

| 模型 | 最佳 epoch | 驗證準確率 | 驗證 loss |
|------|-----:|-----:|-----:|
| v1（3 epochs） | 3 | **89.18%** | **0.4330** |
| v2（10 epochs） | 5 | 89.06% | 0.5916 |

### 測試句比較：第 9 節的三句

| 句子 | v1 | v2 |
|------|-----|-----|
| 我昨天夢到有一隻超大蜘蛛在追著我跑！ | 驚奇語調 0.6735 | **平淡語氣** 0.9971 |
| 你覺得我今天哪裡不一樣？ | 平淡語氣 0.7888 | 平淡語氣 0.9979 |
| 今天吃了麥當勞跟肯德基當晚餐,吃得好滿足. | 開心語調 0.7141 | **平淡語氣** 0.9689 |

v2 的信心分數雖然接近 1，但「蜘蛛」與「好滿足」兩句從原本合理的判斷改判為「平淡語氣」，也就是「很有把握地錯」。

### 測試句比較：五句新句子

以下句子皆不在 `data.csv` 中（已逐句比對）。

| 句子 | v1 | v2 |
|------|-----|-----|
| 我每天都能跟她一起上學，我好開心！ | 開心語調 0.9872 | 開心語調 0.9987 |
| 最好的朋友要離開臺灣了，以後可能不容易再見面... | 悲傷語調 0.9787 | 悲傷語調 0.9981 |
| 我覺得我快不行了... | 悲傷語調 0.9589 | 悲傷語調 0.9022 |
| 剛剛收到研究所錄取的通知書！ | 開心語調 0.9785 | 開心語調 0.9976 |
| 今年的冬天好像比較晚來。 | 平淡語氣 0.9731 | 平淡語氣 0.9986 |

觀察：

- 兩個模型的判斷完全一致且合理；情緒明確的句子，v1 的信心分數本來就在 0.95 以上。
- v2 多半只是把 0.97 推到 0.99，實際判斷沒有差別。
- 「我覺得我快不行了...」是唯一 v2 分數較低的句子，「平淡語氣」分到 0.09；這句可能指身體不適，也可能指情緒低落，本身較模糊。

### 結論

- 增加 epoch **沒有提升準確率**，反而造成過擬合，讓模型在模糊句子上偏向「平淡語氣」且過度自信。
- **v1 仍是較好的模型。**
- v1 信心分數偏低的句子（0.67–0.79）本身情緒較模糊，增加訓練並不能解決。
- 後續可嘗試：將 `metric_for_best_model` 改為 `eval_loss` 以避免選到過擬合的 checkpoint，或整理「平淡語氣」與其他類別之間不一致的標註（v2 偏向平淡語氣的原因為推測，尚未驗證）。

---

## 輸出檔案

```
bert_emotion_model/
├── config.json              # 模型設定（含 id2label / label2id）
├── model.safetensors        # 模型權重
├── tokenizer.json
├── tokenizer_config.json
├── training_args.bin
└── checkpoint-624/          # 最佳 checkpoint（含 optimizer、scheduler、trainer_state）

bert_emotion_model_v2/       # v2（10 epochs），結構相同
└── checkpoint-1040/         # 最佳 checkpoint（epoch 5）
```

> 模型檔超過 GitHub 單檔 100 MB 上限，未上傳至 repo。
