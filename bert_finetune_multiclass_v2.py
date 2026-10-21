# 微調 BERT：多元分類（Chinese_Multi-Emotion_Dialogue_Dataset，8 種情緒語調）
import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline,
)

MODEL_NAME = "bert-base-chinese"
OUTPUT_DIR = "./bert_emotion_model_v2"

# 1. 讀取資料集
# 方法一：直接從 Hugging Face 讀取（官方 pandas 範例）
# df = pd.read_csv("hf://datasets/Johnson8187/Chinese_Multi-Emotion_Dialogue_Dataset/data.csv")
# 方法二：讀取已下載的 data.csv
df = pd.read_csv("data.csv")
df = df.dropna(subset=["text", "emotion"])
print(df["emotion"].value_counts())

# 2. 文字 label 轉換成整數 0, 1, 2, ...
labels = sorted(df["emotion"].unique())
label2id = {label: i for i, label in enumerate(labels)}
id2label = {i: label for label, i in label2id.items()}
df["label"] = df["emotion"].map(label2id)
print("label2id:", label2id)

# 3. 切分訓練 / 測試資料（依類別比例分層抽樣）
train_df, test_df = train_test_split(
    df[["text", "label"]], test_size=0.2, random_state=42, stratify=df["label"]
)
train_ds = Dataset.from_pandas(train_df, preserve_index=False)
test_ds = Dataset.from_pandas(test_df, preserve_index=False)

# 4. Tokenize
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=128)

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# 5. 建立模型：num_labels 由 2 改成類別數
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

def compute_metrics(eval_pred):
    logits, y_true = eval_pred
    y_pred = np.argmax(logits, axis=-1)
    return {"accuracy": (y_pred == y_true).mean()}

# 6. 訓練
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=10,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    learning_rate=2e-5,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=test_ds,
    compute_metrics=compute_metrics,
)
trainer.train()
print(trainer.evaluate())

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

# 7. 預測
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
