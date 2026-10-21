"""用 LoRA 微調 Hugging Face 上的對話模型，讓它成為「土味情話」專家。

只對「專家（assistant）」的回答計算 loss，system / user 部分不學，
這樣模型學到的是「怎麼接話」，而不是背使用者的問題。
"""
import argparse
import json
import math
import random

import torch
from peft import LoraConfig, get_peft_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["messages"] for line in f if line.strip()]


def encode(tokenizer, messages, max_len):
    """把一段多輪對話轉成 input_ids，並把非 assistant 的 token 標成 -100（不算 loss）。"""
    def ids_of(msgs, add_generation_prompt=False):
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=add_generation_prompt)
        return tokenizer(text, add_special_tokens=False)["input_ids"]

    input_ids = ids_of(messages)
    labels = [-100] * len(input_ids)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        start = len(ids_of(messages[:i], add_generation_prompt=True))
        end = len(ids_of(messages[: i + 1]))
        labels[start:end] = input_ids[start:end]
    return {"input_ids": input_ids[:max_len], "labels": labels[:max_len]}


def collate(batch, pad_id):
    width = max(len(x["input_ids"]) for x in batch)
    ids, labels, mask = [], [], []
    for x in batch:
        pad = width - len(x["input_ids"])
        ids.append(x["input_ids"] + [pad_id] * pad)
        labels.append(x["labels"] + [-100] * pad)
        mask.append([1] * len(x["input_ids"]) + [0] * pad)
    return {k: torch.tensor(v) for k, v in (("input_ids", ids), ("labels", labels), ("attention_mask", mask))}


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    losses = [model(**{k: v.to(device) for k, v in b.items()}).loss.item() for b in loader]
    model.train()
    return sum(losses) / max(len(losses), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--train_file", default="data/train.jsonl")
    ap.add_argument("--eval_file", default="data/eval.jsonl")
    ap.add_argument("--output_dir", default="outputs/lora-tuwei")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    print(f"載入模型 {args.model}（{device}, {dtype}）")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype).to(device)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model.config.use_cache = False

    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    train_data = [encode(tokenizer, m, args.max_len) for m in load_jsonl(args.train_file)]
    eval_data = [encode(tokenizer, m, args.max_len) for m in load_jsonl(args.eval_file)]
    pad_id = tokenizer.pad_token_id
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                              collate_fn=lambda b: collate(b, pad_id))
    eval_loader = DataLoader(eval_data, batch_size=args.batch_size, collate_fn=lambda b: collate(b, pad_id))

    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=args.lr, weight_decay=0.0)
    total_steps = args.epochs * math.ceil(len(train_data) / args.batch_size)
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * total_steps), total_steps)

    print(f"訓練資料 {len(train_data)} 段、驗證資料 {len(eval_data)} 段、共 {total_steps} steps")
    print(f"訓練前 eval loss: {evaluate(model, eval_loader, device):.4f}")
    model.train()
    step = 0
    for epoch in range(1, args.epochs + 1):
        for batch in train_loader:
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            step += 1
            if step % 5 == 0:
                print(f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f}")
        print(f"== epoch {epoch} 結束，eval loss: {evaluate(model, eval_loader, device):.4f}")

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"LoRA adapter 已存到 {args.output_dir}")


if __name__ == "__main__":
    main()
