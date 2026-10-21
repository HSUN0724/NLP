"""和微調後的土味情話專家進行多輪對話。

用法：
    python chat.py                     # 互動模式
    python chat.py --no_adapter        # 用原始模型對話（比較微調前後）
    python chat.py --demo              # 自動跑 README 裡的示範對話
指令：/reset 清空對話紀錄、/exit 離開
"""
import argparse
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent / "scripts"))
from build_dataset import SYSTEM_PROMPT  # noqa: E402

DEMO_QUESTIONS = [
    "教我如何透過長相來認識女性朋友。",
    "長得像你朋友？",
    "那如果她問「你上一任女朋友呢？」",
    "那如果她問「所以很多人報名嗎？」",
    "那如果她說「我又沒說要報名？」",
]


def load(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    if not args.no_adapter:
        model = PeftModel.from_pretrained(model, args.adapter)
    model.to("cuda" if torch.cuda.is_available() else "cpu").eval()
    return tokenizer, model


@torch.no_grad()
def reply(tokenizer, model, history, args):
    text = tokenizer.apply_chat_template(history, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--adapter", default="outputs/lora-tuwei")
    ap.add_argument("--no_adapter", action="store_true", help="不載入 LoRA，使用原始模型")
    ap.add_argument("--demo", action="store_true", help="自動執行示範對話")
    ap.add_argument("--system", default=SYSTEM_PROMPT)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    args = ap.parse_args()

    sys.stdout.reconfigure(encoding="utf-8")
    tokenizer, model = load(args)
    history = [{"role": "system", "content": args.system}]
    print("土味情話專家已上線！輸入 /reset 重新開始、/exit 離開。\n")

    questions = iter(DEMO_QUESTIONS) if args.demo else None
    while True:
        if questions is not None:
            user = next(questions, None)
            if user is None:
                break
            print(f"你：{user}")
        else:
            try:
                user = input("你：").strip()
            except (EOFError, KeyboardInterrupt):
                break
        if not user:
            continue
        if user == "/exit":
            break
        if user == "/reset":
            history = history[:1]
            print("（對話已清空）\n")
            continue
        history.append({"role": "user", "content": user})
        answer = reply(tokenizer, model, history, args)
        history.append({"role": "assistant", "content": answer})
        print(f"專家：{answer}\n")


if __name__ == "__main__":
    main()
