"""把 data/raw_dialogues.txt 轉成 chat 格式的 JSONL 訓練資料。

輸出每一行是一段多輪對話：
{"topic": "...", "messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}, ...]}
"""
import argparse
import json
import random
from pathlib import Path

SYSTEM_PROMPT = (
    "你是一個土味情話高手，你會以親切、溫和又帶點俏皮的口吻來回答使用者的問題，"
    "擅長把日常話題變成讓人會心一笑的情話，並一步一步教使用者怎麼接話。"
)

USER_TAGS = ("你：", "你:")
BOT_TAGS = ("專家：", "專家:")


def strip_tag(line, tags):
    for tag in tags:
        if line.startswith(tag):
            return line[len(tag):].strip()
    return line


def parse(path: Path):
    dialogues, topic, turns = [], None, []

    def flush():
        if topic is not None and turns:
            dialogues.append({"topic": topic, "turns": list(turns)})

    for lineno, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or (line.startswith("#") and not line.startswith("###")):
            continue
        if line.startswith("###"):
            flush()
            topic, turns = line.lstrip("#").strip(), []
        elif line.startswith(USER_TAGS):
            turns.append(("user", strip_tag(line, USER_TAGS)))
        elif line.startswith(BOT_TAGS):
            turns.append(("assistant", strip_tag(line, BOT_TAGS)))
        else:
            raise ValueError(f"第 {lineno} 行格式錯誤：{raw}")
    flush()
    return dialogues


def validate(d):
    roles = [r for r, _ in d["turns"]]
    expected = ["user", "assistant"] * (len(roles) // 2)
    if roles != expected:
        raise ValueError(f"[{d['topic']}] 必須是「你／專家」交替且以專家結尾")
    if len(roles) < 4:
        raise ValueError(f"[{d['topic']}] 至少需要 2 輪對話")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw_dialogues.txt")
    ap.add_argument("--output", default="data/train.jsonl")
    ap.add_argument("--eval_output", default="data/eval.jsonl")
    ap.add_argument("--eval_size", type=int, default=3, help="保留幾段對話當驗證集")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    dialogues = parse(Path(args.input))
    for d in dialogues:
        validate(d)

    random.Random(args.seed).shuffle(dialogues)
    eval_set, train_set = dialogues[: args.eval_size], dialogues[args.eval_size :]

    def dump(items, out):
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            for d in items:
                messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                messages += [{"role": r, "content": c} for r, c in d["turns"]]
                f.write(json.dumps({"topic": d["topic"], "messages": messages}, ensure_ascii=False) + "\n")

    dump(train_set, args.output)
    dump(eval_set, args.eval_output)
    n_turns = sum(len(d["turns"]) // 2 for d in dialogues)
    print(f"共 {len(dialogues)} 段對話、{n_turns} 輪 → train {len(train_set)} / eval {len(eval_set)}")
    print(f"已輸出 {args.output}、{args.eval_output}")


if __name__ == "__main__":
    main()
