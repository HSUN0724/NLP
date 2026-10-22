"""檢索評估：用 testset.json 的問題測試向量檢索是否找到正確新聞。

指標：
    Hit@1  第 1 名就是正確新聞的比例
    Hit@3  前 3 名內有正確新聞的比例
    MRR    正確新聞排名倒數的平均

用法：
    python evaluate.py              # 只評估檢索
    python evaluate.py --answer 2   # 另外讓模型實際回答前 2 題
"""
import argparse
import json

from rag import Retriever, answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--answer", type=int, default=0, help="實際產生回答的題數")
    args = parser.parse_args()

    with open("testset.json", encoding="utf-8") as f:
        tests = json.load(f)
    retriever = Retriever()

    hit1 = hit3 = mrr = 0
    for i, t in enumerate(tests, 1):
        docs = retriever.search(t["question"], k=len(retriever.ids), min_score=-1)
        ranked = [d["id"] for d in docs]
        rank = ranked.index(t["expected_id"]) + 1 if t["expected_id"] in ranked else None
        hit1 += rank == 1
        hit3 += rank is not None and rank <= 3
        mrr += 1 / rank if rank else 0
        mark = "✔" if rank == 1 else ("△" if rank and rank <= 3 else "✘")
        print(f"{mark} Q{i:02d} 正確 #{t['expected_id']} 排第 {rank}｜Top1 #{docs[0]['id']} "
              f"({docs[0]['score']:.3f})｜{t['question']}")

    n = len(tests)
    print(f"\n題數 {n}｜Hit@1 {hit1 / n:.1%}｜Hit@3 {hit3 / n:.1%}｜MRR {mrr / n:.3f}")

    for t in tests[:args.answer]:
        docs = retriever.search(t["question"])
        print(f"\n=== 問：{t['question']}\n{answer(t['question'], [], docs)}")
        for d in docs:
            print(f"- {d['title']}｜{d['url']}｜{d['published_at']}")


if __name__ == "__main__":
    main()
