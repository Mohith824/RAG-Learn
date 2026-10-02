"""
Evaluation harness for your RAG system.

Usage (from the same folder as RagV1.py):
    python eval.py --tag baseline-v2              # retrieval metrics only (cheap, fast)
    python eval.py --tag baseline-v2 --answers    # also generate and check answers (uses Gemini)

- Answers are cached in answers_<tag>.json, so if the Gemini server is busy and some
  questions fail, just run the SAME command again: finished questions are skipped.
- Use a NEW tag for every experiment (the cache belongs to the tag).
- A row is added to results.csv only when every question has been processed.
"""
import argparse
import csv
import json
import os
import time

from RagV1 import PROMPT, TOP_K, generate_with_retry, retrieve


def check_answer(item, answer):
    text = answer.lower()
    if item.get("should_refuse"):
        return "couldn't find" in text or "could not find" in text
    if item.get("keywords"):
        return all(k.lower() in text for k in item["keywords"])
    return None  # no automatic check for this question


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="run")
    parser.add_argument("--answers", action="store_true")
    parser.add_argument("--k", type=int, default=TOP_K)
    args = parser.parse_args()

    with open("Goldenset.json", encoding="utf-8") as f:
        golden = json.load(f)

    cache_file = f"answers_{args.tag}.json"
    cache = {}
    if args.answers and os.path.exists(cache_file):
        with open(cache_file, encoding="utf-8") as f:
            cache = json.load(f)

    hits, rr_sum, scored = 0, 0.0, 0
    ans_pass, ans_total, errors = 0, 0, 0

    print(f"{'#':<3} {'result':<6} {'rank':<5} {'retrieved pages':<22} question")
    for i, item in enumerate(golden, 1):
        results = retrieve(item["q"], k=args.k)
        pages = [c["page"] for _, c in results]

        rank = None
        if item["pages"]:
            for r, p in enumerate(pages, 1):
                if p in item["pages"]:
                    rank = r
                    break
            scored += 1
            if rank:
                hits += 1
                rr_sum += 1 / rank
            status = "HIT" if rank else "MISS"
        else:
            status = "n/a"
        print(f"{i:<3} {status:<6} {str(rank or '-'):<5} {str(pages):<22} {item['q'][:60]}")

        if args.answers:
            answer = cache.get(item["q"])
            if answer is None:
                context = "\n\n".join(f"[page {c['page']}]\n{c['text']}" for _, c in results)
                try:
                    answer = generate_with_retry(
                        PROMPT.format(context=context, question=item["q"])
                    ).text or ""
                except Exception as e:
                    errors += 1
                    print(f"      answer error (skipped): {str(e)[:90]}")
                    continue
                cache[item["q"]] = answer
                with open(cache_file, "w", encoding="utf-8") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=1)
                time.sleep(2)  # be gentle with the API
            ok = check_answer(item, answer)
            if ok is not None:
                ans_total += 1
                ans_pass += int(ok)
                print(f"      answer check: {'PASS' if ok else 'FAIL'}")

    recall = hits / scored if scored else 0
    mrr = rr_sum / scored if scored else 0
    print(f"\nRecall@{args.k}: {recall:.2f}  ({hits}/{scored})")
    print(f"MRR:       {mrr:.2f}")
    if args.answers and ans_total:
        print(f"Answer accuracy (auto-checked): {ans_pass}/{ans_total}")

    if errors:
        print(f"\n{errors} question(s) failed because the server was busy.")
        print("Run the same command again to finish them. Nothing was added to results.csv yet.")
        return

    new_file = not os.path.exists("results.csv")
    with open("results.csv", "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if new_file:
            w.writerow(["tag", "k", "recall", "mrr", "answer_pass", "answer_total"])
        w.writerow([args.tag, args.k, f"{recall:.3f}", f"{mrr:.3f}", ans_pass, ans_total])


if __name__ == "__main__":
    main()