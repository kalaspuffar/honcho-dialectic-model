#!/usr/bin/env python3
"""gen_rejected.py — stage 2: the base model answers each context -> REJECTED.

Runs the student (qwen3.5:9b on Ollama by default) on the exact Honcho trajectory
(system prompt, question, tool calls, tool results) and records its natural
verbose answer. This is the DPO "rejected" side — the real distribution we are
training away from (PLAN §3.1: never replace this with a teacher-written
"verbose" answer).

  python3 gen_rejected.py --contexts data/contexts.jsonl --out data/rejected.jsonl
  python3 gen_rejected.py --contexts data/contexts.jsonl --out data/rejected.jsonl --only c00001,c00002
  OLLAMA_BASE=http://node7.ea.org:11434/v1 OLLAMA_MODEL=qwen3.5:9b python3 gen_rejected.py ...

Resume-safe: good rows in --out are kept, failed/missing rows are (re)generated.
Output rows: {id, category, answer, words, extra_calls, forced}.
"""
import argparse
import concurrent.futures
import os
import time

import llm_backend as be
from trajectory import answer_with_ollama


def one(base, model, ctx, max_rounds, temperature):
    for attempt in range(2):
        try:
            r = answer_with_ollama(base, model, ctx, max_rounds=max_rounds, temperature=temperature)
            if r["answer"]:
                return {"id": ctx["id"], "category": ctx.get("category"), "answer": r["answer"],
                        "words": len(r["answer"].split()), "extra_calls": r["extra_calls"], "forced": r["forced"]}
            err = "empty answer"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        time.sleep(2)
    return {"id": ctx["id"], "category": ctx.get("category"), "answer": f"__FAILED__: {err}", "words": 0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out", default="data/rejected.jsonl")
    ap.add_argument("--base", default=be.setting("OLLAMA_BASE", "http://node7.ea.org:11434/v1"))
    ap.add_argument("--model", default=be.setting("OLLAMA_MODEL", "qwen3.5:9b"))
    ap.add_argument("--only", default="", help="comma-separated context ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1, help="parallel requests to Ollama")
    ap.add_argument("--max-rounds", type=int, default=3, help="extra tool-call rounds before forcing an answer")
    ap.add_argument("--temperature", type=float, default=0.3)
    a = ap.parse_args()

    ctxs = [c for c in be.read_jsonl(a.contexts) if not be.failed(c)]
    if a.only:
        keep = set(a.only.split(","))
        ctxs = [c for c in ctxs if c["id"] in keep]
    if a.limit:
        ctxs = ctxs[:a.limit]
    existing = {r["id"]: r for r in be.read_jsonl(a.out)}
    todo = [c for c in ctxs if c["id"] not in existing or be.failed(existing[c["id"]])]
    print(f"{a.model} @ {a.base}: {len(todo)} to answer ({len(ctxs) - len(todo)} already good in {a.out})")
    rows, t0 = [], time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        for rec in ex.map(lambda c: one(a.base, a.model, c, a.max_rounds, a.temperature), todo):
            rows.append(rec)
            flag = "" if not rec.get("forced") else " (forced)"
            print(f"  [{len(rows)}/{len(todo)}] {rec['id']} {rec['words']:4d}w{flag}  {rec['answer'][:70]!r}", flush=True)
            if len(rows) % 20 == 0:
                be.write_jsonl(a.out, sorted(be.merge_rows(a.out, rows), key=lambda r: r["id"]))
    merged = sorted(be.merge_rows(a.out, rows), key=lambda r: r["id"])
    be.write_jsonl(a.out, merged)
    failed = sum(1 for r in merged if be.failed(r))
    good = [r["words"] for r in merged if not be.failed(r)]
    med = sorted(good)[len(good) // 2] if good else None
    print(f"saved {a.out}: {len(merged)} rows, {len(rows)} new, {failed} failed, median {med} words, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
