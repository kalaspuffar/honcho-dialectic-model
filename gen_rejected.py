#!/usr/bin/env python3
"""gen_rejected.py — stage 2: the base model answers each context -> REJECTED.

Runs the student (qwen3.5:9b on Ollama by default) on the exact Honcho trajectory
(system prompt, question, tool calls, tool results) and records its natural
verbose answer. This is the DPO "rejected" side — the real distribution we are
training away from (PLAN §3.1: never replace this with a teacher-written
"verbose" answer).

The student can also run on OpenRouter (same weights, `qwen/qwen3.5-9b`) when the
Ollama host is busy: give an OpenRouter alias or `vendor/model` id as --model and the
script switches provider, prints the cost estimate and tracks live spend.

  python3 gen_rejected.py --contexts data/contexts.jsonl --out data/rejected.jsonl
  python3 gen_rejected.py --contexts data/contexts.jsonl --out data/rejected.jsonl --only c00001,c00002
  OLLAMA_BASE=http://localhost:11434/v1 OLLAMA_MODEL=qwen3.5:9b python3 gen_rejected.py ...
  python3 gen_rejected.py --contexts data/contexts.jsonl --out data/rejected.jsonl \
      --model qwen9b --concurrency 8 [--max-usd 2]              # OpenRouter (OPENROUTER_API_KEY)

Resume-safe: good rows in --out are kept, failed/missing rows are (re)generated.
Output rows: {id, category, answer, words, extra_calls, forced}.
"""
import argparse
import concurrent.futures
import threading
import time

import llm_backend as be
from trajectory import answer_with_ollama, build_messages

OUT_TOKENS_PER_ROW = 500        # estimate only: verbose base answers + occasional extra round


def one(base, model, ctx, max_rounds, temperature, max_tokens, api_key, usage=None):
    for attempt in range(2):
        try:
            r = answer_with_ollama(base, model, ctx, max_rounds=max_rounds, temperature=temperature,
                                   max_tokens=max_tokens, api_key=api_key)
            if usage is not None:
                usage.append(r.get("usage") or {})
            if r["answer"]:
                return {"id": ctx["id"], "category": ctx.get("category"), "answer": r["answer"],
                        "words": len(r["answer"].split()), "extra_calls": r["extra_calls"], "forced": r["forced"]}
            err = "empty answer"
        except Exception as e:  # noqa: BLE001
            err = f"{type(e).__name__}: {e}"
        time.sleep(2)
    return {"id": ctx["id"], "category": ctx.get("category"), "answer": f"__FAILED__: {err}", "words": 0}


def estimate(spec, ctxs, max_tokens):
    """Printed before any OpenRouter spend (cost policy: estimates are always printed)."""
    jobs = [{"system": "", "user": "".join(str(m.get("content") or "") for m in build_messages(c))}
            for c in ctxs]
    return be.estimate_usd(spec, jobs, min(OUT_TOKENS_PER_ROW, max_tokens))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--out", default="data/rejected.jsonl")
    ap.add_argument("--base", default=None,
                    help="OpenAI-compatible base URL (default: OLLAMA_BASE, or OPENROUTER_BASE for OpenRouter models)")
    ap.add_argument("--model", default=be.setting("OLLAMA_MODEL", be.OLLAMA_DEFAULT_MODEL),
                    help="Ollama tag (qwen3.5:9b) or OpenRouter alias/id (qwen9b, openrouter:qwen/qwen3.5-9b)")
    ap.add_argument("--only", default="", help="comma-separated context ids")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--concurrency", type=int, default=1, help="parallel requests (Ollama: 1; OpenRouter: 8 is fine)")
    ap.add_argument("--max-rounds", type=int, default=3, help="extra tool-call rounds before forcing an answer")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=1500, help="per-request completion cap (includes reasoning tokens)")
    ap.add_argument("--max-usd", type=float, default=None,
                    help="OpenRouter only: stop starting new rows once usage-based spend exceeds this")
    a = ap.parse_args()
    provider, base, model_id, api_key, spec = be.student_endpoint(a.model, a.base)

    ctxs = [c for c in be.read_jsonl(a.contexts) if not be.failed(c)]
    if a.only:
        keep = set(a.only.split(","))
        ctxs = [c for c in ctxs if c["id"] in keep]
    if a.limit:
        ctxs = ctxs[:a.limit]
    existing = {r["id"]: r for r in be.read_jsonl(a.out)}
    todo = [c for c in ctxs if c["id"] not in existing or be.failed(existing[c["id"]])]
    print(f"{provider} {model_id} @ {base}: {len(todo)} to answer ({len(ctxs) - len(todo)} already good in {a.out})")
    spent, lock, stop = [0.0], threading.Lock(), threading.Event()
    if provider == "openrouter":
        usd, tin, tout = estimate(spec, todo, a.max_tokens)
        print(f"estimate: ~${usd:.2f} for {len(todo)} rows ({tin:,} in / {tout:,} out tokens at list price)"
              + (f"; live cap ${a.max_usd:.2f}" if a.max_usd is not None else "; no cap (--max-usd)"))

    def work(c):
        if stop.is_set():
            return {"id": c["id"], "category": c.get("category"), "answer": "__FAILED__: cost cap", "words": 0}
        u = []
        rec = one(base, model_id, c, a.max_rounds, a.temperature, a.max_tokens, api_key, u)
        if spec is not None:
            with lock:
                spent[0] += sum(be.usage_usd(spec, x) for x in u)
                if a.max_usd is not None and spent[0] > a.max_usd:
                    stop.set()
        return rec

    rows, t0 = [], time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        for rec in ex.map(work, todo):
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
    if provider == "openrouter":
        capped = " (cost cap reached; rerun to resume)" if stop.is_set() else ""
        print(f"openrouter usage-based spend ≈ ${spent[0]:.3f}{capped}")


if __name__ == "__main__":
    main()
