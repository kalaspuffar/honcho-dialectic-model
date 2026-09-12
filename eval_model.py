#!/usr/bin/env python3
"""eval_model.py — run a student model (Ollama, or OpenRouter for the untuned base)
over held-out contexts on the Honcho trajectory and score it with scoring.py
(same rules as build_dataset.py).

  python3 eval_model.py --contexts data/contexts.jsonl --ids-from data/dataset_eval.dpo.jsonl \
      --model qwen3.5:9b --base http://node7.ea.org:11434/v1 --out results/eval_base.jsonl
  python3 eval_model.py --contexts data/contexts.jsonl --ids-from data/dataset_eval.dpo.jsonl \
      --model dialectic-v1 --out results/eval_v1.jsonl
  python3 eval_model.py --contexts ... --ids-from ... --model qwen9b --out results/eval_base_or.jsonl   # OpenRouter
  python3 eval_model.py compare results/eval_base.jsonl results/eval_v1.jsonl

--ids-from restricts to the ids in a dataset file (use the *eval* split — the
model never saw those personas). Without it every good context is used.
"""
import argparse
import concurrent.futures
import json
import os
import sys

import llm_backend as be
import scoring
from trajectory import answer_with_ollama


def run(a):
    ctxs = [c for c in be.read_jsonl(a.contexts) if not be.failed(c)]
    if a.ids_from:
        keep = {r["id"] for r in be.read_jsonl(a.ids_from)}
        ctxs = [c for c in ctxs if c["id"] in keep]
    if a.limit:
        ctxs = ctxs[:a.limit]
    if not ctxs:
        sys.exit("no contexts selected")
    provider, base, model_id, api_key, _spec = be.student_endpoint(a.model, a.base)
    print(f"{provider} {model_id} @ {base}: {len(ctxs)} contexts", file=sys.stderr)

    def one(c):
        try:
            r = answer_with_ollama(base, model_id, c, max_rounds=a.max_rounds, temperature=a.temperature,
                                   max_tokens=a.max_tokens, api_key=api_key,
                                   extra_body={"reasoning_effort": a.reasoning_effort} if a.reasoning_effort else None)
        except Exception as e:  # noqa: BLE001
            r = {"answer": "", "extra_calls": 0, "forced": False, "error": f"{type(e).__name__}: {e}"}
        reasoning = r.get("reasoning") or ""
        in_thinking = not r["answer"] and bool(reasoning)      # answered inside <think>, empty content
        answer = reasoning.strip() if (in_thinking and a.answer_from_reasoning) else r["answer"]
        s = scoring.score_answer(c, answer)
        return c, {"id": c["id"], "category": c.get("category"), "answer": answer,
                   "extra_calls": r.get("extra_calls", 0), "forced": r.get("forced", False),
                   "finish_reason": r.get("finish_reason"), "answered_in_thinking": in_thinking,
                   "reasoning_chars": len(reasoning), "error": r.get("error", ""), **s}

    rows, scores, used = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        for i, (c, row) in enumerate(ex.map(one, ctxs), 1):
            rows.append(row); scores.append({k: row[k] for k in ("words", "coverage", "fab", "abst", "hedge")}); used.append(c)
            print(f"[{i:3}/{len(ctxs)}] {c['id']} [{c.get('category', ''):13}] {row['words']:4}w cov={row['coverage']:.2f} "
                  f"fab={row['fab']} abst={row['abst']} hedge={row['hedge']} extra_calls={row['extra_calls']}", file=sys.stderr)
    agg = scoring.aggregate(used, scores)
    agg.update(model=a.model, reasoning_effort=a.reasoning_effort, forced_rows=sum(1 for r in rows if r["forced"]),
               rows_with_extra_tool_calls=sum(1 for r in rows if r["extra_calls"]),
               # content empty, reasoning text present: the model answered inside <think> and stopped
               # (qwen3.5:9b on Ollama). Honcho would see nothing. With --answer-from-reasoning the
               # reasoning text is scored instead so the base still yields words/coverage numbers.
               answered_in_thinking_rows=sum(1 for r in rows if r["answered_in_thinking"]),
               scored_from_reasoning=bool(a.answer_from_reasoning))
    be.write_jsonl(a.out, rows)
    with open(os.path.splitext(a.out)[0] + ".summary.json", "w") as f:
        json.dump(agg, f, indent=2)
    print("\nAGGREGATE\n" + json.dumps(agg, indent=2))


def rescore(a):
    """Re-score an existing results jsonl with the current scoring.py and rewrite its summary
    (scorer changes must not require re-running the model)."""
    ctxs = {c["id"]: c for c in be.read_jsonl(a.contexts)}
    rows = be.read_jsonl(a.file)
    out, scores, used = [], [], []
    for r in rows:
        c = ctxs.get(r["id"])
        if c is None:
            sys.exit(f"{r['id']} not in {a.contexts}")
        s = scoring.score_answer(c, r["answer"])
        r = {**r, **s}
        out.append(r); used.append(c); scores.append({k: r[k] for k in ("words", "coverage", "fab", "abst", "hedge")})
    agg = scoring.aggregate(used, scores)
    old = json.load(open(os.path.splitext(a.file)[0] + ".summary.json")) if os.path.exists(os.path.splitext(a.file)[0] + ".summary.json") else {}
    agg.update({k: v for k, v in old.items() if k not in agg})
    agg["rescored"] = True
    be.write_jsonl(a.file, out)
    with open(os.path.splitext(a.file)[0] + ".summary.json", "w") as f:
        json.dump(agg, f, indent=2)
    print(json.dumps(agg, indent=2))


def compare(a):
    tables = []
    for p in a.files:
        sp = os.path.splitext(p)[0] + ".summary.json"
        if os.path.exists(sp):
            tables.append(json.load(open(sp)))
        else:
            sys.exit(f"missing {sp} (produced by a run)")
    keys = ["model", "reasoning_effort", "rescored", "n", "median_words", "mean_words", "max_words", "mean_coverage",
            "fabrication_rows", "abstention_correct", "hedge_rows", "empty_rows", "answered_in_thinking_rows",
            "scored_from_reasoning", "forced_rows", "rows_with_extra_tool_calls"]
    w = max(len(k) for k in keys)
    print(" " * w + "  " + "  ".join(f"{str(t.get('model', '?')):>18}" for t in tables))
    for k in keys[1:]:
        print(f"{k:{w}}  " + "  ".join(f"{str(t.get(k, '')):>18}" for t in tables))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("run"); p.set_defaults(fn=run)
    p.add_argument("--contexts", required=True)
    p.add_argument("--ids-from", default=None, help="dataset jsonl whose ids select the eval contexts")
    p.add_argument("--model", required=True)
    p.add_argument("--base", default=None, help="default: OLLAMA_BASE, or OPENROUTER_BASE for OpenRouter models")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--reasoning-effort", default=None,
                   help="send reasoning_effort (e.g. none) like Honcho's MODEL_CONFIG__THINKING_EFFORT; "
                        "Ollama's /v1 maps 'none' to think=false")
    p.add_argument("--answer-from-reasoning", action="store_true",
                   help="when content is empty but a reasoning field came back, score the reasoning text "
                        "(baseline column: qwen3.5:9b on Ollama answers inside <think>; the summary still "
                        "reports answered_in_thinking_rows)")
    p = sub.add_parser("compare"); p.set_defaults(fn=compare)
    p.add_argument("files", nargs="+")
    p = sub.add_parser("rescore", help="re-score a results jsonl with the current scoring.py (rewrites it + its summary)")
    p.set_defaults(fn=rescore)
    p.add_argument("file")
    p.add_argument("--contexts", required=True)
    argv = sys.argv[1:]
    if argv and argv[0] not in ("run", "compare", "rescore", "-h", "--help"):
        argv = ["run"] + argv          # `eval_model.py --contexts ...` == `eval_model.py run ...`
    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        ap.print_help(); return 1
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
