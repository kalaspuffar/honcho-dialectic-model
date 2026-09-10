#!/usr/bin/env python3
"""eval_model.py — run an Ollama model over held-out contexts on the Honcho
trajectory and score it with scoring.py (same rules as build_dataset.py).

  python3 eval_model.py --contexts data/contexts.jsonl --ids-from data/dataset_eval.dpo.jsonl \
      --model qwen3.5:9b --base http://node7.ea.org:11434/v1 --out results/eval_base.jsonl
  python3 eval_model.py --contexts data/contexts.jsonl --ids-from data/dataset_eval.dpo.jsonl \
      --model dialectic-v1 --out results/eval_v1.jsonl
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
    print(f"{a.model} @ {a.base}: {len(ctxs)} contexts", file=sys.stderr)

    def one(c):
        try:
            r = answer_with_ollama(a.base, a.model, c, max_rounds=a.max_rounds, temperature=a.temperature,
                                   max_tokens=a.max_tokens)
        except Exception as e:  # noqa: BLE001
            r = {"answer": "", "extra_calls": 0, "forced": False, "error": f"{type(e).__name__}: {e}"}
        s = scoring.score_answer(c, r["answer"])
        return c, {"id": c["id"], "category": c.get("category"), "answer": r["answer"],
                   "extra_calls": r.get("extra_calls", 0), "forced": r.get("forced", False),
                   "error": r.get("error", ""), **s}

    rows, scores, used = [], [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, a.concurrency)) as ex:
        for i, (c, row) in enumerate(ex.map(one, ctxs), 1):
            rows.append(row); scores.append({k: row[k] for k in ("words", "coverage", "fab", "abst", "hedge")}); used.append(c)
            print(f"[{i:3}/{len(ctxs)}] {c['id']} [{c.get('category', ''):13}] {row['words']:4}w cov={row['coverage']:.2f} "
                  f"fab={row['fab']} abst={row['abst']} hedge={row['hedge']} extra_calls={row['extra_calls']}", file=sys.stderr)
    agg = scoring.aggregate(used, scores)
    agg.update(model=a.model, forced_rows=sum(1 for r in rows if r["forced"]),
               rows_with_extra_tool_calls=sum(1 for r in rows if r["extra_calls"]))
    be.write_jsonl(a.out, rows)
    with open(os.path.splitext(a.out)[0] + ".summary.json", "w") as f:
        json.dump(agg, f, indent=2)
    print("\nAGGREGATE\n" + json.dumps(agg, indent=2))


def compare(a):
    tables = []
    for p in a.files:
        sp = os.path.splitext(p)[0] + ".summary.json"
        if os.path.exists(sp):
            tables.append(json.load(open(sp)))
        else:
            sys.exit(f"missing {sp} (produced by a run)")
    keys = ["model", "n", "median_words", "mean_words", "max_words", "mean_coverage",
            "fabrication_rows", "abstention_correct", "hedge_rows", "empty_rows", "forced_rows",
            "rows_with_extra_tool_calls"]
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
    p.add_argument("--base", default=be.setting("OLLAMA_BASE", "http://node7.ea.org:11434/v1"))
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--max-rounds", type=int, default=3)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=0.1)
    p = sub.add_parser("compare"); p.set_defaults(fn=compare)
    p.add_argument("files", nargs="+")
    argv = sys.argv[1:]
    if argv and argv[0] not in ("run", "compare", "-h", "--help"):
        argv = ["run"] + argv          # `eval_model.py --contexts ...` == `eval_model.py run ...`
    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        ap.print_help(); return 1
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
