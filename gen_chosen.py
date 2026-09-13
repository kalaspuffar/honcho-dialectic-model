#!/usr/bin/env python3
"""gen_chosen.py — stage 3: the teacher writes the ideal terse answer (CHOSEN).

The teacher sees exactly what the student sees — the retrieved conclusions as
tool results (trajectory.format_tool_result) — plus the scenario rubric, and
returns {"answer": "..."}. Any OpenRouter or Anthropic model; same two modes as
gen_contexts.py:

  python3 gen_chosen.py estimate --contexts data/contexts.jsonl --model opus
  python3 gen_chosen.py run      --contexts data/contexts.jsonl --model opus --out data/chosen.jsonl
  python3 gen_chosen.py submit   --contexts data/contexts.jsonl --model opus --out data/chosen.jsonl
  python3 gen_chosen.py status
  python3 gen_chosen.py fetch    [--manifest results/batches/chosen/<file>.json]

Output rows: {id, category, answer, words, teacher, score}. Failed rows keep an
"__FAILED__: ..." answer so build_dataset.py drops them and a re-run retries them.
"""
import argparse
import json

import llm_backend as be
import scoring
from trajectory import format_tool_result, peer_name, searches_for

KIND = "chosen"
MAX_TOKENS_OUT = 800

SYSTEM = """You write the IDEAL final answer of Honcho's dialectic — a memory-recall agent that has
just finished searching a memory system — for a fine-tuning dataset. You see the tool results the
agent retrieved, and a rubric. Write the answer the best possible terse recall agent would give.

Hard rules:
1. LENGTH: 1-3 short sentences, at most 50 words. Answer the question asked; do not recite every
   retrieved conclusion. No preamble ("Based on my memory…"), no restating the question, no
   narration of searches, no closing offer. A compact list is allowed only for enumeration.
2. GROUNDED ONLY: every name/date/number/item must come from the retrieved conclusions. No outside
   knowledge, no guessing, no hedging words (likely / probably / seems / might).
3. Refer to the peer the way the question does (by name, or "you"/"the user").
4. CONTRADICTION: name BOTH conflicting values and state plainly that memory holds both.
5. SUPERSESSION: give the newest value as current; mentioning the old value is optional and brief.
6. ABSTENTION: say plainly that there is no information about that topic in memory (name the
   topic). Do not offer the near-miss facts as a substitute.
7. QUOTE exact values (dates, numbers, names) — do not paraphrase them.

Return ONLY a JSON object, no prose, no code fences: {"answer": "..."}"""


def row_block(ctx) -> str:
    obs = ctx.get("observations") or []
    parts = [f"PEER: {peer_name(ctx)} — {ctx.get('persona', {}).get('bio', '')}".rstrip(" —"),
             f"QUESTION: {ctx['question']}", "", "RETRIEVED (in the order the agent saw them):"]
    for s in searches_for(ctx):
        parts.append(f"\n{s['tool']}(query={json.dumps(s['query'])}) ->")
        parts.append(format_tool_result(s["tool"], [obs[i] for i in s["results"]]))
    parts += ["", f"RUBRIC (for you, not for the answer): category={ctx.get('category')}",
              f"  must contain: {ctx.get('required_facts') or '[] (abstention: nothing is relevant)'}",
              f"  must not assert as true: {ctx.get('forbidden_facts') or []}"]
    return "\n".join(parts)


def make_job(ctx):
    return {"custom_id": ctx["id"], "system": SYSTEM, "user": row_block(ctx), "max_tokens": MAX_TOKENS_OUT}


def to_row(ctx, result, teacher):
    if result.get("error") or not result.get("text"):
        ans = f"__FAILED__: {result.get('error') or 'empty response'}"
    else:
        j = be.extract_json(result["text"])
        ans = j.get("answer") if isinstance(j, dict) else None
        ans = ans.strip() if isinstance(ans, str) and ans.strip() else result["text"].strip()
    row = {"id": ctx["id"], "category": ctx.get("category"), "answer": ans,
           "words": scoring.words(ans) if not ans.startswith("__FAILED__") else 0, "teacher": teacher}
    if not ans.startswith("__FAILED__"):
        row["score"] = scoring.score_answer(ctx, ans)
    return row


def load_contexts(path):
    ctxs = [c for c in be.read_jsonl(path) if not be.failed(c)]
    if not ctxs:
        raise SystemExit(f"no good contexts in {path}")
    ids = [c["id"] for c in ctxs]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate context ids in " + path)
    return ctxs


def report(rows, ctxs, path):
    by_id = {c["id"]: c for c in ctxs}
    good = [r for r in rows if not be.failed(r) and r["id"] in by_id]
    print(f"wrote {path}: {len(rows)} rows, {len(good)} good, {len(rows) - len(good)} failed")
    if good:
        agg = scoring.aggregate([by_id[r["id"]] for r in good], [r.get("score") or scoring.score_answer(by_id[r["id"]], r["answer"]) for r in good])
        print("  teacher quality:", json.dumps(agg))
        low = [r["id"] for r in good if r["score"]["coverage"] < 0.5 and by_id[r["id"]]["category"] != "abstention"]
        if low:
            print(f"  {len(low)} rows with coverage < 0.5 (build_dataset will drop those with no required fact): {low[:10]}")


def default_out(contexts_path):
    import os
    d, b = os.path.split(contexts_path)
    return os.path.join(d, b.replace("contexts", "chosen") if "contexts" in b else "chosen.jsonl")


# ------------------------------------------------------------------ commands
def cmd_estimate(a):
    spec = be.resolve_model(a.model)
    ctxs = load_contexts(a.contexts)
    jobs = [make_job(c) for c in ctxs]
    for batch in ((False, True) if spec.provider == "anthropic" else (False,)):
        usd, tin, tout = be.estimate_usd(spec, jobs, out_tokens_per_job=70, batch=batch)
        print(f"{spec}  n={len(ctxs)}  {'batch' if batch else 'sync '}  in≈{tin/1e6:.2f}M out≈{tout/1e6:.3f}M  ≈ ${usd:.2f}"
              + ("  (batch: shared system prompt is also prompt-cached; real cost lower)" if batch else ""))
    return 0


def _todo(a, ctxs):
    out = a.out or default_out(a.contexts)
    existing = {r["id"]: r for r in be.read_jsonl(out)}
    return out, [c for c in ctxs if c["id"] not in existing or be.failed(existing[c["id"]])]


def cmd_run(a):
    spec = be.resolve_model(a.model)
    ctxs = load_contexts(a.contexts)
    out, todo = _todo(a, ctxs)
    jobs = [make_job(c) for c in todo]
    usd, _, _ = be.estimate_usd(spec, jobs, 70)
    print(f"{spec}: {len(todo)} answers to write ({len(ctxs) - len(todo)} already good in {out}); estimate ≈ ${usd:.2f}")
    if not todo:
        return 0
    by_id, rows = {c["id"]: c for c in todo}, []

    def on_result(cid, r):
        row = to_row(by_id[cid], r, a.model)
        rows.append(row)
        print(f"  [{len(rows)}/{len(todo)}] {cid} {row['words']:3d}w  {row['answer'][:70]!r}", flush=True)
        if len(rows) % 25 == 0:
            be.write_jsonl(out, _ordered(be.merge_rows(out, rows), ctxs))

    be.run_concurrent(spec, jobs, concurrency=a.concurrency, effort=a.effort, on_result=on_result, max_usd=a.max_usd)
    merged = _ordered(be.merge_rows(out, rows), ctxs)
    be.write_jsonl(out, merged)
    report(merged, ctxs, out)
    return 0


def _ordered(rows, ctxs):
    order = {c["id"]: i for i, c in enumerate(ctxs)}
    return sorted(rows, key=lambda r: (order.get(r["id"], 10**9), r["id"]))


def cmd_submit(a):
    spec = be.resolve_model(a.model)
    ctxs = load_contexts(a.contexts)
    out, todo = _todo(a, ctxs)
    jobs = [make_job(c) for c in todo]
    usd, _, _ = be.estimate_usd(spec, jobs, 70, batch=True)
    print(f"{spec} batch: {len(todo)} answers; estimate ≈ ${usd:.2f} (reference only, no cap)")
    if not todo:
        return 0
    b = be.batch_submit(spec, jobs, effort=a.effort)
    import os
    path = be.write_manifest(KIND, {"batch_id": b["id"], "model": str(spec), "model_alias": a.model,
                                    "n": len(todo), "out": out, "contexts": os.path.abspath(a.contexts),
                                    "ids": [c["id"] for c in todo], "est_usd": round(usd, 3)})
    print(f"submitted batch {b['id']} ({len(todo)} requests) -> manifest {path}")
    print("next: python3 gen_chosen.py fetch")
    return 0


def cmd_status(a):
    rc = 0
    for p in be.list_manifests(KIND):
        m = json.load(open(p))
        try:
            b = be.batch_status(m["batch_id"])
            st = b.get("processing_status")
            print(f"{p}: {m['model']} n={m['n']} status={st} {b.get('request_counts', {})}")
            rc = rc or (0 if st == "ended" else 1)
        except Exception as e:  # noqa: BLE001
            print(f"{p}: status check failed ({type(e).__name__}: {e})"); rc = 1
    return rc


def cmd_fetch(a):
    path = be.newest_manifest(KIND, a.manifest)
    m = json.load(open(path))
    print(f"manifest {path}: batch {m['batch_id']} {m['model']} n={m['n']} -> {m['out']}")
    if a.no_wait:
        st = be.batch_status(m["batch_id"]).get("processing_status")
        if st != "ended":
            print(f"still {st}; run fetch again later."); return 1
    else:
        be.batch_wait(m["batch_id"], a.poll_interval)
    ctxs = load_contexts(m["contexts"])
    by_id = {c["id"]: c for c in ctxs}
    results = be.batch_results(m["batch_id"])
    spec = be.resolve_model(m.get("model_alias") or m["model"].split(":", 1)[1])
    rows = [to_row(by_id[i], results.get(i, {"error": "no result for this id"}), m.get("model_alias", m["model"]))
            for i in m["ids"] if i in by_id]
    spent = sum(be.usage_usd(spec, r.get("usage") or {}, batch=True) for r in results.values())
    merged = _ordered(be.merge_rows(m["out"], rows), ctxs)
    be.write_jsonl(m["out"], merged)
    report(merged, ctxs, m["out"])
    print(f"usage-based spend for this batch ≈ ${spent:.3f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    def common(p):
        p.add_argument("--contexts", required=True)
        p.add_argument("--out", default=None, help="default: chosen.jsonl next to --contexts")
        p.add_argument("--model", default="opus", help=f"alias ({', '.join(sorted(be.MODELS))}), anthropic:<id> or openrouter:<vendor/model>")
        p.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max", ""],
                       help="Anthropic thinking effort (ignored for OpenRouter)")

    p = sub.add_parser("estimate"); common(p); p.set_defaults(fn=cmd_estimate)
    p = sub.add_parser("run", help="concurrent sync (OpenRouter or Anthropic)"); common(p)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-usd", type=float, default=None)
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("submit", help="Anthropic Message Batch"); common(p); p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("fetch"); p.add_argument("--manifest"); p.add_argument("--no-wait", action="store_true")
    p.add_argument("--poll-interval", type=int, default=60); p.set_defaults(fn=cmd_fetch)

    a = ap.parse_args()
    if not getattr(a, "fn", None):
        ap.print_help(); return 1
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
