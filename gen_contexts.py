#!/usr/bin/env python3
"""gen_contexts.py — stage 1: synthetic memory-recall scenarios ("contexts").

Each context is one Honcho-style situation: a peer (persona), the conclusions a
memory system holds about them (some relevant, some near-miss distractors), the
searches a recall agent would run to retrieve them, the question, and a rubric
(required_facts / forbidden_facts) for automatic scoring. Schema: trajectory.py.

Any OpenRouter or Anthropic model can generate. Two execution modes:

  run      concurrent synchronous requests (both providers); writes --out directly
  submit   Anthropic Message Batches (50% price, async); then `status` / `fetch`

Usage:
  python3 gen_contexts.py estimate --n 3000 --model deepseek
  python3 gen_contexts.py run      --n 3000 --model deepseek --out data/contexts.jsonl --concurrency 8
  python3 gen_contexts.py submit   --n 3000 --model opus     --out data/contexts.jsonl
  python3 gen_contexts.py status
  python3 gen_contexts.py fetch    [--manifest results/batches/contexts/<file>.json]

`run` is resume-safe: rows already good in --out are skipped; failed rows are
retried. Ids are c00000.. (offset with --start so several runs can be appended).
Cost: the estimate is always printed; --max-usd (run only) stops submitting new
requests once the usage-based spend passes it. Batches never abort.
"""
import argparse
import json
import random

import llm_backend as be
from trajectory import SEARCH_TOOLS

KIND = "contexts"
CTX_MAX_TOKENS = 4096   # a full scenario is ~1.5-2.5k tokens; must not truncate

CATEGORIES = {
    "factual":       "a direct question with one exact answer present in the relevant conclusions",
    "preference":    "a question about the peer's stated preferences, habits or standing instructions",
    "enumeration":   "'list all X' / 'how many Y' — the answer needs several distinct items spread over several conclusions",
    "contradiction": "two relevant conclusions give conflicting values for the same fact and neither is marked newer; the answer must name BOTH and say they conflict",
    "supersession":  "an older conclusion holds an old value and a newer one states it changed; the answer must give the NEW value as current",
    "abstention":    "the question asks about something NOT in any conclusion; every conclusion is a plausible near-miss; the correct answer is a clean 'no information about X'",
    "summary":       "a question asking for the pattern or development across several dated conclusions",
}
MIX = [("factual", 7), ("preference", 4), ("enumeration", 6), ("contradiction", 3),
       ("supersession", 3), ("abstention", 3), ("summary", 4)]

DOMAINS = ["health", "career", "music", "gaming", "baking", "travel", "family", "finance",
           "diy", "fitness", "reading", "home-lab", "cooking", "pets", "gardening", "language-learning",
           "photography", "cycling", "parenting", "software-projects", "relationships", "education",
           "cars", "film", "volunteering", "mental-health", "housing", "sports-fandom"]

NAMES = ["Daniel", "Maria", "Alex", "Priya", "Jonas", "Emily", "Kwame", "Sofia", "Liam", "Aisha",
         "Noah", "Hanna", "Mateo", "Yuki", "Oliver", "Chloe", "Ravi", "Elin", "Tomas", "Zara",
         "Felix", "Nadia", "Omar", "Ingrid", "Lucas", "Mei", "Erik", "Amara", "Leo", "Sara",
         "Viktor", "Fatima", "Hugo", "Lena", "Arjun", "Klara", "Samuel", "Ines", "Ben", "Freya"]

SYSTEM = ("You build synthetic training data for Honcho, a memory system that stores derived "
          "CONCLUSIONS about a person (the peer). Return ONLY one JSON object, complete and untruncated, "
          "no prose, no code fences.")

PROMPT = """Write ONE memory-recall scenario for Honcho.

Peer name: {name}
Domain: {domain}
Category: {cat} — {catdef}

A conclusion is a third-person statement about {name} that a memory system derived from their
conversations, with the date it was recorded (2025 or 2026). Make them concrete: names, numbers,
dates, places, product names. Vary sentence shapes. Do not write encyclopedia facts — write
things a person said or did.

Return exactly this JSON shape:
{{
  "persona": {{"name": "{name}", "bio": "<1-2 sentences about {name}>"}},
  "question": "<the question an application asks Honcho about {name}, 5-25 words; refer to the peer as '{name}' or 'the user'>",
  "observations": [
    {{"date": "2026-MM-DD", "text": "<conclusion about {name}, 10-50 words>", "relevant": true}},
    ...
  ],
  "searches": [
    {{"tool": "search_memory", "query": "<what a recall agent would search>", "results": [<indices into observations>]}},
    ...
  ],
  "required_facts": ["<exact value/name/date every correct answer must contain>", ...],
  "forbidden_facts": ["<distractor value a correct answer must NOT assert as true>", ...]
}}

Rules:
- 8 to 12 observations total: the relevant ones needed to answer ("relevant": true) plus 3-5
  near-miss distractors ("relevant": false) — same domain, plausible, but about something else,
  or a wrong / outdated value.
- 2 or 3 searches, tools from {tools}. Every observation index appears in at least one search.
  The first search must not return everything; the queries look like real search strings.
- required_facts: 1-4 short exact strings copied from the relevant observations.
  forbidden_facts: 1-4 short exact strings copied from distractors. For abstention scenarios
  required_facts is [] and ALL observations are distractors.
- Category-specific: contradiction -> exactly two relevant conclusions conflict, both values in
  required_facts. supersession -> the newer conclusion explicitly says the value changed; only the
  new value in required_facts, the old value in forbidden_facts. enumeration -> 4-7 items across
  several conclusions, each item in required_facts (max 4 listed).
"""


def category_sequence():
    """MIX weights interleaved (smooth weighted round-robin) so even a 10-row run
    touches every category in roughly the target proportions."""
    slots = sorted(((i + 0.5) / w, c) for c, w in MIX for i in range(w))
    return [c for _, c in slots]


def plan(n, start, seed):
    rnd = random.Random(seed)
    cats = category_sequence()
    rows = []
    for i in range(n):
        rows.append({"id": f"c{start + i:05d}", "category": cats[i % len(cats)],
                     "domain": rnd.choice(DOMAINS), "name": rnd.choice(NAMES)})
    return rows


def make_job(meta):
    user = PROMPT.format(name=meta["name"], domain=meta["domain"], cat=meta["category"],
                         catdef=CATEGORIES[meta["category"]], tools=", ".join(SEARCH_TOOLS))
    return {"custom_id": meta["id"], "system": SYSTEM, "user": user, "max_tokens": CTX_MAX_TOKENS}


def validate(obj, meta):
    """Coerce the model's JSON into a context row or return (None, reason)."""
    if not isinstance(obj, dict):
        return None, "not an object"
    persona = obj.get("persona")
    if isinstance(persona, str):
        persona = {"name": meta["name"], "bio": persona}
    if not isinstance(persona, dict) or not persona.get("name"):
        persona = {"name": meta["name"], "bio": (persona or {}).get("bio", "") if isinstance(persona, dict) else ""}
    q = obj.get("question")
    obs_in = obj.get("observations") or obj.get("findings") or []
    obs = []
    for o in obs_in:
        if isinstance(o, dict) and o.get("text") and o.get("date"):
            obs.append({"date": str(o["date"])[:10], "text": str(o["text"]).strip(),
                        "relevant": bool(o.get("relevant", True))})
    if not q or len(obs) < 4:
        return None, f"missing question or <4 observations ({len(obs)})"
    searches = []
    for s in obj.get("searches") or []:
        if not isinstance(s, dict) or not s.get("query"):
            continue
        idx = sorted({i for i in (s.get("results") or []) if isinstance(i, int) and 0 <= i < len(obs)})
        searches.append({"tool": s.get("tool") if s.get("tool") in SEARCH_TOOLS else "search_memory",
                         "query": str(s["query"]), "results": idx})
    covered = {i for s in searches for i in s["results"]}
    missing = [i for i in range(len(obs)) if i not in covered]
    if missing:  # every observation must be reachable through some search
        if searches:
            searches[-1]["results"] = sorted(set(searches[-1]["results"]) | set(missing))
        else:
            searches = [{"tool": "search_memory", "query": str(q), "results": list(range(len(obs)))}]
    req = [str(x) for x in (obj.get("required_facts") or []) if str(x).strip()]
    forb = [str(x) for x in (obj.get("forbidden_facts") or []) if str(x).strip()]
    if meta["category"] == "abstention":
        req = []
        for o in obs:
            o["relevant"] = False
    elif not req:
        return None, "no required_facts"
    return {"id": meta["id"], "category": meta["category"], "domain": meta["domain"],
            "persona": {"name": str(persona.get("name") or meta["name"]), "bio": str(persona.get("bio") or "")},
            "question": str(q).strip(), "observations": obs, "searches": searches,
            "required_facts": req, "forbidden_facts": forb}, ""


def to_row(meta, result):
    if result.get("error") or not result.get("text"):
        return {"id": meta["id"], "category": meta["category"], "domain": meta["domain"],
                "__failed__": f"__FAILED__: {result.get('error') or 'empty response'}"}
    ctx, why = validate(be.extract_json(result["text"]), meta)
    if ctx is None:
        return {"id": meta["id"], "category": meta["category"], "domain": meta["domain"],
                "__failed__": f"__FAILED__: invalid context ({why})", "raw": result["text"][:2000]}
    return ctx


def report(rows, path):
    good = [r for r in rows if not be.failed(r)]
    bad = len(rows) - len(good)
    by_cat = {}
    for r in good:
        by_cat[r["category"]] = by_cat.get(r["category"], 0) + 1
    print(f"wrote {path}: {len(rows)} rows, {len(good)} good, {bad} failed")
    print("  per category:", json.dumps(by_cat, sort_keys=True))
    if bad:
        print("  failed rows keep a __failed__ marker; re-run the same command to retry them.")


# ------------------------------------------------------------------ commands
def cmd_estimate(a):
    spec = be.resolve_model(a.model)
    jobs = [make_job(m) for m in plan(a.n, a.start, a.seed)]
    for batch in ((False, True) if spec.provider == "anthropic" else (False,)):
        usd, tin, tout = be.estimate_usd(spec, jobs, out_tokens_per_job=1800, batch=batch)
        print(f"{spec}  n={a.n}  {'batch' if batch else 'sync '}  in≈{tin/1e6:.2f}M out≈{tout/1e6:.2f}M  ≈ ${usd:.2f}")
    return 0


def cmd_run(a):
    spec = be.resolve_model(a.model)
    metas = plan(a.n, a.start, a.seed)
    existing = {r["id"]: r for r in be.read_jsonl(a.out)}
    todo = [m for m in metas if m["id"] not in existing or be.failed(existing[m["id"]])]
    jobs = [make_job(m) for m in todo]
    usd, tin, tout = be.estimate_usd(spec, jobs, 1800)
    print(f"{spec}: {len(todo)} contexts to generate ({len(metas) - len(todo)} already good in {a.out}); "
          f"estimate ≈ ${usd:.2f}" + (f" (cap ${a.max_usd:.2f})" if a.max_usd is not None else ""))
    if not todo:
        return 0
    by_id = {m["id"]: m for m in todo}
    rows = []

    def on_result(cid, r):
        row = to_row(by_id[cid], r)
        rows.append(row)
        tag = "FAIL" if be.failed(row) else "ok  "
        print(f"  [{len(rows)}/{len(todo)}] {cid} {tag} {row.get('question', row.get('__failed__', ''))[:70]}", flush=True)
        if len(rows) % 25 == 0:
            be.write_jsonl(a.out, _ordered(be.merge_rows(a.out, rows), metas))

    be.run_concurrent(spec, jobs, concurrency=a.concurrency, effort=a.effort,
                      on_result=on_result, max_usd=a.max_usd)
    merged = _ordered(be.merge_rows(a.out, rows), metas)
    be.write_jsonl(a.out, merged)
    report(merged, a.out)
    return 0


def _ordered(rows, metas):
    order = {m["id"]: i for i, m in enumerate(metas)}
    return sorted(rows, key=lambda r: (order.get(r["id"], 10**9), r["id"]))


def cmd_submit(a):
    spec = be.resolve_model(a.model)
    metas = plan(a.n, a.start, a.seed)
    existing = {r["id"]: r for r in be.read_jsonl(a.out)}
    todo = [m for m in metas if m["id"] not in existing or be.failed(existing[m["id"]])]
    jobs = [make_job(m) for m in todo]
    usd, tin, tout = be.estimate_usd(spec, jobs, 1800, batch=True)
    print(f"{spec} batch: {len(todo)} contexts; estimate ≈ ${usd:.2f} (reference only, no cap)")
    if not todo:
        return 0
    b = be.batch_submit(spec, jobs, effort=a.effort)
    path = be.write_manifest(KIND, {"batch_id": b["id"], "model": str(spec), "model_alias": a.model,
                                    "n": len(todo), "out": a.out, "est_usd": round(usd, 3),
                                    "metas": todo})
    print(f"submitted batch {b['id']} ({len(todo)} requests) -> manifest {path}")
    print("next: python3 gen_contexts.py fetch      (polls until ended, then writes --out)")
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
            print(f"{p}: status check failed ({type(e).__name__}: {e})")
            rc = 1
    return rc


def cmd_fetch(a):
    path = be.newest_manifest(KIND, a.manifest)
    m = json.load(open(path))
    print(f"manifest {path}: batch {m['batch_id']} {m['model']} n={m['n']} -> {m['out']}")
    if a.no_wait:
        st = be.batch_status(m["batch_id"]).get("processing_status")
        if st != "ended":
            print(f"still {st}; run fetch again later (or without --no-wait).")
            return 1
    else:
        be.batch_wait(m["batch_id"], a.poll_interval)
    results = be.batch_results(m["batch_id"])
    spec = be.resolve_model(m.get("model_alias") or m["model"].split(":", 1)[1])
    rows = [to_row(meta, results.get(meta["id"], {"error": "no result for this id"})) for meta in m["metas"]]
    spent = sum(be.usage_usd(spec, r.get("usage") or {}, batch=True) for r in results.values())
    merged = be.merge_rows(m["out"], rows)
    merged.sort(key=lambda r: r["id"])
    be.write_jsonl(m["out"], merged)
    report(merged, m["out"])
    print(f"usage-based spend for this batch ≈ ${spent:.3f}")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")

    def common(p, needs_model=True):
        p.add_argument("--n", type=int, default=30)
        p.add_argument("--start", type=int, default=0, help="first id number (append runs)")
        p.add_argument("--seed", type=int, default=7)
        p.add_argument("--out", default="data/contexts.jsonl")
        if needs_model:
            p.add_argument("--model", default="deepseek", help=f"alias ({', '.join(sorted(be.MODELS))}), anthropic:<id> or openrouter:<vendor/model>")
            p.add_argument("--effort", default="low", choices=["low", "medium", "high", "xhigh", "max", ""],
                           help="Anthropic thinking effort (ignored for OpenRouter)")

    p = sub.add_parser("estimate", help="local cost estimate, no network"); common(p); p.set_defaults(fn=cmd_estimate)
    p = sub.add_parser("run", help="concurrent sync generation (OpenRouter or Anthropic)"); common(p)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--max-usd", type=float, default=None, help="stop submitting once spend passes this")
    p.set_defaults(fn=cmd_run)
    p = sub.add_parser("submit", help="Anthropic Message Batch (50%% price, async)"); common(p); p.set_defaults(fn=cmd_submit)
    p = sub.add_parser("status", help="status of every submitted contexts batch"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("fetch", help="wait for a batch, write rows to its --out")
    p.add_argument("--manifest"); p.add_argument("--no-wait", action="store_true")
    p.add_argument("--poll-interval", type=int, default=60); p.set_defaults(fn=cmd_fetch)

    a = ap.parse_args()
    if not getattr(a, "fn", None):
        ap.print_help(); return 1
    return a.fn(a) or 0


if __name__ == "__main__":
    raise SystemExit(main())
