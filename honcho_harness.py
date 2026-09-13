#!/usr/bin/env python3
"""honcho_harness.py — the Phase D acceptance test: ask Honcho's dialectic endpoint a fixed set of
questions through the REAL loop (tool calls, prompt loadout, level budget) and record words,
latency and simple quality flags per question. Replaces the vault's honcho_verbosity_test.py
(PLAN §7), which printed words/latency for 5 questions; this one also takes the hard set and can
diff two runs. Stdlib only.

  python3 honcho_harness.py --base http://honcho:8000 --workspace W --peer P --questions harness_questions.json \
                            --level low --label dialectic_s50 [--key JWT] [--target OTHER_PEER] [--session S]
  python3 honcho_harness.py compare results/harness-A.json results/harness-B.json

Questions file: JSON list of {"id", "category", "query", "expect": [strings that a correct answer
contains]} — `expect` is optional; when present, coverage = fraction of `expect` strings found
(scoring.has_entity), and for category "abstention" the answer must be a refusal (scoring.REFUSAL).
The words/latency of the PLAN §1 baseline (median 304 w) came from the original 5 questions; put
those first, with the same ids, so the trail stays comparable.
"""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

import scoring


def ask(base, workspace, peer, query, level, key=None, target=None, session=None, timeout=600):
    url = f"{base.rstrip('/')}/v3/workspaces/{workspace}/peers/{peer}/chat"
    body = {"query": query, "reasoning_level": level, "stream": False}
    if target:
        body["target"] = target
    if session:
        body["session_id"] = session
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers=headers, method="POST")
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read())
        return (data.get("content") or ""), time.time() - t0, ""
    except urllib.error.HTTPError as e:
        return "", time.time() - t0, f"HTTP {e.code}: {e.read()[:300].decode(errors='replace')}"
    except Exception as e:  # noqa: BLE001
        return "", time.time() - t0, f"{type(e).__name__}: {e}"


def score(q, answer):
    w = scoring.words(answer)
    hedge = bool(scoring.HEDGE.search(answer))
    refusal = bool(scoring.REFUSAL.search(answer))
    exp = [e for e in (q.get("expect") or []) if e]
    narration = bool(scoring.NARRATION.search(answer))
    row = {"words": w, "hedge": hedge, "refusal": refusal, "empty": w == 0, "narration": narration}
    if narration:
        row["ok"] = False
    elif q.get("category") == "abstention":
        row["ok"] = refusal and w <= scoring.ABSTENTION_MAX_WORDS and not hedge
    elif exp:
        hits = sum(1 for e in exp if scoring.has_entity(answer, e))
        row["coverage"] = round(hits / len(exp), 3)
        row["ok"] = hits > 0
    return row


def run(a):
    qs = json.load(open(a.questions))
    key = a.key or os.environ.get("HONCHO_API_KEY")
    rows = []
    for i, q in enumerate(qs, 1):
        ans, dt, err = ask(a.base, a.workspace, a.peer, q["query"], a.level, key, a.target, a.session, a.timeout)
        s = score(q, ans)
        rows.append({"id": q.get("id", f"q{i}"), "category": q.get("category", ""), "query": q["query"],
                     "answer": ans, "latency_s": round(dt, 1), "error": err, **s})
        flag = " ".join(k for k in ("hedge", "refusal", "empty", "narration") if s.get(k))
        okm = "" if "ok" not in s else ("OK " if s["ok"] else "MISS ")
        print(f"[{i:2}/{len(qs)}] {rows[-1]['id']:12} {q.get('category', ''):13} {s['words']:4}w {dt:6.1f}s {okm}{flag} {err}",
              file=sys.stderr)
    ws = [r["words"] for r in rows]
    summary = {"label": a.label, "level": a.level, "base": a.base, "workspace": a.workspace, "peer": a.peer,
               "n": len(rows), "median_words": int(statistics.median(ws)) if ws else None,
               "max_words": max(ws) if ws else None,
               "median_latency_s": round(statistics.median(r["latency_s"] for r in rows), 1) if rows else None,
               "empty": sum(1 for r in rows if r["empty"]), "hedge": sum(1 for r in rows if r["hedge"]),
               "narration": sum(1 for r in rows if r["narration"]),
               "errors": sum(1 for r in rows if r["error"]),
               "ok": f"{sum(1 for r in rows if r.get('ok'))}/{sum(1 for r in rows if 'ok' in r)}",
               "ts": time.strftime("%Y%m%d-%H%M%S")}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out = a.out if a.out else f"results/harness-{a.label}-{summary['ts']}.json"
    with open(out, "w") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2))
    print("saved", out, file=sys.stderr)


def compare(a):
    runs = [json.load(open(p)) for p in a.files]
    labels = [r["summary"].get("label", p) for r, p in zip(runs, a.files)]
    print(f"{'question':14}" + "".join(f"{l[:22]:>24}" for l in labels))
    ids = [r["id"] for r in runs[0]["rows"]]
    for qid in ids:
        cells = []
        for r in runs:
            m = next((x for x in r["rows"] if x["id"] == qid), None)
            cells.append("-" if m is None else f"{m['words']}w {m['latency_s']}s" + (" ok" if m.get("ok") else (" MISS" if "ok" in m else "")))
        print(f"{qid:14}" + "".join(f"{c:>24}" for c in cells))
    for k in ("median_words", "max_words", "median_latency_s", "empty", "hedge", "narration", "errors", "ok"):
        print(f"{k:14}" + "".join(f"{str(r['summary'].get(k, '')):>24}" for r in runs))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("run"); p.set_defaults(fn=run)
    p.add_argument("--base", default=os.environ.get("HONCHO_BASE", "http://localhost:8000"))
    p.add_argument("--workspace", required=True)
    p.add_argument("--peer", required=True, help="observer peer (whose perspective)")
    p.add_argument("--target", default=None, help="peer being asked about, if not the observer itself")
    p.add_argument("--session", default=None)
    p.add_argument("--questions", required=True)
    p.add_argument("--level", default="low", choices=["minimal", "low", "medium", "high", "max"])
    p.add_argument("--label", required=True, help="model/config name for the record, e.g. dialectic_s50")
    p.add_argument("--key", default=None, help="JWT if Honcho runs with AUTH_USE_AUTH (or env HONCHO_API_KEY)")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--out", default=None, help="default results/harness-<label>-<ts>.json")
    p = sub.add_parser("compare"); p.set_defaults(fn=compare)
    p.add_argument("files", nargs="+")
    argv = sys.argv[1:]
    if argv and argv[0] not in ("run", "compare", "-h", "--help"):
        argv = ["run"] + argv
    a = ap.parse_args(argv)
    if not getattr(a, "fn", None):
        ap.print_help(); return 1
    a.fn(a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
