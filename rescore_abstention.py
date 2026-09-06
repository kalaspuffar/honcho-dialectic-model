#!/usr/bin/env python3
"""rescore_abstention.py — cheap re-scoring of existing eval outputs using the
corrected abstention rule (refusal+terse+no-hedge; naming the topic is allowed),
without re-running any model calls. Emits corrected A/B numbers for TRAIN.md."""
import json, os, re, sys

HEDGE = re.compile(r"\b(likely|probably|might|maybe|perhaps|possibly|seems?|appears?|presumably)\b", re.I)
REFUSAL = re.compile(
    r"no (stored |existing )?information|nothing (stored|(in|on|about) (my )?(record|notes?|memory)|on|about)|"
    r"not stored|no (note|notes|record|entry|data|evidence)|no way to (know|tell|verify)|"
    r"i can't (confirm|verify|say|be sure|see)|i (have no |doesn't have) |we don't (have|know)|no record", re.I)

def rescore(path):
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    rows = [json.loads(l) for l in open(path) if l.strip()]
    ctxs = {}
    for p in ("results/openrouter/contexts.jsonl",):
        if os.path.exists(p):
            for l in open(p):
                if l.strip():
                    c = json.loads(l); ctxs[c["id"]] = c
    fixed = 0
    for r in rows:
        c = ctxs.get(r.get("id"), {})
        if c.get("category") != "abstention":
            continue
        ans = r.get("answer") or ""
        words = len(ans.split()) if ans else 0
        if words == 0:
            continue
        refusal = bool(REFUSAL.search(ans))
        ok = refusal and (words <= 60) and not HEDGE.search(ans)  # forbidden-word mention allowed
        was = r.get("abst", False)
        r["abst"] = ok
        if ok and not was:
            fixed += 1
    abst_total = sum(1 for c in ctxs.values() if c.get("category") == "abstention")
    abst_correct = sum(1 for r in rows if ctxs.get(r.get("id"), {}).get("category") == "abstention" and r.get("abst"))
    return {"model": os.path.basename(path), "abstention_correct": f"{abst_correct}/{abst_total}",
            "abst_rows_fixed_from_0": fixed}

os.chdir(os.path.dirname(os.path.abspath(__file__)))
for p in ("results/eval_baseline_8b.jsonl", "results/eval_tuned_8b.jsonl"):
    res = rescore(p)
    print(p, "->", res)
