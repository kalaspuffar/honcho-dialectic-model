#!/usr/bin/env python3
"""scoring.py — the ONE scorer used by build_dataset.py, eval_model.py and
gen_chosen.py's report. Edit rules here only.

Rules (PLAN §4):
  coverage    fraction of required_facts present in the answer (substring, lenient)
  fabrication a forbidden_fact asserted while NO required_fact is present
              (contradiction/supersession answers legitimately name the old value
              next to the new one, so forbidden+required together is not fabrication)
  abstention  a genuine refusal, <= 60 words, no hedge; naming the missing topic is allowed
  hedge       likely/probably/might/... anywhere in the answer
"""
import re

HEDGE = re.compile(r"\b(likely|probably|might|maybe|perhaps|possibly|seems?|appears?|presumably)\b", re.I)
REFUSAL = re.compile(
    r"no (stored |existing |recorded )?information|nothing (stored|recorded|(in|on|about) (my |the )?(record|notes?|memory)|on|about)|"
    r"not stored|no (note|notes|record|records|entry|data|evidence|conclusions?|mention)|no way to (know|tell|verify)|"
    r"i can'?t (confirm|verify|say|be sure|see|find)|i (have no |don'?t have |do not have |found no |couldn'?t find |could not find )|"
    r"we don'?t (have|know)|no record|isn'?t (anything|any information)|is not (recorded|stored|in memory)|"
    r"not (captured|present|available) in|"
    # memory-limit phrasings the tuned model uses for false-premise questions (2026-09-12, s50 eval):
    # "memory only has her considering…", "it doesn't say she uses…", "has only mentioned visiting…",
    # "no known pet allergies". A bare "No — she uses X, not Y" (a negative inferred from absence)
    # is deliberately NOT a refusal.
    r"\b(memory|the notes?|the records?|it) only (has|holds|shows?|mentions?|contains?|records?)\b|"
    r"\b(doesn'?t|does not|don'?t|do not|never|didn'?t|did not) (say|mention|state|record|indicate|show|note)\b|"
    r"\bhas only (mentioned|said|noted|talked about|referred to)\b|\bno known\b", re.I)

# Tool-call narration leaking out as the final answer ("Search for X and Y.", "Let me search…"):
# seen once in 75 Honcho harness answers (s150_sft at `max`, 2026-09-12). Honcho shows it to the
# user as the answer, so it counts as a failed row wherever it appears.
NARRATION = re.compile(r"^\s*(search(ing)? (for|memory|messages)|let me (search|look|check|grep)|i('ll| will) (search|look|check|grep)|"
                       r"grep(ping)? (for|messages)|calling |first,? (search|let me))", re.I)

ABSTENTION_MAX_WORDS = 60


def has_entity(text: str, entity: str) -> bool:
    t = (text or "").lower()
    e = (entity or "").lower().strip()
    if not e:
        return False
    if e in t:
        return True
    toks = e.split()
    return len(toks) >= 2 and all(x in t for x in toks[:2])


def words(text: str) -> int:
    return len((text or "").split())


def score_answer(ctx: dict, answer: str) -> dict:
    """{words, coverage, fab, abst, hedge, refusal, req_ok, narration}"""
    a = answer or ""
    w = words(a)
    hedge = bool(HEDGE.search(a))
    refusal = bool(REFUSAL.search(a))
    narration = bool(NARRATION.search(a))
    if narration:   # a search plan is not an answer, whatever it happens to mention
        return {"words": w, "coverage": 0.0, "fab": False, "abst": False,
                "hedge": hedge, "refusal": False, "req_ok": False, "narration": True}
    req = [e for e in (ctx.get("required_facts") or []) if e]
    forb = [e for e in (ctx.get("forbidden_facts") or []) if e]
    if not a.strip():
        return {"words": 0, "coverage": 0.0, "fab": False, "abst": False,
                "hedge": False, "refusal": False, "req_ok": False, "narration": False}
    if ctx.get("category") == "abstention":
        abst = refusal and w <= ABSTENTION_MAX_WORDS and not hedge
        return {"words": w, "coverage": 1.0 if abst else 0.0, "fab": False, "abst": abst,
                "hedge": hedge, "refusal": refusal, "req_ok": abst, "narration": False}
    hits = sum(1 for e in req if has_entity(a, e))
    coverage = hits / len(req) if req else 0.0
    req_ok = hits > 0
    fab = bool(forb) and not req_ok and any(has_entity(a, f) for f in forb)
    return {"words": w, "coverage": round(coverage, 3), "fab": fab, "abst": False,
            "hedge": hedge, "refusal": refusal, "req_ok": req_ok, "narration": False}


def aggregate(ctxs, scores):
    """Summary dict over parallel lists of contexts and score dicts."""
    import statistics
    ws = [s["words"] for s in scores]
    covs = [s["coverage"] for s in scores]
    n_abst = sum(1 for c in ctxs if c.get("category") == "abstention")
    return {
        "n": len(scores),
        "median_words": int(statistics.median(ws)) if ws else None,
        "mean_words": round(statistics.mean(ws), 1) if ws else None,
        "max_words": max(ws) if ws else None,
        "mean_coverage": round(statistics.mean(covs), 3) if covs else None,
        "fabrication_rows": sum(1 for s in scores if s["fab"]),
        "abstention_correct": f"{sum(1 for s in scores if s['abst'])}/{n_abst}",
        "hedge_rows": sum(1 for s in scores if s["hedge"]),
        "empty_rows": sum(1 for s in scores if s["words"] == 0),
        "narration_rows": sum(1 for s in scores if s.get("narration")),
    }
