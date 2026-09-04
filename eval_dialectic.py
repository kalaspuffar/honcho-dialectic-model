#!/usr/bin/env python3
"""eval_dialectic.py — run the 30 smoke contexts through a local Ollama model
and score the same way build_dataset.py does (required_facts present,
no fabricated forbidden_facts, abstention must refuse, hedge check).

Usage:
  # Baseline (what we're trying to beat) — the un-fine-tuned qwen3.5:9b:
  python3 eval_dialectic.py --model qwen3.5:9b --base http://node7.ea.org:11434/v1 \
      --out baseline.jsonl
  # Fine-tuned model (after step 6 in TRAIN.md):
  python3 eval_dialectic.py --model dialectic-qwen3.5-9b \
      --out dialectic-v0.jsonl

Outputs a jsonl with per-row scores + an aggregate. Compare the two:
  - median_words (target: <= 60 after fine-tune; baseline ~150)
  - entity_coverage (target: >= 0.85; baseline ~0.85)
  - fabrication_rows (target: <= 2; baseline ~6 on the trial data)
  - abstention_correct (target: 3/3; baseline varies)
"""
import argparse, json, statistics, urllib.request, re, sys

HEDGE = re.compile(r"\b(likely|probably|might|maybe|perhaps|possibly|seems?|appears?|presumably)\b", re.I)
REFUSAL = re.compile(
    r"no (stored |existing )?information|nothing (stored|(in|on|about) (my )?(record|notes?|memory)|on|about)|"
    r"not stored|no (note|notes|record|entry|data|evidence)|no way to (know|tell|verify)|"
    r"i can't (confirm|verify|say|be sure|see)|i (have no |doesn't have) |we don't (have|know)|no record", re.I)

def chat(base, model, sysp, user, max_tokens=1024, temperature=0.1):
    body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
            "messages": [{"role":"system","content":sysp},{"role":"user","content":user}]}
    req = urllib.request.Request(base.rstrip("/")+"/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]

def score_row(c, ans):
    cat = c["category"]
    words = len(ans.split()) if ans else 0
    if not ans or not ans.strip():
        return {"words":words,"coverage":0.0,"fab":False,"abst":False,"hedge":False}
    coverage, fab, abst, hedge = 0.0, False, False, False
    if cat == "abstention":
        forb = c.get("forbidden_facts") or []
        refusal = bool(REFUSAL.search(ans))
        abst = refusal and (words <= 60) and not HEDGE.search(ans) and not any(f.lower() in ans.lower() for f in forb)
        coverage = 1.0 if abst else 0.0
    else:
        req = c.get("required_facts") or [""]
        ok = sum(1 for e in req if e and e.lower() in ans.lower())
        coverage = ok / max(1, len(req))
        forb = c.get("forbidden_facts") or []
        req_ok = any(e and e.lower() in ans.lower() for e in req)
        if forb and not req_ok and any(f.lower() in ans.lower() for f in forb):
            fab = True
    if HEDGE.search(ans): hedge = True
    return {"words":words,"coverage":coverage,"fab":fab,"abst":abst,"hedge":hedge}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="http://node7.ea.org:11434/v1")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    a = ap.parse_args()
    try:
        from honcho_prompt import agent_system_prompt
        TOOLS = ["search_memory","search_messages","grep_messages","get_reasoning_chain",
                 "get_observation_context","get_messages_by_date_range","search_messages_temporal"]
        def sysp(ctx):
            cache = "\n".join(f"- [{f['date']}] {f['text']}" for f in ctx["findings"])
            return (agent_system_prompt("Daniel","Daniel",None,None,TOOLS)
                    + "\n\n## RETRIEVAL CACHE (results already gathered — do NOT re-search)\n"
                    + cache + "\n").strip()
    except Exception:
        def sysp(ctx):
            return ("You are Honcho's dialectic. Answer tersely, fully grounded in the provided findings.\n## FINDINGS\n"
                    + "\n".join(f"- [{f['date']}] {f['text']}" for f in ctx["findings"]))
    ctxs = [json.loads(l) for l in open(a.contexts) if l.strip()]
    rows, results = [], []
    for i, c in enumerate(ctxs, 1):
        try:
            ans = chat(a.base, a.model, sysp(c), c["question"], a.max_tokens)
        except Exception as e:
            ans = f"__FAILED__: {e}"
        s = score_row(c, ans)
        row = {"id": c["id"], "category": c.get("category"), "answer": ans, **s}
        rows.append(row); results.append(s)
        print(f"[{i:2}/{len(ctxs)}] {c['id']} [{c['category']:12}] {s['words']:4}w cov={s['coverage']:.2f} fab={s['fab']} abst={s['abst']} hedge={s['hedge']}", file=sys.stderr)
    # Aggregate
    words = [r["words"] for r in results]; covs = [r["coverage"] for r in results]
    fabs = sum(1 for r in results if r["fab"])
    absts = sum(1 for r in results if r.get("abst"))
    abst_total = sum(1 for r in results if [c for c in ctxs if c["id"]==r["id"]][0]["category"]=="abstention")
    hedges = sum(1 for r in results if r["hedge"])
    agg = {"model":a.model,"n":len(results),
           "median_words":int(statistics.median(words)) if words else None,
           "max_words":max(words) if words else None,
           "mean_entity_coverage":round(statistics.mean(covs),3) if covs else None,
           "fabrication_rows":fabs,"abstention_correct":f"{absts}/{abst_total}",
           "hedge_rows":hedges}
    with open(a.out,"w") as f:
        for r in rows: f.write(json.dumps(r)+"\n")
    print("\nAGGREGATE\n"+json.dumps(agg,indent=2))

if __name__ == "__main__":
    main()
