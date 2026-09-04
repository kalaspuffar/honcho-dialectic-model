#!/usr/bin/env python3
"""
build_dataset.py (stage 4)
--------------------------
Join contexts + base (rejected) + chosen, apply the PLAN §4 filters, split 90/10
on PERSONA (no question leakage), and emit both SFT-warmup and DPO formats.

Outputs:
  sft.jsonl   {"messages":[system, user, assistant-chosen]}   (SFT warm-up rows)
  dpo.jsonl   {"prompt":[system,user], "chosen":"...", "rejected":"..."}  (DPO pairs)
  train/eval split written to <out>_train.jsonl and <out>_eval.jsonl

Filter gates (from PLAN §4):
  - drop rows where base(chosen-ish) length ratio to chosen < 1.5  (no signal)
  - chosen must contain >=1 required_fact entity
  - chosen must NOT assert any forbidden_fact
  - abstention rows: chosen must contain none of the findings' entities
  - max chosen words = 120 (else drop + log)
  - dedupe on (persona, question)

Usage:
    python build_dataset.py --contexts trial/contexts_50.jsonl \
        --rejected trial/base_rejected.jsonl --chosen trial/chosen.jsonl --out dataset
"""
import argparse, json, re, sys
from collections import defaultdict

HEDGE = re.compile(r"\b(likely|probably|might|maybe|perhaps|possibly|seems?|appears?|presumably)\b", re.I)

def load(p): return {json.loads(l)["id"]: json.loads(l) for l in open(p) if l.strip()}

def has_entity(text, entity):
    t = text.lower()
    if entity.lower() in t: return True
    toks = entity.lower().split()
    return bool(toks) and all(x in t for x in toks[:2])

def findings_entities(ctx):
    ents = set()
    for f in ctx["findings"]:
        ents.add(f["text"])
    return ents

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True); ap.add_argument("--rejected", required=True)
    ap.add_argument("--chosen", required=True); ap.add_argument("--out", default="dataset")
    a = ap.parse_args()
    ctx, rej, cho = load(a.contexts), load(a.rejected), load(a.chosen)

    kept, dropped = [], defaultdict(int)
    for cid, c in ctx.items():
        r_ans = (rej.get(cid, {}) or {}).get("answer", "")
        c_ans = (cho.get(cid, {}) or {}).get("answer", "")
        if not r_ans or not c_ans or r_ans.startswith("__FAILED__") or c_ans.startswith("__FAILED__"):
            dropped["failed_generation"] += 1; continue
        r_w, c_w = len(r_ans.split()), len(c_ans.split())
        if c_w > 120: dropped["chosen_too_long"] += 1; continue
        if r_w / max(1, c_w) < 1.5: dropped["low_ratio(<1.5)"] += 1; continue
        cat = c.get("category")
        if cat == "abstention":
            if any(has_entity(c_ans, fe) for fe in findings_entities(c)):
                dropped["abstention_leakage"] += 1; continue
        else:
            req = c.get("required_facts") or []
            if not req or not any(has_entity(c_ans, e) for e in req):
                dropped["no_required_fact"] += 1; continue
            forb = c.get("forbidden_facts") or []
            if any(has_entity(c_ans, e) for e in forb):
                dropped["asserts_forbidden"] += 1; continue
        if HEDGE.search(c_ans): dropped["hedge_in_chosen"] += 1; continue
        kept.append({"id": cid, "persona": c.get("persona"), "question": c["question"],
                     "category": cat, "chosen": c_ans, "rejected": r_ans})

    # persona-level split: no question leakage
    personas = {}
    for k in kept: personas.setdefault(k["persona"], []).append(k)
    import random
    random.seed(7)
    plist = sorted(personas)
    eval_people = set(plist[:max(1, len(plist)//10)])
    def split(k): return "eval" if k["persona"] in eval_people else "train"

    def write(name, rows, fmt):
        out = []
        for k in rows:
            if fmt == "sft":
                out.append({"messages": [
                    {"role": "system", "content": "You are Honcho's dialectic. Answer tersely (1-3 short sentences), fully grounded in the findings, no preamble, no search narration."},
                    {"role": "user", "content": k["question"]},
                    {"role": "assistant", "content": k["chosen"]}]})
            elif fmt == "dpo":
                out.append({"prompt": k["question"], "chosen": k["chosen"], "rejected": k["rejected"], "category": k["category"]})
        open(f"{a.out}_{name}.{fmt}.jsonl", "w").write("\n".join(json.dumps(o) for o in out) + "\n")
        return len(out)

    train_k = [k for k in kept if split(k)=="train"]; eval_k = [k for k in kept if split(k)=="eval"]
    print(f"kept {len(kept)} / dropped {sum(dropped.values())} {dict(dropped)}")
    print(f"SFT   train={write('train',train_k,'sft')}  eval={write('eval',eval_k,'sft')}")
    print(f"DPO   train={write('train',train_k,'dpo')}  eval={write('eval',eval_k,'dpo')}")
    print("files:", ", ".join(f"{a.out}_{n}.{f}.jsonl" for n in ["train","eval"] for f in ["sft","dpo"]))

if __name__ == "__main__":
    main()
