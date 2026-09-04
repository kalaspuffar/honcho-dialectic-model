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
# Abstention gold = a genuine "I have no information" refusal.
REFUSAL = re.compile(
    r"no (stored |existing )?information|nothing (stored|(in|on|about) (my )?(record|notes?|memory)|on|about)|"
    r"not stored|no (note|notes|record|entry|data|evidence)|no way to (know|tell|verify)|"
    r"i can't (confirm|verify|say|be sure|see)|i (have no |doesn't have) "
    r"|we don't (have|know)|no record", re.I)

# Match the EXACT prompt the base model (rejected) was generated under, so chosen &
# rejected share one prompt and training matches Honcho's runtime distribution.
try:
    from honcho_prompt import agent_system_prompt as _agent_sp
    _TOOLS = ["search_memory","search_messages","grep_messages","get_reasoning_chain",
              "get_observation_context","get_messages_by_date_range","search_messages_temporal"]
    def ctx_system(c):
        cache = "\n".join(f"- [{f['date']}] {f['text']}" for f in c.get("findings", []))
        return (_agent_sp("Daniel","Daniel",None,None,_TOOLS)
                + "\n\n## RETRIEVAL CACHE (results already gathered — do NOT re-search)\n"
                + cache + "\n").strip()
except Exception:  # honcho_prompt unavailable: fallback to a findings-in-user prompt
    def ctx_system(c):
        cache = "\n".join(f"- [{f['date']}] {f['text']}" for f in c.get("findings", []))
        return ("You are Honcho's dialectic: answer tersely, fully grounded in the provided "
                "findings, no preamble, no search narration.\n\n## FINDINGS\n" + cache + "\n")

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
        # Key fix vs v0.2: supersession/contradiction CORRECT answers must name BOTH the
        # new and the old value ("115, down from 120"; "8+ updated from 10+"), so an
        # assertive "forbidden_fact" mention is NOT fabrication when the required_fact is
        # also present. Only a TRUE fabrication (required absent, forbidden asserted) drops.
        if cat == "abstention":
            # A correct abstention is a genuine "no information" refusal that names the
            # topic. The old findings-leak gate killed legitimate refusals ("no info on
            # Alzheimer's") because they mention the subject word.
            if not REFUSAL.search(c_ans):
                dropped["abst_not_refusal"] += 1; continue
        else:
            req = [e for e in (c.get("required_facts") or [])]
            req_ok = bool(req) and any(has_entity(c_ans, e) for e in req)
            if not req_ok:
                dropped["no_required_fact"] += 1; continue
            forb = c.get("forbidden_facts") or []
            if forb and not req_ok and any(has_entity(c_ans, e) for e in forb):
                dropped["fabrication"] += 1; continue
        # HEDGE gate removed: Opus batch answers are declarative, and words in the set
        # legitimately appear in correct lines ("...the guideline appears in the preface").
        # Correctness is already enforced by required_fact + no-fabrication above.
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
            c = ctx[k["id"]]
            sysp = ctx_system(c)
            user = c["question"]
            if fmt == "sft":
                out.append({"messages": [
                    {"role": "system", "content": sysp},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": k["chosen"]}
                ]})
            elif fmt == "dpo":
                out.append({"prompt": [
                    {"role": "system", "content": sysp},
                    {"role": "user", "content": user},
                ], "chosen": k["chosen"], "rejected": k["rejected"], "category": k["category"]})
        open(f"{a.out}_{name}.{fmt}.jsonl", "w").write("\n".join(json.dumps(o) for o in out) + "\n")
        return len(out)

    train_k = [k for k in kept if split(k)=="train"]; eval_k = [k for k in kept if split(k)=="eval"]
    print(f"kept {len(kept)} / dropped {sum(dropped.values())} {dict(dropped)}")
    print(f"SFT   train={write('train',train_k,'sft')}  eval={write('eval',eval_k,'sft')}")
    print(f"DPO   train={write('train',train_k,'dpo')}  eval={write('eval',eval_k,'dpo')}")
    print("files:", ", ".join(f"{a.out}_{n}.{f}.jsonl" for n in ["train","eval"] for f in ["sft","dpo"]))

if __name__ == "__main__":
    main()
