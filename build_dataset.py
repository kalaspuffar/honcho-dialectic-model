#!/usr/bin/env python3
"""build_dataset.py — stage 4: join contexts + rejected + chosen, filter, split, emit.

Rows are full Honcho trajectories (trajectory.build_messages): system prompt,
question, the tool calls and tool results, then the answer. Loss is only ever
computed on the final assistant turn (train_dialectic.py masks the rest).

  python3 build_dataset.py --contexts data/contexts.jsonl --rejected data/rejected.jsonl \
      --chosen data/chosen.jsonl --out data/dataset

writes  data/dataset_train.sft.jsonl   {"id", "messages": [..., assistant chosen], "tools": [...]}
        data/dataset_train.dpo.jsonl   {"id", "prompt": [...], "tools": [...], "chosen", "rejected", "category"}
        data/dataset_eval.{sft,dpo}.jsonl  (held-out personas, ~10%)

Filters (PLAN §4):
  failed generation on either side              -> drop
  chosen > --max-chosen-words (120)             -> drop
  rejected/chosen word ratio < --min-ratio (1.5)-> drop (no length signal)
  non-abstention: no required_fact in chosen    -> drop
  non-abstention: forbidden asserted & no required (fabrication) -> drop
  abstention: chosen is not a refusal, or > 60 words, or hedges  -> drop
  duplicate (peer name, question)               -> drop
Split is by persona name (seeded), so a person never appears in both halves.
"""
import argparse
import json
import random
from collections import Counter

import llm_backend as be
import scoring
from trajectory import TOOL_SCHEMAS, build_messages, peer_name


def load_by_id(path):
    return {r["id"]: r for r in be.read_jsonl(path)}


def filter_rows(ctxs, rej, cho, max_chosen_words=120, min_ratio=1.5):
    kept, dropped, seen = [], Counter(), set()
    for c in ctxs:
        if be.failed(c):
            dropped["failed_context"] += 1; continue
        cid = c["id"]
        r_ans = (rej.get(cid) or {}).get("answer") or ""
        c_ans = (cho.get(cid) or {}).get("answer") or ""
        if not r_ans or r_ans.startswith("__FAILED__"):
            dropped["missing_or_failed_rejected"] += 1; continue
        if not c_ans or c_ans.startswith("__FAILED__"):
            dropped["missing_or_failed_chosen"] += 1; continue
        r_w, c_w = scoring.words(r_ans), scoring.words(c_ans)
        if c_w > max_chosen_words:
            dropped["chosen_too_long"] += 1; continue
        if r_w / max(1, c_w) < min_ratio:
            dropped[f"low_ratio(<{min_ratio})"] += 1; continue
        s = scoring.score_answer(c, c_ans)
        if c.get("category") == "abstention":
            if not s["abst"]:
                dropped["abstention_chosen_not_clean_refusal"] += 1; continue
        else:
            if s["fab"]:                       # forbidden value asserted, no required value present
                dropped["fabrication_in_chosen"] += 1; continue
            if not s["req_ok"]:
                dropped["no_required_fact_in_chosen"] += 1; continue
        key = (peer_name(c).lower(), c["question"].strip().lower())
        if key in seen:
            dropped["duplicate_question"] += 1; continue
        seen.add(key)
        kept.append({"ctx": c, "chosen": c_ans.strip(), "rejected": r_ans.strip()})
    return kept, dropped


def split_by_persona(kept, eval_frac=0.1, seed=7):
    names = sorted({peer_name(k["ctx"]) for k in kept})
    rnd = random.Random(seed)
    rnd.shuffle(names)
    n_eval = max(1, round(len(names) * eval_frac)) if len(names) > 1 else 0
    eval_names = set(names[:n_eval])
    train = [k for k in kept if peer_name(k["ctx"]) not in eval_names]
    ev = [k for k in kept if peer_name(k["ctx"]) in eval_names]
    return train, ev


def sft_row(k):
    return {"id": k["ctx"]["id"], "category": k["ctx"].get("category"),
            "messages": build_messages(k["ctx"]) + [{"role": "assistant", "content": k["chosen"]}],
            "tools": TOOL_SCHEMAS}


def dpo_row(k):
    return {"id": k["ctx"]["id"], "category": k["ctx"].get("category"),
            "prompt": build_messages(k["ctx"]), "tools": TOOL_SCHEMAS,
            "chosen": k["chosen"], "rejected": k["rejected"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--rejected", required=True)
    ap.add_argument("--chosen", required=True)
    ap.add_argument("--out", default="data/dataset")
    ap.add_argument("--max-chosen-words", type=int, default=120)
    ap.add_argument("--min-ratio", type=float, default=1.5)
    ap.add_argument("--eval-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    ctxs = be.read_jsonl(a.contexts)
    kept, dropped = filter_rows(ctxs, load_by_id(a.rejected), load_by_id(a.chosen),
                                a.max_chosen_words, a.min_ratio)
    train, ev = split_by_persona(kept, a.eval_frac, a.seed)
    print(f"contexts {len(ctxs)}  kept {len(kept)}  dropped {sum(dropped.values())} {json.dumps(dict(dropped))}")
    cats = Counter(k["ctx"].get("category") for k in kept)
    print("kept per category:", json.dumps(dict(sorted(cats.items()))))
    cw = sorted(scoring.words(k["chosen"]) for k in kept)
    rw = sorted(scoring.words(k["rejected"]) for k in kept)
    if kept:
        print(f"median words: chosen {cw[len(cw)//2]}  rejected {rw[len(rw)//2]}")
    for name, rows in (("train", train), ("eval", ev)):
        be.write_jsonl(f"{a.out}_{name}.sft.jsonl", [sft_row(k) for k in rows])
        be.write_jsonl(f"{a.out}_{name}.dpo.jsonl", [dpo_row(k) for k in rows])
        print(f"{name}: {len(rows)} rows -> {a.out}_{name}.sft.jsonl / .dpo.jsonl")


if __name__ == "__main__":
    main()
