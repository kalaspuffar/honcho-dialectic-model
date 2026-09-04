#!/usr/bin/env python3
"""
write_chosen.py (stage 3)
-------------------------
Teacher writes the IDEAL terse (CHOSEN) answer per context. Reuses the SAME
briefing prompt + generate() helper as teacher_trial.py (single style source).

Usage:
    python write_chosen.py --contexts trial/contexts_50.jsonl --teacher opus --out trial/chosen.jsonl
"""
import argparse, json, os, time
from teacher_trial import BRIEFING, generate_one, extract_json

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--contexts", required=True)
    ap.add_argument("--teacher", default="opus", choices=["opus", "gemini"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    envkey = {"opus": "ANTHROPIC_API_KEY", "gemini": "GEMINI_API_KEY"}[a.teacher]
    if not os.environ.get(envkey):
        print(f"missing {envkey}; aborting."); return 1

    ctxs = [json.loads(l) for l in open(a.contexts) if l.strip()]
    out = []
    for i, c in enumerate(ctxs, 1):
        user = (BRIEFING
                + "\n\nPERSONA: " + c["persona"]
                + "\n\nFINDINGS:\n" + "\n".join(f"- [{f['date']}] {f['text']}" for f in c["findings"])
                + "\n\nQUESTION: " + c["question"])
        try:
            raw = generate_one(a.teacher, "You write terse, fully grounded recall answers for a memory-recall fine-tuning dataset.", user)
            ans = (extract_json(raw) or {}).get("answer", raw)
        except Exception as e:
            ans = f"__FAILED__: {type(e).__name__}: {e}"
        out.append({"id": c["id"], "category": c.get("category"), "answer": ans, "teacher": a.teacher})
        print(f"[{a.teacher} {i}/{len(ctxs)}] {c['id']} {len(ans.split())}w")
        time.sleep(0.4)
    open(a.out, "w").write("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"saved {a.out} ({len(out)} rows)")

if __name__ == "__main__":
    raise SystemExit(main() or 0)
