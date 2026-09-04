#!/usr/bin/env python3
"""
base_answer_probe.py  (stage 2)
-------------------------------
Run the BASE model (qwen3.5:9b on node7 Ollama) over each context and save its
natural answer as the REJECTED side of each DPO pair. This is the key trick:
the rejected is the model's TRUE verbose style, not a teacher fabrication.

Uses Honcho's exact dialectic system prompt (honcho_prompt.agent_system_prompt)
+ a findings-pool block, so the input distribution matches runtime.

Usage:
    pip install requests
    export OLLAMA_BASE=http://node7.ea.org:11434/v1   OLLAMA_MODEL=qwen3.5:9b
    python base_answer_probe.py --in trial/contexts_50.jsonl --out trial/base_rejected.jsonl
"""
import argparse, json, os, time
import requests

def build_system_prompt(ctx, OBSERVER, OBSERVED):
    from honcho_prompt import agent_system_prompt
    return (
        agent_system_prompt(OBSERVER, OBSERVED, None, None,
                            ["search_memory", "search_messages", "grep_messages",
                             "get_reasoning_chain", "get_observation_context",
                             "get_messages_by_date_range", "search_messages_temporal"])
        + "\n\n## RETRIEVAL CACHE (results already gathered — do NOT re-search)\n"
        + "\n".join(f"- [{f['date']}] {f['text']}" for f in ctx["findings"])
        + "\n"
    )

def probe(ctx, base, model):
    body = {
        "model": model,
        "temperature": 0.3,
        "max_tokens": 1500,
        "messages": [
            {"role": "system", "content": build_system_prompt(ctx, "Daniel", "Daniel").strip()},
            {"role": "user", "content": ctx["question"]},
        ],
    }
    r = requests.post(base.rstrip("/") + "/chat/completions", json=body, timeout=900)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=os.environ.get("OLLAMA_BASE", "http://node7.ea.org:11434/v1"))
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen3.5:9b"))
    a = ap.parse_args()
    ctxs = [json.loads(l) for l in open(a.inp) if l.strip()]
    out = []
    for i, c in enumerate(ctxs, 1):
        try:
            ans = probe(c, a.base, a.model)
        except Exception as e:
            ans = f"__FAILED__: {type(e).__name__}: {e}"
        rec = {"id": c["id"], "category": c.get("category"), "answer": ans,
               "words": len(ans.split())}
        out.append(rec)
        print(f"[base {i:3d}/{len(ctxs)}] {c['id']} {rec['words']}w  {ans[:60]}")
        time.sleep(0.5)
    open(a.out, "w").write("\n".join(json.dumps(r) for r in out) + "\n")
    print(f"saved {a.out}")

if __name__ == "__main__":
    main()
