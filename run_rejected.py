#!/usr/bin/env python3
"""Stage 2 (stdlib-only): run base qwen3.5:9b over each context -> REJECTED side.
Uses Honcho's exact dialectic system prompt + findings cache (matches runtime
distribution). Replaces base_answer_probe.py (which needs `requests`, pip-blocked here).
Output: results/openrouter/chosen/base_rejected.jsonl with {id, category, answer, words}.
"""
import argparse, json, os, time, urllib.request
from honcho_prompt import agent_system_prompt

TOOLS = ["search_memory", "search_messages", "grep_messages",
         "get_reasoning_chain", "get_observation_context",
         "get_messages_by_date_range", "search_messages_temporal"]

def system_prompt(ctx):
    return (
        agent_system_prompt("Daniel", "Daniel", None, None, TOOLS)
        + "\n\n## RETRIEVAL CACHE (results already gathered — do NOT re-search)\n"
        + "\n".join(f"- [{f['date']}] {f['text']}" for f in ctx["findings"])
        + "\n"
    )

def probe(base, model, ctx):
    body = {
        "model": model, "temperature": 0.3, "max_tokens": 1500,
        "messages": [
            {"role": "system", "content": system_prompt(ctx).strip()},
            {"role": "user", "content": ctx["question"]},
        ],
    }
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        data = json.loads(r.read())
    return data["choices"][0]["message"]["content"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=os.environ.get("OLLAMA_BASE", "http://node7.ea.org:11434/v1"))
    ap.add_argument("--model", default=os.environ.get("OLLAMA_MODEL", "qwen3.5:9b"))
    ap.add_argument("--only", default="", help="optional comma-list of row ids (smoke test)")
    a = ap.parse_args()
    ctxs = [json.loads(l) for l in open(a.inp) if l.strip()]
    if a.only:
        keep = set(a.only.split(","))
        ctxs = [c for c in ctxs if c["id"] in keep]
    existing = {}
    if os.path.exists(a.out):
        for line in open(a.out):
            if line.strip():
                r = json.loads(line); existing[r["id"]] = r
    out, fresh = [], 0
    for i, c in enumerate(ctxs, 1):
        rec = existing.get(c["id"])
        if rec and not rec["answer"].startswith("__FAILED__"):
            out.append(rec); continue
        try:
            ans = probe(a.base, a.model, c)
        except Exception as e:
            ans = f"__FAILED__: {type(e).__name__}: {e}"
        rec = {"id": c["id"], "category": c.get("category"), "answer": ans,
               "words": len(ans.split())}
        out.append(rec); fresh += 1
        print(f"[base {i}/{len(ctxs)}] {c['id']} {rec['words']}w  {ans[:70]!r}", flush=True)
        if ans.startswith("__FAILED__") or ans.strip() == "":
            time.sleep(2)
            try:  # one retry on failure/empty (Ollama hiccup)
                ans = probe(a.base, a.model, c)
                rec = {"id": c["id"], "category": c.get("category"), "answer": ans,
                       "words": len(ans.split())}
                out[-1] = rec
                print(f"[retry {c['id']}] {rec['words']}w  {ans[:70]!r}", flush=True)
            except Exception as e:
                print(f"[retry {c['id']}] FAILED again: {e}", flush=True)
        time.sleep(0.5)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    failed = sum(1 for r in out if r["answer"].startswith("__FAILED__"))
    print(f"saved {a.out} — {len(out)} rows, {fresh} new, {failed} failed")

if __name__ == "__main__":
    main()
