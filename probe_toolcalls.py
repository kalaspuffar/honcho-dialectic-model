#!/usr/bin/env python3
"""probe_toolcalls.py — verify the fine-tuned dialectic model still emits valid
Ollama tool_calls (and the base model's behavior as control).

Why this exists: SFT/DPO training is done WITHOUT tool-call examples in the
smoke/eval data (the eval harness feeds a RETRIEVAL CACHE so the model is
never asked to call). LoRA on a handful of rows shouldn't erase tool-calling,
but "shouldn't" is not a number — this probe makes it one: valid_call rate
over N prompts that explicitly require a search.

Usage:
  python3 probe_toolcalls.py --model dialectic-qwen3.5-9b --base http://node7.ea.org:11434
  python3 probe_toolcalls.py --model qwen3:8b --base http://node7.ea.org:11434

PASS = tool_calls present with a JSON-parseable arguments dict on >= 90% of prompts.
"""
import argparse, json, urllib.request, sys

TOOLS = ["search_memory","search_messages","grep_messages","get_reasoning_chain",
         "get_observation_context","get_messages_by_date_range","search_messages_temporal"]

QUESTIONS = [
    "What did Daniel say about the Ceph cluster last month?",
    "Summarize Daniel's communication preferences with his partner.",
    "Which projects was Daniel working on in August 2026?",
]

def call(base, model, sysp, user, tools, temperature=0.1):
    body = {"model": model, "temperature": temperature, "max_tokens": 256,
            "stream": False, "tools": tools,
            "messages": [{"role": "system", "content": sysp}, {"role": "user", "content": user}]}
    req = urllib.request.Request(base.rstrip("/") + "/api/chat",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default="http://node7.ea.org:11434")
    a = ap.parse_args()
    from honcho_prompt import agent_system_prompt
    schemas = [{"type": "function", "function": {"name": t, "description": f"memory tool {t}",
              "parameters": {"type": "object", "properties": {
                  "query": {"type": "string"}, "observer": {"type": "string"}, "observed": {"type": "string"}},
                  "required": ["query"]}}} for t in TOOLS]
    sysp = agent_system_prompt("Daniel", "Daniel", None, None, TOOLS)
    valid = 0
    for i, q in enumerate(QUESTIONS, 1):
        m = call(a.base, a.model, sysp, q, schemas)
        tcs = m.get("tool_calls") or []
        ok = False; why = "no tool_calls"
        if tcs:
            tc = tcs[0]
            name = (tc.get("function") or {}).get("name")
            raw = (tc.get("function") or {}).get("arguments", "{}")
            try:
                args = raw if isinstance(raw, dict) else json.loads(raw)
                ok = name in TOOLS and isinstance(args, dict) and len(args) > 0
                why = f"{name}({args})"
            except Exception as e:
                why = f"args unparseable: {e}"
        else:
            why = f"text response: {str(m.get('content'))[:90]!r}"
        if ok: valid += 1
        print(f"  [{i}/{len(QUESTIONS)}] {'OK  ' if ok else 'FAIL'}  {why}")
    rate = valid / len(QUESTIONS)
    print(f"\nRESULT model={a.model}  valid={valid}/{len(QUESTIONS)}  ({rate:.0%})  "
          f"{'PASS' if rate >= 0.9 else 'FAIL (<90%)'}")
    sys.exit(0 if rate >= 0.9 else 1)

if __name__ == "__main__":
    main()
