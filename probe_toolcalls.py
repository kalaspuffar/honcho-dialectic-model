#!/usr/bin/env python3
"""probe_toolcalls.py — does the model still emit valid tool calls after fine-tuning?

Training only ever puts loss on the final synthesis turn, so tool calling should
survive — this makes "should" a number. Uses the same tool schemas as training
(trajectory.TOOL_SCHEMAS) through Ollama's native /api/chat.

  python3 probe_toolcalls.py --model dialectic-v1 --base http://node7.ea.org:11434
  python3 probe_toolcalls.py --model qwen3.5:9b   --base http://node7.ea.org:11434   # control

PASS = a tool_call with a parseable arguments object on >= 90% of prompts.
"""
import argparse
import json
import os
import sys
import urllib.request

import llm_backend as be
from honcho_prompt import agent_system_prompt
from trajectory import TOOLS, TOOL_SCHEMAS

QUESTIONS = [
    "What did Daniel say about the Ceph cluster last month?",
    "Summarize Daniel's communication preferences with his partner.",
    "Which projects was Daniel working on in August 2026?",
    "How many bikes does Daniel own?",
    "When did Daniel change his gym schedule?",
]


def call(base, model, sysp, user):
    body = {"model": model, "stream": False, "tools": TOOL_SCHEMAS,
            "options": {"temperature": 0.1, "num_predict": 256},
            "messages": [{"role": "system", "content": sysp}, {"role": "user", "content": user}]}
    req = urllib.request.Request(base.rstrip("/") + "/api/chat", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())["message"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--base", default=be.setting("OLLAMA_BASE", "http://node7.ea.org:11434/v1").replace("/v1", ""))
    a = ap.parse_args()
    sysp = agent_system_prompt("Daniel", "Daniel", None, None, TOOLS)
    valid = 0
    for i, q in enumerate(QUESTIONS, 1):
        m = call(a.base, a.model, sysp, q)
        tcs = m.get("tool_calls") or []
        ok, why = False, "no tool_calls"
        if tcs:
            fn = tcs[0].get("function") or {}
            raw = fn.get("arguments", "{}")
            try:
                args = raw if isinstance(raw, dict) else json.loads(raw)
                ok = fn.get("name") in TOOLS and isinstance(args, dict) and bool(args)
                why = f"{fn.get('name')}({args})"
            except Exception as e:  # noqa: BLE001
                why = f"args unparseable: {e}"
        else:
            why = f"text response: {str(m.get('content'))[:90]!r}"
        valid += ok
        print(f"  [{i}/{len(QUESTIONS)}] {'OK  ' if ok else 'FAIL'}  {why}")
    rate = valid / len(QUESTIONS)
    print(f"\nRESULT model={a.model} valid={valid}/{len(QUESTIONS)} ({rate:.0%}) {'PASS' if rate >= 0.9 else 'FAIL (<90%)'}")
    sys.exit(0 if rate >= 0.9 else 1)


if __name__ == "__main__":
    main()
