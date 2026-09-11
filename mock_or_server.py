#!/usr/bin/env python3
"""mock_or_server.py — fake OpenAI-compatible endpoint (OpenRouter AND Ollama /v1) so the
whole pipeline can be exercised at $0 and without a GPU host.

  python3 mock_or_server.py 9911 &
  OPENROUTER_BASE=http://127.0.0.1:9911/v1 OPENROUTER_API_KEY=mock python3 gen_contexts.py run --n 6 --model deepseek --out /tmp/x/contexts.jsonl
  OPENROUTER_BASE=http://127.0.0.1:9911/v1 OPENROUTER_API_KEY=mock python3 gen_chosen.py run --contexts /tmp/x/contexts.jsonl --model deepseek
  python3 gen_rejected.py --contexts /tmp/x/contexts.jsonl --out /tmp/x/rejected.jsonl --base http://127.0.0.1:9911/v1 --model mock

Behaviour by request shape:
  system mentions "synthetic training data"  -> a context JSON (question varies per call)
  system mentions "IDEAL final answer"       -> {"answer": "..."} terse, grounded
  messages contain role=tool (student run)   -> a verbose Honcho-style answer; the FIRST call for a
                                                conversation returns an extra tool_call so the
                                                answer loop is exercised too
  model contains "nullc"                     -> null content (provider failure)
"""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

COUNTER = [0]
SEEN = set()
LOCK = threading.Lock()

CTX = {
    "persona": {"name": "NAME", "bio": "Runs a small home lab and keeps notes about every change."},
    "question": "What deadline did NAME set for the storage migration? #N",
    "observations": [
        {"date": "2026-03-01", "text": "NAME set April 25 as the deadline for migrating the NAS to Ceph.", "relevant": True},
        {"date": "2026-04-01", "text": "NAME moved the Ceph migration deadline from April 25 to April 22 after a vendor call.", "relevant": True},
        {"date": "2026-02-10", "text": "NAME plans to retire the old Synology unit in May.", "relevant": False},
        {"date": "2026-01-15", "text": "NAME budgeted 900 EUR for new drives.", "relevant": False},
        {"date": "2026-03-20", "text": "NAME's backup rehearsal is scheduled for May 3.", "relevant": False},
    ],
    "searches": [
        {"tool": "search_memory", "query": "storage migration deadline", "results": [0, 1]},
        {"tool": "grep_messages", "query": "Ceph", "results": [2, 3, 4]},
    ],
    "required_facts": ["April 22"],
    "forbidden_facts": ["May 3", "April 25"],
}
VERBOSE = ("Based on my memory search, here's what I found about the storage migration deadline:\n\n"
           "**Original deadline:** April 25 was the date initially set for migrating the NAS to Ceph "
           "(recorded 2026-03-01).\n\n**Updated deadline:** After a vendor call the deadline was moved "
           "to **April 22** (recorded 2026-04-01), so April 22 is the current target.\n\nRelated context: the "
           "old Synology unit is planned for retirement in May, 900 EUR was budgeted for drives, and a backup "
           "rehearsal is scheduled for May 3.\n\nIs there anything else you'd like to know about the migration?")


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        msgs = body.get("messages", [])
        model = body.get("model", "")
        system = " ".join(m.get("content", "") for m in msgs if m.get("role") == "system" and isinstance(m.get("content"), str))
        has_tool = any(m.get("role") == "tool" for m in msgs)
        tool_calls = None
        if "nullc" in model:
            content = None
        elif "synthetic training data" in system:
            with LOCK:
                COUNTER[0] += 1
                k = COUNTER[0]
            name = "Daniel"
            for m in msgs:
                if m.get("role") == "user":
                    for line in m["content"].splitlines():
                        if line.startswith("Peer name:"):
                            name = line.split(":", 1)[1].strip()
            ctx = json.loads(json.dumps(CTX).replace("NAME", name).replace("#N", f"(scenario {k})"))
            if "Category: abstention" in " ".join(m.get("content", "") for m in msgs if m.get("role") == "user"):
                ctx["required_facts"] = []
                ctx["question"] = f"What did {name} decide about the four-day workweek? (scenario {k})"
            content = json.dumps(ctx)
        elif "IDEAL final answer" in system:
            user = " ".join(m.get("content", "") for m in msgs if m.get("role") == "user")
            if "category=abstention" in user:
                content = json.dumps({"answer": "There is no information in memory about a four-day workweek decision."})
            else:
                content = json.dumps({"answer": "April 22 — moved from the original April 25 after a vendor call."})
        elif has_tool:
            key = msgs[1].get("content", "") if len(msgs) > 1 else ""
            first = False
            with LOCK:
                if key not in SEEN and body.get("tools"):
                    SEEN.add(key); first = True
            if first:   # exercise the extra-round path once per conversation
                content = ""
                tool_calls = [{"id": "call_extra", "type": "function",
                               "function": {"name": "search_messages", "arguments": json.dumps({"query": "deadline changed"})}}]
            else:
                content = VERBOSE
        else:
            content = "mock: unrecognised request shape"
        msg = {"role": "assistant", "content": content}
        if "thinker" in body.get("model", "") and not tool_calls:   # qwen3.5:9b-on-Ollama shape: answer inside <think>, empty content
            msg["reasoning"] = "Let me verify. " + content
            msg["content"] = ""
        if tool_calls:
            msg["tool_calls"] = tool_calls
        resp = {"choices": [{"message": msg, "finish_reason": "tool_calls" if tool_calls else ("stop" if content is not None else "content_filter")}],
                "usage": {"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": 940}}
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    srv = HTTPServer(("127.0.0.1", int(sys.argv[1]) if len(sys.argv) > 1 else 9911), H)
    print("mock openai-compatible server listening:", srv.server_address, flush=True)
    srv.serve_forever()
