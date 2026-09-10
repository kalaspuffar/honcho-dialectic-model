#!/usr/bin/env python3
"""mock_anth_batch.py — fake Anthropic endpoint (sync Messages + Message Batches) for $0 tests.

  python3 mock_anth_batch.py 9977 &
  ANTHROPIC_API_BASE=http://127.0.0.1:9977 ANTHROPIC_API_KEY=mock python3 gen_chosen.py submit --contexts ... --model opus
  ANTHROPIC_API_BASE=http://127.0.0.1:9977 ANTHROPIC_API_KEY=mock python3 gen_chosen.py fetch

Endpoints (real response shapes as parsed by llm_backend.py):
  POST /v1/messages                        -> one Message ({"content":[{"type":"text","text":...}], "usage":{...}})
  POST /v1/messages/batches                -> batch object, in_progress
  GET  /v1/messages/batches/{id}           -> ended after the first poll
  GET  /v1/messages/batches/{id}/results   -> JSONL {custom_id, result:{type:"succeeded", message:{...}}}
Content is chosen from the request's system text exactly like mock_or_server.py (context JSON
for the context generator, {"answer": ...} for the teacher).
"""
import json
import re
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import mock_or_server as mo

STATE = {}
LOCK = threading.Lock()


def _content_for(params):
    system = params.get("system", "")
    if isinstance(system, list):
        system = " ".join(b.get("text", "") for b in system)
    msgs = [{"role": "system", "content": system}] + params.get("messages", [])
    # reuse the OpenAI mock's decision logic through a tiny fake request
    class Fake:
        pass
    if "synthetic training data" in system:
        with mo.LOCK:
            mo.COUNTER[0] += 1
            k = mo.COUNTER[0]
        name = "Daniel"
        user = " ".join(m.get("content", "") for m in msgs if m.get("role") == "user")
        for line in user.splitlines():
            if line.startswith("Peer name:"):
                name = line.split(":", 1)[1].strip()
        ctx = json.loads(json.dumps(mo.CTX).replace("NAME", name).replace("#N", f"(batch scenario {k})"))
        if "Category: abstention" in user:
            ctx["required_facts"] = []
            ctx["question"] = f"What did {name} decide about the four-day workweek? (batch scenario {k})"
        return json.dumps(ctx)
    if "IDEAL final answer" in system:
        user = " ".join(m.get("content", "") for m in msgs if m.get("role") == "user")
        if "category=abstention" in user:
            return json.dumps({"answer": "There is no information in memory about a four-day workweek decision."})
        return json.dumps({"answer": "April 22 — moved from the original April 25 after a vendor call."})
    return "mock: unrecognised request shape"


def _message(params, i=0):
    return {"id": f"msg_mock_{i}", "type": "message", "role": "assistant", "model": params.get("model"),
            "stop_reason": "end_turn",
            "content": [{"type": "text", "text": _content_for(params)}],
            "usage": {"input_tokens": 900, "output_tokens": 40, "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _batch_obj(self, bid, status):
        with LOCK:
            n = STATE.get(bid, {}).get("n", 0)
        rc = ({"processing": 0, "succeeded": n, "errored": 0, "canceled": 0, "expired": 0} if status == "ended"
              else {"processing": n, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0})
        return {"id": bid, "type": "message_batch", "processing_status": status, "request_counts": rc,
                "created_at": "2026-09-10T00:00:00Z",
                "results_url": f"http://127.0.0.1/v1/messages/batches/{bid}/results"}

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if self.path == "/v1/messages":
            self._send(200, json.dumps(_message(body)))
        elif self.path == "/v1/messages/batches":
            reqs = body.get("requests", [])
            with LOCK:
                bid = f"msgbatch_mock_{len(STATE) + 1:04d}"
                STATE[bid] = {"polls": 0, "n": len(reqs), "reqs": reqs}
            sys.stderr.write(f"MOCK-BATCH created {bid} n={len(reqs)}\n")
            self._send(200, json.dumps(self._batch_obj(bid, "in_progress")))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_GET(self):
        m = re.match(r"^/v1/messages/batches/([^/]+)/results$", self.path)
        if m:
            bid = m.group(1)
            with LOCK:
                reqs = list(STATE.get(bid, {}).get("reqs", []))
            lines = [json.dumps({"custom_id": r["custom_id"],
                                 "result": {"type": "succeeded", "message": _message(r["params"], i)}})
                     for i, r in enumerate(reversed(reqs))]      # any order, like the real API
            self._send(200, ("\n".join(lines) + "\n").encode(), ctype="application/x-ndjson")
            return
        m = re.match(r"^/v1/messages/batches/([^/]+)$", self.path)
        if m:
            bid = m.group(1)
            with LOCK:
                s = STATE.setdefault(bid, {"polls": 0, "n": 0, "reqs": []})
                s["polls"] += 1
                ended = s["polls"] >= 1
            self._send(200, json.dumps(self._batch_obj(bid, "ended" if ended else "in_progress")))
            return
        self._send(404, json.dumps({"error": "not found"}))


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 9977
    srv = HTTPServer(("127.0.0.1", port), H)
    sys.stderr.write(f"mock anthropic listening on {srv.server_address}\n")
    srv.serve_forever()
