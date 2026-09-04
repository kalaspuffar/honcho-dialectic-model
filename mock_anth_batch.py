#!/usr/bin/env python3
"""mock_anth_batch.py — fake Anthropic Message Batches endpoint so
batch-run / batch-fetch can be exercised at $0 against a real key-lookup path.

Implements exactly the three endpoints openrouter_trial.py calls:
  POST /v1/messages/batches            -> 200 batch object (in_progress, then ended)
  GET  /v1/messages/batches/{id}       -> 200 batch object (state flips to ended after 1 poll)
  GET  /v1/messages/batches/{id}/results -> 200 text (JSONL of {custom_id, result})

Responses honour the REAL Anthropic result shape we parse in cmd_batch_fetch:
  {"custom_id": "...", "result": {"type":"succeeded","message":{...,"content":[{"type":"text","text":"..."}],"usage":{...}}}}

Run:  python3 mock_anth_batch.py 9977 &
"""
import json, sys, threading, re
from http.server import BaseHTTPRequestHandler, HTTPServer

STATE = {}  # batch_id -> {"polls": int, "n": int, "cids": [..]}
LOCK = threading.Lock()

def make_answer(cid):
    # return a terse grounded JSON answer (same shape the briefing asks for)
    return json.dumps({"answer": "The deadline was updated to April 22; the prior date April 25 was superseded."})

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass

    def _send(self, code, body, ctype="application/json"):
        data = body if isinstance(body, bytes) else body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path == "/v1/messages/batches":
            n = int(self.headers.get("Content-Length", 0))
            body = json.loads(self.rfile.read(n) or b"{}")
            reqs = body.get("requests", [])
            bid = "msgbatch_mock_" + str(len(STATE) + 1).zfill(4)
            with LOCK:
                STATE[bid] = {"polls": 0, "n": len(reqs),
                              "cids": [r.get("custom_id", f"arm_{i}") for i, r in enumerate(reqs)]}
            sys.stderr.write(f"MOCK-BATCH created {bid} n={len(reqs)}\n")
            obj = {
                "id": bid, "type": "message_batch",
                "processing_status": "in_progress",
                "request_counts": {"processing": len(reqs), "succeeded": 0, "errored": 0,
                                   "canceled": 0, "expired": 0},
                "created_at": "2026-09-04T00:00:00Z",
                "results_url": f"https://api.anthropic.com/v1/messages/batches/{bid}/results",
            }
            self._send(200, json.dumps(obj))
        elif re.match(r"^/v1/messages/batches/(.+)$", self.path):
            m = re.match(r"^/v1/messages/batches/(.+)$", self.path)
            bid = m.group(1)
            with LOCK:
                STATE.setdefault(bid, {"polls": 0})
                STATE[bid]["polls"] += 1
                ended = STATE[bid]["polls"] >= 1
            if ended:
                self._send(200, json.dumps(self._batch_obj(bid, "ended")))
            else:
                self._send(200, json.dumps(self._batch_obj(bid, "in_progress")))
        else:
            self._send(404, json.dumps({"error": "not found"}))

    def do_GET(self):
        m = re.match(r"^/v1/messages/batches/([^/]+)/results$", self.path)
        if m:
            bid = m.group(1)
            sys.stderr.write(f"MOCK-BATCH results for {bid}\n")
            self._send(200, b"\n".join(self._results(bid)) + b"\n", ctype="application/x-ndjson")
            return
        m = re.match(r"^/v1/messages/batches/([^/]+)$", self.path)
        if m:
            bid = m.group(1)
            with LOCK:
                s = STATE.setdefault(bid, {"polls": 0, "n": 2, "cids": []})
                s["polls"] += 1
                ended = s["polls"] >= 1
            self._send(200, json.dumps(self._batch_obj(bid, "ended" if ended else "in_progress")))
            return
        self._send(404, json.dumps({"error": "not found"}))

    def _batch_obj(self, bid, status):
        with LOCK:
            n = STATE.get(bid, {}).get("n", 2)
        if status == "ended":
            rc = {"processing": 0, "succeeded": n, "errored": 0, "canceled": 0, "expired": 0}
        else:
            rc = {"processing": n, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
        return {
            "id": bid, "type": "message_batch", "processing_status": status,
            "request_counts": rc, "created_at": "2026-09-04T00:00:00Z",
            "results_url": f"https://api.anthropic.com/v1/messages/batches/{bid}/results",
        }

    def _results(self, bid):
        with LOCK:
            cids = list(STATE.get(bid, {}).get("cids") or [])
        n = len(cids) if cids else 2
        lines = []
        for i in range(n):
            cid = cids[i] if cids else f"arm_{i:03d}"
            lines.append(json.dumps({
                "custom_id": cid,
                "result": {"type": "succeeded",
                           "message": {"id": f"msg_mock_{bid}_{i}",
                                       "content": [{"type": "text", "text": make_answer(cid)}],
                                       "usage": {"input_tokens": 900, "output_tokens": 40}}}
            }).encode())
        return lines

port = int(sys.argv[1]) if len(sys.argv) > 1 else 9977
srv = HTTPServer(("127.0.0.1", port), H)
sys.stderr.write(f"mock-anth-batch listening on {srv.server_address}\n")
srv.serve_forever()
