#!/usr/bin/env python3
"""mock_or_server.py — fake OpenRouter endpoint so openrouter_trial.py's live
path (chat/parse/score/CSV/cap/collect) can be exercised without spending money.
Run: python3 mock_or_server.py 9911 &
Then: OPENROUTER_API_KEY=mock python3 openrouter_trial.py gen --context-model mock/x --n 5
      OPENROUTER_API_KEY=mock python3 openrouter_trial.py run --arms deepseek --max-usd 1
"""
import json, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer

class H(BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        sys.stderr.write(f"MOCK req model={body['model']} tokens_in={sum(len(m['content']) for m in body['messages'])//4}\n")
        last_content = body["messages"][-1]["content"]
        model = body.get("model", "")
        if "nullc" in model:
            content = None  # simulate provider returning null content
        elif "Hard rules" in last_content:
            # answer arm: return a grounded terse answer JSON
            content = '{"answer": "the deadline was updated to April 22; the prior date April 25 was superseded."}'
        else:
            # context gen arm
            content = ('{"id": "000", "persona": "A hobbyist with stable facts", "question": "What is the exact deadline for the project?", '
                       '"findings": [{"text": "the deadline is April 25", "date": "2026-03-01"}, '
                       '{"text": "the deadline was changed to April 22", "date": "2026-04-01"}], '
                       '"distractors": ["May 3"], "required_facts": ["April 22"], "forbidden_facts": ["May 3"]}')
        resp = {
            "choices": [{"message": {"role": "assistant", "content": content},
                         "finish_reason": "stop" if content is not None else "content_filter"}],
            "usage": ({"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
                      if "zerou" in model else
                      {"prompt_tokens": 900, "completion_tokens": 40, "total_tokens": 940}),
        }
        data = json.dumps(resp).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

srv = HTTPServer(("127.0.0.1", int(sys.argv[1] if len(sys.argv) > 1 else 9911)), H)
print("mock openrouter listening:", srv.server_address, flush=True)
srv.serve_forever()
