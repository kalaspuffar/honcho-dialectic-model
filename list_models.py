#!/usr/bin/env python3
"""list_models.py -- public /models endpoint (no key, no cost).
Lists 'writer' candidate models on OpenRouter with real IDs + USD/M pricing.
Run this before choosing trial arms so you know what you're paying per run.
"""
import json
import urllib.request

CANDID_FAMILIES = [
    ("anthropic", ["claude-opus", "claude-sonnet", "claude-haiku"]),
    ("openai", ["gpt-5", "gpt-4.1", "o4-mini", "o3-mini"]),
    ("google", ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-3"]),
    ("x-ai", ["grok-4", "grok-3"]),
    ("mistral", ["mistral-large", "magistral-large"]),
    ("deepseek", ["deepseek-chat", "deepseek-reasoner"]),
    ("qwen", ["qwen3-max", "qwen3.5"]),
    ("meta", ["llama-4-maverick", "llama-3.3-70b"]),
]


def main():
    req = urllib.request.Request(
        "https://openrouter.ai/api/v1/models",
        headers={"User-Agent": "honcho-dialectic-model/0.1"},
    )
    data = json.load(urllib.request.urlopen(req, timeout=30))["data"]
    print(f"total models listed on OpenRouter: {len(data)}\n")

    rows = []
    for m in data:
        mid = m["id"].lower()
        for fam, subs in CANDID_FAMILIES:
            if fam in mid and any(s in mid for s in subs):
                p = m.get("pricing", {})
                pin = p.get("prompt")
                pout = p.get("completion")
                f2 = lambda v: float(v) if v not in (None, "") else None
                pin = f2(pin); pout = f2(pout)
                rows.append({"id": m["id"], "name": m.get("name", ""),
                             "ctx": m.get("context_length", 0),
                             "in_usd_m": pin, "out_usd_m": pout})
                break
    rows.sort(key=lambda r: (r["in_usd_m"] or 99, r["id"]))
    hdr = f"{'id':48s} {'in $/M':>8s} {'out $/M':>8s} {'ctx':>10s}"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        def fm(v): return f"{v:g}" if v is not None else "-"
        print(f"{r['id']:48s} {fm(r['in_usd_m']):>8s} {fm(r['out_usd_m']):>8s} {r['ctx']:>10}")


if __name__ == "__main__":
    main()
