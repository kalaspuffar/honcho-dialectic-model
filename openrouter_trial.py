#!/usr/bin/env python3
"""
openrouter_trial.py — self-contained, stdlib-only teacher A/B benchmark via OpenRouter.

WHY THIS EXISTS
  Pick the best "terse grounded answer" teacher for the dialectic DPO dataset.
  One OpenRouter key is all you need (context generation AND every answer arm
  go through the same endpoint). No pip installs: python3 + network + a key.

MODES (all read the same 30 shared contexts so arms are directly comparable)
  1. estimate   (NO network, NO cost)  -> prints predicted $ per arm + total
  2. gen        (ONE teacher, small)   -> builds the shared contexts.jsonl
  3. run        (per-arm, cost-capped) -> answers each context, scores, saves
  4. collect    (local)                -> tarballs results/ for copying back

SAFETY (the "don't blow my wallet" bits)
  - `estimate` is pure-local: it computes tokens from char counts and multiplies
    by live $/M from the model list; it never hits the API.
  - `run` enforces --max-usd (default 5.00). It pre-estimates, and if the
    estimate exceeds the cap AND you did not pass --yes, it ABORTS before
    spending anything.
  - Every call has a max_tokens ceiling and a hard timeout; errors are counted,
    never silently retried in a loop.

OUTPUT (all under results/openrouter/, plain files you can scp/copy over)
  contexts.jsonl            shared (persona, question, findings, rubric) x N
  arms/<arm>/<id>.json      one file per answer: {id, arms, answer, words, ok}
  summary.json              per-arm: median/max words, entity coverage, fabrication,
                            abstention-correct, hedge count, rows, est_$, actual_$
  blind-review.csv          10 rows, arms anonymized as A/B/C for your 1-5 review
  run-manifest.json         exact args + timestamp + which arm -> which OpenRouter id
  (collect) results_openrouter_<stamp>.tar.gz

USAGE
  export OPENROUTER_API_KEY=sk-or-...     # or put it in keys.env next to this file
  python3 openrouter_trial.py estimate
  python3 openrouter_trial.py gen        --context-model anthropic/claude-opus-5 --n 30
  python3 openrouter_trial.py run        --arms opus,sonnet,gemini-pro,gpt5 --max-usd 5
  python3 openrouter_trial.py collect

  (run on any machine that has python3 + outbound https)
"""
import argparse, csv, datetime, json, os, re, statistics, sys, tarfile, time, urllib.request, urllib.error

OPENROUTER = os.environ.get("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
MAX_TOKENS_OUT = 800          # terse answers; hard ceiling per call (was 400 — too low, caused truncation)
CTX_MAX_TOKENS = 4096        # contexts need 7-9 findings + distractors + facts; must not truncate
HTTP_TIMEOUT = 120            # seconds per request

# Live model IDs + $/M pulled from OpenRouter /models on 2026-09-03 (re-check with `estimate`).
# name -> (openrouter_id, $/M in, $/M out, note)
ARMS = {
    "opus":        ("anthropic/claude-opus-5",      5.00, 25.0, "Claude Opus 5 (strong style, pricier)"),
    "sonnet":      ("anthropic/claude-sonnet-5",    2.00, 10.0, "Claude Sonnet 5 (quality/cost)"),
    "gemini-pro":  ("google/gemini-3.1-pro-preview",2.00, 12.0, "Gemini 3.1 Pro (frontier, good cost)"),
    "gemini-2.5pro":("google/gemini-2.5-pro",       1.25, 10.0, "Gemini 2.5 Pro"),
    "gpt5":        ("openai/gpt-5",                 1.25, 10.0, "GPT-5"),
    "o4mini":      ("openai/o4-mini",               1.10,  4.4, "o4-mini (reasoning, cheap)"),
    "deepseek":    ("deepseek/deepseek-chat",       0.32,  0.89,"DeepSeek V3-chat (very cheap)"),
    "qwen3max":    ("qwen/qwen3-max",               0.78,  3.90,"Qwen3-Max (same family as base)"),
    "fable":       ("anthropic/claude-fable-5",    10.00, 50.0, "Claude Fable 5 (top tier, priciest; :batch = 5/25)"),
    "grok46":      ("x-ai/grok-4.6",                2.00,  6.0, "Grok 4.6"),
    "llama70":     ("meta-llama/llama-3.3-70b-instruct", 0.10, 0.32, "Llama 3.3 70B (cheapest frontier-class)"),
    # --- test-only arms (mock server) ---
    "zerou":       ("mock/zerou", 0.00, 0.00, "TEST: mock, zero usage"),
    "nullc":       ("mock/nullc", 0.00, 0.00, "TEST: mock, returns null content"),
}

# ---- single briefing prompt: one source of style truth for every arm ----
BRIEFING = """You are writing the IDEAL answer to a memory-recall query for a fine-tuning dataset.
Below is a pool of FINDINGS (derived facts / raw messages) from a memory system plus a user QUESTION.
Write the answer the best possible terse recall agent would give.

Hard rules:
1. LENGTH: 1-3 short sentences, at most 50 words. No preamble, no "based on what I found", no re-stating
   the question, no bullet lists unless there are more than 2 items to enumerate.
2. GROUNDED ONLY: every name/date/number/item must come from the findings. Do NOT add outside knowledge,
   do NOT guess, do NOT soften with "likely"/"probably"/"seems".
3. CONTRADICTION: name BOTH conflicting values and state the conflict explicitly.
4. SUPERSESSION: give only the most recent value as current.
5. ABSTENTION: say plainly that no information about this is stored; mention none of the distractor facts.
6. QUOTE exact values (dates/numbers/names) — do not paraphrase them.

Return ONLY a JSON object, no prose, no code fences: {"answer": "..."}
"""

CTX_GEN = """Write ONE synthetic memory-recall context. Domain: {domain}. Category: {cat}
(category definition: {catdef})
Return ONLY a JSON object, no prose, no code fences:
{{
  "id": "{id}",
  "persona": "<2-line bio of the person whose memory this is>",
  "question": "<user question, 5-25 words>",
  "findings": [{{"text": "<a fact or short message, 10-60 words>", "date": "2026-0X-XX"}}, ... 7-9 findings ...],
  "distractors": ["<plausible WRONG fact in the same domain>", ... 3-5 ...],
  "required_facts": ["<entity/value every correct answer must contain>", ...],
  "forbidden_facts": ["<distractor values a correct answer must NOT assert>", ...]
}}
Rules: internally consistent with the category. 'contradiction' -> exactly two findings conflict.
'supersession' -> one finding states the change, another holds the new value. 'abstention' -> the question
targets a topic ABSENT from findings while distractors are plausible near-misses."""

DOMAIS = ["health","career","music","gaming","baking","travel","family","finance","diy","fitness","reading","home-lab"]
CATS = {
 "factual":"direct factual question, one exact answer present in findings",
 "preference":"question about stated preferences or standing instructions",
 "enumeration":"'list all X / how many Y' needing several distinct items",
 "contradiction":"two findings conflict for the same fact; answer must name both and refuse to pick one",
 "supersession":"old value + newer updated value; answer must give only the newer",
 "abstention":"NO answer in findings; correct answer is a clean 'no information' (distractors are plausible)",
 "summary":"summary of patterns over several findings across different dates",
}
MIX = [("factual",7),("preference",4),("enumeration",6),("contradiction",3),
       ("supersession",3),("abstention",3),("summary",4)]  # weights for any n (~1..: sum 30)
HEDGE = re.compile(r"\b(likely|probably|might|maybe|perhaps|possibly|seems?|appears?|presumably)\b", re.I)

R = "results/openrouter"


def load_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return os.environ["OPENROUTER_API_KEY"]
    for f in ("keys.env", os.path.join(os.path.dirname(os.path.abspath(__file__)), "keys.env")):
        if os.path.exists(f):
            for line in open(f):
                if line.strip().startswith("OPENROUTER_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return None


def chat(key, model, system, user, max_tokens=None):
    body = {
        "model": model,
        "max_tokens": max_tokens or MAX_TOKENS_OUT,
        "temperature": 0.2,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }
    req = urllib.request.Request(
        OPENROUTER + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={
            "Authorization": "Bearer " + key,
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/kalaspuffar/honcho-dialectic-model",
            "X-Title": "honcho-dialectic-model-trial",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT) as r:
        return json.load(r)


def extract_json(text):
    """Robust JSON object extractor.
    Scans for the first '{' whose full balanced parse succeeds (handles nested
    strings with escaped quotes, arrays, numbers). If none, returns None.
    """
    text = (text or "").strip()
    # strip code fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    # find candidate start positions
    start = 0
    while True:
        i = text.find("{", start)
        if i == -1:
            return None
        # attempt parse at i
        obj, end = _try_parse_at(text, i)
        if obj is not None:
            return obj
        start = i + 1
    return None

def _try_parse_at(text, i):
    """Try to parse a JSON object starting at text[i]. Return (obj, end) or (None, -1).
    Handles escaped quotes, nested braces inside strings, numbers, arrays."""
    n = len(text)
    depth = 0
    in_str = False
    esc = False
    j = i
    while j < n:
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    slice_ = text[i:j+1]
                    try:
                        return json.loads(slice_), j+1
                    except json.JSONDecodeError:
                        # tolerant retry: strip trailing commas (common in LLM JSON)
                        try:
                            import re as _re
                            fixed = _re.sub(r",\s*([}\]])", r"\1", slice_)
                            return json.loads(fixed), j+1
                        except Exception:
                            return None, -1
        j += 1
    return None, -1


def tokens_estimate(text):
    return (len(text or "") + 3) // 4  # ~4 chars/token, upper-ish bound


def user_block(c):
    return (BRIEFING
            + "\n\nPERSONA: " + c["persona"]
            + "\n\nFINDINGS:\n" + "\n".join(f"- [{f['date']}] {f['text']}" for f in c["findings"])
            + "\n\nDistractors (WRONG - do not assert):\n" + "\n".join("- " + d for d in c["distractors"])
            + "\n\nQUESTION: " + c["question"])


def est_cost(in_m, out_m, ctxs, n_out_words=55):
    tin = sum(6 + tokens_estimate(user_block(c)) for c in ctxs)  # +6 for JSON wrapper
    tout = len(ctxs) * n_out_words
    return tin * in_m / 1e6 + tout * out_m / 1e6, tin, tout


def build_contexts(key, model, n, raw_dir=None):
    cats = []
    for name, w in MIX:
        cats += [name] * w
    out = []
    for i in range(n):
        prompt = CTX_GEN.format(domain=DOMAIS[i % len(DOMAIS)], cat=cats[i % len(cats)],
                                catdef=CATS[cats[i % len(cats)]], id=f"{i:03d}")
        for attempt in range(3):
            try:
                resp = chat(key, model, "You build training contexts for an LLM fine-tuning dataset.\nReturn ONLY one JSON object, complete and untruncated.", prompt, max_tokens=CTX_MAX_TOKENS)
                content = resp["choices"][0]["message"]["content"]
                obj = extract_json(content)
                if obj and isinstance(obj, dict) and obj.get("persona") and obj.get("question") and obj.get("findings"):
                    obj["category"] = cats[i % len(cats)]
                    out.append(obj)
                    print(f"[ctx {len(out):2d}/{n}] {obj.get('id',i)} {str(obj['question'])[:58]}")
                    break
                else:
                    print(f"[ctx {i}] attempt {attempt+1}: got response but missing required keys. raw saved.")
                    if raw_dir:
                        try:
                            os.makedirs(raw_dir, exist_ok=True)
                            with open(os.path.join(raw_dir, f"ctx{i:03d}_att{attempt+1}.json"), "w") as dh:
                                json.dump({"model": model, "content": content, "usage": resp.get("usage", {})}, dh, indent=2, default=str)
                        except Exception:
                            pass
                    time.sleep(2)
            except Exception as e:
                print(f"[ctx {i}] attempt {attempt+1} failed: {type(e).__name__}: {e}")
                time.sleep(2)
        else:
            obj = {"id": f"{i:03d}", "persona": "synthetic user", "question": "x",
                   "findings": [{"text": "placeholder", "date": "2026-01-01"}],
                   "distractors": [], "required_facts": ["x"], "forbidden_facts": [],
                   "category": cats[i % len(cats)], "__failed_ctx": True}
            out.append(obj)
    return out


def score(ctxs, recs):
    words, coverage, fab, abst, hedge = [], [], 0, 0, 0
    for c in ctxs:
        r = recs.get(c["id"])
        a = (r or {}).get("answer")
        if not isinstance(a, str):
            a = ""
        cat = c.get("category")
        words.append(len(a.split()))
        if not a.strip():
            # model produced no answer: for non-abstention, that's a miss.
            # For abstention, an empty answer is also a miss (should have said "no information").
            coverage.append(0.0)
            continue
        forb = (c.get("forbidden_facts") or [])
        if cat == "abstention":
            if len(a.split()) <= 40 and not HEDGE.search(a) and not any(f.lower() in a.lower() for f in forb):
                abst += 1
            coverage.append(1.0)
        else:
            req = c.get("required_facts") or [""]
            ok = sum(1 for e in req if e and (e.lower() in a.lower()))
            coverage.append(ok / len(req))
            if forb and any(f.lower() in a.lower() for f in forb):
                fab += 1
        if HEDGE.search(a):
            hedge += 1
    n = max(1, len(ctxs))
    return {
        "n": len(ctxs),
        "median_words": int(statistics.median(words)) if words else None,
        "max_words": max(words) if words else None,
        "entity_coverage": round(statistics.mean(coverage), 3) if coverage else 0.0,
        "fabrication_rows": fab,
        "abstention_correct": abst,
        "hedge_rows": hedge,
    }


def cmd_estimate(a):
    ctxs_file = os.path.join(R, "contexts.jsonl")
    if os.path.exists(ctxs_file):
        ctxs = [json.loads(l) for l in open(ctxs_file) if l.strip()]
        src = "existing contexts.jsonl"
    else:
        # build a representative in-memory sample of size a.n for estimation
        random = __import__("random")
        random.seed(1); ctxs = [{"persona": "synthetic user with stable facts",
            "question": "What is the exact deadline?",
            "findings": [{"text": "the deadline was moved to April 22", "date": "2026-04-01"}] * 8,
            "distractors": ["April 25"], "required_facts": [], "forbidden_facts": []} for _ in range(a.n)]
        src = f"synthetic sample (n={a.n}) — contexts.jsonl not generated yet"
    arms = [k.strip() for k in a.arms.split(",") if k.strip()]
    print(f"estimate over {len(ctxs)} contexts (source: {src})\n")
    tot = 0
    for k in arms:
        if k not in ARMS:
            print(f"  {k:14s} UNKNOWN arm id"); continue
        mid, inm, outm, note = ARMS[k]
        c, tin, tout = est_cost(inm, outm, ctxs)
        tot += c
        print(f"  {k:14s} {mid:40s} ~${c:7.3f}   ({tin:,} tok in, {tout:,} tok out)  {note}")
    print(f"\n  TOTAL (one pass of each arm): ${tot:.3f}")
    print("  NOTE: context generation is a separate (small, one-time) cost on the --context-model you pick.")
    return 0


PRICE_BY_ID = {mid: (inm, outm) for mid, inm, outm, _ in ARMS.values()}

# ---- Anthropic native Message Batches (no BYOK, 50% of standard $) ----
# Daniel's Claude account is the teacher source; the deriver already uses this API.
# Prices are the Batch rates ($/MTok) = 50% of standard, identical to OpenRouter's :batch.
# NOTE: no `temperature`/`top_k`/`top_p` — post-Opus-4.6 models REJECT them (400).
ANTHROPIC = os.environ.get("ANTHROPIC_API_BASE", "https://api.anthropic.com")
ANTHROPIC_VERSION = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")
ANTHROPIC_ARMS = {
    "opus":   "claude-opus-5",
    "sonnet": "claude-sonnet-5",
    "fable":  "claude-fable-5",
}
ANTHROPIC_BATCH_PRICE = {
    "claude-opus-5":   (2.50, 12.50),
    "claude-sonnet-5": (1.00, 5.00),
    "claude-fable-5":  (5.00, 25.00),
}
BATCH_ANSWER_SYSTEM = "You write terse, fully grounded recall answers."

def load_anth_key():
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    for f in ("keys.env", os.path.join(os.path.dirname(os.path.abspath(__file__)), "keys.env")):
        if os.path.exists(f):
            for line in open(f):
                if line.strip().startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip()
    return None

def anth_headers(key):
    return {
        "Content-Type": "application/json",
        "anthropic-version": ANTHROPIC_VERSION,
        "x-api-key": key,
    }

def anth_get(url, key, timeout=120):
    req = urllib.request.Request(url, headers={"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()

def row_block(c):
    """Per-row user content (BRIEFING lives in the cached system block in batch mode)."""
    print(c['id'])
    return ("PERSONA: " + c["persona"]
            + "\n\nFINDINGS:\n" + "\n".join(f"- [{f['date']}] {f['text']}" for f in c["findings"])
            + "\n\nDistractors (WRONG - do not assert):\n" + "\n".join("- " + d for d in c["distractors"])
            + "\n\nQUESTION: " + c["question"])

def batch_build_payload(arm, ctxs):
    model = ANTHROPIC_ARMS[arm]
    shared = BATCH_ANSWER_SYSTEM + "\n\n" + BRIEFING
    return model, {
        "requests": [
            {
                "custom_id": f"{arm}_{c['id']}",  # must match ^[a-zA-Z0-9_-]{1,64}$
                "params": {
                    "model": model,
                    "max_tokens": MAX_TOKENS_OUT,  # >= 1 required; 800 cap
                    # Shared prefix across all rows -> prompt-cache hit (stacks with batch 50%).
                    "system": [{"type": "text", "text": shared,
                                "cache_control": {"type": "ephemeral"}}],
                    "messages": [{"role": "user", "content": row_block(c)}],
                },
            }
            for c in ctxs
        ]
    }

def batch_estimate(arm, ctxs):
    """(worst, best, tin_total, tout) — worst = zero cache hits;
    best = shared system prefix cached on (n-1) rows (cache reads = 10% of input)."""
    inm, outm = ANTHROPIC_BATCH_PRICE[ANTHROPIC_ARMS[arm]]
    shared_tok = tokens_estimate(BATCH_ANSWER_SYSTEM + "\n\n" + BRIEFING)
    row_toks = [6 + tokens_estimate(row_block(c)) for c in ctxs]
    n = len(ctxs)
    tin_worst = n * shared_tok + sum(row_toks)
    tin_best = shared_tok + (n - 1) * int(shared_tok * 0.10) + sum(row_toks)
    tout = n * 55  # terse answers ~55 words; cap 800 tokens
    worst = (tin_worst * inm + tout * outm) / 1e6
    best = (tin_best * inm + tout * outm) / 1e6
    return worst, best, tin_worst, tout

def batches_dir():
    d = os.path.join(R, "batches")
    os.makedirs(d, exist_ok=True)
    return d

def cmd_batch_run(a):
    key = load_anth_key()
    if not key:
        print("ANTHROPIC_API_KEY not set (env or keys.env). Aborting — nothing sent.")
        print("This uses your Anthropic account directly (50% batch rate), not OpenRouter.")
        return 1
    ctxs_file = os.path.join(R, "contexts.jsonl")
    if not os.path.exists(ctxs_file):
        print(f"no contexts file at {ctxs_file} — run `gen` first (OpenRouter, DeepSeek).")
        return 1
    ctxs = [json.loads(l) for l in open(ctxs_file) if l.strip() and not json.loads(l).get("__failed_ctx")]
    if not ctxs:
        print(f"no usable (non-failed) contexts in {ctxs_file} — regenerate them first.")
        return 1
    arms = [k.strip() for k in a.arms.split(",") if k.strip()]
    bad = [k for k in arms if k not in ANTHROPIC_ARMS]
    if bad:
        print(f"unknown arm(s) for Anthropic batch: {bad} — valid: {sorted(ANTHROPIC_ARMS)}")
        return 1

    print("pre-spend estimate (Anthropic batch 50% rate, NO network yet):")
    total = 0.0
    for k in arms:
        worst, best, tin, tout = batch_estimate(k, ctxs)
        total += worst
        print(f"  {k:9s} {ANTHROPIC_ARMS[k]:20s} {len(ctxs):>5d} rows  ~${best:.4f} (w/ cache) – ${worst:.4f} (no cache)")
    print(f"  EST TOTAL ${total:.4f}   cap ${a.max_usd}")
    if total > a.max_usd and not a.yes:
        print(f"\nABORT: estimate (no-cache worst case) ${total:.4f} exceeds --max-usd {a.max_usd}.\n"
              f"Real cost is lower (shared prompt is cacheable). Proceed anyway with --yes "
              f"or raise the cap. Nothing has been sent.")
        return 1

    for k in arms:
        model, payload = batch_build_payload(k, ctxs)
        worst, best, tin, tout = batch_estimate(k, ctxs)
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            ANTHROPIC + "/v1/messages/batches", data=body, headers=anth_headers(key), method="POST")
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                b = json.load(r)
        except urllib.error.HTTPError as e:
            print(f"  {k}: SUBMIT FAILED HTTP {e.code}: {e.read()[:300].decode(errors='replace')}")
            return 1
        except Exception as e:
            print(f"  {k}: SUBMIT FAILED {type(e).__name__}: {e}")
            return 1
        stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        mani = {
            "arm": k, "model": model, "n": len(ctxs), "est_usd_low": round(best, 5),
            "est_usd_high": round(worst, 5), "batch_id": b["id"],
            "processing_status": b.get("processing_status"),
            "submitted_at": datetime.datetime.now().isoformat(),
            "expires_at": b.get("expires_at"),
            "poll_url": f"{ANTHROPIC}/v1/messages/batches/{b['id']}",
            "results_url_template": f"{ANTHROPIC}/v1/messages/batches/{{id}}/results",
            "base_estimate_tokens_in": tin, "base_estimate_tokens_out": tout,
        }
        mp = os.path.join(batches_dir(), f"batch_{k}_{stamp}.json")
        open(mp, "w").write(json.dumps(mani, indent=2))
        print(f"  {k}: submitted batch {b['id']} ({len(ctxs)} requests). Manifest: {mp}")
    print("\nBatch processing is async (most finish in < 1h, 24h max). When ready, or any time:\n"
          f"  python3 openrouter_trial.py batch-fetch --manifest results/openrouter/batches/batch_<arm>_*.json\n"
          f"(without --no-wait it polls until ended, then scores and writes arms/<arm>/ as before.)")
    return 0


def cmd_batch_fetch(a):
    key = load_anth_key()
    if not key:
        print("ANTHROPIC_API_KEY not set (env or keys.env)."); return 1
    manifests = []
    if a.manifest:
        manifests = [a.manifest]
    else:
        bd = os.path.join(R, "batches")
        if not os.path.isdir(bd):
            print(f"no batch manifests under {bd} — run `batch-run` first."); return 1
        manifests = sorted(os.path.join(bd, f) for f in os.listdir(bd) if f.endswith(".json"))
    if not manifests:
        print("no batch manifests found."); return 1
    if len(manifests) > 1:
        print(f"multiple manifests; passing --manifest picks one. Using newest: {manifests[-1]}")
        manifests = [manifests[-1]]
    mp = manifests[0]
    m = json.load(open(mp))
    arm, model = m["arm"], m["model"]
    print(f"batch {m['batch_id']}  arm={arm} model={model} n={m['n']}")

    done_url = f"{ANTHROPIC.rstrip('/')}/v1/messages/batches/{m['batch_id']}/results"
    if not a.no_wait:
        while True:
            b = json.loads(anth_get(m["poll_url"], key))
            rc = b.get("request_counts", {})
            st = b.get("processing_status")
            print(f"  status={st} {rc}")
            if st == "ended":
                break
            if st in ("canceled", "cancelled"):
                print("batch was canceled — no results."); return 1
            time.sleep(a.poll_interval)
    else:
        b = json.loads(anth_get(m["poll_url"], key))
        if b.get("processing_status") != "ended":
            print(f"still {b.get('processing_status')}; use without --no-wait when it ends.")
            return 1
    if not done_url:
        print("could not build results_url. Check --manifest and base."); return 1

    # stream results (JSONL). Batch arms live under results/openrouter/batch/<arm>/ —
    # KEEP SEPARATE from arms/<arm>/ so they never clobber sync-run or other-source rows.
    ctxs_file = os.path.join(R, "contexts.jsonl")
    ctxs = {json.loads(l)["id"]: json.loads(l)
            for l in open(ctxs_file) if l.strip() and not json.loads(l).get("__failed_ctx")}
    batch_root = os.path.join(R, "batch")
    arm_dir = os.path.join(batch_root, arm)
    os.makedirs(arm_dir, exist_ok=True)
    raw_data = anth_get(done_url, key, timeout=300)
    lines = [l for l in (s.decode("utf-8", errors="replace") for s in raw_data.split(b"\n")) if l.strip()]
    recs, n_err = {}, 0
    for line in lines:
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            n_err += 1; continue
        cid = obj.get("custom_id", "")
        ctx_id = cid.split("_", 1)[1] if "_" in cid else cid
        res = obj.get("result", {})
        rec = {"id": ctx_id, "arm": arm, "model": model, "ok": False, "answer": ""}
        if res.get("type") == "succeeded":
            msg = res.get("message", {})
            text = ""
            for blk in (msg.get("content") or []):
                if isinstance(blk, dict) and blk.get("type") == "text":
                    text += blk.get("text", "")
            if not text and isinstance(msg.get("content"), str):
                text = msg["content"]
            u = msg.get("usage") or {}
            rec["usage"] = u
            j = extract_json(text)
            ans = (j or {}).get("answer") if isinstance(j, dict) else None
            rec["answer"] = ans if isinstance(ans, str) and ans.strip() else (text.strip() if text else "")
            rec["ok"] = bool(rec["answer"].strip())
            if not rec["ok"]:
                rec["error"] = "no answer text in batch result"
        else:
            rec["error"] = f"batch result type={res.get('type')}: " \
                           f"{json.dumps(res.get('error', {}))[:300]}"
            n_err += 1
        recs[ctx_id] = rec
        open(os.path.join(arm_dir, f"{ctx_id}.json"), "w").write(json.dumps(rec, indent=1))
    print(f"fetched {len(recs)} rows ({n_err} failed/expired/errored) -> {arm_dir}")

    # score + write a SEPARATE batch_summary.json (never merge into the sync-run summary.json)
    scored_ctxs = [c for c in (ctxs.get(k) for k in recs) if c]
    st = score(scored_ctxs, recs)
    sp = os.path.join(batch_root, "batch_summary.json")
    summary = json.load(open(sp)) if os.path.exists(sp) else {"ts": None, "arms": {}}
    inm, outm = ANTHROPIC_BATCH_PRICE.get(model, (0, 0))
    usage = sum((r.get("usage", {}).get("input_tokens", 0) * inm +
                 r.get("usage", {}).get("output_tokens", 0) * outm) for r in recs.values()) / 1e6
    summary["arms"][arm] = {"model": model, "source": f"anthropic-batch {m['batch_id']}",
                            "actual_usd_upper": round(usage, 5), **st,
                            "note": "usage excludes cached-input discount (billed at 10%); treat as worst case"}
    summary["ts"] = datetime.datetime.now().isoformat()
    open(sp, "w").write(json.dumps(summary, indent=1))
    print(f"-- {arm}: {st['n']} rows, median {st['median_words']}w, coverage {st['entity_coverage']}, "
          f"fab(flags) {st['fabrication_rows']}, abst {st['abstention_correct']}, hedges {st['hedge_rows']}")
    print(f"summary saved to {sp}")
    print(f"NOTE: batch rows live under results/openrouter/batch/ — separate from sync arms/. "
          f"Review batch/opus vs arms/opus (round 4) before deciding.")
    return 0

def cmd_gen(a):
    key = load_key()
    if not key:
        print("OPENROUTER_API_KEY not set (env or keys.env). Aborting — nothing sent."); return 1
    os.makedirs(R, exist_ok=True)
    # wallet: estimate context-gen spend before any call.
    # Contexts come out roughly 1.5-2.5k output tokens (4096 cap); use ~2000 for the estimate.
    model = a.context_model
    inm, outm = PRICE_BY_ID.get(model, (0.0, 0.0))
    est_ctx = a.n * (600 * inm + 2000 * outm) / 1e6
    if inm == 0 and outm == 0:
        print(f"NOTE: '{model}' not in local pricing table ({sorted(PRICE_BY_ID)}).")
        print("      Estimate is 0 — I cannot enforce the cap precisely for this model.")
    else:
        print(f"pre-spend estimate for context-gen: {a.n} contexts via {model} ≈ ${est_ctx:.4f} (cap ${a.max_usd})")
    if est_ctx > a.max_usd and not a.yes:
        print(f"ABORT: context-gen estimate {est_ctx:.4f} exceeds --max-usd {a.max_usd}. Use --yes to proceed anyway."); return 1
    ctxs = build_contexts(key, model, a.n, raw_dir=os.path.join(R, "raw-dumps"))
    p = os.path.join(R, "contexts.jsonl")
    open(p, "w").write("\n".join(json.dumps(c) for c in ctxs) + "\n")
    ok = sum(1 for c in ctxs if not c.get("__failed_ctx"))
    print(f"\nsaved {p}  ({ok} good / {len(ctxs)-ok} failed)  via {a.context_model}")
    return 0


def cmd_run(a):
    key = load_key()
    if not key:
        print("OPENROUTER_API_KEY not set (env or keys.env). Aborting — nothing sent."); return 1
    ctxs_file = os.path.join(R, "contexts.jsonl")
    if not os.path.exists(ctxs_file):
        print(f"no contexts file at {ctxs_file} — run `gen` first."); return 1
    ctxs = [json.loads(l) for l in open(ctxs_file) if l.strip() and not json.loads(l).get("__failed_ctx")]
    if not ctxs:
        print(f"no usable (non-failed) contexts in {ctxs_file} — regenerate them first."); return 1

    arms = [k.strip() for k in a.arms.split(",") if k.strip()]
    est_total = 0.0
    print("pre-spend estimate vs --max-usd cap:")
    for k in arms:
        mid, inm, outm, _ = ARMS[k]
        c, _, _ = est_cost(inm, outm, ctxs)
        est_total += c
        print(f"  {k:14s} {mid:40s} ~${c:7.3f}")
    print(f"  EST TOTAL ~${est_total:.3f}   cap ${a.max_usd}")
    if est_total > a.max_usd and not a.yes:
        print(f"\nABORT: estimate ${est_total:.3f} exceeds --max-usd {a.max_usd}.\n"
              f"Proceed anyway with --yes, or lower the arm set / raise the cap.")
        return 1

    os.makedirs(R, exist_ok=True)
    actual = {}
    running_dollars = 0.0
    cap = a.max_usd
    for k in arms:
        mid, *_ = ARMS[k]
        arm_dir = os.path.join(R, "arms", k)
        os.makedirs(arm_dir, exist_ok=True)
        recs = {}
        used = 0.0
        for c in ctxs:
            rec = {"id": c["id"], "arm": k, "model": mid, "ok": False, "answer": ""}
            try:
                resp = chat(key, mid, "You write terse, fully grounded recall answers.", user_block(c))
                choice = resp["choices"][0]["message"]["content"]
                # guard: some providers return null content (e.g. refusals / reasoning-only)
                if choice is None:
                    rec["error"] = "provider returned null content"
                    rec["answer"] = ""
                else:
                    j = extract_json(choice)
                    ans = (j or {}).get("answer") if isinstance(j, dict) else None
                    rec["answer"] = ans if isinstance(ans, str) and ans.strip() else (choice if isinstance(choice, str) else "")
                    rec["ok"] = bool(rec["answer"].strip())
                u = resp.get("usage") or {}
                inm, outm = ARMS[k][1], ARMS[k][2]
                used += (u.get("prompt_tokens", 0) * inm + u.get("completion_tokens", 0) * outm) / 1e6 if u else 0
                running_dollars += used
            except urllib.error.HTTPError as e:
                rec["error"] = f"HTTP {e.code}: {e.read()[:200].decode(errors='replace')}"
                print(f"  {k} {c['id']} HTTP {e.code}  ({e.read()[:120]})")
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
            recs[c["id"]] = rec
            open(os.path.join(arm_dir, f"{c['id']}.json"), "w").write(json.dumps(rec, indent=1))
            tag = "ok" if rec["ok"] else "ERR"
            print(f"  [{k} {c['id']}] {tag} {len((rec.get('answer') or '').split())}w  (running ${running_dollars:.4f})")
            if running_dollars > cap and not a.yes:
                print(f"\nLIVE CAP HIT: running spend ${running_dollars:.4f} exceeded --max-usd {cap}.")
                print(f"Partial results saved (arms done so far). Rerun with --yes to continue, or lower arms.")
                return 1
            time.sleep(a.sleep)
        actual[k] = {"model": mid, "estimated_usd": round(est_cost(ARMS[k][1], ARMS[k][2], ctxs)[0], 4),
                     "actual_usd": round(used, 4)}
        st = score(ctxs, recs); actual[k].update(st)
        print(f"-- {k}: {st['n']} rows, median {st['median_words']}w, coverage {st['entity_coverage']}, "
              f"fab {st['fabrication_rows']}, abst {st['abstention_correct']}, hedges {st['hedge_rows']}")

    summary = {"ts": datetime.datetime.now().isoformat(), "arms": actual}
    open(os.path.join(R, "summary.json"), "w").write(json.dumps(summary, indent=1))

    # blind review csv
    import random as _r
    pick = _r.Random(7).sample(ctxs, min(10, len(ctxs)))
    with open(os.path.join(R, "blind-review.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["id", "category", "question"] + [f"answer_{chr(65+i)}" for i in range(len(arms))])
        for c in pick:
            row = [c["id"], c.get("category"), c["question"]]
            for k in arms:
                rec = json.load(open(os.path.join(R, "arms", k, f"{c['id']}.json")))
                row.append(rec.get("answer", ""))
            w.writerow(row)

    # manifest
    mani = {"ts": summary["ts"], "arms": {k: ARMS[k][0] for k in arms},
            "arms_order": arms, "n_contexts": len(ctxs), "max_tokens_out": MAX_TOKENS_OUT}
    open(os.path.join(R, "run-manifest.json"), "w").write(json.dumps(mani, indent=1))

    blinding = ", ".join(f"{i+1}:{chr(65+i)}={k}" for i, k in enumerate(arms))
    print(f"\nDONE. Files in {R}/ : summary.json, blind-review.csv, run-manifest.json, arms/...")
    print(f"Blind-review legend (A/B/C... = {blinding}). Review the 10 rows 1-5; tell me the picks and I unblind.")
    return 0


def cmd_collect(a):
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    base = "openrouter"
    if not os.path.exists(R):
        print(f"nothing to collect at {R} yet."); return 1
    src = os.path.dirname(R.rstrip("/"))  # results/
    out = f"{src}/{base}_{stamp}.tar.gz"
    with tarfile.open(out, "w:gz") as t:
        t.add(R, arcname="results/openrouter")
    print(f"collected -> {out}  ({os.path.getsize(out)} bytes)\nCopy this file back to the host and I'll fold it into PLAN.md.")
    return 0


def main():
    ap = argparse.ArgumentParser(description="OpenRouter teacher A/B for the dialectic brevity model")
    sub = ap.add_subparsers(dest="cmd")

    e = sub.add_parser("estimate"); e.add_argument("--arms", default="opus,sonnet,gemini-pro,gpt5,deepseek,qwen3max")
    e.add_argument("--n", type=int, default=50, help="contexts if not yet generated (estimation sample size)")
    g = sub.add_parser("gen"); g.add_argument("--context-model", default="deepseek/deepseek-chat")
    g.add_argument("--n", type=int, default=30)
    g.add_argument("--max-usd", type=float, default=5.0); g.add_argument("--yes", action="store_true")
    r = sub.add_parser("run"); r.add_argument("--arms", default="deepseek,opus,sonnet,gemini-pro,gpt5,qwen3max")
    r.add_argument("--max-usd", type=float, default=5.0); r.add_argument("--yes", action="store_true")
    r.add_argument("--sleep", type=float, default=0.5)
    br = sub.add_parser("batch-run", help="submit Anthropic Message Batches (50% rate, your Claude account)")
    br.add_argument("--arms", default="opus")
    br.add_argument("--max-usd", type=float, default=5.0); br.add_argument("--yes", action="store_true")
    bf = sub.add_parser("batch-fetch", help="poll a batch until ended, then write/score arms/<arm>/")
    bf.add_argument("--manifest", help="specific batch_*.json manifest (default: newest)")
    bf.add_argument("--poll-interval", type=int, default=60)
    bf.add_argument("--no-wait", action="store_true", help="check once; fail if not ended yet")
    sub.add_parser("collect")

    a = ap.parse_args()
    handlers = {"estimate": cmd_estimate, "gen": cmd_gen, "run": cmd_run, "collect": cmd_collect,
                "batch-run": cmd_batch_run, "batch-fetch": cmd_batch_fetch}
    if a.cmd not in handlers:
        ap.print_help(); return 1
    return handlers[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
