#!/usr/bin/env python3
"""llm_backend.py — one stdlib-only client for every teacher model we use.

Two providers, one interface:

  openrouter   sync HTTP, run concurrently with a thread pool ("batch" here means
               N parallel requests with retries + resume; OpenRouter has no
               asynchronous batch endpoint).
  anthropic    either the same concurrent sync path (`run`) or the Message
               Batches API (`submit` / `status` / `fetch`, 50% price, async,
               finishes in minutes to hours).

Raw HTTP on purpose: the machines that run the data scripts have no pip access,
so the official SDKs are not an option here (CLAUDE.md "stdlib-only by design").

Model selection (`--model`):
  alias from MODELS below            opus | sonnet | fable | haiku | deepseek | qwen3max | ...
  provider-qualified id              anthropic:claude-opus-5   openrouter:deepseek/deepseek-chat
  bare id                            claude-*  -> anthropic,   vendor/model -> openrouter

Keys: ANTHROPIC_API_KEY / OPENROUTER_API_KEY from the environment or keys.env
next to this file. Key values are never printed.
"""
import concurrent.futures
import datetime
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OPENROUTER_BASE = os.environ.get("OPENROUTER_BASE", "https://openrouter.ai/api/v1")
ANTHROPIC_BASE = os.environ.get("ANTHROPIC_API_BASE", "https://api.anthropic.com")
ANTHROPIC_VERSION = os.environ.get("ANTHROPIC_VERSION", "2023-06-01")
HTTP_TIMEOUT = 180
BATCH_DISCOUNT = 0.5          # Anthropic Message Batches bill 50% of list price

# alias -> (provider, model id, list $/M input, list $/M output)
# Anthropic prices: first-party list (2026-06). OpenRouter prices: /models catalog
# 2026-09-03; refresh with `python3 list_models.py`.
MODELS = {
    "opus":       ("anthropic",  "claude-opus-5",                   5.00, 25.00),
    "sonnet":     ("anthropic",  "claude-sonnet-5",                 2.00, 10.00),
    "fable":      ("anthropic",  "claude-fable-5-1",               10.00, 50.00),
    "haiku":      ("anthropic",  "claude-haiku-4-5",                1.00,  5.00),
    "deepseek":   ("openrouter", "deepseek/deepseek-chat",          0.32,  0.89),
    "qwen3max":   ("openrouter", "qwen/qwen3-max",                  0.78,  3.90),
    "gemini-pro": ("openrouter", "google/gemini-3.1-pro-preview",   2.00, 12.00),
    "gpt5":       ("openrouter", "openai/gpt-5",                    1.25, 10.00),
    "grok46":     ("openrouter", "x-ai/grok-4.6",                   2.00,  6.00),
    "llama70":    ("openrouter", "meta-llama/llama-3.3-70b-instruct", 0.10, 0.32),
    # student on OpenRouter (stage 2 / eval when the Ollama host is busy)
    "qwen9b":     ("openrouter", "qwen/qwen3.5-9b",                 0.10,  0.15),
}
OLLAMA_DEFAULT_BASE = "http://node7.ea.org:11434/v1"
OLLAMA_DEFAULT_MODEL = "qwen3.5:9b"


class ModelSpec:
    def __init__(self, provider, model_id, in_usd_m, out_usd_m, alias=None):
        self.provider, self.id = provider, model_id
        self.in_usd_m, self.out_usd_m = in_usd_m, out_usd_m
        self.alias = alias or model_id

    def __repr__(self):
        return f"{self.provider}:{self.id}"


def resolve_model(spec: str) -> ModelSpec:
    s = (spec or "").strip()
    if s in MODELS:
        p, mid, i, o = MODELS[s]
        return ModelSpec(p, mid, i, o, alias=s)
    if ":" in s and s.split(":", 1)[0] in ("anthropic", "openrouter"):
        p, mid = s.split(":", 1)
    elif s.startswith("claude-"):
        p, mid = "anthropic", s
    elif "/" in s:
        p, mid = "openrouter", s
    else:
        raise SystemExit(f"unknown model '{spec}'. Use an alias ({', '.join(sorted(MODELS))}), "
                         "'anthropic:<id>' or 'openrouter:<vendor/model>'.")
    known = {mid_: (i, o) for _, (pp, mid_, i, o) in MODELS.items() if pp == p}
    i, o = known.get(mid, (0.0, 0.0))
    if (i, o) == (0.0, 0.0):
        print(f"[backend] no price on file for {p}:{mid}; cost estimates will read $0")
    return ModelSpec(p, mid, i, o)


# ---------------------------------------------------------------- keys / util
def load_key(name: str):
    """Env var first, then keys.env next to this file. Never printed."""
    if os.environ.get(name):
        return os.environ[name]
    for f in ("keys.env", os.path.join(HERE, "keys.env")):
        if os.path.exists(f):
            for line in open(f):
                line = line.strip()
                if line.startswith(name + "="):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    return None


def setting(name: str, default: str) -> str:
    """Non-secret setting (e.g. OLLAMA_BASE) from the environment or keys.env."""
    return load_key(name) or default


def student_endpoint(model: str, base=None):
    """Where the *student* (stage 2 rejected / eval) runs. Returns
    (provider, base_url, model_id, api_key, ModelSpec-or-None).

    Ollama (free, default): a bare Ollama tag such as 'qwen3.5:9b'.
    OpenRouter: an alias with provider openrouter ('qwen9b'), 'openrouter:<vendor/model>'
    or a bare '<vendor/model>'. Base comes from OPENROUTER_BASE (mockable); the key from
    OPENROUTER_API_KEY. An explicit `base` always wins."""
    m = (model or "").strip()
    is_or = (m in MODELS and MODELS[m][0] == "openrouter") or m.startswith("openrouter:") or "/" in m
    if is_or:
        spec = resolve_model(m)
        if spec.provider != "openrouter":
            raise SystemExit(f"student model must be an Ollama tag or an OpenRouter id, got {spec}")
        return "openrouter", (base or OPENROUTER_BASE), spec.id, key_for(spec), spec
    return "ollama", (base or setting("OLLAMA_BASE", OLLAMA_DEFAULT_BASE)), m, None, None


def key_for(spec: ModelSpec):
    name = "ANTHROPIC_API_KEY" if spec.provider == "anthropic" else "OPENROUTER_API_KEY"
    key = load_key(name)
    if not key:
        raise SystemExit(f"{name} not set (environment or keys.env). Nothing sent.")
    return key


def tokens_estimate(text) -> int:
    return (len(text or "") + 3) // 4


def estimate_usd(spec: ModelSpec, jobs, out_tokens_per_job: int, batch: bool = False):
    """(usd, in_tokens, out_tokens). Batch pricing only applies to Anthropic."""
    tin = sum(tokens_estimate(j["system"]) + tokens_estimate(j["user"]) + 8 for j in jobs)
    tout = len(jobs) * out_tokens_per_job
    usd = (tin * spec.in_usd_m + tout * spec.out_usd_m) / 1e6
    if batch and spec.provider == "anthropic":
        usd *= BATCH_DISCOUNT
    return usd, tin, tout


def usage_usd(spec: ModelSpec, usage: dict, batch: bool = False) -> float:
    if not usage:
        return 0.0
    tin = usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0
    tout = usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0
    cr = usage.get("cache_read_input_tokens", 0) or 0
    cw = usage.get("cache_creation_input_tokens", 0) or 0
    usd = (tin * spec.in_usd_m + cr * spec.in_usd_m * 0.1 + cw * spec.in_usd_m * 1.25
           + tout * spec.out_usd_m) / 1e6
    return usd * (BATCH_DISCOUNT if batch and spec.provider == "anthropic" else 1.0)


def extract_json(text):
    """Tolerant JSON-object extractor: strips code fences, scans for the first
    '{' whose balanced slice parses (retrying without trailing commas)."""
    text = (text or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text).strip()
    start = 0
    while True:
        i = text.find("{", start)
        if i == -1:
            return None
        obj = _try_parse_at(text, i)
        if obj is not None:
            return obj
        start = i + 1


def _try_parse_at(text, i):
    depth, in_str, esc = 0, False, False
    for j in range(i, len(text)):
        ch = text[j]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                s = text[i:j + 1]
                try:
                    return json.loads(s)
                except json.JSONDecodeError:
                    try:
                        return json.loads(re.sub(r",\s*([}\]])", r"\1", s))
                    except Exception:
                        return None
    return None


def _http(req, timeout=HTTP_TIMEOUT):
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _post_json(url, headers, body, timeout=HTTP_TIMEOUT):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    return json.loads(_http(req, timeout))


# --------------------------------------------------------- single completions
def _anth_headers(key):
    return {"Content-Type": "application/json", "anthropic-version": ANTHROPIC_VERSION,
            "x-api-key": key}


def _anth_params(spec, job, effort):
    p = {"model": spec.id, "max_tokens": job["max_tokens"],
         # Shared system text first -> prompt-cache hit across rows (stacks with batch 50%).
         "system": [{"type": "text", "text": job["system"], "cache_control": {"type": "ephemeral"}}],
         "messages": [{"role": "user", "content": job["user"]}]}
    if effort:
        p["output_config"] = {"effort": effort}   # thinking depth; no temperature on Claude 5 models
    return p


def _anth_text(msg):
    text = ""
    for blk in msg.get("content") or []:
        if isinstance(blk, dict) and blk.get("type") == "text":
            text += blk.get("text", "")
    return text


def complete(spec: ModelSpec, key: str, job: dict, effort=None, temperature=0.2):
    """One synchronous completion. Returns {"text", "usage", "error"}."""
    if spec.provider == "anthropic":
        body = _anth_params(spec, job, effort)
        data = _post_json(ANTHROPIC_BASE + "/v1/messages", _anth_headers(key), body)
        if data.get("stop_reason") == "refusal":
            return {"text": "", "usage": data.get("usage"), "error": "refusal"}
        return {"text": _anth_text(data), "usage": data.get("usage"), "error": ""}
    body = {"model": spec.id, "max_tokens": job["max_tokens"], "temperature": temperature,
            "messages": [{"role": "system", "content": job["system"]},
                         {"role": "user", "content": job["user"]}]}
    headers = {"Authorization": "Bearer " + key, "Content-Type": "application/json",
               "HTTP-Referer": "https://github.com/kalaspuffar/honcho-dialectic-model",
               "X-Title": "honcho-dialectic-model"}
    data = _post_json(OPENROUTER_BASE + "/chat/completions", headers, body)
    if "error" in data and not data.get("choices"):
        return {"text": "", "usage": data.get("usage"), "error": str(data["error"])[:300]}
    msg = (data.get("choices") or [{}])[0].get("message") or {}
    text = msg.get("content")
    if not isinstance(text, str):
        return {"text": "", "usage": data.get("usage"), "error": "null content"}
    return {"text": text, "usage": data.get("usage"), "error": ""}


def _retryable(e):
    if isinstance(e, urllib.error.HTTPError):
        return e.code in (408, 409, 429) or e.code >= 500
    return isinstance(e, (urllib.error.URLError, TimeoutError, ConnectionError))


def complete_with_retry(spec, key, job, effort=None, attempts=4):
    last = None
    for k in range(attempts):
        try:
            return complete(spec, key, job, effort)
        except Exception as e:  # noqa: BLE001 — classify below
            last = e
            if not _retryable(e):
                break
            time.sleep(min(60, 2 ** k + 0.5 * k))
    detail = ""
    if isinstance(last, urllib.error.HTTPError):
        try:
            detail = last.read()[:200].decode(errors="replace")
        except Exception:
            pass
    return {"text": "", "usage": None, "error": f"{type(last).__name__}: {last} {detail}".strip()}


def run_concurrent(spec: ModelSpec, jobs, concurrency=4, effort=None, on_result=None, max_usd=None):
    """Run every job (dicts with custom_id/system/user/max_tokens) in parallel.
    Returns {custom_id: {"text","usage","error"}}. If `max_usd` is given, stops
    submitting new work once the running usage-based spend exceeds it (jobs not
    started are reported with error 'cost cap')."""
    key = key_for(spec)
    results, spent, lock = {}, [0.0], threading.Lock()
    stop = threading.Event()

    def work(job):
        if stop.is_set():
            return job["custom_id"], {"text": "", "usage": None, "error": "cost cap"}
        r = complete_with_retry(spec, key, job, effort)
        with lock:
            spent[0] += usage_usd(spec, r.get("usage") or {})
            if max_usd is not None and spent[0] > max_usd:
                stop.set()
        return job["custom_id"], r

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        for cid, r in ex.map(work, jobs):
            results[cid] = r
            if on_result:
                on_result(cid, r)
    if stop.is_set():
        print(f"[backend] cost cap ${max_usd:.2f} reached at ~${spent[0]:.2f}; remaining jobs skipped")
    else:
        print(f"[backend] done, usage-based spend ≈ ${spent[0]:.3f}")
    return results


# --------------------------------------------------------- Anthropic batches
def batch_submit(spec: ModelSpec, jobs, effort=None) -> dict:
    if spec.provider != "anthropic":
        raise SystemExit("Message Batches exist only for Anthropic models; use `run` for OpenRouter.")
    key = key_for(spec)
    payload = {"requests": [{"custom_id": j["custom_id"], "params": _anth_params(spec, j, effort)}
                            for j in jobs]}
    for j in jobs:
        if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", j["custom_id"]):
            raise SystemExit(f"custom_id '{j['custom_id']}' violates ^[a-zA-Z0-9_-]{{1,64}}$")
    try:
        return _post_json(ANTHROPIC_BASE + "/v1/messages/batches", _anth_headers(key), payload)
    except urllib.error.HTTPError as e:
        raise SystemExit(f"batch submit failed HTTP {e.code}: {e.read()[:400].decode(errors='replace')}")


def batch_status(batch_id: str) -> dict:
    key = load_key("ANTHROPIC_API_KEY")
    if not key:
        raise SystemExit("ANTHROPIC_API_KEY not set (environment or keys.env).")
    req = urllib.request.Request(f"{ANTHROPIC_BASE}/v1/messages/batches/{batch_id}",
                                 headers=_anth_headers(key))
    return json.loads(_http(req))


def batch_results(batch_id: str) -> dict:
    """{custom_id: {"text","usage","error"}} from the JSONL results stream.
    Results come back in any order — always keyed by custom_id."""
    key = load_key("ANTHROPIC_API_KEY")
    req = urllib.request.Request(f"{ANTHROPIC_BASE}/v1/messages/batches/{batch_id}/results",
                                 headers=_anth_headers(key))
    raw = _http(req, timeout=600)
    out = {}
    for line in raw.decode("utf-8", errors="replace").split("\n"):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        cid, res = obj.get("custom_id", ""), obj.get("result", {})
        if res.get("type") == "succeeded":
            msg = res.get("message", {})
            if msg.get("stop_reason") == "refusal":
                out[cid] = {"text": "", "usage": msg.get("usage"), "error": "refusal"}
            else:
                out[cid] = {"text": _anth_text(msg), "usage": msg.get("usage"), "error": ""}
        else:
            out[cid] = {"text": "", "usage": None,
                        "error": f"{res.get('type')}: {json.dumps(res.get('error', {}))[:300]}"}
    return out


def batch_wait(batch_id: str, poll_seconds=60) -> dict:
    while True:
        try:
            b = batch_status(batch_id)
        except Exception as e:  # noqa: BLE001
            print(f"  status poll error ({type(e).__name__}: {e}); retrying in {poll_seconds}s")
            time.sleep(poll_seconds)
            continue
        st = b.get("processing_status")
        print(f"  batch {batch_id}: {st} {b.get('request_counts', {})}", flush=True)
        if st == "ended":
            return b
        if st in ("canceled", "cancelled"):
            raise SystemExit("batch was canceled — no results.")
        time.sleep(poll_seconds)


# ------------------------------------------------------------- manifests
def results_dir() -> str:
    """results/ next to this file; DIALECTIC_RESULTS_DIR overrides (tests)."""
    return os.environ.get("DIALECTIC_RESULTS_DIR") or os.path.join(HERE, "results")


def manifest_dir(kind: str) -> str:
    d = os.path.join(results_dir(), "batches", kind)
    os.makedirs(d, exist_ok=True)
    return d


def write_manifest(kind: str, data: dict) -> str:
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = os.path.join(manifest_dir(kind), f"{stamp}_{data.get('batch_id', 'x')}.json")
    data = dict(data, kind=kind, submitted_at=datetime.datetime.now().isoformat())
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return path


def list_manifests(kind: str):
    d = manifest_dir(kind)
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith(".json"))


def newest_manifest(kind: str, explicit=None):
    if explicit:
        return explicit
    ms = list_manifests(kind)
    if not ms:
        raise SystemExit(f"no batch manifests under {manifest_dir(kind)} — run `submit` first.")
    return ms[-1]


# ------------------------------------------------------------- jsonl helpers
def read_jsonl(path):
    if not path or not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def write_jsonl(path, rows):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def failed(row) -> bool:
    return str(row.get("answer", row.get("__failed__", ""))).startswith("__FAILED__") or bool(row.get("__failed__"))


def merge_rows(path, new_rows, key="id"):
    """Resume-safe merge: rows already good on disk win over new failures;
    new good rows replace old failures. Returns the merged list in the new order."""
    old = {r[key]: r for r in read_jsonl(path)}
    merged = []
    seen = set()
    for r in new_rows:
        k = r[key]
        prev = old.get(k)
        if prev is not None and not failed(prev) and failed(r):
            merged.append(prev)
        else:
            merged.append(r)
        seen.add(k)
    for k, r in old.items():
        if k not in seen:
            merged.append(r)
    return merged
