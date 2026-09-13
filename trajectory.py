#!/usr/bin/env python3
"""trajectory.py — build the Honcho-shaped conversation every stage shares.

Honcho's dialectic model never sees "findings in the system prompt". At runtime
it sees:

    system    agent_system_prompt(observer, observed, ...)   (honcho_prompt.py, verbatim copy)
    user      the question
    assistant tool_calls: search_memory(...)                  (one or more rounds)
    tool      the retrieved conclusions
    ...
    assistant the synthesized answer                         <- the only turn we train on

Every stage (rejected generation, chosen generation, dataset build, eval) builds
its messages through `build_messages(ctx)` so chosen and rejected share one
prompt and training matches the runtime distribution (PLAN risk R1/R2).

PARITY POINTS (re-copy from Honcho when it changes, then regenerate data):
  * honcho_prompt.py                — verbatim `src/dialectic/prompts.py`
  * TOOL_SCHEMAS below              — parameter names of the dialectic tools
  * format_tool_result() below      — how a tool result is rendered to the model

Context row schema (output of gen_contexts.py):
  {id, category, domain,
   persona: {name, bio},
   question,
   observations: [{date, text, relevant: bool}],
   searches:     [{tool, query, results: [observation index, ...]}],
   required_facts: [...], forbidden_facts: [...]}
"""
import json
import urllib.request

from honcho_prompt import agent_system_prompt

TOOLS = ["search_memory", "search_messages", "grep_messages", "get_reasoning_chain",
         "get_observation_context", "get_messages_by_date_range", "search_messages_temporal"]

SEARCH_TOOLS = ("search_memory", "search_messages", "grep_messages")

_Q = {"type": "string", "description": "Search query"}
_PAIR = {"observer": {"type": "string", "description": "Peer whose model this is"},
         "observed": {"type": "string", "description": "Peer the conclusions are about"}}
TOOL_SCHEMAS = [
    {"type": "function", "function": {"name": "search_memory",
     "description": "Semantic search over conclusions about this pair.",
     "parameters": {"type": "object", "properties": {"query": _Q, **_PAIR,
                    "top_k": {"type": "integer", "description": "Max results"}},
                    "required": ["query", "observer", "observed"]}}},
    {"type": "function", "function": {"name": "search_messages",
     "description": "Semantic search over messages in this query's scope.",
     "parameters": {"type": "object", "properties": {"query": _Q,
                    "top_k": {"type": "integer"}}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "grep_messages",
     "description": "Exact text search. Use for names, dates, keywords.",
     "parameters": {"type": "object", "properties": {"query": _Q}, "required": ["query"]}}},
    {"type": "function", "function": {"name": "get_reasoning_chain",
     "description": "Premises and downstream conclusions for a specific conclusion.",
     "parameters": {"type": "object", "properties": {"conclusion_id": {"type": "string"}},
                    "required": ["conclusion_id"]}}},
    {"type": "function", "function": {"name": "get_observation_context",
     "description": "Messages around a specific conclusion.",
     "parameters": {"type": "object", "properties": {"conclusion_id": {"type": "string"}},
                    "required": ["conclusion_id"]}}},
    {"type": "function", "function": {"name": "get_messages_by_date_range",
     "description": "Messages in a time window.",
     "parameters": {"type": "object", "properties": {"start_date": {"type": "string"},
                    "end_date": {"type": "string"}}, "required": ["start_date", "end_date"]}}},
    {"type": "function", "function": {"name": "search_messages_temporal",
     "description": "Semantic search with a date filter.",
     "parameters": {"type": "object", "properties": {"query": _Q,
                    "start_date": {"type": "string"}, "end_date": {"type": "string"}},
                    "required": ["query"]}}},
]

NO_RESULTS = "No results found."


def peer_name(ctx) -> str:
    p = ctx.get("persona")
    if isinstance(p, dict):
        return p.get("name") or "User"
    return "User"


def system_prompt(ctx) -> str:
    n = peer_name(ctx)
    return agent_system_prompt(n, n, None, None, TOOLS).strip()


def format_tool_result(tool: str, observations) -> str:
    """Render retrieved conclusions/messages the way the model sees a tool result."""
    if not observations:
        return NO_RESULTS
    if tool == "search_memory":
        rows = [{"content": o["text"], "created_at": f"{o['date']}T{_fake_time(i)}Z",
                 "level": "explicit"} for i, o in enumerate(observations)]
    else:  # message-style hits
        rows = [{"content": o["text"], "created_at": f"{o['date']}T{_fake_time(i)}Z"}
                for i, o in enumerate(observations)]
    return json.dumps(rows, ensure_ascii=False, indent=1)


def _fake_time(i: int) -> str:
    return f"{9 + (i * 3) % 12:02d}:{(i * 17) % 60:02d}:00"


def tool_arguments(ctx, search) -> dict:
    n = peer_name(ctx)
    tool = search.get("tool", "search_memory")
    args = {"query": search.get("query", ctx["question"])}
    if tool == "search_memory":
        args.update(observer=n, observed=n, top_k=15)
    elif tool == "search_messages":
        args["top_k"] = 15
    return args


def searches_for(ctx):
    """Validated search plan; falls back to one search_memory over everything."""
    obs = ctx.get("observations") or []
    plan = []
    for s in ctx.get("searches") or []:
        idx = [i for i in (s.get("results") or []) if isinstance(i, int) and 0 <= i < len(obs)]
        tool = s.get("tool") if s.get("tool") in SEARCH_TOOLS else "search_memory"
        if s.get("query"):
            plan.append({"tool": tool, "query": s["query"], "results": idx})
    if not plan:
        plan = [{"tool": "search_memory", "query": ctx["question"], "results": list(range(len(obs)))}]
    return plan


def build_messages(ctx, arguments_as_string=False):
    """[system, user, (assistant tool_calls, tool)*] — the prefix before the answer.
    `arguments_as_string=True` for OpenAI-compatible HTTP (Ollama); dicts for datasets."""
    obs = ctx.get("observations") or []
    msgs = [{"role": "system", "content": system_prompt(ctx)},
            {"role": "user", "content": ctx["question"]}]
    for k, s in enumerate(searches_for(ctx)):
        call_id = f"call_{ctx.get('id', 'x')}_{k}"
        args = tool_arguments(ctx, s)
        msgs.append({"role": "assistant", "content": "",
                     "tool_calls": [{"id": call_id, "type": "function",
                                     "function": {"name": s["tool"],
                                                  "arguments": json.dumps(args) if arguments_as_string else args}}]})
        msgs.append({"role": "tool", "tool_call_id": call_id, "name": s["tool"],
                     "content": format_tool_result(s["tool"], [obs[i] for i in s["results"]])})
    return msgs


def stringify_tool_args(msgs):
    out = []
    for m in msgs:
        if m.get("tool_calls"):
            m = dict(m, tool_calls=[
                dict(tc, function=dict(tc["function"], arguments=(
                    tc["function"]["arguments"] if isinstance(tc["function"]["arguments"], str)
                    else json.dumps(tc["function"]["arguments"])))) for tc in m["tool_calls"]])
        out.append(m)
    return out


# ------------------------------------------------ student answering (OpenAI-compatible)
OPENROUTER_HEADERS = {"HTTP-Referer": "honcho-dialectic-model",
                      "X-Title": "honcho-dialectic-model"}


def _chat(base, body, timeout=900, api_key=None):
    """POST /chat/completions on any OpenAI-compatible endpoint (Ollama, OpenRouter).
    Returns (message, usage). `api_key` adds the bearer header OpenRouter needs."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = "Bearer " + api_key
        headers.update(OPENROUTER_HEADERS)
    req = urllib.request.Request(base.rstrip("/") + "/chat/completions",
                                 data=json.dumps(body).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    if not data.get("choices"):          # OpenRouter reports provider errors in-body with HTTP 200
        raise RuntimeError(f"no choices: {str(data.get('error') or data)[:300]}")
    ch = data["choices"][0]
    m = dict(ch["message"])
    m["_finish_reason"] = ch.get("finish_reason")
    # Ollama's /v1 and OpenRouter split the model's <think> text into a side field; qwen3.5:9b on
    # Ollama puts its whole answer there and stops with EMPTY content (TRAIN.md §7, 2026-09-11).
    m["_reasoning"] = (m.get("reasoning") or m.get("reasoning_content") or "")
    return m, data.get("usage") or {}


def _add_usage(total, usage):
    for k in ("prompt_tokens", "completion_tokens"):
        total[k] = total.get(k, 0) + int(usage.get(k) or 0)


def answer_with_ollama(base, model, ctx, max_rounds=3, temperature=0.3, max_tokens=1500, api_key=None, extra_body=None):
    """Run the student model on the trajectory, the way Honcho's loop would.

    Works against any OpenAI-compatible /chat/completions endpoint: Ollama's /v1
    (no key) or OpenRouter (pass `api_key`) — see `llm_backend.student_endpoint`.
    If the model asks for more tool calls we answer each with NO_RESULTS (the
    scenario's retrieval is already complete) for up to `max_rounds`, then force
    a synthesis turn by dropping the tool schemas. Returns
    {"answer", "extra_calls", "forced", "finish_reason", "reasoning",
     "usage": {"prompt_tokens", "completion_tokens"}}. A non-empty `reasoning` with an empty
    `answer` means the model wrote its answer inside the thinking block and stopped — what
    qwen3.5:9b does on Ollama, and what the fine-tune is meant to remove."""
    msgs = build_messages(ctx, arguments_as_string=True)
    extra, forced, usage = 0, False, {}
    # extra_body: request fields Honcho can also send, e.g. {"reasoning_effort": "none"} — Ollama's /v1
    # maps that to think=false (closed <think> block in the prompt); Honcho sends it from
    # DIALECTIC_LEVELS__<level>__MODEL_CONFIG__THINKING_EFFORT=none (TRAIN.md §12, 2026-09-12).
    xb = extra_body or {}
    for _ in range(max_rounds):
        m, u = _chat(base, {"model": model, "temperature": temperature, "max_tokens": max_tokens,
                            "messages": msgs, "tools": TOOL_SCHEMAS, **xb}, api_key=api_key)
        _add_usage(usage, u)
        calls = m.get("tool_calls") or []
        if not calls:
            return _result(m, extra, forced, usage)
        msgs.append({"role": "assistant", "content": m.get("content") or "", "tool_calls": calls})
        for tc in calls:
            extra += 1
            msgs.append({"role": "tool", "tool_call_id": tc.get("id", f"call_extra_{extra}"),
                         "name": (tc.get("function") or {}).get("name", ""), "content": NO_RESULTS})
    forced = True
    m, u = _chat(base, {"model": model, "temperature": temperature, "max_tokens": max_tokens,
                        "messages": msgs, **xb}, api_key=api_key)
    _add_usage(usage, u)
    return _result(m, extra, forced, usage)


def _result(m, extra, forced, usage):
    return {"answer": (m.get("content") or "").strip(), "extra_calls": extra, "forced": forced,
            "finish_reason": m.get("_finish_reason"), "reasoning": m.get("_reasoning", ""),
            "usage": usage}


answer_on_trajectory = answer_with_ollama   # provider-neutral name
