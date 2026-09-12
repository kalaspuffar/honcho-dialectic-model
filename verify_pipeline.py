#!/usr/bin/env python3
"""verify_pipeline.py — offline self-test of the data pipeline. No key, no GPU, $0.

  python3 verify_pipeline.py          # static checks + unit checks + mock end-to-end run
  python3 verify_pipeline.py --quick  # skip the mock end-to-end run

The end-to-end part starts mock_or_server.py and mock_anth_batch.py on free ports and runs:
gen_contexts (sync + batch) -> gen_chosen (sync + batch) -> gen_rejected -> build_dataset ->
eval_model -> train_dialectic data prep with a fake tokenizer.
"""
import json
import os
import re
import socket
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)
PROBLEMS = []


def ok(name, cond, detail=""):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  — {detail}" if detail and not cond else ""))
    if not cond:
        PROBLEMS.append(name)
    return cond


def free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close(); return p


def wait_port(port, timeout=10):
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.5).close(); return True
        except OSError:
            time.sleep(0.1)
    return False


def sh(cmd, env=None, check=True):
    e = dict(os.environ); e.update(env or {})
    r = subprocess.run(cmd, cwd=HERE, env=e, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(r.stdout[-2000:]); print(r.stderr[-2000:])
    return r


# ------------------------------------------------------------------ 1. static
print("== 1. every .py parses ==")
for f in sorted(os.listdir(HERE)):
    if f.endswith(".py"):
        try:
            compile(open(f).read(), f, "exec"); ok(f, True)
        except SyntaxError as e:
            ok(f, False, str(e))

print("== 2. single scorer / single prompt ==")
defs = {f: open(f).read() for f in os.listdir(HERE) if f.endswith(".py")}
hedge_defs = [f for f, s in defs.items() if re.search(r"^HEDGE\s*=", s, re.M)]
ok("HEDGE regex defined only in scoring.py", hedge_defs == ["scoring.py"], str(hedge_defs))
ref_defs = [f for f, s in defs.items() if re.search(r"^REFUSAL\s*=", s, re.M)]
ok("REFUSAL regex defined only in scoring.py", ref_defs == ["scoring.py"], str(ref_defs))
sp_users = [f for f, s in defs.items() if "agent_system_prompt(" in s and f not in ("honcho_prompt.py", "verify_pipeline.py")]
ok("agent_system_prompt used only via trajectory.py (+probe)", set(sp_users) <= {"trajectory.py", "probe_toolcalls.py"}, str(sp_users))

# ------------------------------------------------------------------ 3. units
print("== 3. unit checks ==")
import llm_backend as be  # noqa: E402
import scoring  # noqa: E402
import trajectory  # noqa: E402
import build_dataset  # noqa: E402
import train_dialectic as td  # noqa: E402

for name, t, want in [("plain", '{"a": 1}', True), ("prose", 'x:\n{"q":1,"f":[]}', True),
                      ("trailing_comma", '{"a":[{"b":1},],"c":1,}', True), ("fenced", '```json\n{"a":1}\n```', True),
                      ("escaped", '{"q":"he said \\"yes\\""}', True), ("none", "nothing", False)]:
    ok(f"extract_json[{name}]", (be.extract_json(t) is not None) == want)

ok("resolve_model alias", be.resolve_model("opus").id == "claude-opus-5")
ok("resolve_model qualified", be.resolve_model("openrouter:x/y").provider == "openrouter")
ok("resolve_model bare claude", be.resolve_model("claude-sonnet-5").provider == "anthropic")
ok("student_endpoint ollama tag", be.student_endpoint("qwen3.5:9b", "http://h:1/v1")[:3] == ("ollama", "http://h:1/v1", "qwen3.5:9b"))
os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY") or "mock"
ok("student_endpoint openrouter alias", be.student_endpoint("qwen9b")[0] == "openrouter" and be.student_endpoint("qwen9b")[2] == "qwen/qwen3.5-9b")
ok("student_endpoint openrouter id", be.student_endpoint("openrouter:qwen/qwen3.5-9b")[1] == be.OPENROUTER_BASE)
u, _, _ = be.estimate_usd(be.resolve_model("opus"), [{"system": "a" * 4000, "user": "b" * 4000}], 100)
ub, _, _ = be.estimate_usd(be.resolve_model("opus"), [{"system": "a" * 4000, "user": "b" * 4000}], 100, batch=True)
ok("batch estimate is half of sync", abs(ub * 2 - u) < 1e-9)

ctx = {"id": "c1", "category": "supersession", "persona": {"name": "Maria", "bio": "x"},
       "question": "When is Maria's deadline?",
       "observations": [{"date": "2026-03-01", "text": "Maria set April 25.", "relevant": True},
                        {"date": "2026-04-01", "text": "Maria moved it to April 22.", "relevant": True},
                        {"date": "2026-02-01", "text": "Maria's rehearsal is May 3.", "relevant": False}],
       "searches": [{"tool": "search_memory", "query": "deadline", "results": [0, 1]},
                    {"tool": "grep_messages", "query": "May", "results": [2]}],
       "required_facts": ["April 22"], "forbidden_facts": ["May 3", "April 25"]}
msgs = trajectory.build_messages(ctx)
ok("trajectory: system,user then (assistant,tool)*", [m["role"] for m in msgs] == ["system", "user", "assistant", "tool", "assistant", "tool"])
ok("trajectory: system prompt is Honcho's", "Honcho's dialectic" in msgs[0]["content"] and "Maria" in msgs[0]["content"])
ok("trajectory: no findings in system prompt", "Maria set April 25" not in msgs[0]["content"])
ok("trajectory: tool result carries the observation", "April 22" in msgs[3]["content"])
ok("trajectory: tool_call args are a dict with observer/observed", msgs[2]["tool_calls"][0]["function"]["arguments"]["observer"] == "Maria")
s_msgs = trajectory.build_messages(ctx, arguments_as_string=True)
ok("trajectory: string args for HTTP", isinstance(s_msgs[2]["tool_calls"][0]["function"]["arguments"], str))
ok("trajectory: fallback search when plan missing", len(trajectory.searches_for({**ctx, "searches": []})[0]["results"]) == 3)

s = scoring.score_answer(ctx, "April 22, moved from April 25.")
ok("scoring: supersession with old value is NOT fabrication", s["req_ok"] and not s["fab"] and s["coverage"] == 1.0)
s = scoring.score_answer(ctx, "The deadline is May 3.")
ok("scoring: forbidden without required IS fabrication", s["fab"] and not s["req_ok"])
abst = {**ctx, "category": "abstention", "required_facts": []}
ok("scoring: clean refusal naming topic passes", scoring.score_answer(abst, "There is no information about a four-day workweek in memory.")["abst"])
ok("scoring: hedged refusal fails", not scoring.score_answer(abst, "There is probably no information about that.")["abst"])
ok("scoring: long refusal fails", not scoring.score_answer(abst, "I have no information about that. " + "word " * 70)["abst"])
for _a in ("No — memory only has her considering training for a half-marathon, not completing a full marathon.",
           "Memory holds that Chloe bought Montessori toys in November 2025, but it doesn't say she uses Montessori methods.",
           "Sofia has only mentioned visiting a friend in a high-rise, and she disliked the noise.",
           "No — Chloe herself has no known pet allergies. Her son Liam is allergic to cats."):
    ok("scoring: memory-limit phrasing counts as abstention: " + _a[:40], scoring.score_answer(abst, _a)["abst"])
for _a in ("No — Chloe owns a Centrifuge 3000, not a Centrifuge 5000.",
           "No — she tried a Peloton bike at a friend's house but found it too expensive.",
           "Yes, she uses a Thermo Fisher centrifuge."):
    ok("scoring: inferred negative / assertion is NOT an abstention: " + _a[:40], not scoring.score_answer(abst, _a)["abst"])

kept, dropped = build_dataset.filter_rows(
    [ctx, {**ctx, "id": "c2", "question": "Other?"}, {**ctx, "id": "c3"}],
    {"c1": {"answer": "long " * 40}, "c2": {"answer": "long " * 40}, "c3": {"answer": "long " * 40}},
    {"c1": {"answer": "April 22, moved from April 25."}, "c2": {"answer": "The deadline is May 3."},
     "c3": {"answer": "April 22."}})
ok("build_dataset: keeps good, drops fabrication, drops duplicate question",
   len(kept) == 1 and dropped.get("fabrication_in_chosen") == 1 and dropped.get("duplicate_question") == 1, str(dict(dropped)))
row = build_dataset.sft_row(kept[0])
ok("build_dataset: sft row ends with chosen + carries tools", row["messages"][-1]["content"] == "April 22, moved from April 25." and row["tools"])


class FakeTok:
    """Minimal chat-template tokenizer: 1 token per whitespace word."""
    pad_token_id = 0

    chat_template = ""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, tools=None, **kw):
        parts = ["<tools>" if tools else ""]
        for m in msgs:
            c = m.get("content") or ""
            if m.get("tool_calls"):
                c += " <tool_call> " + json.dumps(m["tool_calls"][0]["function"]["arguments"], sort_keys=True).replace(" ", "")
            parts.append(f"<|{m['role']}|> {c} <|end|>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
        return " ".join(p for p in parts if p)

    def __call__(self, text, add_special_tokens=False):
        vocab = {}
        return {"input_ids": [abs(hash(w)) % 30000 + 1 for w in text.split()]}

    def decode(self, ids):
        return f"<{len(ids)} tokens>"


class ThinkTok(FakeTok):
    """Qwen3.5-shaped: generation prompt ends '<think>\n' unless enable_thinking=False; a full
    assistant turn renders the closed block. Tokens: whitespace words, so '<think>' '</think>' are tokens."""
    chat_template = "... {%- if enable_thinking is defined and enable_thinking is false %} ..."

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=False, tools=None, enable_thinking=None, **kw):
        prefix = [m for m in msgs if m["role"] != "assistant" or m is not msgs[-1]]
        if add_generation_prompt:
            base = FakeTok.apply_chat_template(self, msgs, tools=tools, add_generation_prompt=True)
            return base + (" <think> \n\n </think> \n\n" if enable_thinking is False else " <think> \n")
        base = FakeTok.apply_chat_template(self, msgs[:-1], tools=tools, add_generation_prompt=True)
        m = msgs[-1]; c = m.get("content") or ""
        if m.get("tool_calls"):
            c += " <tool_call> " + json.dumps(m["tool_calls"][0]["function"]["arguments"], sort_keys=True).replace(" ", "")
        return base + " <think> \n\n </think> \n\n " + c + " <|end|>"

_te = td.encode_example(ThinkTok(), msgs, "April 22.", trajectory.TOOL_SCHEMAS, 100000)
_served = ThinkTok()(ThinkTok().apply_chat_template(msgs, tools=trajectory.TOOL_SCHEMAS, add_generation_prompt=True))["input_ids"]
_want = ThinkTok()("\n</think>\n\n April 22. <|end|>")["input_ids"]
ok("train prep: think-switch template -> prompt tokenized as served ('<think>\\n'), completion closes the block",
   _te["input_ids"][:len(_served)] == _served and _te["labels"][:len(_served)] == [-100] * len(_served)
   and _te["input_ids"][len(_served):] == _want and _te["labels"][len(_served):] == _want, str(_te)[:200])
ok("train prep: think-switch over-long dropped", td.encode_example(ThinkTok(), msgs, "April 22.", None, 5) is None)

enc = td.encode_example(FakeTok(), msgs, "April 22.", trajectory.TOOL_SCHEMAS, 100000)
n_train = sum(1 for l in enc["labels"] if l != -100)
ok("train prep: labels cover answer + end marker only", n_train == 3, f"trainable={n_train}")  # 'April' '22.' '<|end|>'
ok("train prep: prompt tokens masked", enc["labels"][0] == -100 and len(enc["labels"]) == len(enc["input_ids"]))
ok("train prep: over-long rows dropped, not truncated", td.encode_example(FakeTok(), msgs, "April 22.", None, 5) is None)
n_tool_turns = sum(1 for m in row["messages"] if m["role"] == "assistant" and m.get("tool_calls"))
samples, dropped, kinds = td.prepare_sft([row], FakeTok(), 100000)
ok("train prep: prepare_sft trains answer + every tool-call turn",
   kinds == {"answer": 1, "tool": n_tool_turns} and len(samples) == 1 + n_tool_turns and dropped == 0 and n_tool_turns >= 1,
   f"kinds={kinds} tool_turns={n_tool_turns}")
tool_prefix, tool_target, kind = td.sft_targets(row)[0]
ok("train prep: first target is the opening tool call with no results in the prefix",
   kind == "tool" and tool_target.get("tool_calls") and all(m["role"] != "tool" for m in tool_prefix))
tenc = td.encode_example(FakeTok(), tool_prefix, tool_target, trajectory.TOOL_SCHEMAS, 100000)
t_train = [t for t, l in zip(tenc["input_ids"], tenc["labels"]) if l != -100]
t_ids = FakeTok()("<tool_call> " + json.dumps(tool_target["tool_calls"][0]["function"]["arguments"], sort_keys=True).replace(" ", ""))["input_ids"]
ok("train prep: tool-call turn labels cover the rendered call only", t_train[:len(t_ids)] == t_ids and len(t_train) == len(t_ids) + 1,
   f"trainable={len(t_train)} expected={len(t_ids) + 1}")
s_first, _, k_first = td.prepare_sft([row], FakeTok(), 100000, tool_turns="first")
s_none, _, k_none = td.prepare_sft([row], FakeTok(), 100000, tool_turns="none")
ok("train prep: --tool-turns first/none", k_first == {"answer": 1, "tool": 1} and k_none == {"answer": 1, "tool": 0} and len(s_none) == 1)
pairs, dropped = td.prepare_dpo([build_dataset.dpo_row(kept[0])], FakeTok(), 100000)
ok("train prep: prepare_dpo shares the prefix", pairs and pairs[0][0]["input_ids"][:10] == pairs[0][1]["input_ids"][:10])
batch = td.pad_collator(0)([samples[0], {k: v[:5] for k, v in samples[0].items()}])
lab = batch["labels"]
lab = lab.tolist() if hasattr(lab, "tolist") else lab
ok("train prep: collator pads labels with -100", lab[1][-1] == -100 and len(lab[0]) == len(lab[1]))
ok("train: no hardcoded smoke split", "samples[:7]" not in defs["train_dialectic.py"])
_tail = ("{%- if add_generation_prompt %}\n    {{- '<|im_start|>assistant\\n' }}\n"
         "    {%- if enable_thinking is defined and enable_thinking is false %}\n        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
         "    {%- else %}\n        {{- '<think>\\n' }}\n    {%- endif %}\n{%- endif %}")
_t, _p = td.close_thinking_template(_tail)
ok("train: export closes the think block by default (TRAIN.md §12)", _p and td.THINK_SWITCH not in _t and td.THINK_SWITCH_CLOSED in _t)
ok("train: unknown template left alone", td.close_thinking_template("{{ .Prompt }}") == ("{{ .Prompt }}", False))
# 1e-5 (v0.8.0) saturated the 500-row run by step 25/126; the right value scales with 1/steps (TRAIN.md §10).
ok("train: DPO lr in a plausible range", 3e-7 <= td.DPO_DEFAULTS["lr"] <= 2e-5)
ok("train: DPO stops on saturation by default", td.DPO_STOP["loss"] > 0 and td.DPO_STOP["patience"] > 0)
ok("train: SFT merges the best eval epoch", "load_best_model_at_end=bool(eval_s)" in defs["train_dialectic.py"])

# ------------------------------------------------------------------ 4. e2e
if "--quick" not in sys.argv:
    print("== 4. mock end-to-end ==")
    p_or, p_an = free_port(), free_port()
    procs = [subprocess.Popen([sys.executable, "mock_or_server.py", str(p_or)], cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL),
             subprocess.Popen([sys.executable, "mock_anth_batch.py", str(p_an)], cwd=HERE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)]
    tmp = tempfile.mkdtemp(prefix="dialectic-e2e-")
    try:
        ok("mock servers up", wait_port(p_or) and wait_port(p_an))
        env_or = {"OPENROUTER_BASE": f"http://127.0.0.1:{p_or}/v1", "OPENROUTER_API_KEY": "mock"}
        env_an = {"ANTHROPIC_API_BASE": f"http://127.0.0.1:{p_an}", "ANTHROPIC_API_KEY": "mock",
                  "OPENROUTER_API_KEY": "mock"}
        ctxf = f"{tmp}/contexts.jsonl"
        # keep manifests out of the repo's results/ during the test
        r = sh([sys.executable, "gen_contexts.py", "run", "--n", "8", "--model", "deepseek", "--out", ctxf, "--concurrency", "3"], env_or)
        rows = be.read_jsonl(ctxf)
        ok("gen_contexts run (openrouter mock): 8 good rows", r.returncode == 0 and len(rows) == 8 and not any(be.failed(x) for x in rows))
        ok("gen_contexts: category mix applied", {x["category"] for x in rows} >= {"factual", "abstention"})
        ok("gen_contexts: abstention rows have empty required_facts", all(x["required_facts"] == [] for x in rows if x["category"] == "abstention"))
        r = sh([sys.executable, "gen_contexts.py", "run", "--n", "8", "--model", "deepseek", "--out", ctxf], env_or)
        ok("gen_contexts run is resume-safe (0 to generate)", "0 contexts to generate" in r.stdout)
        # anthropic sync + batch
        r = sh([sys.executable, "gen_contexts.py", "run", "--n", "2", "--start", "100", "--model", "opus", "--out", f"{tmp}/ctx_an.jsonl"], env_an)
        ok("gen_contexts run (anthropic sync mock)", r.returncode == 0 and len(be.read_jsonl(f"{tmp}/ctx_an.jsonl")) == 2)
        env_batch = dict(env_an, DIALECTIC_RESULTS_DIR=tmp)
        r = sh([sys.executable, "gen_contexts.py", "submit", "--n", "3", "--start", "200", "--model", "opus", "--out", f"{tmp}/ctx_batch.jsonl"], env_batch)
        ok("gen_contexts submit (anthropic batch mock)", r.returncode == 0 and "submitted batch" in r.stdout, r.stdout[-300:])
        r = sh([sys.executable, "gen_contexts.py", "status"], env_batch)
        ok("gen_contexts status", r.returncode == 0 and "status=ended" in r.stdout, r.stdout[-300:])
        r = sh([sys.executable, "gen_contexts.py", "fetch", "--poll-interval", "1"], env_batch)
        ok("gen_contexts fetch writes 3 rows", r.returncode == 0 and len(be.read_jsonl(f"{tmp}/ctx_batch.jsonl")) == 3, r.stdout[-300:])
        # chosen: openrouter sync, anthropic batch
        chof = f"{tmp}/chosen.jsonl"
        r = sh([sys.executable, "gen_chosen.py", "run", "--contexts", ctxf, "--model", "deepseek", "--out", chof], env_or)
        cho = be.read_jsonl(chof)
        ok("gen_chosen run (openrouter mock)", r.returncode == 0 and len(cho) == 8 and all("score" in x for x in cho))
        r = sh([sys.executable, "gen_chosen.py", "submit", "--contexts", ctxf, "--model", "opus", "--out", f"{tmp}/chosen_b.jsonl"], env_batch)
        r2 = sh([sys.executable, "gen_chosen.py", "fetch", "--poll-interval", "1"], env_batch)
        ok("gen_chosen submit+fetch (anthropic batch mock)", r.returncode == 0 and r2.returncode == 0 and len(be.read_jsonl(f"{tmp}/chosen_b.jsonl")) == 8, r2.stdout[-300:])
        # rejected via the OpenAI-compatible mock acting as Ollama
        rejf = f"{tmp}/rejected.jsonl"
        r = sh([sys.executable, "gen_rejected.py", "--contexts", ctxf, "--out", rejf, "--base", f"http://127.0.0.1:{p_or}/v1", "--model", "mock", "--concurrency", "2"])
        rej = be.read_jsonl(rejf)
        ok("gen_rejected: 8 verbose answers", r.returncode == 0 and len(rej) == 8 and all(x["words"] > 50 for x in rej), r.stdout[-300:])
        ok("gen_rejected: extra tool-call round handled", any(x.get("extra_calls") for x in rej))
        # rejected via OpenRouter (same mock behind OPENROUTER_BASE, bearer key, estimate + spend printed)
        rejf2 = f"{tmp}/rejected_or.jsonl"
        r = sh([sys.executable, "gen_rejected.py", "--contexts", ctxf, "--out", rejf2, "--model", "openrouter:mock/qwen", "--concurrency", "4"], env_or)
        rej2 = be.read_jsonl(rejf2)
        ok("gen_rejected (openrouter mock): 8 answers, estimate + spend printed",
           r.returncode == 0 and len(rej2) == 8 and not any(be.failed(x) for x in rej2)
           and "estimate:" in r.stdout and "usage-based spend" in r.stdout, r.stdout[-400:])
        r = sh([sys.executable, "gen_rejected.py", "--contexts", ctxf, "--out", f"{tmp}/rejected_cap.jsonl", "--model", "qwen9b", "--max-usd", "0"], env_or)
        ok("gen_rejected (openrouter mock): --max-usd cap stops early", r.returncode == 0 and "cost cap" in r.stdout, r.stdout[-300:])
        # dataset
        r = sh([sys.executable, "build_dataset.py", "--contexts", ctxf, "--rejected", rejf, "--chosen", chof, "--out", f"{tmp}/ds"])
        tr, ev = be.read_jsonl(f"{tmp}/ds_train.dpo.jsonl"), be.read_jsonl(f"{tmp}/ds_eval.dpo.jsonl")
        ok("build_dataset: emits train+eval, all kept", r.returncode == 0 and len(tr) + len(ev) == 8, r.stdout[-400:])
        ok("build_dataset: no persona in both splits",
           not ({x["prompt"][0]["content"] for x in tr} & {x["prompt"][0]["content"] for x in ev}) or not ev)
        sft = be.read_jsonl(f"{tmp}/ds_train.sft.jsonl")
        ok("build_dataset: sft rows are trajectories", sft and any(m["role"] == "tool" for m in sft[0]["messages"]))
        # eval
        r = sh([sys.executable, "eval_model.py", "--contexts", ctxf, "--ids-from", f"{tmp}/ds_eval.dpo.jsonl", "--model", "mock",
                "--base", f"http://127.0.0.1:{p_or}/v1", "--out", f"{tmp}/eval_mock.jsonl"])
        ok("eval_model run", r.returncode == 0 and os.path.exists(f"{tmp}/eval_mock.summary.json"), r.stderr[-300:])
        r = sh([sys.executable, "eval_model.py", "compare", f"{tmp}/eval_mock.jsonl", f"{tmp}/eval_mock.jsonl"])
        ok("eval_model compare", r.returncode == 0 and "median_words" in r.stdout)
        r = sh([sys.executable, "eval_model.py", "rescore", f"{tmp}/eval_mock.jsonl", "--contexts", ctxf])
        ok("eval_model rescore", r.returncode == 0 and json.load(open(f"{tmp}/eval_mock.summary.json")).get("rescored") is True, r.stderr[-300:])
        # baseline shape: answer inside <think>, empty content -> counted, and scored from reasoning only when asked
        r = sh([sys.executable, "eval_model.py", "--contexts", ctxf, "--ids-from", f"{tmp}/ds_eval.dpo.jsonl", "--model", "thinker",
                "--base", f"http://127.0.0.1:{p_or}/v1", "--out", f"{tmp}/eval_think.jsonl"])
        agg = json.load(open(f"{tmp}/eval_think.summary.json")) if r.returncode == 0 else {}
        ok("eval_model: answered-in-thinking rows counted, content empty", agg.get("answered_in_thinking_rows", 0) > 0
           and agg.get("empty_rows") == agg.get("answered_in_thinking_rows") and not agg.get("scored_from_reasoning"), r.stderr[-300:])
        r = sh([sys.executable, "eval_model.py", "--contexts", ctxf, "--ids-from", f"{tmp}/ds_eval.dpo.jsonl", "--model", "thinker",
                "--base", f"http://127.0.0.1:{p_or}/v1", "--answer-from-reasoning", "--out", f"{tmp}/eval_think2.jsonl"])
        agg = json.load(open(f"{tmp}/eval_think2.summary.json")) if r.returncode == 0 else {}
        ok("eval_model: --answer-from-reasoning scores the reasoning text", agg.get("scored_from_reasoning") is True
           and agg.get("empty_rows") == 0 and agg.get("answered_in_thinking_rows", 0) > 0, r.stderr[-300:])
        # train prep on the real emitted rows with the fake tokenizer
        samples, dropped, kinds = td.prepare_sft(sft, FakeTok(), 100000)
        ok("train prep on emitted sft rows", kinds["answer"] == len(sft) and kinds["tool"] >= len(sft) and dropped == 0,
           f"kinds={kinds} rows={len(sft)}")
    finally:
        for p in procs:
            p.terminate()
        print(f"  (e2e artefacts in {tmp})")

print()
if PROBLEMS:
    print(f"NO-GO — {len(PROBLEMS)} failing: {PROBLEMS}")
    sys.exit(1)
print("GO — all checks passed.")
