# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

Synthetic-data + fine-tuning pipeline that produces a terse "dialectic" (memory-recall) model for
Honcho from a Qwen 8B/9B base. Flat directory of standalone Python 3 scripts, no package, no build
system, no pytest. Every data script is **stdlib-only by design** (the machines that run them have
no pip access); only `train_dialectic.py` needs the Unsloth/torch/transformers/peft venv on the GPU
host. Raw HTTP to OpenRouter and Anthropic is deliberate for the same reason — do not introduce SDKs.

`PLAN.md` is the single source of truth (goals, decision log, phase status, trained-model
inventory). `TRAIN.md` is the training runbook and failure log. Append to their logs rather than
rewriting them. `README.md` is the usage guide and must stay in step with the CLI.

Active work is on branch `first_impl`; `main` only has the initial commit.

## Commands

Run everything as `python3 <script>.py ...` from the repo root (scripts import each other by module
name: `llm_backend`, `trajectory`, `scoring`, `honcho_prompt`).

```bash
python3 verify_pipeline.py            # $0 self-test: static + unit + mock end-to-end. Must print GO before a commit.
python3 verify_pipeline.py --quick    # skip the mock end-to-end part

# stage 1  scenarios          (run = concurrent sync, both providers; submit/status/fetch = Anthropic batches)
python3 gen_contexts.py estimate|run|submit|status|fetch --n N --model <alias|anthropic:id|openrouter:id> --out data/contexts.jsonl
# stage 2  base model answers (Ollama, or OpenRouter when --model is an alias/vendor id) -> rejected
python3 gen_rejected.py --contexts data/contexts.jsonl --out data/rejected.jsonl [--base URL --model NAME|qwen9b --only id1,id2 --max-usd N]
# stage 3  teacher answers -> chosen
python3 gen_chosen.py estimate|run|submit|status|fetch --contexts data/contexts.jsonl --model opus --out data/chosen.jsonl
# stage 4  join + filters + persona split -> data/dataset_{train,eval}.{sft,dpo}.jsonl
python3 build_dataset.py --contexts ... --rejected ... --chosen ... --out data/dataset
# stage 5  (GPU host)
python3 train_dialectic.py --stage check|strip|sft|dpo|export ...
# eval
python3 eval_model.py --contexts ... --ids-from data/dataset_eval.dpo.jsonl --model <ollama-name> --out results/x.jsonl
python3 eval_model.py compare results/a.jsonl results/b.jsonl
python3 probe_toolcalls.py --model <ollama-name> --base http://host:11434
bash verify_all.sh                    # BASE= TUNED= BASELINE= env knobs
```

"Run a single test" means `verify_pipeline.py --quick`, or one script against the mock servers
(`mock_or_server.py <port>` is OpenAI-compatible and stands in for both OpenRouter and Ollama;
`mock_anth_batch.py <port>` serves Anthropic sync Messages and Message Batches).

## Secrets and endpoints

`keys.env` (git-ignored, template `keys.env.example`) holds `ANTHROPIC_API_KEY`,
`OPENROUTER_API_KEY` and optionally `OLLAMA_BASE` / `OLLAMA_MODEL`. `llm_backend.load_key` /
`setting` read the environment first, then `keys.env`. `OPENROUTER_BASE` and `ANTHROPIC_API_BASE`
point the backends at the mocks; `DIALECTIC_RESULTS_DIR` relocates batch manifests (tests). Never
print key values.

Generated data (`data/`, `results/`, `*_train.*.jsonl`, `*_eval.*.jsonl`, `*.gguf`) is
git-ignored. Only the 10-row `smoke10_sft.jsonl` / `smoke10_dpo.jsonl` are committed.

## Architecture: trajectory-shaped data

```
gen_contexts.py   contexts.jsonl  {id, category, domain, persona{name,bio}, question,
                                   observations[{date,text,relevant}], searches[{tool,query,results[idx]}],
                                   required_facts, forbidden_facts}
gen_rejected.py   base model answers ON THE HONCHO TRAJECTORY -> REJECTED (its real verbose style)
gen_chosen.py     teacher answers from the same tool results  -> CHOSEN  (terse, grounded)
build_dataset.py  join + filters + persona split -> SFT rows {messages:[system,user,assistant(tool_calls),tool,...,assistant], tools}
                                                 -> DPO rows {prompt:[same prefix], tools, chosen, rejected}
train_dialectic.py SFT -> DPO -> GGUF (loss on the final assistant turn only)
```

Invariants that span files:

- **One conversation builder.** `trajectory.build_messages(ctx)` is the only place that assembles
  the Honcho conversation (system prompt from `honcho_prompt.agent_system_prompt(name, name, None,
  None, TOOLS)`, the question, synthetic `search_*` tool calls and their results). Stages 2, 3, 4 and
  eval all go through it. Findings never go into the system prompt. Parity points with Honcho:
  `honcho_prompt.py` (verbatim copy, never hand-edit), `trajectory.TOOL_SCHEMAS`,
  `trajectory.format_tool_result`. If Honcho changes any of them, re-copy and regenerate all data.
- **One scorer.** `scoring.py` owns `HEDGE`, `REFUSAL`, `score_answer`, `aggregate`.
  `verify_pipeline.py` fails if those regexes are defined anywhere else.
- **One backend.** `llm_backend.py` owns the model/price table (`MODELS`), key loading, OpenRouter
  concurrency with retries, Anthropic sync + Message Batches, manifests, and resume-safe JSONL
  merging. Both generation scripts share its CLI shape (`estimate/run/submit/status/fetch`).
  `student_endpoint(model, base)` decides where the *student* runs (Ollama tag vs OpenRouter
  alias/id) for `gen_rejected.py` and `eval_model.py`; `trajectory.answer_with_ollama` is the one
  student caller and accepts any OpenAI-compatible endpoint plus an optional bearer key.
- **Rejected comes from the base model, not a teacher** (PLAN §3.1). Never replace stage 2 with
  teacher-written "verbose" answers.
- **Cost policy.** Estimates are always printed. `run --max-usd` is an opt-in live cap on the
  concurrent path; Anthropic batches never abort (owner's request).
- **Failure markers.** Contexts: `{"id", "__failed__": "__FAILED__: ..."}`. Answers: `answer =
  "__FAILED__: ..."`. `llm_backend.failed(row)` recognises both; every stage skips/retries them.
- **Category mix** is `gen_contexts.MIX`, interleaved by `category_sequence()` so small runs still
  cover every category.

## Training gotchas (history in TRAIN.md §7)

- `import unsloth` must precede any transformers/peft import; it is wrapped in try/except so
  `--stage check` and `verify_pipeline.py` can import the module without Unsloth.
- v0.8.0 (2026-09-10) fixed the bugs that made the 500/1000/2000-row models identical: the SFT
  stage trained on a hardcoded `samples[:7]`, the DPO learning rate was 5e-7 (full-fine-tune
  scale) on a LoRA adapter, and the DPO log-prob gathered token *t* against logits at position *t*
  (off by one). Do not reintroduce a fixed-size split; `verify_pipeline.py` checks for it.
- Rows longer than `--max-seq` are dropped, never truncated (truncation cuts the answer).
  `--stage check` reports lengths; trajectory rows are ~4–5k tokens, default `--max-seq 6144`.
- Adapters are merged with Unsloth's `save_pretrained_merged(..., "merged_16bit")` when available;
  `merge_and_unload` on a 4-bit base re-quantises and is only the fallback.
- `Qwen/Qwen3.5-9B` on the Hub is a VL checkpoint; use `Qwen/Qwen3-8B` or `--stage strip` first.
- DPO needs `remove_unused_columns=False`; `pad_collator` must return int64 tensors and pad labels
  with -100; GGUF export passes the tokenizer as the second positional and introspects the quant kwarg.
