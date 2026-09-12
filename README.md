# honcho-dialectic-model

Fine-tune a small Qwen model so Honcho's dialectic (memory-recall) answers are **terse by
default**: grounded, exact values, no preamble, no search narration. Synthetic data from a
teacher model, the base model's own verbose answers as the DPO "rejected" side, Unsloth QLoRA
SFT → DPO → GGUF → Ollama.

Plan, decisions and status live in [PLAN.md](PLAN.md); the training runbook and failure log in
[TRAIN.md](TRAIN.md). Read those before changing pipeline behaviour.

## Layout

| file | role |
|---|---|
| `gen_contexts.py` | **stage 1** — scenarios (peer, conclusions, searches, question, rubric) from any OpenRouter/Anthropic model |
| `gen_rejected.py` | **stage 2** — the base model (Ollama, or the same weights on OpenRouter) answers each scenario on the real Honcho trajectory → *rejected* |
| `gen_chosen.py` | **stage 3** — the teacher writes the ideal terse answer → *chosen* (any OpenRouter/Anthropic model) |
| `build_dataset.py` | **stage 4** — join, filter, persona split, emit SFT + DPO JSONL |
| `train_dialectic.py` | **stage 5** — `check` / `strip` / `sft` / `merge` / `dpo` / `export` (GPU host, Unsloth venv) |
| `eval_model.py` | score any Ollama model on held-out scenarios; `compare` two runs |
| `probe_toolcalls.py` | confirm the tuned model still emits valid tool calls |
| `verify_all.sh` | post-training A/B: probe + baseline vs tuned on the eval split |
| `SUMMARY_PLAN.md` | proposal for a Honcho *summary* model (to move to its own repo): measure-first gate, ledger-based chains, SFT-only recipe |
| `honcho_harness.py` | Phase D: questions through the REAL Honcho loop (`/v3/.../peers/{peer}/chat`), words/latency/flags per question, `compare` two runs; `harness_questions.example.json` is the template (5 original + 5 hard) |
| `llm_backend.py` | shared: model table, keys, OpenRouter concurrency, Anthropic Message Batches, JSONL helpers |
| `trajectory.py` | shared: builds the Honcho conversation (system → user → tool calls → tool results); tool schemas; Ollama answer loop |
| `scoring.py` | shared: the one scorer (coverage, fabrication, abstention, hedge) |
| `honcho_prompt.py` | verbatim copy of Honcho `src/dialectic/prompts.py` — re-copy on Honcho upgrades, then regenerate data |
| `verify_pipeline.py`, `mock_or_server.py`, `mock_anth_batch.py` | $0 offline self-test (unit checks + mock end-to-end run) |
| `list_models.py` | refresh OpenRouter ids/prices for `llm_backend.MODELS` |
| `smoke10_*.jsonl` | 10 committed rows in the current format, for training plumbing smoke tests |
| `Modelfile` | Ollama wrapper for the exported GGUF |

All data scripts are **stdlib-only** (no pip). Only `train_dialectic.py` needs the Unsloth venv.

## Why the trajectory format

Honcho's model never sees findings in its system prompt. At runtime it sees the Honcho system
prompt, the question, its own tool calls, the tool results, and *then* writes the answer. Every
stage here builds exactly that conversation through `trajectory.build_messages`, so chosen and
rejected share one prompt and training matches the runtime distribution. Loss is only computed
on the final assistant turn. Three parity points to re-check whenever Honcho changes:
`honcho_prompt.py`, `trajectory.TOOL_SCHEMAS`, `trajectory.format_tool_result`.

## Setup

```bash
cp keys.env.example keys.env      # ANTHROPIC_API_KEY, OPENROUTER_API_KEY, optional OLLAMA_BASE
python3 verify_pipeline.py        # no key, no network, no GPU — must print GO
```

Every script reads `keys.env` next to it (or the environment). Key values are never printed.

`--model` accepts an alias (`opus`, `sonnet`, `fable`, `haiku`, `deepseek`, `qwen3max`,
`gemini-pro`, `gpt5`, `grok46`, `llama70`), `anthropic:<id>` or `openrouter:<vendor/model>`.

Two execution modes on the generation scripts:

* `run` — concurrent synchronous requests (OpenRouter or Anthropic). Writes the output file as it
  goes, resume-safe (re-run the same command to retry failed rows). `--max-usd` stops submitting
  once the usage-based spend passes the cap; without it there is no cap.
* `submit` / `status` / `fetch` — Anthropic **Message Batches**: half price, asynchronous, usually
  done within an hour. The estimate is printed for reference and never aborts. Manifests live in
  `results/batches/<contexts|chosen>/`.

## Run the pipeline

```bash
# 0. what will it cost?
python3 gen_contexts.py estimate --n 3000 --model deepseek
python3 gen_chosen.py   estimate --contexts data/contexts.jsonl --model opus

# 1. scenarios (DeepSeek via OpenRouter, 8 in flight; or --model opus + submit/fetch)
python3 gen_contexts.py run --n 3000 --model deepseek --out data/contexts.jsonl --concurrency 8

# 2. the base model's own answers = rejected (Ollama host; slow — one request at a time by default)
python3 gen_rejected.py --contexts data/contexts.jsonl --out data/rejected.jsonl \
    --base http://node7.ea.org:11434/v1 --model qwen3.5:9b
#    Ollama host busy? Same weights on OpenRouter (qwen/qwen3.5-9b, ~$0.10/$0.15 per M tokens, ≈ $0.5 per
#    1 000 rows). Any OpenRouter alias or vendor/model id switches provider; the estimate is printed,
#    --max-usd is an opt-in live cap, resume works across providers because the output file is the same.
python3 gen_rejected.py --contexts data/contexts.jsonl --out data/rejected.jsonl \
    --model qwen9b --concurrency 8 --max-usd 3

# 3. teacher answers = chosen (Opus, batch = 50% price)
python3 gen_chosen.py submit --contexts data/contexts.jsonl --model opus --out data/chosen.jsonl
python3 gen_chosen.py status
python3 gen_chosen.py fetch                 # polls until ended, writes data/chosen.jsonl

# 4. filter + split (prints every drop reason and the per-category counts)
python3 build_dataset.py --contexts data/contexts.jsonl --rejected data/rejected.jsonl \
    --chosen data/chosen.jsonl --out data/dataset
#   -> data/dataset_{train,eval}.{sft,dpo}.jsonl
```

Sizing: a 3 000-scenario run costs roughly $5 for contexts on DeepSeek and $10–25 for Opus
batch answers; the estimate commands print current numbers. Expect 20–30 % of rows to be dropped
by the filters; generate accordingly.

## Train (GPU host)

```bash
# token lengths and the trainable tail of one row — no GPU; do this before every run
python3 train_dialectic.py --stage check --model /data/smoke/qwen35-9b-text --data data/dataset_train.sft.jsonl --max-seq 6144

python3 train_dialectic.py --stage sft --model /data/smoke/qwen35-9b-text --data data/dataset_train.sft.jsonl \
    --eval-data data/dataset_eval.sft.jsonl --out runs/v1-sft
python3 train_dialectic.py --stage dpo --sft runs/v1-sft/merged --data data/dataset_train.dpo.jsonl --out runs/v1-dpo
python3 train_dialectic.py --stage export --model runs/v1-dpo/merged --out runs/v1-gguf
# edit Modelfile FROM -> runs/v1-gguf/*.gguf, then:  ollama create dialectic-v1 -f Modelfile

# start DPO from a different SFT epoch than the one that got merged (checkpoints are kept per epoch):
python3 train_dialectic.py --stage merge --adapter runs/v1-sft/checkpoint-125 --out runs/v1-sft-ep1
python3 train_dialectic.py --stage dpo --sft runs/v1-sft-ep1/merged --data data/dataset_train.dpo.jsonl --out runs/v1-dpo-ep1
```

* 12 GB card: defaults (`--load-bits 4 --max-seq 6144`). 48 GB A6000: `--load-bits 16 --max-seq 8192`.
* Rows longer than `--max-seq` are **dropped, never truncated** (`check` tells you how many).
* **SFT trains the tool-call turns too** (`--tool-turns all`, default). Each trajectory yields one sample per
  `search_*` call (prefix so far → the call) plus the final answer, so the model learns to search when it has a
  question and no results. v0.8.0 trained the answer turn only and the 500-row model stopped calling tools
  (`probe_toolcalls.py` 0/5 vs 5/5 for the base) and fabricated instead. Expect ~3× the SFT samples per epoch;
  `--tool-turns first` trains only the opening call, `none` is the old behaviour. Watch
  `rows_with_extra_tool_calls` in the eval for over-searching.
* Defaults (v0.8.1): SFT 2 epochs at 2e-4, and with `--eval-data` the epoch with the **lowest eval loss** is the
  one merged (the 500-row run went .442 → .461 → .596 over 3 epochs). DPO 1 epoch at 3e-6, β 0.1, summed
  log-probs, and it **stops early** once the loss has stayed under `--dpo-stop-loss` (0.01) for
  `--dpo-stop-patience` (10) steps — past that point the margin only drifts. The step log shows `d_chosen` /
  `d_rejected` (log-ratio vs the reference per side): `d_chosen` going negative while the margin grows means
  the model is only pushing the verbose answer down, not the terse one up. Adapters merge to 16-bit.
* For the real Qwen3.5 9B text backbone run `--stage strip --model Qwen/Qwen3.5-9B --out /path/qwen35-9b-text` first
  and pass that directory as `--model`.

## Evaluate

```bash
# baseline vs tuned on the held-out personas + tool-call probe, one command
BASE=http://node7.ea.org:11434 TUNED=dialectic-v1 BASELINE=qwen3.5:9b bash verify_all.sh   # baseline scored with --answer-from-reasoning (qwen3.5:9b answers inside <think>) — TRAIN.md §7 2026-09-11

# or by hand
python3 eval_model.py --contexts data/contexts.jsonl --ids-from data/dataset_eval.dpo.jsonl --model qwen3.5:9b   --out results/eval_base.jsonl
python3 eval_model.py --contexts data/contexts.jsonl --ids-from data/dataset_eval.dpo.jsonl --model dialectic-v1 --out results/eval_v1.jsonl
python3 eval_model.py compare results/eval_base.jsonl results/eval_v1.jsonl
python3 probe_toolcalls.py --model dialectic-v1 --base http://node7.ea.org:11434
```

The final gate is the real Honcho loop (PLAN §6): point one `DIALECTIC_LEVELS__*` model at the
new Ollama name and run the verbosity harness.

## Offline testing

`python3 verify_pipeline.py` starts the two mock servers on free ports and drives every stage
(sync + batch, both providers, rejected, dataset, eval, training data prep with a fake tokenizer).
To poke at one script by hand:

```bash
python3 mock_or_server.py 9911 &        # OpenAI-compatible: OpenRouter and Ollama /v1 shapes
python3 mock_anth_batch.py 9977 &       # Anthropic sync Messages + Message Batches
OPENROUTER_BASE=http://127.0.0.1:9911/v1 OPENROUTER_API_KEY=mock python3 gen_contexts.py run --n 6 --model deepseek --out /tmp/t/contexts.jsonl
ANTHROPIC_API_BASE=http://127.0.0.1:9977 ANTHROPIC_API_KEY=mock python3 gen_chosen.py submit --contexts /tmp/t/contexts.jsonl --model opus
```

## Data formats

```
contexts.jsonl   {id, category, domain, persona:{name,bio}, question,
                  observations:[{date,text,relevant}], searches:[{tool,query,results:[idx]}],
                  required_facts:[..], forbidden_facts:[..]}          (failed: {id, "__failed__": "..."} )
rejected.jsonl   {id, category, answer, words, extra_calls, forced}
chosen.jsonl     {id, category, answer, words, teacher, score}        (failed: answer = "__FAILED__: ...")
*.sft.jsonl      {id, category, messages:[system,user,assistant(tool_calls),tool,...,assistant], tools:[..]}
*.dpo.jsonl      {id, category, prompt:[...same prefix...], tools:[..], chosen, rejected}
```
