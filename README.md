# honcho-dialectic-model

A concise dialectic (recall) model for Honcho, fine-tuned to be terse by default.
Mirrors the conventions of `kalaspuffar/honcho-deriver-model` (JSONL pipeline, `--count/--out`, Modelfile at root, sanity-check gates).

**Plan, findings, decisions, and status:** see [PLAN.md](PLAN.md) — that is the single source of truth for this project.

## Pipeline

| stage | script | does |
|---|---|---|
| 1 | `openrouter_trial.py gen` (Phase B) — or its `teacher_trial.py` equivalent | generates shared (persona, question, findings, rubric) contexts |
| 2 | `base_answer_probe.py` | runs **qwen3.5:9b on node7 Ollama** over each context → REJECTED style |
| 3 | `write_chosen.py --teacher <opus\|gemini>` | teacher writes the terse CHOSEN answer |
| 4 | `build_dataset.py` | filters (PLAN §4), persona-split 90/10, SFT + DPO JSONL |
| 5 | Unsloth QLoRA SFT → DPO → merge → GGUF | training runbook = deriver repo (TBD values after smoke test) |

`honcho_prompt.py` — verbatim copy of Honcho `src/dialectic/prompts.py` @ main, 2026-09-03 (runtime parity; re-copy on any Honcho upgrade before regenerating).

## Setup (trial / Phase B) — stdlib only, no pip installs

```
cp keys.env.example keys.env   # paste OPENROUTER_API_KEY only, never commit
bash run_trial.sh              # estimate -> gen -> run -> collect
```

`run_trial.sh` loads `keys.env` itself. Wallet caps: per-step pre-estimate abort
(`--max-usd`) plus a live running-dollar check inside `run` (default cap $5/step).
Results land in `results/openrouter/` and a tarball in `results/` for copying back.

## Key files

- `PLAN.md` — project doc, decision log, phase status
- `run_trial.sh` / `openrouter_trial.py` — Phase B teacher A/B via OpenRouter (wallet-safe)
- `mock_or_server.py` + `verify_pipeline.py` — self-tests (zero cost)
- `results/openrouter/` — Phase B outputs (contexts, per-arm answers, `summary.json`, `blind-review.csv`)
- `dataset_{train,eval}_{sft,dpo}.jsonl` — outputs of `build_dataset.py` (Phase C)
- `Modelfile` — Ollama wrapper for the final model
