# Honcho Dialectic Brevity Model — Project Plan (single source of truth)

**Status:** ACTIVE — Phase A (data pipeline) ready, Phase B (teacher A/B) blocked on API keys
**Owner:** Daniel (train/verify) + Mya (build/scaffold/scripts)
**Started:** 2026-09-03
**Supersedes:** the Tier-0/Tier-1/Tier-3 plan in `active-projects/Honcho-Dialectic-Verbosity-Report.md` — that report's *benchmark data and code findings* remain valid; its "fix it with config" conclusion is **abandoned** (see Decision Log).
**Companion repo (this dir):** `~/honcho-dialectic-model/` — training data + Modelfile, mirrors `kalaspuffar/honcho-deriver-model` conventions.
**Predecessor work:** `kalaspuffar/honcho-deriver-model` (deriver-qwen3, synthetic data via Opus, Unsloth, GGUF→Ollama) — proven pipeline we are copying.

---

## 1. Goal

A fine-tuned dialectic model (base **qwen3.5:9b**, same family as production `low` level) that answers Honcho dialectic queries **terse by default** — grounded, exact values, no preamble, no search narration, no restate-and-elaborate.

**Success:** median ≤ 150 words on the 5-question harness (baseline 304, v3.1.1, same session), fact retention ≥ 90% of baseline, and *no* regression on enumeration/contradiction/abstention behaviour. Target deployment: one model across all dialectic levels (see §3.3 for the trade-off we deliberately need to test).

## 2. Why we're here (established facts, 2026-09-03)

1. **Baseline (measured, `honcho_verbosity_results-20260903-191208.json`):** level `low` = qwen3.5:9b, max_tool_iterations 5 → per-question 168 / 267 / 306 / 416 / 304 words (**median 304**), latency 54–145 s.
2. **Bigger model ≠ better:** level `medium` = qwen3.8:27b, max_tool_iterations 2 → median **326**, max 448 (i.e. *more* verbose, plus 50–125 s/query, one timeout). Model size is not the verbosity lever.
3. **Smaller model is not a lever either:** `minimal` = qwen3.5:4b returned **empty answers** on all 5 harness questions (budget of 1 or parsing). Downgrading is off the table.
4. **Config knobs are a red herring for our median:** answers never near the 8192-token ceiling (max observed ~450 tokens); a max-token cap trims the tail only. `MAX_TOOL_ITERATIONS` is a loop *ceiling*, not a quota (loop exits early when the model stops calling tools; per-level defaults are 1/2/4/5/10 in `src/config.py`) — reducing it cuts latency, not median length, and *hurts* enumeration recall.
5. **Root cause (accepted):** the verbose 250–450-word band is the Qwen family's default synthesis style under Honcho's thoroughness-driven system prompt (`src/dialectic/prompts.py` mandates multi-search, verification, cross-referencing — and ends with "be as specific as possible", which 9B/27B both interpret as "say more"). Honcho team (in Daniel's mail thread) independently confirms: known local-model tendency, fix = post-training. **Decision: train; do not patch their code (Daniel — upgrade/maintenance nightmare).**

## 3. Decisions

### 3.1 Algorithm — DPO (paired preferred/short vs rejected/verbose), with SFT warm-up
- Daniel's documented position (Zhao et al., arXiv:2505.12843): DPO over GRPO; naive length penalties reward-hack against correctness.
- **Rejected side comes from our own base model** (rejection sampling): run qwen3.5:9b on node7 Ollama on each context → its natural verbose answer is the *rejected*. This beats a teacher-fabricated "verbose" because it is the true distribution we're fighting.
- **Chosen side** = teacher-written ideal terse answer (teacher TBD by A/B, Phase B).
- **Recommended schedule:** ~500-row SFT warm-up (task format: findings→answer) → DPO on ~1–2k pairs. If plain SFT on chosen reaches target it's also acceptable (cheaper eval); DPO is the default per 3.1.
- VRAM fit: QLoRA 9B on the A6000 48GB fits with room; reuse the deriver runbook for Unsloth notebook params (r=16, bf16, 4-bit base).

### 3.2 Teacher generation — A/B (Phase B), via OpenRouter
- **Access (2026-09-04, supersedes earlier per-vendor plan):** one `OPENROUTER_API_KEY` in `keys.env`. Candidate arms already priced in `openrouter_trial.py` (Opus 4.8/5, Sonnet, Gemini 3.1 Pro / 2.5 Pro, GPT-5, o4-mini, DeepSeek, Qwen3-Max, Grok, Llama 70B). Live catalog check: `fetch_openrouter_models.py`.
- **Trial design (fair):** one fixed set of **30 contexts** (generated once, teacher-independent). All arms answer the *same* 30 with the *same* briefing prompt and word budget. Scores: median word count, rubric entity-coverage, fabrication check (claim not present in findings), abstention correctness (3/30 contexts have **no** answer → correct = clean abstain), hedge-rate. `blind-review.csv` export for Daniel's blind 1–5 review.
- **Wallet safety:** `estimate` (no network), `--max-usd` hard cap with pre-abort, `max_tokens` per call, no retry loops, `collect` for transfer. Mock-verified end-to-end 2026-09-04.
- **Accept rule:** best entity-coverage at median ≤ 120 words AND zero fabrication passes; tiebreak on cost. Full trial ≈ $0.15–0.45.
- Status: **ready, BLOCKED on real run** — v0.2.0 fixes (truncation + null guard) shipped; Daniel runs `bash run_trial.sh` and brings back the results tarball.

### 3.3 One model for all levels (minimal/low/medium/high/max)
- Rationale (Daniel's + our data): levels differ mainly in tool-iteration budget and prompt loadout, **not** in answer quality we need — the 27B is *more* verbose, not better. One fine-tune, one `.env` swap (set all five `DIALECTIC_LEVELS__*` model lines to `dialectic-qwen3.5-9b`), zero per-level retraining.
- **Risk we must test, not assume:** `high`/`max` queries are harder (multi-hop, enumeration, cross-contradiction). The 9B may regress reasoning there. Mitigation: the eval harness (§7) includes those hard categories; if 27B-beats-9B gap on *correctness* (not verbosity) > 10 points on hard questions, keep 27B for medium/high/max and ship the 9b only for minimal/low.

### 3.4 Base model — qwen3.5:9b with documented fallback
- Production `low` already runs qwen3.5:9b; fine-tuning it maximises continuity.
- **Known risk (Daniel):** qwen3.5 hit a LoRA-format issue in the deriver run (we fell back to Qwen3-9B/GGUF then). **Checkpoint 1 of Phase C (training):** run a 10-row Unsloth SFT smoke test on qwen3.5:9b *before* generating full volume. Fallback base: Qwen3-9B (proven in deriver pipeline).

## 4. Data pipeline (3 stages)

```
stage1  generate_contexts.py      teacher (Opus/fixed): (persona, question, findings-pool, rubric) × N
stage2  base_answer_probe.py      local: qwen3.5:9b on node7 Ollama answers each context  → REJECTED (its natural verbose style)
stage3  write_chosen.py           teacher (A/B winner): ideal terse answer per context   → CHOSEN
stage4  build_dataset.py          join + filters + 90/10 split + SFT/DPO formats          → train.jsonl / dpo.jsonl
```
All in `~/honcho-dialectic-model/`. Conventions copied from the deriver repo (JSONL, `--count --out`, `extract_json` tolerant parser, `sanity_check.py`-style gates, Modelfile at root).

**Context categories (fixed mix per 50-batch):**
- 20% factual recall (single fact, exact value)
- 15% preferences/style
- 20% enumeration ("all X", "how many Y")
- 10% contradiction (two conflicting values in findings → chosen must name **both**)
- 10% supersession (old + updated value → chosen must give the **latest**)
- 10% abstention (nothing in findings → clean "no information", no hedge)
- 15% summary/pattern over time

Each context embeds: 4–8 relevant findings + 4–6 plausible distractors (same domain, wrong value) so "pick the right one" is non-trivial, plus a `required_facts` rubric + `forbidden_facts` (the distractor values) for automated scoring.

**Filters (sanity_check.py, enforced in build_dataset.py):**
- chosen ≤ 120 words; rejected/chosen length ratio ≥ 1.5 (drop pairs where the base model was already terse — no signal)
- ≥ 1 required_fact entity present in chosen; **zero** forbidden_fact values in chosen
- no hedge words in chosen (likely/probably/might/seem)
- abstention rows: chosen must not contain any entity from the findings pool
- dedupe on (question, persona); 90/10 split on persona not on question (no question leakage between train/eval)

**Volumes (targets, Phase C):** 50 (A/B sample) → 1,200 contexts → ~1,000 pairs after filters (drop-rate budget 20%).

## 5. Training (Phase C)

1. Unsloth QLoRA SFT warm-up: 500 rows, base qwen3.5:9b (or Qwen3-9b fallback per §3.4), r=16, 4-bit, ~3 epochs, early-stop on eval loss.
2. DPO: dpo_beta 0.1, lr 5e-7, 1–2 epochs on pairs. Early-stop monitoring: eval DPO reward margin + harness median words + fact-retention. (params mirror the deriver run; exact notebook values filled in during smoke test)
3. Merge LoRA → quantize GGUF (Q4_K_M) → `ollama create` from `Modelfile` → name `dialectic-qwen3.5-9b`.
4. **Modelfile (root, already written):** `FROM qwen3.5-9b` / `temperature 0.1` / `num_ctx 8192` — adjust FROM tag after smoke test confirms the exact base id Unsloth needs.

## 6. Deployment A/B (Phase D)

- In `.env.honcho`: temporarily `DIALECTIC_LEVELS__low__MODEL_CONFIG__MODEL=dialectic-qwen3.5-9b`, restart Honcho.
- Run `honcho_verbosity_test.py` (vault, active-projects/honcho-verbosity/) → new JSON row in the baseline trail.
- Pass gates: median ≤ 150 words (baseline 304); fact retention ≥ 90% (rubric); no new abstention failures; latency ≤ 90 s median (5 iters × shorter answers).
- If pass → set **all five** level lines to the new model (§3.3), run the hard-question harness at `high` to confirm no reasoning regression, keep the 27B only where 3.3 risk materialises.
- Rollback = one env line + restart. Previous model keeps living in Ollama until the trial week passes.

## 7. Evaluation harness (unchanged, proven)

`honcho_verbosity_test.py` (5 questions, any level, prints words/latency, saves JSON) + the extended hard-question set (enumeration ×2, contradiction ×1, abstention ×1, supersession ×1) — **TODO Mya:** add the extended set to the harness before Phase D (small script edit).

## 8. Phases & status

| Phase | Scope | Owner | Status |
|---|---|---|---|
| A | Pipeline scripts + prompt parity file (`honcho_prompt.py` = verbatim `agent_system_prompt`) + repo skeleton + Modelfile + README | Mya | **DONE** 2026-09-03 (scripts untested — need keys/base-model probe) |
| B | Teacher A/B trial: run `run_trial.sh` on Daniel's box (OpenRouter) → copy `results/openrouter_*.tar.gz` back → I score + log teacher choice in §3.2/§10 | Daniel (run, blind-review 10) + Mya (fold results) | **READY** (wallet-safe: estimate/cap/collect built & mock-verified); BLOCKED on `OPENROUTER_API_KEY` |
| C | Smoke-test base LoRA → generate 1.2k contexts → stage2/3 → build_dataset → SFT → DPO → GGUF | Daniel (GPU) + Mya (scripts/monitor) | pending, needs Phase B winner |
| D | Deploy to `low`, harness A/B, hard-set at `high`, all-level swap or split | Daniel | pending |

## 9. Risks & open questions

- **R1 Prompt drift:** if Honcho ships a dialectic prompt change, the trained style is tied to the *current* prompt text (baked into data as system message). Mitigation: `honcho_prompt.py` records the exact version hash (`git rev` of plastic-labs/honcho@main at 2026-09-03); re-run stage1–4 (data, free with the same teachers) on any upstream prompt change. **Action for Daniel:** file the "brevity mode knob" feature request with the Honcho team regardless — a knob they ship makes us prompt-drift-immune and the training effort becomes just the dataset.
- **R2 Tool-protocol drift:** v1 (SFT/DPO on *synthesis* rows only, not tool-call rows) does not teach the model its loop behaviour — Honcho's harness drives tool calls around it. Residual risk that a fine-tuned 9B degrades tool-calling on `low`. Mitigation: the §6 harness runs through the real Honcho loop, so we see it live; if tool-call behaviour regresses, escalate to Option-B data (full multi-turn trajectories) — expensive, last resort.
- **R3 LoRA format on qwen3.5:** handled by §3.4 checkpoint (10-row smoke test first).
- **R4 Fabrication:** DPO can trade correctness for brevity (the exact failure the ACL paper warns about). Mitigation: forbidden_fact filter + retention gate in §6 + abstention category in rubric.
- **OQ1:** ~~Gemini key~~ → superseded: OpenRouter key now covers Google models (gemini-3.1-pro-preview etc.).
- **OQ2:** teacher cost — now measured: 30-context trial $0.15–0.45, full-volume $1–9 (§9.5). Pick by quality, cost range is small.

## 9.5 Teacher access — OpenRouter (added 2026-09-04)

Daniel has no Gemini key and wants wallet-safe testing on his own machine. OpenRouter replaces
per-vendor keys: **one `OPENROUTER_API_KEY`** covers context generation + every answer arm, and
gives a broader candidate set (Claude Opus/Sonnet, GPT-5/o4-mini, Gemini Pro, DeepSeek, Qwen3-Max,
Grok 4.6, Llama 3.3 70B). Live IDs + $/M verified from OpenRouter's public `/models` endpoint on
2026-09-03 (`fetch_openrouter_models.py`, arm table in `openrouter_trial.py`).

**Wallet safety (verified end-to-end against a local mock API):**
- `estimate` subcommand: pure-local cost projection, no network, no spend
- `run` pre-estimates and hard-aborts unless `estimate ≤ --max-usd` (default $5) or `--yes`
- Every call: `max_tokens` ceiling, 120 s timeout, no auto-retry loops
- `collect` tars `results/openrouter/` for copy-over

**Cost reality:** the 30-context trial ≈ **$0.15–0.40 total**; full-volume 1,200-row chosen-writes
≈ **$1–9** (DeepSeek ~$0.35; qwen3-max ~$3; Opus ~$9). So the whole teacher phase is small;
the expensive thing is still our GPU time + curation.

**Files (in `~/honcho-dialectic-model/`):**
- `openrouter_trial.py` — self-contained (stdlib only: `estimate/gen/run/collect`)
- `run_trial.sh` — Daniel's one-shot runner (estimate → gen → run → collect)
- `fetch_openrouter_models.py` — refresh the candidate table from the live catalog
- `mock_or_server.py` — local fake OpenRouter for testing the trial script with $0 (used for this verification)

**Handshake:** Daniel sets `OPENROUTER_API_KEY` in `keys.env`, runs `bash run_trial.sh` on his
machine, copies `results/openrouter_YYYYMMDD-HHMMSS.tar.gz` (or the `results/openrouter/` folder)
back to this host → I fold `summary.json` + your blind-review picks into §3.2 and lock the teacher.

## 10. Decision log

- **2026-09-03** Baseline captured: 304 median (low/9b), 326 median (medium/27b), minimal/4b empty. → Config-knob plan abandoned as primary fix; training plan adopted. (report v2 → superseded by this doc)
- **2026-09-03** `MAX_TOOL_ITERATIONS` investigated: ceiling-not-quota, per-level defaults, not our verbosity driver. → Removed from fix plan, kept as known latency lever.
- **2026-09-03** Daniel: do not patch Honcho's code (maintenance risk). → Tier-1 prompt-patch plan dropped; training + optional upstream feature request only.
- **2026-09-03** Daniel: one 9b-based dialectic model across all levels is acceptable. → §3.3 adopted with hard-question regression gate.
- **2026-09-03** Daniel: synthetic-data pipeline mirroring the deriver repo is the way. → §4 adopted.
- **2026-09-03** Daniel: not a one-shot task → this document is the living project doc (§8 status, §10 log).
- **2026-09-04** OpenRouter adopted as the teacher-API gateway (one key, wider model set). Trial script
  `openrouter_trial.py` built and **verified end-to-end on a mock API** (gen/run/score/CSV/cap/collect all pass).
- **2026-09-04 (round 3, after $1.14 live attempt — root cause found via raw-dumps)** Daniel's live run:
  30 contexts gen'd, 11 failed, arms crashed. `results/openrouter/raw-dumps/` showed the truth:
  **context-gen was truncated at `completion_tokens: 400`** (= old `MAX_TOKENS_OUT`) — the JSON never
  closed, so every "missing required keys" was a *truncation*, not a format quirk. Second crash:
  one arm returned `content: null` → `rec["answer"] = None` → `score()` died on `None.split()`.
  Fixes shipped + verified: (a) `CTX_MAX_TOKENS=4096` for gen, answers `MAX_TOKENS_OUT` 400→800,
  (b) null-content guard in `run` (recorded as `ERR` + error field, scored 0 coverage — no crash),
  (c) `score()` robust to empty answers, (d) `cmd_gen` pricing lookup fixed (was KeyError-prone
  on context-model ID), (e) test-only mock arms `zerou`/`nullc` + mock null/zero-usage modes,
  (f) E2E mock re-run: null-content arm now `ERR 0w` instead of process death.
  **Open:** evaluate which model is best *for context generation* (DeepSeek worked but cheap —
  trial arms to compare gen quality next) and lock per-step model assignments before full volume.
  Committed + tagged `v0.2.0`.

## 11. Immediate next actions (owner: Daniel unless noted)

1. **(done, Daniel)** `keys.env` has an `OPENROUTER_API_KEY` (used in first live run 2026-09-04).
2. **Daniel** Re-run on the same box as before:
   ```
   cd honcho-dialectic-model
   bash run_trial.sh          # contexts.jsonl from the failed run is overwritten (all rows were __failed_ctx)
   ```
   DeepSeek is now the context generator (~$0.02 for 30). If any context STILL fails parse, its raw
   response is dumped to `results/openrouter/raw-dumps/` — copy that folder over too, it tells us exactly
   what the model is sending. Wallet: cap $5/step, abort-before-spend + live running-dollar check.
3. **Daniel** Look at `results/openrouter/blind-review.csv`; write your 1–5 picks per row.
4. **Mya (on Daniel's call)** Fold Daniel's picks + `summary.json` into §3.2 and §10, lock the teacher,
   and move Phase C to ready (smoke-test base LoRA, then full generation).
5. **(Mya on request)** Extend `honcho_verbosity_test.py` with the hard-question set (Phase D gate).
6. **(Daniel, optional — the one I most want)** File the brevity-knob feature request with the Honcho
   team; even a plain `ANSWER_MAX_WORDS` would make the fine-tune a safety net rather than our only lever.
   Text draft lives in the vault's `active-projects/Honcho-Dialectic-Verbosity-Report.md` (old Tier-1).
