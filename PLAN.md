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

### 3.2 Teacher generation — A/B (Phase B), via OpenRouter — **findings logged 2026-09-04**
- **Generator:** DeepSeek (`deepseek/deepseek-chat`) — proven 30/30 clean in the live run (round-4),
  ~$0.01/context. Round-3 failures (11/30) were the 400-token truncation, not the model.
- **Trial design (fair):** 30 shared contexts, each arm answered the same 30 with the same briefing
  prompt. Scored on: median words, rubric entity coverage (substring match against required_facts),
  fabrication (asserts a forbidden_facts value), abstention correctness, hedge rate.
  **Caveat:** the substring check overcounts false-positive "fab" — a *correct* abstention that
  mentions the missing topic (e.g. "no record of a four-day workweek") is flagged the same as a
  wrong one that asserts the absent fact. Read the raw arms in `results/openrouter/arms/` before
  trusting any single arm's fab number. `abstention_correct` for all arms is 1/3 or 2/3 in the data
  — see the leak analysis in PLAN for the real pattern.
- **Wallet:** estimate (no network), per-step `--max-usd` pre-abort + live running-$ cap inside
  `run`, `max_tokens` per call. Full 6-arm × 30-context trial ≈ $0.15–0.45 (round-4 actual: $2.11
  accumulated across rounds 2-4 including the $0.72 R2 misparse run and the $1.14 R3 gen-abort run;
  R4 the clean run was ~$0.40 for all 6 arms × 30 contexts + ~$0.02 for 30 DeepSeek contexts).
- **Accept rule (updated):** pick the arm with best *real* entity-coverage at median ≤ 40 words,
  zero *verifiable* fabrications (human-review pass on the top-5 fabrication-flagged rows),
  no format failures, and best cost/quality for the intended row count.
- **Status: findings below; teacher map locked for Phase C — see §3.5.**

### 3.5 Per-step teacher map (locked 2026-09-04, round-4 data + Daniel's decision)

**DECISION (Daniel, 2026-09-04):** correctness order Opus > Sonnet > Qwen3-Max; one-time
generation, so pay for best = **Opus 5 for the full answer set**. Route: **Anthropic direct
Message Batches API** (Daniel's Claude account — same 50% batch price as OpenRouter `:batch`,
no third-party key, no new wallet). Pilot first on the existing 30 contexts, then full set.

| role | model | path | pricing |
|---|---|---|---|
| Context generator | DeepSeek | OpenRouter sync (`gen`) | $0.32 / $0.89 per M — proven 30/30 clean |
| **Teacher (ALL rows)** | **Opus 5** | **Anthropic direct batch** (`batch-run`/`batch-fetch`) | **$2.50 / $12.50 per M** (batch 50%), +prompt-cache on shared briefing (10% of input on cached rows) |
| Fallback / cheaper bulk | Sonnet 5 | Anthropic batch | $1.00 / $5.00 per M |
| optional top-tier slice | Fable 5 | Anthropic batch | $5.00 / $25.00 per M |
| — dropped — | GPT-5 | — | 64% null-content + 6.4× actual cost |
| — dropped — | Gemini 3.1 Pro | — | 0.49 coverage + reasoning-text leak |

**Cost model (measured from the 30 real contexts, ~550 tok in / ~50 tok out per row):**
- 30-row Opus pilot: **$0.05–0.13**
- 500 rows: ~$1.10–1.60
- 2,000 rows: ~$4.10–6.50 (plus ~$0.65–2.60 for contexts via DeepSeek)
- Cache note: Anthropic cache *writes* cost 1.25× input once per 5-min window; at batch
  concurrency cache hits are best-effort. Budget with the **no-cache** column, treat hits as upside.

**Batch API facts (from Anthropic docs, 2026-09-04):**
- `POST /v1/messages/batches` with `requests:[{custom_id, params}]`; cap 100k requests or 256MB.
- 24h processing window (most < 1h); results downloadable 29 days; **download before then**.
- No `temperature`/`top_k`/`top_p` on post-Opus-4.6 models (400) — batch code omits them.
- Result rows: `succeeded` (billed) / `errored`+`canceled`+`expired` (NOT billed).
- One batch cannot be modified; cancel + resubmit if prompt needs fixing.

**Not used further:** Gemini 3.1 Pro (0.492 coverage + 1/30 code-fence response), GPT-5 (64% nulls).

**Round-4 arms table (real data, 30 contexts each):**
- opus:       coverage 0.859, median 38w, contradiction 3/3 — **longest, priciest**
- sonnet:     coverage 0.794, median 29w, contradiction 3/3 — **best short+grounded+clean+cheap**
- qwen3max:   coverage 0.735, median 27w, $0.04/1k rows — cheapest + **same family as 9B base**
- deepseek:   coverage 0.723, median **19w (shortest, cleanest)**
- gemini-pro: coverage 0.492 + **reasoning-leak** ("Let's refine Attempt 1…" leaked into the answer)
- gpt5:       19/30 null-content (64% fail) — **actual cost 6.4× its estimate** (reasoning tokens)

**CORRECTIONS to my round-4 auto-scores (read the raw arms, not the metric):**
- The "fabrication" counts were **false positives** — every flagged hit is a correct
  negation/reference to a distractor ("avoiding cryptocurrencies", "switched from oak",
  "they do not include Luigi", "over fine dining"), not an assertion of the wrong fact.
  **Real fabrication ≈ 0 across all arms.** Don't use the raw `fabrication_rows` number.
- `abstention_correct` was **under-counted** — my check flagged correct abstentions for
  *naming the topic* ("no record of a four-day workweek"). All 6 arms actually abstained
  correctly 3/3; that row should read 3/3, not 1/3.
- Gemini's 0.492 is real (it under-covered + leaked its own reasoning text).
- GPT-5's nulls are real (64%) and cost 6.4× the estimate — it burns reasoning tokens that
  OpenRouter charges even on null output. This is why my "$0.15–0.45" estimate was wrong.

**Open (before Phase C):**
- Human-review pass on the top-5 fabrication-flagged rows per arm to separate true fab from
  false-positive substrings, so the fab count is trustworthy before the "zero fabrication" gate.
- If the hard-category slice (contradiction/supersession) shows Opus clearly pulling ahead of
  Sonnet on ≥2/3 of those rows, lock Opus for those rows. If not, Sonnet for all (simplest,
  ~40% cheaper).

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

- **2026-09-10** Stage 2 (rejected) can run the student on **OpenRouter** (`qwen/qwen3.5-9b`,
  alias `qwen9b`, $0.10/$0.15 per M) when the Ollama host is busy. Same weights, same trajectory,
  same output file, so runs resume across providers. Estimate printed, `--max-usd` opt-in cap, spend
  tracked from usage. `eval_model.py` accepts the same model spec. Caveat: a different serving stack
  (quantisation, sampler, reasoning budget) is a mild distribution shift vs the Ollama deployment;
  keep one provider per dataset where possible and note the provider in the run log.

- **2026-09-11 — first real 500-row run (v0.8.0): overfit SFT, saturated DPO (Claude review of
  Daniel's logs).** SFT eval loss .442 / .461 / .596 over 3 epochs (train .40 → .07): the last,
  worst checkpoint was merged. DPO at 1e-5 hit zero loss by step 25/126 and then drifted the margin
  from 3 to 25. Category mix of the 500-row slice was fine. v0.8.1: SFT merges the best epoch by eval
  loss (default 2 epochs), DPO 3e-6 × 1 epoch for 500 rows (rate scales with 1/steps: ~7e-7 at 3000 rows, so
  Daniel's point that 5e-7 fits the full set stands; the v0.7 5e-7 result was confounded by the
  off-by-one loss) with early stop on saturation and per-side log-ratio logging, `--stage merge` to reuse an earlier SFT checkpoint. Daniel's decision: evaluate this model,
  retrain the same 500 rows with the new defaults, and bring in no more data until a 500-row model
  shows a gap that more rows could plausibly widen. Details TRAIN.md §10.

- **2026-09-11 — `dialectic_500` (v0.8.0) is unusable: it stopped calling tools.** Probe 0/5 vs base
  5/5; without results in context it fabricates facts about the peer. The synthesis-turn eval looked
  excellent (coverage .924, 0 fabrication, 5/5 abstention, median 32 words) precisely because eval
  rows always contain the tool results. Root cause: SFT loss on the final turn only. v0.8.1 trains
  every `tool_calls` turn of the trajectory as well (`--tool-turns all`, default). Gate order from
  now on: `probe_toolcalls.py` ≥ 90 % first, eval second. Base eval column on node7 Ollama was
  invalid (31/50 empty answers) — open item. Details TRAIN.md §11.

## 11. Trained model inventory (Daniel, 2026-09-09)

| Ollama name | records trained | note |
|---|---|---|
| `dialectic_1` | **500** | |
| `dialectic_2` | **2000** | names are NOT in record-count order |
| `dialectic_3` | **1000** | |
| `dialectic-qwen3.5-9b` | 25 | process-verification run |
| `dialectic_500` | **500** (v0.8.0, 2026-09-11) | first run that actually learned the data; overfit SFT (3 ep) + saturated DPO (1e-5); synthesis eval coverage .924 / 0 fabrication / 5/5 abstention on 50 held-out rows, but **probe 0/5 — no tool calls, fabricates without context. Not deployable.** |

Validation (2026-09-09): the three are behaviorally indistinguishable — head-to-head 10 wins/9 losses each across 152 common rows; all ~0.68–0.70 coverage vs base 0.663; all ~1.5× more terse than base; all serve 16k/32k context fine via `num_ctx` override (no retrain needed). Full write-up + open questions (why 500≈2000: method vs data-prep vs premise) in vault `active-projects/Honcho-Dialectic-Verbosity-Report.md` §11.

## 12. Immediate next actions (owner: Daniel unless noted)

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

- **2026-09-10 — why dialectic_1/2/3 were identical (Claude review, Daniel's question).** Not the
  premise. `train_dialectic.py` (since v0.5.1) split SFT as `samples[:7]` train / rest eval whenever
  a file had >8 rows — every run SFT-trained on 7 rows. DPO then ran at lr 5e-7 (a full-fine-tune
  value; LoRA needs ~1e-5), with per-token length-normalised log-probs (removes the length signal)
  and an off-by-one in the log-prob gather. Net: the three models = base + 7-row SFT. The 500 ≈
  1000 ≈ 2000 result carries no information about data volume. **Decisions (Daniel):**
  (a) fix training (v0.8.0), (b) move all rows to the **tool-trajectory format** — findings arrive as
  tool results after synthetic `search_*` calls, exactly as in Honcho's loop, never in the system
  prompt (closes R2), (c) contexts become peer-style conclusions about a named persona instead of
  encyclopedia facts, (d) reorganise: `gen_contexts.py` / `gen_chosen.py` (any OpenRouter or
  Anthropic model, `run` concurrent or Anthropic `submit/fetch` batches), `gen_rejected.py`,
  `build_dataset.py`, `train_dialectic.py`, `eval_model.py`; shared `llm_backend.py`,
  `trajectory.py`, `scoring.py`; `openrouter_trial.py`, `write_chosen_batch.py`, `run_rejected.py`,
  `eval_dialectic.py`, `rescore_abstention.py`, `estimate_cost.py`, `teacher_trial.py`,
  `write_chosen.py`, `base_answer_probe.py`, `run_trial.sh` removed. Also fixed: the build_dataset
  fabrication gate was unreachable; the persona split ignored its seed. §4/§5 above describe the
  old pipeline; README.md is now the usage reference.

## 13. Next actions after the 2026-09-10 rewrite

1. Regenerate data in the trajectory format (old `results/openrouter/contexts.jsonl` rows are the
   old schema and are not reused): `gen_contexts.py run --n 3000 --model deepseek`, then
   `gen_rejected.py`, then `gen_chosen.py submit/fetch --model opus`, then `build_dataset.py`.
2. `train_dialectic.py --stage check` on the emitted SFT file to pick `--max-seq`.
3. Train ONE model (500 rows is enough for the first signal) with v0.8.0 defaults, run
   `verify_all.sh` against the base on the eval split. Only if that shows a clear gap, sweep sizes.
   *2026-09-11:* first run done (overfit SFT + saturated DPO, TRAIN.md §10); evaluate it, then
   retrain the same 500 rows with v0.8.1 defaults before any size sweep.
4. Then the real gate: Phase D through Honcho's loop (§6).
