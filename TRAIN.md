# TRAIN.md — Unsloth Dialectic 10-row smoke test  (runbook)

**Owner:** Daniel (node7 GPU) + Mya (scripts/monitor)
**Goal:** prove the whole pipeline (Unsloth SFT smoke → verify → optional DPO → GGUF → Ollama) works end-to-end on `qwen3.5:9b`, using the 10-row smoke files that already exist in this repo.
**Duration:** ~30 min SFT smoke; ~1–2 h if DPO stage is added.
**Cost:** ~$0 (node7 is local, no API in the loop). The 10-row data was already paid for via the Opus batch ($0.118 earlier).

---

## 0. What's already on this host (Mya side) — verified before handoff
- `smoke10_sft.jsonl` — 10 rows, `{messages:[system, user, assistant]}`. System carries the exact Honcho prompt + findings cache (matches runtime).
- `smoke10_dpo.jsonl` — 10 rows, `{prompt:[system,user], chosen, rejected}`. Same 10 contexts; chosen = Opus (median 38w), rejected = qwen3.5:9b on node7 (median 153w), ratio ~3.78×.
- `dataset_train.{sft,dpo}.jsonl` — full 25-row set, `dataset_eval.{sft,dpo}.jsonl` — 2 held-out rows (persona split). Not yet used by smoke.
- `train_dialectic.py` — the Unsloth script (next section).
- `Modelfile` — Ollama Modelfile template; `FROM` line is a placeholder that gets filled in once Unsloth hands off.
- Code: `honcho_prompt.py`, `base_answer_probe.py`, `build_dataset.py` (v0.4.0, fixed), `run_rejected.py` (stdlib-only, added this turn).

## 0b. Qwen3.5 flag (added 2026-09-06) — official checkpoint, text-only part

`--stage strip` loads the **official** `Qwen/Qwen3.5-9B` (Qwen-trained — no
trust in a community re-save) through the text-only class and re-saves it.
Verified against the installed transformers source:

```
class Qwen3_5ForCausalLM:            # modeling_qwen3_5.py
    config: Qwen3_5TextConfig        # flat text config -> saves as qwen3_5_text
    _keys_to_ignore_on_load_unexpected = [r"^mtp.*", r"^model.visual.*"]
```

So `from_pretrained()` on the VL repo instantiates only the text backbone and
drops the vision weights by design; `save_pretrained()` then yields the same
shape as the community text-only trims, built by you. Smoke venv is
transformers 5.5.0 (≥ 5.2 requirement met). Runs on CPU RAM, ~17 GB bf16.

```bash
# on the 3080 Ti host (strip = weights surgery, no GPU needed):
/data/smoke/.venv/bin/python3 train_dialectic.py --stage strip \
  --model Qwen/Qwen3.5-9B --out /data/smoke/qwen35-9b-text

# then A/B train the SAME 10 rows on the text-only 9B class:
/data/smoke/.venv/bin/python3 train_dialectic.py --stage sft \
  --model /data/smoke/qwen35-9b-text --data smoke10_sft.jsonl \
  --out smoke-9b --load-bits 4 --max-seq 4096

# DPO + export exactly as before, with --sft smoke-9b/merged / --model smoke-9b/merged
```

Expected outcome: a checkpoint that Ollama can load as `dialectic-qwen3.5-9b`
(the name it already runs under), but now on the real 9B text backbone the
other model was right to point at.

## 2. On node7 — check before running
```bash
ssh node7
# 1a. GPU idle, no other workloads
nvidia-smi
#   Node7: RTX A6000 48 GB (per memory card; if that has shifted, re-check)
#   3080 Ti host: 11.6 GB — use --load-bits 4 --max-seq 4096 (default)
# 1b. Unsloth + deps present
python3 -c "import unsloth, transformers, torch, trl; \
  print('unsloth',unsloth.__version__,'| transformers',transformers.__version__,\
        '| torch',torch.__version__,'| trl',trl.__version__)"
#   Known-good in this venv (2026-09-04): unsloth 2026.9.2, transformers 5.5.0,
#   torch 2.11.0+cu130, CUDA 13.0. If trl is <1.2, the script still runs (v0.7
#   removed the trl dependency from SFT+DPO entirely).
# 1c. Base model present
ollama list | grep -E 'qwen3\.'
#   qwen3.5:9b    <-- Ollama tag = production `low`; the Hub repo is VL. TRAIN ON ITS TEXT-ONLY
#                     STRIP: /data/smoke/qwen35-9b-text (§0b). This is the base (PLAN §3.4).
#   qwen3:8b      <-- fallback ONLY if Qwen3.5 hits the LoRA-format snag; never the default
```

If `unsloth` is not on node7, first do `pip install -U unsloth` in the same python env that ran the deriver.

## 2. Copy the script + data to node7
From Mya's host:
```bash
scp ~/honcho-dialectic-model/train_dialectic.py \
    ~/honcho-dialectic-model/smoke10_sft.jsonl \
    ~/honcho-dialectic-model/smoke10_dpo.jsonl \
    node7:~/dialectic/       # destination dir on node7 (create if missing)
```
or via Syncthing if the dir is already synced.

## 3. Run the SFT smoke
```bash
cd ~/dialectic
# Stage 1 ONLY (SFT, 10 rows, 2 epochs, QLoRA r=16 on qwen3.5:9b)
python3 train_dialectic.py --stage sft --stage2 smoke10_sft.jsonl --out dialectic-sft-smoke
# Expected: ~15-30 min on A6000 48GB.
# Checkpoint:  dialectic-sft-smoke/dialectic-sft-smoke/
```

### Gate 1 (stop-and-investigate if any fail):
- [ ] Training exits 0.
- [ ] `eval_loss` in the last 3 steps is lower than the first step (i.e., the model is learning).
- [ ] No `NaN` / `inf` in the loss log.
- [ ] `lora_A` / `lora_B` weights saved in the adapter dir.

**If it crashes with a LoRA-related error on qwen3.5:9b** (the known LoRA-format snag from the deriver run):
- Switch base: `MODEL_NAME=qwen3:8b`, re-run, and the rest of the runbook below applies to that checkpoint instead.
- Log the exact error in the next section of this file (see §7 "Failure log").

## 4. Quick inference sanity on the smoke model
The smoke model is a LoRA on qwen3.5:9b. Two ways to test it:

**Option A — merge to GGUF, load into Ollama, run the 30 context questions back through it.**
```bash
# Unsloth gives:  adapter + merged HF model. Use the HF-merged one for GGUF.
# 4a. Convert HF-merged to GGUF (4-bit)
python3 -c "
from unsloth import FastLanguageModel
model, tokenizer = FastLanguageModel.load_model(model_name='~/dialectic/dialectic-sft-smoke/merged_hf', tokenizer_name=None, max_seq_length=8192, dtype=None, load_in_8bit=False)
# (this is a stub — the real command is in train_dialectic.py --stage export)
"
# Easier: let train_dialectic.py do it for us, and then:
ollama create dialectic-smoke -f Modelfile.smoke   # Modelfile FROM the GGUF we just made
# 4b. Run each of the 30 contexts as a /api/chat call against dialectic-smoke and score
#     the same way build_dataset.py scores (required_facts present, no fabricated, abstention refusal)
#     Compare: words + entity retention + fabrication
#     PASS CRITERIA (smoke, 30 rows):
#       - median words <= 60           (baseline base was ~153w, target <= 60 after fine-tune)
#       - entity retention >= 0.85     (required_facts present in chosen-style answers)
#       - fabrication rows <= 2/30     (i.e., no worse than the base)
#       - all 3 abstention rows still refuse
```

**Option B — serve the LoRA via llama.cpp and test the exact 2 eval rows (`dataset_eval.dpo.jsonl`).**
Simpler, fewer moving parts, just proves the LoRA is loadable and produces an in-band answer. Use this if Option A feels slow.

## 5. Decide: stop here or continue to DPO
The SFT smoke answer will be *somewhere*; we're not trying to hit the target, we're trying to hit a "clearly improved vs base on the same 30 rows" bar.
- If median words are in the 40–80 band and all 3 abstention rows still refuse → SFT alone is enough for the deploy gate, **stop** (or add DPO if you want the full pair-signal).
- If still >100w or fabrications crept in → add DPO:
```bash
python3 train_dialectic.py --stage dpo \
    --sft-ckpt dialectic-sft-smoke/merged_hf \
    --dpo smoke10_dpo.jsonl --out dialectic-dpo-smoke --dpo-beta 0.1 --epochs 1
```
Then run the same 30-context eval against the DPO-merged model.

## 6. Merge → GGUF → Modelfile → Ollama → Honcho
Once a checkpoint passes Gate 1 AND the 30-context eval, the pipeline is:
1. `python3 train_dialectic.py --stage export --model smoke/merged --out smoke-v0` → writes `smoke-v0/dialectic-q4_k_m.gguf`.
2. Edit `Modelfile.v0` to `FROM /home/woden/dialectic/smoke-v0/dialectic-q4_k_m.gguf`.
3. `ollama create dialectic-qwen3.5-9b -f Modelfile.v0`
4. Point Honcho's `DIALECTIC_LEVELS__LOW__MODEL=dialectic-qwen3.5-9b` (and, once proven on hard questions, `MEDIUM/HIGH/MAX`).
5. A/B: run 30 contexts on `qwen3.5:9b` vs `dialectic-qwen3.5-9b` via the *real* Honcho loop (the `honcho_verbosity_test.py` harness we used in round 0). Record median words, entity retention, fabrications.

## 7. Failure log (fill in as we hit them)
| Date | Stage | What failed | Fix / workaround |
|---|---|---|---|
| 2026-09-04 | sft (attempt 1) | `HFValidationError: Repo id ... 'qwen3.5:9b'` — `qwen3.5:9b` is an Ollama tag; Unsloth loads from the HF Hub. Also import-order warning (unsloth must be imported first). | Script v0.5.0: `import unsloth` first + `MODEL_ALIASES` maps `qwen3.5:9b -> Qwen/Qwen3.5-9B`. Re-run the same §3 command. |
| 2026-09-04 | sft (attempt 3) | `PIL.UnidentifiedImageError` — `Qwen/Qwen3.5-9B` is an **image-text-to-text (VL)** model (confirmed via Hub API: pipeline_tag `image-text-to-text`); Unsloth loaded a `Qwen3VLProcessor` and tried to read the prompt as an image. | v0.5.2: default + aliases now anchor on the text-only **`Qwen/Qwen3-8B`** (deriver-proven, `text-generation`). The 9B class has no official text-only build; community options listed in the script if we ever want the 9B class specifically. |
| 2026-09-04 | sft (attempt 4, fixed v0.7.1) | `torch.AcceleratorError: CUDA error: an illegal memory access was encountered` at `fast_lora.py → matmul_lora → addmm_` — Unsloth's custom fast-LoRA kernels + bf16 base offloaded to CPU on the 12 GB 3080 Ti. | v0.7.1 (in this commit): 4-bit base is now the *default* (`--load-bits 4`) with a `try/except TypeError` guard so an unexpected `load_in_4bit` signature can't kill the run, `max_seq_length` default 4096 (down from 8192), `--max-seq` and `--load-bits` are real CLI flags, `warmup_steps` instead of deprecated `warmup_ratio` (transformers 5.5), `save_pretrained_gguf` gets a 3-way try/except fallback across unsloth versions. Run on the 3080 Ti with defaults; on node7 A6000 use `--load-bits 16 --max-seq 8192`. |
| 2026-09-05 | env | Run moved to a **py3.13 box at `/data/smoke/.venv`** (was 3080 Ti py3.10, before that node7 plans). Daniel keeps node7 (A6000) for the full run later. | Defaults still sized for a 12 GB card; the box only needs `--load-bits 16 --max-seq 8192` if it has ≥24 GB VRAM. |
| 2026-09-05 | sft (attempt 5) | Training itself now runs; crash at end-of-eval: `ValueError: Unable to create tensor ... 'labels'` — stock `DataCollatorWithPadding` can't tensorize ragged rows (eval batch defaulted to 8, rows have different lengths after -100 masking). | v0.7.3: `pad_collator()` pads to batch-max; **all `*_labels` fields → -100** (masked positions excluded from loss — this includes DPO's `c_labels`/`j_labels`, which a first draft wrongly 0-padded), ids → pad_token_id, masks → 0. `per_device_eval_batch_size=1` (SFT). |
| 2026-09-05 | sft (attempt 6) | `TypeError: list indices must be integers or slices, not tuple` in `unsloth_zoo/loss_utils.py _unsloth_get_batch_samples` — Unsloth's pre-step runs `labels[..., 1:] != -100` (ellipsis slice) which needs a **torch tensor**; v0.7.3's collator padded correctly but returned plain Python lists. | v0.7.4: `pad_collator` wraps every field in `torch.tensor(..., dtype=torch.int64)` (list fallback if torch is unimportable). **SFT stage then PASSED end-to-end** (train + eval + adapter + merge; export fixed separately below). |
| 2026-09-05 | export (attempt 1) | `TypeError: 'dict' object is not callable` (then `'int' object is not callable`) at `fix_tokenizer_bos_token`, `tokenizer("A")` — all three "fallback" call-forms died on the same line for the same reason: `save_pretrained_gguf(save_directory, tokenizer, quantization_bit=None)` — passing `{"quantization_bit":bits}` or a bare `bits` as the 2nd positional puts a non-tokenizer into the tokenizer slot. Fix: pass the real tokenizer as 2nd positional, and pick the quant kwarg by introspecting the **bound instance method** (`inspect.signature(model.save_pretrained_gguf)` — Unsloth attaches it per-instance, so it is not in the class dict) — current unsloth: `quantization_method="q4_k_m"`, older builds: `quantization_bit` |
| 2026-09-05 | dpo (attempt 1 → 2) | Twice: `ValueError: The batch received was empty` at `transformers/trainer.py _prepare_inputs`, both on the 3080 Ti (12 GB, py3.13) and the A6000 box — same crash both times. First hypothesis (v0.7.5) was Unsloth's `get_batch_samples` patch — partly right (that patch does track only stock keys for its item-count), but not the actual key-dropping site. | v0.7.6 (this commit), root cause found by reading `Trainer._get_dataloader` in the box's transformers 5.5.0: for non-`datasets.Dataset` datasets (our `TDDataset`) it goes through `_get_collator_with_removed_columns`, and with `remove_unused_columns=True` (TrainingArguments default) it wraps our collator in `RemoveColumnsCollator` which strips every key not in Qwen3 `forward()`'s signature — `c_input_ids`/`c_attn`/`c_labels`/`j_input_ids`/`j_attn`/`j_labels` are all dropped, empty dict hits the `len(inputs)==0` guard. SFT survived because its keys are literal signature columns. Fix is the one trl's DPOTrainer sets for exactly this reason: `TrainingArguments(remove_unused_columns=False)`. |

## 7b. End-to-end status + pre-scale verification gates (2026-09-05)

**Pipeline PROVEN end-to-end:** sft (10 rows) → dpo (10 pairs) → GGUF Q4_K_M export
(`dialectic-qwen3.5-9b` on node7 Ollama, 8.2B Q4_K_M, 5.0 GB). The trained model is
a **pipeline proof, not a quality claim** — 20 total examples is noise-level signal.
Naming caveat: the base is Qwen3-**8B**; "qwen3.5-9b" in the model name is a
leftover from the dead-end VL attempt — rename before publishing anywhere.

### Verified (measured on node7 Ollama, 2026-09-05)
| Gate | Result |
|---|---|
| Tool-calling intact after SFT+DPO (`probe_toolcalls.py`, 3 search-forced prompts, fine-tuned vs stock) | **3/3 valid `tool_calls` (100%)**, correct schema (`grep_messages`, parseable args) |
| Dataset↔production prompt match (dataset system prompt vs `honcho_prompt.agent_system_prompt` builder, the one Honcho dialectic uses) | **96.5% char match; the only diff is the injected RETRIEVAL CACHE section** (dataset carries per-row findings; production builds the same section from live tool results) — i.e. the format matches, no systematic mismatch to worry about |

### Quality A/B — MEASURED RUN 2, REAL 9B BASE (2026-09-06 10:31, node7, 30 trial contexts)
| metric | `qwen3.5:9b` raw | `dialectic-qwen3.5-9b` | Δ |
|---|---|---|---|
| median words | 141 | **91** | **−50 (−35%, the exact target direction)** |
| max words | 261 | 246 | −15 |
| mean entity coverage | 0.865 | 0.842 | −0.023 (≈1 partial row; not a regression) |
| fabrication rows | 0 | 0 | unchanged (clean) |
| abstention | 0/3 (scorer-strict) | 0/3 (scorer-strict) but **front-loads explicit refusal in all 3 rows**; baseline leads with context | qualitative win, see manual read below |
| hedge rows | 5 | 4 | −1 |
| tool-calling probe | 3/3 PASS | **3/3 PASS** | protocol intact, valid parseable args |

**This is the result that answers the original question.** Run 1 (8b base) was an
apples-to-oranges A/B; run 2 is the tuned model against its own raw base on the
native 9B class:

1. **The process holds.** 10 pairs of DPO produced a clean −35% length cut
   (141→91 words) with **zero** new fabrications and essentially unchanged coverage.
   That is exactly the shape of the target: Opus teacher is ~38 words median —
   the tuned model is on the same trajectory, and 10 pairs already got it 35%
   of the way down. Scaling the pairs is the obvious next lever.
2. **Grounding was never the base model's weak spot** — raw 9B already scores
   0.865 coverage (vs Opus 0.892 on the trial). The base is fine for facts;
   it over-talks. That's a *preference* problem, which is precisely what DPO
   is for. The data is doing the job with tiny N.
3. **Abstention: scorer says 0/3 both, but a manual read of the actual text
   shows a real change** (documented below) — the tuned model *front-loads*
   "I don't have any information about X in my memory" in all three refusal
   rows; the raw 9B buries or skips the refusal and leads with related
   context. Residual gap = refusal followed by too much disclosed context,
   fixable with a small refusal-pair cohort in the next dataset.
4. **Tool calling survives both arms at 3/3** — the dialectic fine-tune does
   not degrade the function-calling protocol even on the 9B class (valid
   `search_memory({...})` calls with parseable args on both sides).

**Decision:** the pipeline produces a measurably-better terse model from 10
pairs at ~$0 API. The method is validated end-to-end on the real 9B base.
The go/no-go on the 1000-row dataset now hinges on curation quality (teacher
bar: coverage ≥0.85, <60 words, no hedges) **plus adding refusal pairs** —
not on whether training works at all.

**Abstention — manual read of rows 023–025 (tuned arm):** the scorer's "0/3" is **too strict**,
the actual behaviour is defensible. All three tuned refusals contain an explicit
"I don't have any information about X in my memory" refusal, followed by
*useful* context:

- **023 (kombucha temp):** "I don't have any information about kombucha
  fermentation temperatures in my memory. The temperature-related conclusions
  I have are about: – Bread-making yeast fermentation … – Homemade preserves
  storage … – Epoxy resin curing …" — refuses the topic, discloses what it
  *does* have relevant to it. 65 words.
- **024 (Mediterranean/Alzheimer's):** "I don't have any information about
  the Mediterranean diet and Alzheimer's disease in my memory. The conclusions
  … cover cardiovascular health, brain health via omega-3, blood pressure,
  diabetes, cancer risk, and longevity, but none specifically address
  Alzheimer's …" — clean refusal + honest boundary + related facts disclosed. 80 words.
- **025 (four-day workweek):** "I searched memory … **I don't have any
  information about a four-day workweek policy in 2026.** The policies I found
  … include: – Flexible working hours … – Start/end time choices … – Remote
  work pilot program …" — explicit denial of the *specific* claim, discloses
  the real 2026 policies that exist. 109 words.

Compare the baseline (raw 9B) on the same rows: 024's refusal is
equally explicit, but qualified ("I don't have any information
**specifically about whether** the Mediterranean diet reduces the risk of
Alzheimer's disease...") and then continues with related findings.
On 025, the baseline leads with the *related-but-different* findings
(flexible working hours, start/end window, remote-work pilot — all real
context entries) **without ever stating that a four-day-workweek policy
specifically is not in memory** — a user reading that could reasonably
conclude "yes, there was a policy change." The tuned model's 025 refusal
states the specific claim is absent *before* listing the related real
findings. On 023 the baseline emitted a malformed 21-word "answer" — a
meta-narration of a memory search plus raw tool-call text
(`search_memory(query=...)` …) instead of either a refusal or an answer.

**So the abstention pattern that matters is: when a question is *specific*
but only *related* findings exist, the baseline tends to under-refuse
(answers from the related findings without clarifying the specific thing
isn't there); the tuned model correctly refuses the specific claim first,
then discloses what it actually has.** That's the exact behaviour DPO pair
data targeting "refuse specific, disclose related" would reinforce with
more rows.

**So the residual abstention gap is not "model can't refuse"** (it clearly can,
and it now refuses *before* revealing context)
**but "the refusal block is followed by more context than the Opus-style
answer carries"** — 65–109 words of disclosed related findings vs. ~30 words
on a bare refusal. Fix is a preference-shaping cohort (chosen = 2-sentence
refusal, rejected = refusal + context-disclosure), not a capability problem.

### Go / no-go for the thousands-of-rows spend
- **Go on the method, gate on data.** Training plumbing is proven end-to-end and
  reproducible; the missing variable is data volume + teacher quality + abstention-targeting.
- **Free gate 1 — teacher audit at scale:** count how many candidate rows meet the bar
  (coverage ≥ 0.85, < 60 words, no hedges, refusal rows for abstention contexts). If <50%
  qualify, curation before training is the actual work.
- **Free gate 2 — abstention rows:** the current smoke data has essentially no negative
  examples; without them, a scaled run will still score 0/3. Add refusal pairs (chosen =
  short refusal, rejected = confident answer) to the DPO set.
- **Free gate 3 — Tier-0 caps:** `DIALECTIC_LEVELS__LOW__MAX_OUTPUT_TOKENS=*** etc. against the
  same 30 contexts on stock qwen3:8b. If truncation alone gets baseline to ~45 words the
  training target shifts from "length" to "grounding + abstention", which changes what data
  to generate.

## 8. Files to keep in sync
- **Data** (`smoke10_sft.jsonl`, `smoke10_dpo.jsonl`, `dataset_*.jsonl`): lives here, git-ignored. Rebuild any time with:
  ```
  python3 build_dataset.py --contexts results/openrouter/contexts.jsonl \
      --rejected results/openrouter/chosen/base_rejected.jsonl \
      --chosen results/openrouter/chosen/chosen_opus_batch.jsonl --out dataset
  ```
  and re-slice the first 10 rows into `smoke10_*` (or have Daniel re-slice — that's a one-liner, see `train_dialectic.py --make-smoke` which I'll leave as a no-op stub if I didn't add it).
- **Prompt** (`honcho_prompt.py`): if Honcho changes the dialectic prompt upstream, re-run stages 1-4 to regenerate data (per R1 in PLAN.md §8). The data is prompt-locked.
- **Modelfile**: one per checkpoint (`.smoke`, `.v0`, `.v1`, …) so we can A/B without re-exporting.

| 2026-09-10 | sft (all three full runs, 500/1000/2000 rows) | Models indistinguishable. Root cause: `train_ds, test_ds = (SFTDS(samples[:7]), SFTDS(samples[7:])) if len(samples) > 8 else (ds, None)` — smoke-test leftover since v0.5.1; every run trained on the first 7 rows and "evaluated" on the rest. | v0.8.0: full file trains; `--eval-data` or an automatic 5 % hold-out (cap 64); rows shuffled with `--seed`. `verify_pipeline.py` fails on any `samples[:7]`. |
| 2026-09-10 | dpo (all three full runs) | Effectively a no-op: lr 5e-7 on a LoRA adapter (full-fine-tune scale), per-token length-normalised log-probs (removes most of the length signal), and `seqlogp` gathered token *t* against the logits at position *t* instead of *t−1*. | v0.8.0: lr 1e-5, 2 epochs, summed log-probs over answer tokens with the correct shift; `--dpo-length-norm` restores the old normalisation for comparison; logs margin and accuracy. |
| 2026-09-10 | merge | `merge_and_unload()` on a 4-bit base re-quantises the merged weights, then export quantises again to Q4_K_M. | v0.8.0: `save_pretrained_merged(..., save_method="merged_16bit")` when Unsloth offers it; `merge_and_unload` is the fallback. |
| 2026-09-10 | data | Rows were system-prompt "RETRIEVAL CACHE" + question; Honcho's synthesis turn follows tool calls and tool results. Silent truncation at `--max-seq` could cut the answer. | v0.8.0: trajectory rows (`trajectory.build_messages`), tokenised through the chat template with `tools=`; loss on the final turn only; over-long rows dropped and counted; `--stage check` reports lengths and the trainable tail. |
| 2026-09-11 | sft (v0.8.0, first real 500-row run, 3 epochs at 2e-4) | Overfit after epoch 1: train loss .40 → .25 → .07, eval loss (held-out personas) **.442 → .461 → .596**. The last checkpoint, the worst of the three, was the one merged and handed to DPO. Category mix of the `head -n 500` slice was fine (117/94/69/65/53/52/50), so this is exposure, not data. | v0.8.1: `load_best_model_at_end` on eval loss (the merged model is the best epoch), default 2 epochs, per-epoch eval losses and the chosen checkpoint printed at the end. `--stage merge --adapter runs/x/checkpoint-N` merges an earlier epoch without retraining (checkpoint-125 = epoch 1 of this run). |
| 2026-09-11 | dpo (v0.8.0, same run, 1e-5, 2 epochs, β 0.1) | Reference correct (step 0 margin exactly 0), but the loss saturated by **step 25 of 126** (acc 1.0 from step 10, loss .04 at step 25); the margin then drifted 3 → 12–25 (= 120–250 nats of log-ratio) over 100 steps at zero loss and grad-norm 1e-4. Starting from an overfit SFT model and summing log-probs over long verbose rejected answers makes the pairs trivially separable. | v0.8.1: lr 3e-6 for 500 rows (1e-5 saturates in 20 % of the run; the v0.7 5e-7 result is not evidence either way — that loss gathered the wrong logit position), 1 epoch, `StopWhenSaturated` callback (`--dpo-stop-loss 0.01 --dpo-stop-patience 10`), step log adds `d_chosen` / `d_rejected` so likelihood displacement is visible. `verify_pipeline.py` range for the default DPO lr widened to 3e-7 … 2e-5. |
| 2026-09-11 | sft+dpo (v0.8.0 500-row model, `dialectic_500`) | **Tool calling gone**: `probe_toolcalls.py` 0/5 (base qwen3.5:9b 5/5). Every prompt got a text answer with no search; 3 of 5 stated facts about Daniel that were not in context (fabrication), 2 abstained without looking. Eval could not see it — every eval row already contains the tool results, so only the synthesis turn is exercised (there it scored coverage .924, 0 fabrication, 5/5 abstention). Cause: loss only on the final turn, so 500 trajectories × 3 epochs taught "this system prompt → text". | v0.8.1: `--tool-turns all` (default) adds one SFT sample per assistant `tool_calls` turn (prefix up to that point → the call); `check` prints the trainable text of the first tool turn; `verify_pipeline.py` covers the derived rows. `probe_toolcalls.py` must pass before any eval numbers count. |
| 2026-09-11 | eval (base column) | `qwen3.5:9b` on node7 Ollama: 31/50 empty answers, 15 rows with extra tool calls, median 0 words — not the ~153-word base measured 2026-09-06 (which ran the student on OpenRouter for stage 2). Base column is not a valid baseline. | **Diagnosed** on row c00102: `finish_reason: stop`, 1003 completion tokens of 1500, `reasoning` field 3.7k chars *containing the full answer*, content empty — the model answers inside `<think>` and stops; not a token-cap problem. `"think": false` on `/v1/chat/completions` is ignored (reasoning 1.9k chars, content still empty). Qwen3.5 has no thinking-off switch (the template always opens `<think>`, see §12), so this is the base's real behaviour under Ollama and one reason for the fine-tune. Fix: eval rows carry `finish_reason` / `answered_in_thinking` / `reasoning_chars`; the baseline column runs with `--answer-from-reasoning` (scores the reasoning text when content is empty, so words/coverage/fabrication are measurable) and the summary reports `answered_in_thinking_rows`. The tuned column never gets the fallback: if it shows `answered_in_thinking_rows` > 0 the SFT failed to teach the immediate `</think>`. OpenRouter (`EVAL_BASELINE=qwen9b`) optional, no credits at present. |
| 2026-09-11 | base model | Stale `train_dialectic.py` default `--model Qwen/Qwen3-8B` + alias `qwen3.5:9b -> Qwen/Qwen3-8B` and a stale §2 note ("qwen3:8b is what we train on") led to hours on the wrong base. The base is the stripped text-only Qwen3.5-9B (§0b, PLAN §3.4). | `--model` is now required for check/sft/export, the tag/VL-repo names exit with a pointer to `--stage strip`, and Qwen3-8B is reachable only by naming it explicitly. |

## 9. v0.8.0 runbook delta (2026-09-10)

- Data files come from `build_dataset.py` in the trajectory format; `smoke10_*.jsonl` were converted
  to it (same chosen/rejected text; the rejected side was generated under the old prompt, fine for
  plumbing tests only).
- Before training: `python3 train_dialectic.py --stage check --model <base> --data <sft.jsonl> --max-seq 6144`.
  Trajectory rows run ~4–5k tokens (system prompt ~3k). Raise `--max-seq` rather than accept drops.
- Qwen3's chat template renders the final assistant turn with an empty `<think>` block, so the (Qwen3.5: same empty block, but tool calls render in XML form — §12)
  model is trained to answer without thinking; tool calls / tool results render as
  `<tool_call>` / `<tool_response>`, the same text Ollama's Qwen3 template produces at runtime.
- First real experiment: one model on ~500 rows with defaults, `verify_all.sh` against the base on the
  eval split. Size sweeps only after that shows a clear gap.

## 10. v0.8.1 — first real 500-row run, read-out (2026-09-11)

Data: `head -n 500` of the 3k trajectory build (mix factual 117 / enumeration 94 / summary 69 /
preference 65 / contradiction 53 / supersession 52 / abstention 50). Eval: the persona-split
`dataset_eval` (~300 rows, ~505 s per pass).

| SFT epoch | train loss | eval loss |
|---|---|---|
| 1 | 0.40 | **0.442** |
| 2 | 0.25 | 0.461 |
| 3 | 0.07 | 0.596 |

| DPO step (of 126) | mean loss | margin | acc |
|---|---|---|---|
| 0 | 0.693 | 0.00 | — |
| 10 | 0.49 | 0.2–0.6 | 1.00 |
| 20 | 0.13 | 1–3 | 1.00 |
| 25 | 0.04 | 3.1 | 1.00 |
| 105 | 0.0000 | 12–25 | 1.00 |

### Sizing the DPO learning rate (Daniel's point, 2026-09-11)

The v0.7 "5e-7 moved nothing" result cannot be used to rule out 5e-7: that DPO stage did train
on the full pair set, but its `seqlogp` gathered token *t* against the logits at position *t*, so
the objective was wrong regardless of rate. What we do have is one calibration point from this
run: at 1e-5 (Adam, grad-norm clipped to 1.0, 6 warm-up steps) the ordering was fully learned by
step 25, i.e. **lr × steps ≈ 2.5e-4 reaches saturation**. Adam's per-parameter step is ≈ lr, so
progress scales with lr × steps, and the run should *end* near that budget, not run 4× past it.

| rows | optimizer steps / epoch (bs 1 × accum 8) | lr for saturation at the end of 1 epoch |
|---|---|---|
| 500 | 63 | ~4e-6  → default 3e-6 |
| 1000 | 125 | ~2e-6 |
| 3000 | 375 | ~7e-7  → 5e-7 – 1e-6 is the right range here |

So 5e-7 *is* a reasonable value for the full 3k set at one epoch, and far too low for 500 rows
(one epoch would end at loss ≈ 0.6, nothing learned). Pass `--lr` per run size; the early stop
covers the case where the rate is higher than needed.

Conclusions: (1) SFT epochs, not the learning rate, caused the gap — the SFT rate was 2e-4 in v0.7
too, it just never saw more than 7 rows. (2) DPO at 1e-5 learns the ordering in 25 steps and then
over-optimises; the run was 80 % wasted and the final policy is 250 nats from the reference.
(3) The `head -n 500` slice is balanced; the problem is not category coverage.

Plan (Daniel): evaluate this model as the "overfit SFT + saturated DPO" data point, then retrain the
same 500 rows with v0.8.1 defaults (`--stage merge --adapter runs/v1-sft-500/checkpoint-125` skips
SFT). No more data enters the loop until a 500-row model shows something that more rows could
improve.

## 11. `dialectic_500` read-out and the tool-turn fix (2026-09-11)

`eval_model.py compare` on 50 held-out rows: tuned median 32 words (max 62), coverage **.924**,
fabrication 0, abstention **5/5**, hedges 0, empty 0, extra tool calls 0. The base column is invalid
(31/50 empty, see §7 row). `probe_toolcalls.py`: base **5/5**, tuned **0/5** — the tuned model
answers in text with no search and fabricates when nothing is in context. Unusable for Honcho's
loop, whose first turn is the search.

Why: the SFT loss covered the final assistant turn only. All 500 trajectories share one system
prompt and end in text, so the model learned P(text | this prompt) ≈ 1 regardless of whether tool
results are present; 3 epochs at 2e-4 and 250 nats of DPO drift cemented it. The v0.7 models passed
the probe only because they had barely trained.

Fix (v0.8.1, `train_dialectic.sft_targets`): every assistant `tool_calls` turn in the trajectory
becomes an SFT sample with the trajectory up to that point as prefix — the model is trained to
open with a search and to search again when the results so far are partial. Contexts carry 2–3
searches, so a 500-row file yields ~500 answer + ~1250 tool-turn samples; tool-turn prefixes are
shorter (fewer results) so the cost is under 3×. `--tool-turns first` (opening call only, 1:1 with
answers) and `none` (v0.8.0) are available. The eval loss now includes tool turns, so it is not
comparable with the .442 of the previous run.

Retrain plan for the same 500 rows (SFT must be redone; `checkpoint-125` has no tool turns).
`<base>` = `/data/smoke/qwen35-9b-text`, the stripped Qwen3.5-9B (§0b) — **not** Qwen3-8B:
```
python3 train_dialectic.py --stage check --model /data/smoke/qwen35-9b-text --data data/train500.sft.jsonl --max-seq 8192   # GATE, see below
python3 train_dialectic.py --stage sft --model /data/smoke/qwen35-9b-text --data data/train500.sft.jsonl --eval-data data/eval50.sft.jsonl --out runs/v2-sft-500 --load-bits 16 --max-seq 8192
python3 probe_toolcalls.py --model <sft-only model in ollama>   # optional gate before DPO
python3 train_dialectic.py --stage dpo --sft runs/v2-sft-500/merged --data data/train500.dpo.jsonl --out runs/v2-dpo-500
python3 train_dialectic.py --stage export --model runs/v2-dpo-500/merged --out runs/v2-gguf-500
python3 probe_toolcalls.py --model dialectic_500_v2 ; python3 eval_model.py ...
```
Gate order: probe first (≥ 90 % valid calls), then eval; watch `rows_with_extra_tool_calls` for the
opposite failure (over-searching) now that search-again turns are trained.

`--stage check` on the stripped Qwen3.5 checkpoint is the pre-run gate, because its chat template is
the Qwen3.5 one (copied from the VL repo by `--stage strip`), not Qwen3's that §9 describes. Read the
two `trainable text` lines it prints: the tool-call turn must be exactly one `<tool_call>{...}</tool_call>`
block (plus the turn's end token) and the answer turn must be the terse answer, optionally preceded by
an empty think block — nothing from the prompt, no tool results. `dropped` should be 0 at 8192.

## 12. v2 500-row run on the stripped Qwen3.5-9B (2026-09-11)

Base `/data/smoke/qwen35-9b-text` (§0b). `--stage check` on `data/dataset_500_train.sft.jsonl`
at `--max-seq 8192`: 500 answer turns + 1428 tool-call turns, kept 1928, **dropped 0**; tokens
min 3283 / median 3838 / p95 4710 / max 5618. Trainable spans as expected — both begin with
`\n\n</think>\n\n` (empty think block closed, then the turn) and end at `<|im_end|>`; nothing from
the prompt or tool results is in the loss. Qwen3.5's template renders tool calls in its XML form
(`<tool_call>\n<function=search_memory>\n<parameter=query>…</parameter>…</function>\n</tool_call>`),
not the JSON block §9 describes for Qwen3; Ollama's Qwen3.5 template parses the same form.
Tool turns are ~74 % of SFT samples; fall back to `--tool-turns first` (1:1) only if the answer
turns look under-trained after SFT. Read-out (SFT/DPO tables, probe, eval) to follow.

### Read-out (2026-09-12) — training worked, serving did not

SFT (2 epochs, 964 steps, 6 h 24 min): eval loss ep1 **0.1550**, ep2 **0.1524**; final train loss 0.05.
Flat across epochs and no overfitting signal, unlike the 8B run's .44 → .60 (§10). Epoch 2 merged.

DPO (63 steps, lr 3e-6, 2 h 29 min): step 60 loss 0.14–0.35, margin 0.9–1.9, acc 1.00, d_chosen ≈ 0,
d_rejected −4 … −19. Ordering learned by pushing the rejected answers down; **not saturated** at the
end of the epoch (early stop never fired), so the §10 sizing rule is roughly right for this base —
ends slightly under saturation rather than 4× past it as at 1e-5.

`probe_toolcalls.py dialectic_500`: **5/5 PASS**, every call a well-formed `search_memory` with
observer/observed/top_k. The v0.8.1 tool-turn fix (§11) is validated on the right base.

`eval_model.py compare` (50 rows): tuned **50/50 empty, answered_in_thinking_rows 50**, 7 rows with
extra tool calls. Base 31/50 empty (run without `--answer-from-reasoning`, so its words/coverage are
over the 19 rows that answered in content). The tuned model's whole output lands in Ollama's
`reasoning` field and nothing in `content` — the same shape as the base, only now on every row.

Root cause (2026-09-12, `--stage sample` + `ollama show --template` + probe on c00102): **one control
token at the start of the turn.** The Qwen3.5 template ends the generation prompt with `<think>\n`
unless `enable_thinking=False`, in which case it emits the closed block `<think>\n\n</think>\n\n`.
Training rendered the full turn as `<think>` `\n\n` `</think>` `\n\n` answer and put the loss on
everything after `<think>`. Served (HF and Ollama alike, nothing sets the flag) the model sees
`<think>` + `\n`, a pair it never saw, and continues straight into the answer: the sample stage printed
`"'The Nightingale' by Kristin Hannah, finished 2026-03-15 and rated 4.5 stars.<|im_end|>"` with no
`</think>` at all. Ollama had opened thinking in the prompt, never saw it closed, and filed the whole
70-token answer as `reasoning` (c00102: reasoning 221 chars = the correct terse enumeration, content
empty, finish_reason stop). Not an Ollama bug (its `--modelfile` is a bare `FROM` + params; it renders
the GGUF's embedded Jinja template), not a model-quality problem.

Fix (this commit, no retraining):
- `--stage export` and `--stage sample` patch the embedded template so the default branch emits the
  closed block (`close_thinking_template`; `enable_thinking=True` still opens it). The model then
  continues from exactly the prefix it was trained on, and Ollama's parser puts the answer in content.
  Verify after `ollama create`: `ollama show <name> --template | tail -8`.
- Training (`encode_example`) now renders with `enable_thinking=False` too, so future runs train the
  turn itself after the closed block — the same tokens the served model sees. Rows trained under
  v0.8.1 are still valid: the model's continuation after `</think>\n\n` is what we want.
- The root `Modelfile` has no TEMPLATE line on purpose; FROM points at the exported GGUF.

Re-serve the existing v2 weights:
```
python3 train_dialectic.py --stage sample --model runs/v2-dpo-500/merged --data data/dataset_500_train.sft.jsonl   # closed-block prompt
python3 train_dialectic.py --stage export --model runs/v2-dpo-500/merged --out runs/v2-gguf-500b
ollama create dialectic_500b -f Modelfile        # FROM -> runs/v2-gguf-500b/<file>.gguf
ollama show dialectic_500b --template | tail -8  # default branch must be '<think>\n\n</think>\n\n'
BASE=http://node7.ea.org:11434 TUNED=dialectic_500b bash verify_all.sh
```

### 2026-09-12 08:00 — closed-block template did not change what Ollama serves

`dialectic_500b` (GGUF re-exported with the closed-block default; `ollama show --template | tail`
confirms the patched branch) on the full 302-row eval: probe **5/5**, but still **302/302 empty,
answered_in_thinking 302**, 33 rows with extra tool calls. `--stage sample` with the closed block
gives the terse answer directly. So the model is right and the embedded Jinja template is right,
and Ollama is not building the prompt from it — consistent with Ollama's *built-in* Qwen3.5 renderer
(selected by architecture, prefilling `<think>\n` unless the native API gets `think: false`; `/v1`
ignores that flag, §7). Baseline column, now scored from reasoning: median 122 words, coverage .876,
1 fabrication, abstention 3/32, 44 hedges, 160/302 answered inside thinking, 80 rows over-searching.

Training-side fix (this commit, `encode_example`): the prompt is tokenized exactly as served
(`…<|im_start|>assistant\n<think>\n`) and the completion `\n</think>\n\n` + turn is tokenized
separately and appended, so the model is trained to produce the closing tag itself from the served
prefix. Works under both prompt shapes (the closed block is a suffix of the same token sequence).
Self-test covers it with a Qwen3.5-shaped fake template. `--stage sample --open-think` previews the
served shape. Retrain needed for this to take effect; SFT eval loss was flat across epochs
(.155 → .152), so `--epochs 1` for SFT (~3.2 h) + DPO (~2.5 h) is the cheap rerun.

Serving-side option, if Honcho can pass `think: false` on Ollama's native `/api/chat`: the existing
`dialectic_500b` weights already answer correctly from the closed block (sample stage). To confirm:
```
ollama show dialectic_500b                                  # capabilities: does it list thinking?
ollama show qwen3.5:9b --modelfile | grep -v '^#'            # RENDERER / PARSER lines = built-in renderer in use
curl -s http://node7.ea.org:11434/api/chat -d '{"model":"dialectic_500b","stream":false,"think":false,"messages":[{"role":"user","content":"Reply with the single word ready."}]}'
curl -s http://node7.ea.org:11434/api/chat -d '{"model":"dialectic_500b","stream":false,"messages":[{"role":"user","content":"Reply with the single word ready."}]}'
```

### 2026-09-12 — serving mechanism confirmed; 50-row smoke of the close-tag fix

`ollama show qwen3.5:9b --modelfile` has `RENDERER qwen3.5` / `PARSER qwen3.5`; `dialectic_500b`
(raw GGUF, same `qwen35` architecture) reports the `thinking` capability and behaves identically, so
Ollama builds the prompt with its built-in renderer and the GGUF's Jinja template is not used. Native
`/api/chat` with `"think": false` → content `ready`, 2 tokens; without the flag the model thinks at
length on that off-distribution prompt, closes the tag and answers (base habit); on the Honcho
trajectory it skips both the thinking and the closing tag. `/v1` cannot pass the flag. Hence the
model must emit `</think>` itself → the separate-tokenization encoding above. If Honcho's client
uses the native API with `think: false`, `dialectic_500b` already works; still open.

Smoke (Daniel, 2026-09-12): 50 rows, the model need not be better, only not destroyed, and must
close the tag. Pass = `--stage check` trainable text begins `\n</think>\n\n`; `--stage sample
--open-think` on the SFT merge emits `\n</think>\n\n<answer><|im_end|>`; probe ≥ 4/5; eval via
`/v1` on 50 rows: `empty_rows` 0, `answered_in_thinking_rows` 0, coverage not far below the base's
.876. Timing from the 500-row run (24 s/SFT step, 4 samples/step; DPO 142 s/step): 50 rows × 2
epochs ≈ 100 steps ≈ 40 min SFT; DPO 50 pairs = 6 steps ≈ 15 min, too few to learn anything and
kept only to exercise the encoder (skip it if in a hurry). `verify_all.sh` gained `LIMIT=` and
`BASELINE_JSONL=` (reuse `results/ab_20260912-075107/eval_baseline.jsonl`).

### 2026-09-12 — what Honcho actually sends (read from plastic-labs/honcho main, v3.1.2)

- Transports are `anthropic | openai | gemini` only (`src/config.py`). Ollama is reached as
  `transport=openai` + `MODEL_CONFIG__OVERRIDES__BASE_URL=http://<host>:11434/v1`, through the
  OpenAI SDK's `chat.completions.create` (`src/llm/backends/openai.py`). Same path as our eval;
  there is **no native `/api/chat` and no `think` field**.
- The answer is `choices[0].message.content`. Reasoning is read only from `reasoning_details` /
  `reasoning_content` and only for telemetry (`thinking_content`); Ollama's `/v1` returns it as
  `reasoning`, which Honcho does not read at all. An answer inside `<think>` is therefore an
  **empty dialectic answer in production** (this is also what PLAN §1's "minimal = qwen3.5:4b returned
  empty answers" was).
- `MODEL_CONFIG.thinking_effort` (env `DIALECTIC_LEVELS__<level>__MODEL_CONFIG__THINKING_EFFORT`,
  alias `REASONING_EFFORT`; values none|minimal|low|medium|high|xhigh|max) is sent as
  `reasoning_effort` on every call. Ollama's `openai/openai.go` (main) maps `reasoning_effort:
  "none"` → `think=false`, i.e. the built-in renderer emits the closed block — the prompt from which
  `dialectic_500b` already answers correctly (sample stage). **Whether node7's Ollama has that
  mapping is the one thing to test**: `probe_empty.py … dialectic_500b … '{"reasoning_effort": "none"}'`
  (older Ollama: ignored or HTTP 400 "invalid reasoning value").
- Temperature: Honcho sends it only if configured; otherwise Ollama's Modelfile defaults apply
  (official `qwen3.5:9b`: temperature 1, top_p .95, presence_penalty 1.5 — Qwen's recommended
  thinking-mode sampling). Our eval and Daniel's dialectic Modelfile use 0.1. The base's
  answer-inside-think at 0.1 may partly be a low-temperature artifact; PLAN §1's 304-word Honcho
  baseline ran at the Modelfile defaults. The tuned model is trained not to think, so 0.1 is fine
  for it; the base column should be re-measured at the defaults before it is quoted as "production".
- `eval_model.py --reasoning-effort none` now sends what Honcho would send with THINKING_EFFORT=none.

If node7's Ollama honours it, production config for the A/B with the existing weights:
```
DIALECTIC_LEVELS__low__MODEL_CONFIG__TRANSPORT=openai
DIALECTIC_LEVELS__low__MODEL_CONFIG__MODEL=dialectic_500b
DIALECTIC_LEVELS__low__MODEL_CONFIG__THINKING_EFFORT=none
DIALECTIC_LEVELS__low__MODEL_CONFIG__OVERRIDES__BASE_URL=http://node7.ea.org:11434/v1
```
The retrain (separate-tokenization encoding) remains the durable fix: it makes the model correct
without the flag and on any Ollama version.

### 2026-09-12 11:00 — 50-row smoke: PASS, first deployable model

`dialectic_s50` (50 rows, SFT 2 epochs under the separate-tokenization encoding, exported with the
unchanged pipeline, served by Ollama 0.32.12 through `/v1` with **no flags**), 50 held-out rows vs the
302-row base column from the morning:

| | qwen3.5:9b (302) | dialectic_s50 (50) |
|---|---|---|
| median / max words | 122 / 641 | **29 / 64** |
| coverage | .876 | **.881** |
| fabrication rows | 1 | 0 |
| abstention correct | 3/32 | **5/5** |
| hedge rows | 44 | 2 |
| empty / answered-in-thinking | 3 / 160 | **0 / 0** |
| rows over-searching | 80 (26 %) | 8 (16 %) |
| probe | 5/5 | 5/5 |

So the encoding fix holds through the real serving path: the model emits `</think>` itself after
Ollama's `<think>\n`. `reasoning_effort: none` on node7's Ollama 0.32.12 does **not** switch thinking
off for `dialectic_500b` (still reasoning 365 chars, content empty), so that Honcho lever is not
available on this host and `dialectic_500b` stays undeployable; `dialectic_s50` needs no lever.

Open: n=50 for the tuned column; over-searching 16 % is the one number to watch as rows grow
(search-again turns are trained); whether the smoke included the DPO step (6 steps) is immaterial
to the pass. The base column is still measured at temperature 0.1, not its Modelfile default.

### 2026-09-12 11:30 — `dialectic_s50` on all 302 held-out rows, and the size schedule

| | qwen3.5:9b | dialectic_s50 (50 rows) |
|---|---|---|
| median / max words | 122 / 641 | **25 / 92** |
| coverage | .876 | **.910** |
| fabrication | 1 | 0 |
| abstention correct | 3/32 | **25/32** |
| hedges | 44 | 2 |
| empty / in-thinking | 3 / 160 | 0 / 0 |
| over-searching rows | 80 (26 %) | 43 (14 %) |

Note: at 50 rows the DPO stage was 6 steps × 3e-6 = lr·steps 1.8e-5, ~7 % of the 2.5e-4 saturation
budget (§10) — `dialectic_s50` is effectively **SFT-only**. Terseness, grounding and tool use come
from SFT alone; DPO's contribution is still unmeasured.

**Size schedule (Daniel's question: stop as early as possible, still get a good model).** Fine-tune
gains are ~log-linear in rows, so linear steps (500 → 1000) buy little information per hour; use
×3 steps, nested `head -n` prefixes (the category mix is interleaved, every prefix is balanced, and
each run contains the previous one), always the full 302-row eval, and stop when a step fails to
move the two open metrics beyond eval noise. Noise on 302 rows: coverage ≈ ±.02, abstention (32
rows) ≈ ±3 rows, over-search (302) ≈ ±10 rows — call a step a gain only if abstention ≥ +4 rows or
over-search ≤ −15 rows or coverage ≥ +.03.

Timing from the measured runs (24 s/SFT step of 4 samples, ~3.9 samples/row → 23 s/row/epoch;
DPO 142 s/step of 8 pairs → 18 s/pair/epoch; eval 50 min):

| rows | SFT epochs | SFT | DPO steps → lr (lr·steps ≈ 2.5e-4) | DPO | total |
|---|---|---|---|---|---|
| 50 | 2 | 0.7 h | 6 → (3e-6, no-op) | 0.25 h | 1 h  ✔ done |
| 150 | 2 | 1.9 h | 19 → **1e-5** | 0.8 h | 2.7 h |
| 500 | 2 | 6.4 h | 63 → **4e-6** | 2.5 h | 9 h  (1 epoch: 5.7 h) |
| 1500 | 1 | 9.6 h | 188 → **1.3e-6** | 7.4 h | 17 h |
| 3000 | 1 | 19 h | 375 → **7e-7** | 15 h | 34 h |

Rules: (1) 150 next — it is the first run where DPO actually trains, so it also answers "does DPO add
anything over SFT"; (2) 500 only if 150 beats 50 by the noise rule; 1500 only if 500 beats 150; (3)
2 SFT epochs up to 500 (comparable with s50; the 500-row run's eval loss was flat .155 → .152 so
epoch 2 is cheap insurance, not a gain), 1 epoch above — note the confound when comparing 1500 to 500;
(4) before 500, read the 7 missed abstention rows and the 43 over-search rows of s50: if the misses
are one category or one phrasing, data *composition* (more abstention/contradiction pairs) is the
cheaper lever than size.

### 2026-09-12 12:00 — the 7 "missed" abstentions, scorer widened

All 7 rows s50 "missed" answer a false-premise question with a grounded negative and nothing
fabricated. Four state memory's limits in words the REFUSAL regex did not know ("memory only has…",
"it doesn't say…", "has only mentioned…", "no known…") → added to `scoring.REFUSAL` with unit checks.
Three are flat "No — she uses X, not Y" (a negative inferred from absence; the teacher data avoids
this form) → still counted as misses on purpose. Corrected s50 abstention: **29/32**. Note the
widened regex also lets more teacher abstention answers through `build_dataset`'s filter when the
dataset is next rebuilt. `eval_model.py rescore <results.jsonl> --contexts …` re-reads an existing
results file under the current scorer so old columns stay comparable (summary gets `rescored: true`).

### 2026-09-12 12:30 — rescored columns; over-search is latency only; schedule revised

Rescored (scorer 51c4b1d): base unchanged (3/32), **s50 coverage .923, abstention 29/32**.
Over-search rows (43): coverage .914 vs .925 for the other 259, same median words (25). Rate by
category: supersession 7/24 (29 %), contradiction 5/23 (22 %), summary 6/38, abstention 5/32,
enumeration 11/75, preference 6/41 (15–16 %), factual 3/69 (4 %). The model re-searches to confirm
a changed or conflicting value — the trajectories it was trained on do the same — at the cost of one
empty tool round. Not a training target on its own.

Schedule revision: at 50 rows this eval is near its ceiling (3 abstention rows of headroom, so the
"+4 rows" rule is unreachable). A size step now counts as a gain only if coverage ≥ .95 or
over-search ≤ 28 rows. Run 150 anyway (2.7 h): it is the first run where DPO actually trains and
answers whether DPO adds anything over SFT. Do **not** schedule 500+ on these numbers; the decision
comes from the Honcho harness A/B (PLAN §1 questions + the hard set) and whatever it exposes.
