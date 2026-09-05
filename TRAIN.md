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
#   qwen3.5:9b    <-- VL model, do NOT use for text DPO (see v0.5.2 notes)
#   qwen3:8b      <-- deriver-proven text base (this is what we train on)
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
| 2026-09-05 | export (attempt 1) | `TypeError: 'dict' object is not callable` (then `'int' object is not callable`) at `fix_tokenizer_bos_token`, `tokenizer("A")` — all three "fallback" call-forms died on the same line for the same reason: `save_pretrained_gguf(save_directory, tokenizer, quantization_bit=None)` — passing `{"quantization_bit":bits}` or a bare `bits` as the 2nd positional puts a non-tokenizer into the tokenizer slot. Fix: pass the real tokenizer as 2nd positional, and pick the quant kwarg by introspecting `inspect.signature(...).parameters` (current unsloth: `quantization_method="q4_k_m"`; older builds: `quantization_bit`) |
| 2026-09-05 | dpo (attempt 1 → 2) | Twice: `ValueError: The batch received was empty` at `transformers/trainer.py _prepare_inputs`, both on the 3080 Ti (12 GB, py3.13) and the A6000 box — same crash both times. First hypothesis (v0.7.5) was Unsloth's `get_batch_samples` patch — partly right (that patch does track only stock keys for its item-count), but not the actual key-dropping site. | v0.7.6 (this commit), root cause found by reading `Trainer._get_dataloader` in the box's transformers 5.5.0: for non-`datasets.Dataset` datasets (our `TDDataset`) it goes through `_get_collator_with_removed_columns`, and with `remove_unused_columns=True` (TrainingArguments default) it wraps our collator in `RemoveColumnsCollator` which strips every key not in Qwen3 `forward()`'s signature — `c_input_ids`/`c_attn`/`c_labels`/`j_input_ids`/`j_attn`/`j_labels` are all dropped, empty dict hits the `len(inputs)==0` guard. SFT survived because its keys are literal signature columns. Fix is the one trl's DPOTrainer sets for exactly this reason: `TrainingArguments(remove_unused_columns=False)`. |

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
