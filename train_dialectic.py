#!/usr/bin/env python3
"""train_dialectic.py — Unsloth QLoRA SFT -> DPO -> GGUF for the Honcho dialectic model.

Stages
  check   tokenizer only, no GPU: token lengths of a dataset, how many rows exceed --max-seq,
          and the exact trainable tail of one row (sanity-check the label mask)
  strip   official Qwen3.5 VL checkpoint -> text-only checkpoint (CPU RAM)
  sft     supervised warm-up on the chosen answers (loss on the final assistant turn only)
  dpo     preference training on chosen/rejected pairs (hand-rolled, no trl)
  export  merged HF dir -> GGUF Q4_K_M for `ollama create`

Usage (GPU host, Unsloth venv):
  python3 train_dialectic.py --stage check  --model Qwen/Qwen3-8B --data data/dataset_train.sft.jsonl
  python3 train_dialectic.py --stage sft    --model Qwen/Qwen3-8B --data data/dataset_train.sft.jsonl \
                                            --eval-data data/dataset_eval.sft.jsonl --out runs/v1-sft
  python3 train_dialectic.py --stage dpo    --sft runs/v1-sft/merged --data data/dataset_train.dpo.jsonl --out runs/v1-dpo
  python3 train_dialectic.py --stage export --model runs/v1-dpo/merged --out runs/v1-gguf
  12 GB card: --load-bits 4 --max-seq 6144.   48 GB A6000: --load-bits 16 --max-seq 8192.

Data format (build_dataset.py): SFT rows {"messages": [system, user, assistant(tool_calls), tool, ..., assistant], "tools": [...]}
DPO rows {"prompt": [...same prefix...], "tools": [...], "chosen": str, "rejected": str}. The chat
template renders tool calls / tool results, so training text matches what Ollama renders at runtime.

History (TRAIN.md §7): v0.8.0 (2026-09-10) fixed the SFT split that trained every run on the
first 7 rows, raised the DPO learning rate to a LoRA-appropriate value, made the DPO loss the
standard summed log-prob form, merged adapters to 16-bit, and moved to trajectory-shaped rows.
"""
try:
    import unsloth  # noqa: F401  — MUST precede any transformers/peft import (Unsloth patches them)
    from unsloth import FastLanguageModel
except ImportError:                  # lets `--stage check` and verify_pipeline.py import this file
    unsloth = FastLanguageModel = None

import argparse
import json
import os
import random
import sys

STRIP_DEFAULT_REPO = "Qwen/Qwen3.5-9B"
MODEL_ALIASES = {
    "qwen3:8b": "Qwen/Qwen3-8B",      # deriver-proven text base (PLAN §3.4)
    "qwen3.5:9b": "Qwen/Qwen3-8B",    # the Hub 9B is a VL checkpoint; run --stage strip for the real 9B text backbone
}
SFT_DEFAULTS = dict(epochs=3, lr=2e-4)
DPO_DEFAULTS = dict(epochs=2, lr=1e-5)   # LoRA DPO: 5e-7 (v0.7) is a full-fine-tune value and moved nothing


def resolve_base(name: str) -> str:
    n = (name or "").strip()
    if n.endswith(".gguf") or n.startswith("/") or os.path.isdir(n):
        return n
    return MODEL_ALIASES.get(n, n)


# ----------------------------------------------------------------- data prep
def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def encode_example(tokenizer, prefix_msgs, answer, tools, max_len):
    """Tokenize prefix + answer through the chat template; labels = answer tokens only.
    Returns None when the full sequence exceeds max_len (we never truncate the answer)."""
    msgs = list(prefix_msgs) + [{"role": "assistant", "content": answer}]
    kw = {"tools": tools} if tools else {}
    full = tokenizer.apply_chat_template(msgs, tokenize=False, **kw)
    prompt = tokenizer.apply_chat_template(prefix_msgs, tokenize=False, add_generation_prompt=True, **kw)
    full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if len(full_ids) > max_len:
        return None
    k = 0
    while k < min(len(full_ids), len(prompt_ids)) and full_ids[k] == prompt_ids[k]:
        k += 1
    labels = [-100] * k + full_ids[k:]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


def prepare_sft(rows, tokenizer, max_len):
    samples, dropped = [], 0
    for r in rows:
        msgs = r["messages"]
        assert msgs[-1]["role"] == "assistant", "SFT row must end with the assistant answer"
        enc = encode_example(tokenizer, msgs[:-1], msgs[-1]["content"], r.get("tools"), max_len)
        if enc is None:
            dropped += 1
        else:
            samples.append(enc)
    return samples, dropped


def prepare_dpo(rows, tokenizer, max_len):
    pairs, dropped = [], 0
    for r in rows:
        c = encode_example(tokenizer, r["prompt"], r["chosen"], r.get("tools"), max_len)
        j = encode_example(tokenizer, r["prompt"], r["rejected"], r.get("tools"), max_len)
        if c is None or j is None:
            dropped += 1
        else:
            pairs.append((c, j))
    return pairs, dropped


def pad_collator(pad_id=None):
    """Right-pad ragged rows to batch max; labels with -100, ids with pad_id, masks with 0.
    Returns int64 tensors (Unsloth's get_batch_samples slices labels with an ellipsis)."""
    def collate(batch):
        out = {}
        for k in batch[0]:
            rows = [b[k] for b in batch]
            L = max(len(r) for r in rows)
            if k == "labels" or k.endswith("_labels"):
                fill = -100
            elif k in ("input_ids", "c_input_ids", "j_input_ids"):
                fill = pad_id if pad_id is not None else 0
            else:
                fill = 0
            padded = [r + [fill] * (L - len(r)) for r in rows]
            try:
                import torch
                out[k] = torch.tensor(padded, dtype=torch.int64)
            except Exception:
                out[k] = padded
        return out
    return collate


# ----------------------------------------------------------------- model io
def load_model(base, max_seq_length, bits):
    kwargs = dict(model_name=base, max_seq_length=max_seq_length, dtype=None, token=None)
    try:
        return FastLanguageModel.from_pretrained(**kwargs, load_in_4bit=(bits == 4))
    except TypeError as e:
        if "load_in_4bit" in str(e):
            return FastLanguageModel.from_pretrained(**kwargs)
        raise


def add_lora(model, r=16):
    return FastLanguageModel.get_peft_model(
        model, r=r, lora_alpha=32, lora_dropout=0, bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth")


def save_outputs(model, tokenizer, out):
    """adapter/ + merged/ (16-bit). Unsloth's save_pretrained_merged dequantizes a 4-bit base
    properly; peft's merge_and_unload on a 4-bit model re-quantizes the merged weights (rounding
    error on top of the LoRA delta) — used only as a fallback."""
    adapter_dir, merged_dir = os.path.join(out, "adapter"), os.path.join(out, "merged")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    fn = getattr(model, "save_pretrained_merged", None)
    if fn is not None:
        try:
            fn(merged_dir, tokenizer, save_method="merged_16bit")
            print(f"[save] adapter={adapter_dir} merged_16bit={merged_dir}")
            return merged_dir
        except Exception as e:  # noqa: BLE001
            print(f"[save] save_pretrained_merged failed ({type(e).__name__}: {e}); falling back to merge_and_unload")
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"[save] adapter={adapter_dir} merged={merged_dir} (merge_and_unload)")
    return merged_dir


class ListDS:
    def __init__(self, items): self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]


# ----------------------------------------------------------------- check
def run_check(base, data, max_seq):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(resolve_base(base))
    rows = read_jsonl(data)
    is_dpo = "prompt" in rows[0]
    if is_dpo:
        pairs, dropped = prepare_dpo(rows, tok, max_seq)
        lens = [len(c["input_ids"]) for c, _ in pairs] + [len(j["input_ids"]) for _, j in pairs]
        ex = pairs[0][0] if pairs else None
        kept = len(pairs)
    else:
        samples, dropped = prepare_sft(rows, tok, max_seq)
        lens = [len(s["input_ids"]) for s in samples]
        ex = samples[0] if samples else None
        kept = len(samples)
    lens.sort()
    print(f"[check] {data}: {len(rows)} rows, kept {kept}, dropped {dropped} (> {max_seq} tokens)")
    if lens:
        print(f"[check] tokens/sequence: min {lens[0]}  median {lens[len(lens)//2]}  p95 {lens[int(len(lens)*0.95)]}  max {lens[-1]}")
    if ex:
        tail = [t for t, l in zip(ex["input_ids"], ex["labels"]) if l != -100]
        print(f"[check] trainable tokens in first row: {len(tail)} of {len(ex['input_ids'])}")
        print("[check] trainable text:", repr(tok.decode(tail)))


# ----------------------------------------------------------------- SFT
def run_sft(data, out, base, epochs, lr, max_seq, bits, eval_data=None, seed=42):
    from transformers import Trainer, TrainingArguments
    base = resolve_base(base)
    print(f"[sft] base={base} bits={bits} max_seq={max_seq} epochs={epochs} lr={lr}")
    model, tokenizer = load_model(base, max_seq, bits)
    model = add_lora(model)

    rows = read_jsonl(data)
    random.Random(seed).shuffle(rows)
    train_s, dropped = prepare_sft(rows, tokenizer, max_seq)
    if eval_data:
        eval_s, ed = prepare_sft(read_jsonl(eval_data), tokenizer, max_seq)
        dropped += ed
    elif len(train_s) >= 20:                       # hold out 5% (cap 64) when no eval file
        n_eval = min(64, max(1, len(train_s) // 20))
        train_s, eval_s = train_s[n_eval:], train_s[:n_eval]
    else:
        eval_s = []
    print(f"[sft] train={len(train_s)} eval={len(eval_s)} dropped(too long)={dropped}  (answer-only loss)")
    if not train_s:
        sys.exit("no training samples fit --max-seq; raise it")

    args = TrainingArguments(
        output_dir=out, per_device_train_batch_size=1, gradient_accumulation_steps=4,
        warmup_steps=max(2, len(train_s) // 40), num_train_epochs=epochs, learning_rate=lr,
        lr_scheduler_type="cosine", logging_steps=5,
        eval_strategy="epoch" if eval_s else "no", per_device_eval_batch_size=1,
        save_strategy="epoch", bf16=True, gradient_checkpointing=True, report_to="none",
        optim="adamw_8bit", weight_decay=0.01, max_grad_norm=0.3, seed=seed)
    trainer = Trainer(model=model, args=args, train_dataset=ListDS(train_s),
                      eval_dataset=ListDS(eval_s) if eval_s else None, processing_class=tokenizer,
                      data_collator=pad_collator(pad_id=tokenizer.pad_token_id))
    trainer.train()
    return save_outputs(model, tokenizer, out)


# ----------------------------------------------------------------- DPO
def run_dpo(data, out, base, beta, epochs, lr, max_seq, bits, length_norm=False, seed=42):
    """DPO without trl. loss = -logsigmoid(beta * ((pi_c - ref_c) - (pi_j - ref_j))), sequence
    log-probs SUMMED over answer tokens (standard DPO). Reference = same weights with the
    adapter disabled. --dpo-length-norm divides by answer length instead (v0.7 behaviour), which
    removes most of the length signal we are training for."""
    import torch
    from transformers import Trainer, TrainingArguments
    base = resolve_base(base)
    print(f"[dpo] base={base} beta={beta} lr={lr} epochs={epochs} bits={bits} max_seq={max_seq} length_norm={length_norm}")
    model, tokenizer = load_model(base, max_seq, bits)
    model = add_lora(model)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    rows = read_jsonl(data)
    random.Random(seed).shuffle(rows)
    pairs, dropped = prepare_dpo(rows, tokenizer, max_seq)
    print(f"[dpo] {len(pairs)} pairs, dropped(too long)={dropped}")
    if not pairs:
        sys.exit("no pairs fit --max-seq; raise it")

    class PairDS:
        def __len__(self): return len(pairs)
        def __getitem__(self, i):
            c, j = pairs[i]
            return {"c_input_ids": c["input_ids"], "c_attn": c["attention_mask"], "c_labels": c["labels"],
                    "j_input_ids": j["input_ids"], "j_attn": j["attention_mask"], "j_labels": j["labels"]}

    def seqlogp(m, ids, attn, labels):
        logits = m(input_ids=ids, attention_mask=attn).logits[:, :-1].float()
        tgt = labels[:, 1:]
        valid = (tgt != -100)
        tok = torch.log_softmax(logits, -1).gather(-1, torch.where(valid, tgt, 0).unsqueeze(-1)).squeeze(-1)
        n = valid.sum(-1).clamp(min=1)
        s = (tok * valid).sum(-1)
        return (s / n) if length_norm else s

    class DPO(Trainer):
        def get_batch_samples(self, epoch_iterator, num_batches, device=None, *a, **kw):
            # stock behaviour; Unsloth's patched version drops our c_*/j_* keys
            batch = []
            for _ in range(num_batches):
                try:
                    batch.append(next(epoch_iterator))
                except StopIteration:
                    break
            return batch, None

        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            ci, ca, cl = inputs["c_input_ids"], inputs["c_attn"], inputs["c_labels"]
            ji, ja, jl = inputs["j_input_ids"], inputs["j_attn"], inputs["j_labels"]
            with torch.no_grad():
                model.disable_adapter_layers()
                try:
                    ref_c, ref_j = seqlogp(model, ci, ca, cl), seqlogp(model, ji, ja, jl)
                finally:
                    model.enable_adapter_layers()
            pol_c, pol_j = seqlogp(model, ci, ca, cl), seqlogp(model, ji, ja, jl)
            margin = beta * ((pol_c - ref_c) - (pol_j - ref_j))
            loss = -torch.nn.functional.logsigmoid(margin).mean()
            if self.state.global_step % 5 == 0:
                print(f"  [dpo step {self.state.global_step}] loss={loss.item():.4f} margin={margin.mean().item():.3f} "
                      f"acc={(margin > 0).float().mean().item():.2f}", flush=True)
            return loss

    args = TrainingArguments(
        output_dir=out, per_device_train_batch_size=1, gradient_accumulation_steps=8,
        remove_unused_columns=False,   # keep c_*/j_* keys (RemoveColumnsCollator strips non-forward() args)
        warmup_steps=max(2, len(pairs) // 80), num_train_epochs=epochs, learning_rate=lr,
        lr_scheduler_type="cosine", logging_steps=5, save_strategy="epoch", bf16=True, fp16=False,
        report_to="none", optim="adamw_8bit", weight_decay=0.0, max_grad_norm=1.0, seed=seed)
    trainer = DPO(model=model, args=args, train_dataset=PairDS(), processing_class=tokenizer,
                  data_collator=pad_collator(pad_id=tokenizer.pad_token_id))
    trainer.train()
    return save_outputs(model, tokenizer, out)


# ----------------------------------------------------------------- strip / export
def run_strip(repo, out):
    """Official Qwen3.5 VL checkpoint -> text-only Qwen3_5ForCausalLM (drops model.visual.* / mtp.*)."""
    import torch
    try:
        from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM
    except ImportError as e:
        sys.exit("Qwen3_5ForCausalLM missing — need transformers >= 5.2.\n  " + str(e))
    print(f"[strip] {repo} -> {out}")
    model = Qwen3_5ForCausalLM.from_pretrained(repo, torch_dtype=torch.bfloat16)
    n = sum(p.numel() for p in model.parameters())
    print(f"[strip] text parameters: {n/1e9:.2f}e9")
    os.makedirs(out, exist_ok=True)
    model.save_pretrained(out, safe_serialization=True)
    from transformers import AutoTokenizer
    AutoTokenizer.from_pretrained(repo).save_pretrained(out)
    print(f"[strip] DONE {out}")


def run_export(hf_dir, out, bits=4):
    import inspect
    base = resolve_base(hf_dir)
    print(f"[export] {base} -> {out} q{bits}_k_m")
    model, tokenizer = FastLanguageModel.from_pretrained(model_name=base, max_seq_length=8192, dtype=None, token=None)
    os.makedirs(out, exist_ok=True)
    # tokenizer is the 2nd POSITIONAL; quant kwarg name differs across Unsloth releases -> introspect the bound method
    params = inspect.signature(model.save_pretrained_gguf).parameters
    if "quantization_method" in params:
        model.save_pretrained_gguf(out, tokenizer, quantization_method=f"q{bits}_k_m")
    elif "quantization_bit" in params:
        model.save_pretrained_gguf(out, tokenizer, quantization_bit=bits)
    else:
        model.save_pretrained_gguf(out, tokenizer)
    print(f"[export] DONE — point Modelfile FROM at the .gguf in {out}, then `ollama create <name> -f Modelfile`")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True, choices=["check", "sft", "dpo", "export", "strip"])
    ap.add_argument("--data", help="jsonl (sft or dpo rows)")
    ap.add_argument("--eval-data", help="sft: held-out sft rows (build_dataset *_eval.sft.jsonl)")
    ap.add_argument("--sft", help="dpo: merged HF dir from the sft stage")
    ap.add_argument("--model", default="Qwen/Qwen3-8B", help="HF repo id / local dir (sft, check, export, strip)")
    ap.add_argument("--out", help="output dir")
    ap.add_argument("--epochs", type=int, default=None, help=f"default sft {SFT_DEFAULTS['epochs']}, dpo {DPO_DEFAULTS['epochs']}")
    ap.add_argument("--lr", type=float, default=None, help=f"default sft {SFT_DEFAULTS['lr']}, dpo {DPO_DEFAULTS['lr']}")
    ap.add_argument("--max-seq", type=int, default=6144, help="max tokens per sequence; longer rows are DROPPED, never truncated")
    ap.add_argument("--load-bits", type=int, default=4, choices=[4, 16])
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--dpo-length-norm", action="store_true", help="per-token normalised DPO (v0.7 behaviour)")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8], help="GGUF quant bits (export)")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    if a.stage == "check":
        if not a.data: sys.exit("--data required")
        return run_check(a.model, a.data, a.max_seq)
    if a.stage != "check" and FastLanguageModel is None and a.stage != "strip":
        sys.exit("unsloth is not installed in this environment (only --stage check / strip work without it)")
    if not a.out:
        sys.exit("--out required")
    if a.stage == "strip":
        repo = a.model if a.model != "Qwen/Qwen3-8B" else STRIP_DEFAULT_REPO
        run_strip(repo, a.out)
    elif a.stage == "sft":
        if not a.data: sys.exit("--data required for sft")
        run_sft(a.data, a.out, a.model, a.epochs or SFT_DEFAULTS["epochs"], a.lr or SFT_DEFAULTS["lr"],
                a.max_seq, a.load_bits, a.eval_data, a.seed)
    elif a.stage == "dpo":
        if not a.sft or not a.data: sys.exit("--sft and --data required for dpo")
        run_dpo(a.data, a.out, a.sft, a.dpo_beta, a.epochs or DPO_DEFAULTS["epochs"], a.lr or DPO_DEFAULTS["lr"],
                a.max_seq, a.load_bits, a.dpo_length_norm, a.seed)
    elif a.stage == "export":
        run_export(a.model, a.out, a.bits)
    print("DONE")


if __name__ == "__main__":
    main()
