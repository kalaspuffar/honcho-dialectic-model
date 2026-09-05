#!/usr/bin/env python3
"""train_dialectic.py — Unsloth SFT -> optional DPO -> GGUF export for Honcho dialectic model.

v0.5.1 (2026-09-04) — after 2 node7 trl-compat bugs:
  * SFT uses plain transformers.Trainer + pre-tokenized rows with manual label masking.
    No trl dependency. Loss is on the assistant answer only.
  * DPO is implemented manually (no trl.DPOTrainer) as a Trainer subclass that
    computes seqlogp and the DPO loss from beta * (dchosen - drejected). Uses
    PEFT's disable_adapter_layers to get the reference policy cheaply.
  * `import unsloth` is still the first import (required by Unsloth's patches).
  * `MODEL_ALIASES` maps common Ollama tags (qwen3.5:9b, qwen3:8b) to the real
    HuggingFace repo ids (Qwen/Qwen3.5-9B, Qwen/Qwen3-8B).

Usage (on node7, GPU host):
  python3 train_dialectic.py --stage sft  --data smoke10_sft.jsonl --out smoke
  python3 train_dialectic.py --stage dpo  --sft smoke/merged --data smoke10_dpo.jsonl --out dpo
  python3 train_dialectic.py --stage export --model smoke/merged --out smoke-v0
"""
import unsloth  # noqa: F401  — MUST be first (Unsloth patches transformers/peft on load)
import argparse, json, os, sys

from unsloth import FastLanguageModel  # noqa: E402

MODEL_ALIASES = {
    # Qwen/Qwen3.5-9B is an image-text-to-text (VL) model — Unsloth loads a
    # Qwen3VLProcessor on it and chokes on text-only rows (attempt-3). Do NOT
    # point a text SFT/DPO at it. The Qwen3.5 9B class has no official text-only
    # build at the Qwen org, so the smoke anchors on the deriver-proven text base.
    "qwen3.5:9b":  "Qwen/Qwen3-8B",   # -> text-only Qwen3-8B (see note above)
    "qwen3:8b":    "Qwen/Qwen3-8B",   # deriver-proven text base (PLAN §3.4)
    "qwen3.5:4b":  "Qwen/Qwen3-8B",
    "qwen3.6:27b": "Qwen/Qwen3-8B",
    # If you specifically need the 9B class, a community text-only build exists:
    # "principled-intelligence/Qwen3.5-9B-text-only" / "techwithsergiu/Qwen3.5-text-9B"
    "unsloth/qwen3.5-9b-gguf": "Qwen/Qwen3-8B",
}

def resolve_base(name: str) -> str:
    n = (name or "").strip()
    if n.endswith(".gguf") or n.startswith("/"):
        return n
    return MODEL_ALIASES.get(n, n)

def load_model(base, max_seq_length=8192, bits=16):
    # load_in_4bit / dtype are the stable documented knobs; guard with a
    # try/except TypeError so an unexpected signature on this build (2026.9.2)
    # can't kill the run — the base repo is bf16 either way, we just prefer 4-bit
    # on a 12 GB card.
    kwargs = dict(
        model_name=base,
        max_seq_length=max_seq_length,
        dtype=None,
        token=None,
    )
    try:
        return FastLanguageModel.from_pretrained(**kwargs, **({"load_in_4bit": bits == 4} if bits is not None else {}))
    except TypeError as e:
        if "load_in_4bit" in str(e):
            print(f"[load] load_in_4bit kwarg rejected ({e}); retrying without", flush=True)
            return FastLanguageModel.from_pretrained(**kwargs)
        raise

def add_lora(model, r=16):
    # NOTE: unsloth 2026.9.2's get_peft_model has no use_fast_lora flag (checked
    # against the official QLoRA guide). The crash in attempt 4 (fast_lora.py
    # matmul_lora illegal memory access) is addressed here by keeping the base at
    # 4-bit (no CPU offload) and capping max_seq_length, not by patching kernels.
    return FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

# ---------- SFT ----------
def run_sft(data, out, base, epochs=3, max_seq_length=4096, bits=4, max_seq=None):
    from torch.utils.data import Dataset as TDDataset
    from transformers import Trainer, TrainingArguments
    base = resolve_base(base)
    print(f"[sft] base = {base}")
    if max_seq is not None: max_seq_length = max_seq
    model, tokenizer = load_model(base, max_seq_length, bits)
    model = add_lora(model)
    print(f"[sft] base={base} bits={bits} max_seq={max_seq_length}")

    rows = [json.loads(l) for l in open(data) if l.strip()]
    samples = []
    for r in rows:
        msgs = r["messages"]
        assert msgs[-1]["role"] == "assistant", "expected [system,user,assistant]"
        full_text   = tokenizer.apply_chat_template(msgs, tokenize=False)
        prompt_text = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        enc = tokenizer(full_text, truncation=True, max_length=max_seq_length)
        input_ids, at_mask = enc["input_ids"], enc["attention_mask"]
        prompt_len = len(tokenizer(prompt_text, truncation=True, max_length=max_seq_length)["input_ids"])
        labels = [-100] * len(input_ids)
        labels[prompt_len:] = input_ids[prompt_len:]
        samples.append({"input_ids": input_ids, "attention_mask": at_mask, "labels": labels})
    print(f"[sft] prepared {len(samples)} samples (answer-only loss)")

    class SFTDS(TDDataset):
        def __init__(self, items): self.items = items
        def __len__(self): return len(self.items)
        def __getitem__(self, i): return self.items[i]
    ds = SFTDS(samples)
    train_ds, test_ds = (SFTDS(samples[:7]), SFTDS(samples[7:])) if len(samples) > 8 else (ds, None)

    args = TrainingArguments(
        output_dir=out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        warmup_steps=2,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_strategy="steps" if test_ds else "no",
        eval_steps=5,
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        report_to="none",
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=0.3,
        seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                      eval_dataset=test_ds, processing_class=tokenizer)
    trainer.train()
    adapter_dir  = os.path.join(out, "adapter")
    merged_dir   = os.path.join(out, "merged")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"[sft] DONE  adapter={adapter_dir}  merged_hf={merged_dir}")
    return merged_dir

# ---------- DPO ----------
def run_dpo(data, out, base, beta=0.1, epochs=1, max_seq_length=4096, bits=4, max_seq=None):
    """Trainers-agnostic DPO. No trl. Loss = -logsigmoid( beta*(d_chosen - d_rejected) )
    where d = seqlogp(policy) - seqlogp(reference), length-normalized."""
    import torch
    from torch.utils.data import Dataset as TDDataset
    from transformers import Trainer, TrainingArguments
    base = resolve_base(base)
    if max_seq is not None: max_seq_length = max_seq
    print(f"[dpo] base={base} beta={beta} bits={bits} max_seq={max_seq_length}")
    model, tokenizer = load_model(base, max_seq_length, bits)
    model = add_lora(model)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    def tokenize(prompt_msgs, completion_text):
        p = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full = p + completion_text + tokenizer.eos_token
        enc  = tokenizer(full, truncation=True, max_length=max_seq_length)
        input_ids, at_mask = enc["input_ids"], enc["attention_mask"]
        labels = [-100] * len(input_ids)
        pl = len(tokenizer(p, truncation=True, max_length=max_seq_length)["input_ids"])
        labels[pl:] = input_ids[pl:]
        return {"input_ids": input_ids, "attention_mask": at_mask, "labels": labels}

    rows = [json.loads(l) for l in open(data) if l.strip()]
    pairs = [(tokenize(r["prompt"], r["chosen"]), tokenize(r["prompt"], r["rejected"])) for r in rows]
    print(f"[dpo] {len(pairs)} chosen/rejected pairs")

    class DPOTS(TDDataset):
        def __len__(self): return len(pairs)
        def __getitem__(self, i):
            c, j = pairs[i]
            return {
                "c_input_ids": c["input_ids"], "c_attn": c["attention_mask"], "c_labels": c["labels"],
                "j_input_ids": j["input_ids"], "j_attn": j["attention_mask"], "j_labels": j["labels"],
            }
    ds = DPOTS()

    def seqlogp(model, input_ids, at_mask, labels):
        out = model(input_ids=input_ids, attention_mask=at_mask)
        logp = torch.log_softmax(out.logits.float(), dim=-1)
        tgt  = torch.where(labels == -100, torch.zeros_like(labels), labels)
        tok  = logp.gather(-1, tgt.unsqueeze(-1)).squeeze(-1)
        valid = (labels != -100).float()
        return (tok * valid).sum(-1), valid.sum(-1)

    class DPO(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kw):
            ci, ca, cl = inputs["c_input_ids"], inputs["c_attn"], inputs["c_labels"]
            ji, ja, jl = inputs["j_input_ids"], inputs["j_attn"], inputs["j_labels"]
            with torch.no_grad():
                model.disable_adapter_layers()
                try:
                    r_chosen,   r_cn = seqlogp(model, ci, ca, cl)
                    r_rejected, r_jn = seqlogp(model, ji, ja, jl)
                finally:
                    model.enable_adapter_layers()
            pol_chosen,   pcn = seqlogp(model, ci, ca, cl)
            pol_rejected, pjm = seqlogp(model, ji, ja, jl)
            # length-normalized log probability difference (stable when chosen is short,
            # rejected is long — our DPO data)
            d_c = (pol_chosen   - r_chosen)   / torch.clamp(pcn, min=1.0)
            d_j = (pol_rejected - r_rejected) / torch.clamp(pjm, min=1.0)
            margin = beta * (d_c - d_j)
            loss   = -torch.nn.functional.logsigmoid(margin).mean()
            if self.state.global_step % 5 == 0:
                print(f"  [dpo step {self.state.global_step}] loss={loss.item():.4f} margin={margin.mean().item():.4f}", flush=True)
            return loss
    args = TrainingArguments(
        output_dir=out,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        warmup_steps=2,
        num_train_epochs=epochs,
        learning_rate=5e-7,
        lr_scheduler_type="cosine",
        logging_steps=1,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=0.3,
        seed=42,
        fp16=False,
    )
    trainer = DPO(model=model, args=args, train_dataset=ds, processing_class=tokenizer)
    trainer.train()
    adapter_dir = os.path.join(out, "adapter")
    merged_dir  = os.path.join(out, "merged")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    merged = model.merge_and_unload()
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"[dpo] DONE  adapter={adapter_dir}  merged_hf={merged_dir}")
    return merged_dir

# ---------- GGUF export ----------
def run_export(hf_dir, out, bits=4):
    base = resolve_base(hf_dir)
    print(f"[export] hf = {base}  quant = Q{bits}_K_M")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base, max_seq_length=8192, dtype=None, token=None)
    os.makedirs(out, exist_ok=True)
    # save_pretrained_gguf signature differs across unsloth builds;
    # try the modern dict-form first, fall back to positional bit count,
    # then to a quant-method-only kwarg
    try:
        model.save_pretrained_gguf(out, {"quantization_bit": bits})  # >=2025
    except TypeError:
        try:
            model.save_pretrained_gguf(out, bits)                    # older positional
        except TypeError:
            model.save_pretrained_gguf(out, {"quantization_method": f"Q{bits}_K_M"})
    tokenizer.save_pretrained(os.path.join(out, "tokenizer"))
    print(f"[export] DONE  GGUF in {out} — Modelfile FROM points at the .gguf file")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["sft", "dpo", "export"])
    ap.add_argument("--data", help="jsonl (sft or dpo rows)")
    ap.add_argument("--sft",  help="path to merged HF dir (dpo stage: smoke/merged)")
    ap.add_argument("--model", default="Qwen/Qwen3-8B",
                    help="HF repo id. Default is the text-only Qwen3-8B (deriver-proven). "
                         "qwen3.5:9b maps to Qwen3-8B because Qwen/Qwen3.5-9B is a VL model.")
    ap.add_argument("--out", required=True, help="output dir on node7")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-seq", type=int, default=4096,
                    help="max tokens per sample. 4096 for 12 GB cards; raise to 8192 on 48 GB")
    ap.add_argument("--load-bits", type=int, default=4, choices=[4, 16],
                    help="load base at 4-bit (fit 12 GB 3080 Ti) or 16-bit (needs ~24 GB)")
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8], help="GGUF bits for export")
    a = ap.parse_args()
    if a.stage == "sft":
        if not a.data: sys.exit("--data required for sft")
        run_sft(data=a.data, out=a.out, base=a.model, epochs=a.epochs,
        bits=a.load_bits, max_seq=a.max_seq)
    elif "dpo" == a.stage:
        if not a.sft or not a.data: sys.exit("--sft and --data required for dpo")
        run_dpo(data=a.data, out=a.out, base=a.sft, beta=a.dpo_beta, epochs=a.epochs,
        bits=a.load_bits, max_seq=a.max_seq)
    elif a.stage == "export":
        if not a.model: sys.exit("--model (merged HF dir) required for export")
        run_export(a.model, a.out, a.bits)
    print("DONE — next gate: TRAIN.md §4 (30-context eval via eval_dialectic.py)")

if __name__ == "__main__":
    main()
