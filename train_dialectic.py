#!/usr/bin/env python3
"""train_dialectic.py — Unsloth SFT -> optional DPO -> GGUF export for Honcho dialectic model.

v0.5.0 changes (after first node7 attempt):
  * `import unsloth` is now the FIRST import (Unsloth must patch before transformers/peft load).
  * BASE is a HuggingFace repo id, not an Ollama tag. Ollama tags in ~/github/honcho-dialectic-model
    fail:  qwen3.5:9b  ->  Qwen/Qwen3.5-9B     (alias table below auto-maps common tags)
  * SFT now uses trl.SFTTrainer + DataCollatorForCompletionOnlyLM so the loss is computed
    ONLY on the assistant answer (prompt tokens are masked) — the previous version masked nothing.
  * Export writes a 4-bit GGUF via model.save_pretrained_gguf().

Usage (on node7, the local GPU host):
  python3 train_dialectic.py --stage sft  --data smoke10_sft.jsonl --out smoke
  python3 train_dialectic.py --stage dpo  --sft smoke/merged --data smoke10_dpo.jsonl --out dpo
  python3 train_dialectic.py --stage export --model smoke/merged --out smoke-v0    # 4-bit GGUF
"""
import unsloth  # noqa: F401  — MUST be the very first import (Unsloth patches transformers/peft on load)
import argparse, json, os, sys

from unsloth import FastLanguageModel  # noqa: E402

# Ollama tag -> HuggingFace repo id  (Unsloth loads from HF, not Ollama)
MODEL_ALIASES = {
    "qwen3.5:9b": "Qwen/Qwen3.5-9B",
    "qwen3:8b":   "Qwen/Qwen3-8B",     # deriver-proven fallback base (PLAN §3.4)
    "qwen3.5:4b": "Qwen/Qwen3.5-4B",
    "qwen3.6:27b": "Qwen/Qwen3.6-27B",
    "unsloth/qwen3.5-9b-gguf": "unsloth/Qwen3.5-9B-GGUF",  # 4-bit GGUF load, less RAM
}

def resolve_base(name: str) -> str:
    n = (name or "").strip()
    if n.endswith(".gguf") or n.startswith("/"):
        return n  # local file path
    return MODEL_ALIASES.get(n, n)

def load_model(base, max_seq_length=8192):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base,
        max_seq_length=max_seq_length,
        dtype=None,          # Unsloth picks bf16
        load_in_8bit=False,
        token=None,          # pull from ~/.cache/huggingface/token if the repo is gated
    )
    return model, tokenizer

def add_lora(model, r=16):
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )
    return model

def prep_chat_rows(data_path):
    """rows: {messages:[system,user,assistant]} -> {prompt(str), completion(str)}"""
    rows = [json.loads(l) for l in open(data_path) if l.strip()]
    return rows

def run_sft(data, out, base, epochs=3, max_seq_length=8192):
    from datasets import Dataset
    from trl import SFTTrainer, DataCollatorForCompletionOnlyLM, SFTConfig
    base = resolve_base(base)
    print(f"[sft] base = {base}")
    model, tokenizer = load_model(base, max_seq_length)
    model = add_lora(model)

    rows = prep_chat_rows(data)
    formatted = []
    for r in rows:
        msgs = r["messages"]
        assert msgs[-1]["role"] == "assistant", "expected [system,user,assistant] rows"
        # apply_chat_template gives prompt+answer as ONE string; completion-only collator
        # masks everything before the completion column's completion marker.
        full = tokenizer.apply_chat_template(msgs, tokenize=False)
        prompt_only = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)
        formatted.append({"prompt": prompt_only, "completion": full[len(prompt_only):]})
    ds = Dataset.from_list(formatted)
    test = None
    if len(ds) > 4:
        ds = ds.train_test_split(test_size=min(2, max(1, len(ds) // 10)), seed=7)
        test = ds["test"]
    train = ds["train"] if "train" in ds else ds
    train = train.train_test_split(test_size=max(1, len(train) // 10), seed=7)["train"] if len(train) > 6 else train

    training_args = SFTConfig(
        output_dir=out,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        warmup_ratio=0.05,
        num_train_epochs=epochs,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        logging_steps=1,
        eval_strategy="steps" if test else "no",
        eval_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        optim="adamw_8bit",
        weight_decay=0.01,
        max_grad_norm=0.3,
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        data_collator=DataCollatorForCompletionOnlyLM(
            completion_columns=["prompt", "completion"],
            label_pad_token_id=-100,
            tokenizer=tokenizer,
        ),
        processing_class=tokenizer,
    )
    trainer.train()
    adapter_dir = os.path.join(out, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    merged = model.merge_and_unload()
    merged_dir = os.path.join(out, "merged")
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"[sft] DONE  adapter={adapter_dir}  merged_hf={merged_dir}")
    return merged_dir

def run_dpo(data, out, sft_merged, beta=0.1, epochs=1, max_seq_length=8192):
    from datasets import Dataset
    from trl import DPOTrainer, DPOConfig
    base = resolve_base(sft_merged)
    print(f"[dpo] base = {base}")
    model, tokenizer = load_model(base, max_seq_length)
    model = add_lora(model)
    rows = [json.loads(l) for l in open(data) if l.strip()]
    adapted = []
    for r in rows:
        prompt = tokenizer.apply_chat_template(r["prompt"], tokenize=False, add_generation_prompt=True)
        adapted.append({"prompt": prompt, "chosen": r["chosen"], "rejected": r["rejected"]})
    ds = Dataset.from_list(adapted)
    test = None
    if len(ds) > 4:
        ds = ds.train_test_split(test_size=min(2, max(1, len(ds) // 10)), seed=7)
        test = ds["test"]; train = ds["train"]
    else:
        train = ds
    cfg = DPOConfig(
        output_dir=out,
        beta=beta,
        loss_type="sigmoid",
        max_length=max_seq_length,
        max_prompt_length=max_seq_length - 2048,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=5e-7,
        num_train_epochs=epochs,
        bf16=True,
        optim="adamw_8bit",
        eval_strategy="steps" if test else "no",
        eval_steps=10,
        logging_steps=1,
        save_strategy="epoch",
        warmup_ratio=0.1,
        report_to="none",
        seed=42,
    )
    trainer = DPOTrainer(model=model, args=cfg, train_dataset=train,
                         eval_dataset=test, processing_class=tokenizer)
    trainer.train()
    adapter_dir = os.path.join(out, "adapter")
    model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    merged = model.merge_and_unload()
    merged_dir = os.path.join(out, "merged")
    merged.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    print(f"[dpo] DONE  merged_hf={merged_dir}")
    return merged_dir

def run_export(hf_dir, out, bits=4):
    base = resolve_base(hf_dir)
    print(f"[export] hf = {base}  quant = Q{bits}_K_M")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base, max_seq_length=8192, dtype=None, token=None)
    os.makedirs(out, exist_ok=True)
    try:
        model.save_pretrained_gguf(out, {"quantization_bit": bits})      # unsloth >= 2025 style
    except TypeError:
        model.save_pretrained_gguf(out, bits)                             # older style: position bit
    tokenizer.save_pretrained(os.path.join(out, "tokenizer"))
    print(f"[export] DONE  GGUF in {out} — point a Modelfile FROM line at the .gguf file there")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["sft", "dpo", "export"])
    ap.add_argument("--data", help="jsonl (sft or dpo rows)")
    ap.add_argument("--sft",  help="path to merged HF dir (dpo stage: smoke/merged)")
    ap.add_argument("--model", default="Qwen/Qwen3.5-9B",
                    help="HF base repo id (qwen3.5:9b / qwen3:8b tags accepted and mapped)")
    ap.add_argument("--out", required=True, help="output dir on node7")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--dpo-beta", type=float, default=0.1)
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8], help="GGUF bits for export")
    a = ap.parse_args()
    if a.stage == "sft":
        if not a.data: sys.exit("--data required for sft")
        run_sft(a.data, a.out, a.model, a.epochs)
    elif a.stage == "dpo":
        if not a.sft or not a.data: sys.exit("--sft and --data required for dpo")
        run_dpo(a.data, a.out, a.sft, a.dpo_beta, a.epochs)
    elif a.stage == "export":
        if not a.model: sys.exit("--model (merged HF dir) required for export")
        run_export(a.model, a.out, a.bits)
    print("DONE — next gate: TRAIN.md §4 (30-context eval via eval_dialectic.py)")

if __name__ == "__main__":
    main()
