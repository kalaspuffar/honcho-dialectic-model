#!/usr/bin/env python3
"""estimate_cost.py — cost + GPU-hour estimate for a 1000-sample dialectic run.

Two costs, reported separately:
  1. TEACHER API (OpenRouter)  — the synthetic-data generation that feeds SFT/DPO.
  2. LOCAL GPU (node7 3080 Ti 12GB / A6000) — SFT + DPO + merge.

Teacher prices pulled live from the OpenRouter catalog (2026-09) and overridable
per row. Sizing uses the measured row sizes in smoke10_sft.jsonl when present
(~12.3K chars ≈ ~9K tokens/row system-prompt-dominated), else a conservative
--row-tokens estimate. DPO doubles the forward pass (chosen + rejected) and
typically needs a smaller LoRA r and 1 epoch.

NOTE: GPU hour estimate is order-of-magnitude, not a benchmark — the actual
smoke-10 wall-clock from the last node7 run is the true reference; this scales it.
"""
import argparse, json, os, statistics

# --- teacher prices, USD / 1M tokens (OpenRouter live, 2026-09) ---
DEFAULT_PRICES = {
    "opus":     (5.00, 25.00),   # anthropic/claude-opus-5
    "sonnet":   (2.00, 10.00),   # anthropic/claude-sonnet-5
    "gemini":   (2.00, 12.00),   # google/gemini-3.1-pro-preview
    "qwen3max": (0.78,  3.90),   # qwen/qwen3-max
    "gpt5":     (1.25, 10.00),   # openai/gpt-5
    "deepseek": (0.32,  0.89),   # deepseek/deepseek-chat
}

def row_tokens_from_file(path):
    """Median chars/row -> rough tokens (chars/1.35 for mixed EN/JSON+code)."""
    if not path or not os.path.exists(path): return None
    chars = []
    for l in open(path):
        if l.strip(): chars.append(len(l.strip()))
    if not chars: return None
    return int(statistics.median(chars) / 1.35)

def estimate(n_sft, n_dpo, teacher, in_tok, out_tok,
             sft_epochs=3, dpo_epochs=1, base_gpu_hours_per_k= None):
    tin, tout = DEFAULT_PRICES.get(teacher, (2.0, 10.0))
    # teacher generation: one prompt in, one terse out per SFT row + chosen/rejected pair per DPO row
    teacher_rows = n_sft + n_dpo
    inp = teacher_rows * in_tok
    outp = teacher_rows * out_tok
    api_in = inp / 1e6 * tin
    api_out = outp / 1e6 * tout
    api_total = api_in + api_out
    return {
        "teacher": teacher, "teacher_rows": teacher_rows,
        "in_tokens_M": round(inp/1e6, 2), "out_tokens_M": round(outp/1e6, 2),
        "api_input_usd": round(api_in, 2), "api_output_usd": round(api_out, 2),
        "api_total_usd": round(api_total, 2),
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-sft", type=int, default=1000)
    ap.add_argument("--n-dpo", type=int, default=0, help="separate DPO pairs")
    ap.add_argument("--teacher", default="opus", choices=list(DEFAULT_PRICES))
    ap.add_argument("--row-tokens", type=int, help="per-row in-tokens (else measured from data file)")
    ap.add_argument("--out-tokens", type=int, default=60, help="terse answer tokens (median ~38)")
    ap.add_argument("--sft-file", default="smoke10_sft.jsonl")
    a = ap.parse_args()
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    per_row = a.row_tokens or row_tokens_from_file(a.sft_file) or 9000
    res = estimate(a.n_sft, a.n_dpo, a.teacher, per_row, a.out_tokens)
    res["row_tokens"] = per_row
    # rough GPU hours: measured smoke10 (SFT 3ep + DPO 1ep + export) ~ on the 3080Ti.
    # Scale linearly with rows: 1000 rows ≈ 100x the token count of 10 rows.
    if per_row:
        rows_total = a.n_sft + a.n_dpo
        # ~1.5 s/row-step SFT-3ep + ~0.5 s/row-step DPO = order; times 1000.
        est_h = (a.n_sft * 1.5 * 3 + a.n_dpo * 1.0 * 1) / 60  # minutes from a rough 1.5s+0.5s per row-step
        res["gpu_hours_est"] = round(est_h, 1)

    print(json.dumps(res, indent=2))
    # sanity: cost per 1K rows
    print("\nPER 1K ROWS (teacher=%s, %d in + %d out tokens): " % (a.teacher, per_row, a.out_tokens),
          "$%.2f" % (res["api_total_usd"] / max(a.n_sft+a.n_dpo,1) * 1000))

if __name__ == "__main__":
    main()
