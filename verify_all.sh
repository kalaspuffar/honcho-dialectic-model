#!/usr/bin/env bash
# verify_all.sh — post-training A/B on the held-out split + tool-call probe.
#   BASE=http://localhost:11434 TUNED=dialectic-v1 BASELINE=qwen3.5:9b bash verify_all.sh
# Baseline column: qwen3.5:9b cannot switch thinking off and, through Ollama's /v1, writes its answer
# inside <think> and returns empty content on most rows (TRAIN.md §7, 2026-09-11). Honcho would see
# nothing — that is the base's real behaviour and one reason for the fine-tune. To still get words /
# coverage / fabrication numbers for the base, its eval scores the reasoning text when content is empty
# (--answer-from-reasoning); the summary reports answered_in_thinking_rows. The tuned column never
# gets that fallback. EVAL_BASELINE=qwen9b runs the baseline on OpenRouter instead (needs credits).
set -uo pipefail
cd "$(dirname "$0")"
BASE="${BASE:-http://localhost:11434}"
TUNED="${TUNED:-dialectic-v1}"
BASELINE="${BASELINE:-qwen3.5:9b}"
EVAL_BASELINE="${EVAL_BASELINE:-$BASELINE}"
CTX="${CTX:-data/contexts.jsonl}"
EVAL_IDS="${EVAL_IDS:-data/dataset_eval.dpo.jsonl}"
LIMIT="${LIMIT:-0}"                      # >0: first N eval rows only (smoke runs)
BASELINE_JSONL="${BASELINE_JSONL:-}"     # reuse an earlier results/ab_*/eval_baseline.jsonl instead of re-running the base
OUT=results/ab_$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
echo "=== verify_all @ $BASE  tuned=$TUNED baseline=$BASELINE -> $OUT ==="

echo "[1/4] tool-calling probe (tuned)"
python3 probe_toolcalls.py --model "$TUNED" --base "$BASE" 2>&1 | tee "$OUT/toolcalls_tuned.txt"
echo "[2/4] baseline on the eval split ($EVAL_BASELINE)"
if [[ -n "$BASELINE_JSONL" ]]; then
  cp "$BASELINE_JSONL" "$OUT/eval_baseline.jsonl"; cp "${BASELINE_JSONL%.jsonl}.summary.json" "$OUT/eval_baseline.summary.json"
  echo "  reused $BASELINE_JSONL"
elif [[ "$EVAL_BASELINE" == *"/"* || "$EVAL_BASELINE" == openrouter:* || "$EVAL_BASELINE" == qwen9b ]]; then
  BL_BASE=()                       # OpenRouter: llm_backend.student_endpoint picks the base + key
else
  BL_BASE=(--base "$BASE/v1")      # an Ollama tag
fi
if [[ -z "$BASELINE_JSONL" ]]; then
  python3 eval_model.py --contexts "$CTX" --ids-from "$EVAL_IDS" --model "$EVAL_BASELINE" "${BL_BASE[@]}" --answer-from-reasoning --limit "$LIMIT" --out "$OUT/eval_baseline.jsonl" 2>"$OUT/eval_baseline.log"
fi
echo "[3/4] tuned on the eval split"
python3 eval_model.py --contexts "$CTX" --ids-from "$EVAL_IDS" --model "$TUNED" --base "$BASE/v1" --limit "$LIMIT" --out "$OUT/eval_tuned.jsonl" 2>"$OUT/eval_tuned.log"
echo "[4/4] tool-calling probe (baseline control)"
python3 probe_toolcalls.py --model "$BASELINE" --base "$BASE" 2>&1 | tee "$OUT/toolcalls_baseline.txt"

echo
echo "=============================================="
python3 eval_model.py compare "$OUT/eval_baseline.jsonl" "$OUT/eval_tuned.jsonl"
grep -h RESULT "$OUT"/toolcalls_*.txt
echo "details: $OUT/"
