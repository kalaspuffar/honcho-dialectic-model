#!/usr/bin/env bash
# verify_all.sh — post-training A/B on the held-out split + tool-call probe.
#   BASE=http://node7.ea.org:11434 TUNED=dialectic-v1 BASELINE=qwen3.5:9b bash verify_all.sh
# The baseline EVAL column runs on OpenRouter by default (EVAL_BASELINE=qwen9b, needs OPENROUTER_API_KEY):
# qwen3.5:9b through Ollama's /v1 answers inside its <think> block and returns empty content on most
# rows, and /v1 ignores "think": false (TRAIN.md §7, 2026-09-11). Set EVAL_BASELINE to an Ollama tag
# to force the local path anyway. BASELINE stays the Ollama tag for the tool-call probe control.
set -uo pipefail
cd "$(dirname "$0")"
BASE="${BASE:-http://localhost:11434}"
TUNED="${TUNED:-dialectic-v1}"
BASELINE="${BASELINE:-qwen3.5:9b}"
EVAL_BASELINE="${EVAL_BASELINE:-qwen9b}"
CTX="${CTX:-data/contexts.jsonl}"
EVAL_IDS="${EVAL_IDS:-data/dataset_eval.dpo.jsonl}"
OUT=results/ab_$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
echo "=== verify_all @ $BASE  tuned=$TUNED baseline=$BASELINE -> $OUT ==="

echo "[1/4] tool-calling probe (tuned)"
python3 probe_toolcalls.py --model "$TUNED" --base "$BASE" 2>&1 | tee "$OUT/toolcalls_tuned.txt"
echo "[2/4] baseline on the eval split ($EVAL_BASELINE)"
if [[ "$EVAL_BASELINE" == *"/"* || "$EVAL_BASELINE" == openrouter:* || "$EVAL_BASELINE" == qwen9b ]]; then
  BL_BASE=()                       # OpenRouter: llm_backend.student_endpoint picks the base + key
else
  BL_BASE=(--base "$BASE/v1")      # an Ollama tag
fi
python3 eval_model.py --contexts "$CTX" --ids-from "$EVAL_IDS" --model "$EVAL_BASELINE" "${BL_BASE[@]}" --out "$OUT/eval_baseline.jsonl" 2>"$OUT/eval_baseline.log"
echo "[3/4] tuned on the eval split"
python3 eval_model.py --contexts "$CTX" --ids-from "$EVAL_IDS" --model "$TUNED" --base "$BASE/v1" --out "$OUT/eval_tuned.jsonl" 2>"$OUT/eval_tuned.log"
echo "[4/4] tool-calling probe (baseline control)"
python3 probe_toolcalls.py --model "$BASELINE" --base "$BASE" 2>&1 | tee "$OUT/toolcalls_baseline.txt"

echo
echo "=============================================="
python3 eval_model.py compare "$OUT/eval_baseline.jsonl" "$OUT/eval_tuned.jsonl"
grep -h RESULT "$OUT"/toolcalls_*.txt
echo "details: $OUT/"
