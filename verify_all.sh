#!/usr/bin/env bash
# verify_all.sh — one-command verification suite for the dialectic model.
# Run when node7 has headroom:  bash verify_all.sh
# From this box (remote API):   BASE=http://node7.ea.org:11434 bash verify_all.sh
# Send the RESULT block at the bottom back to Mya.
set -uo pipefail
cd "$(dirname "$0")"
BASE="${BASE:-http://localhost:11434}"
OUT=results/ab_$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
echo "=== verify_all @ $BASE -> $OUT ==="

echo "[1/3] tool-calling probe (tuned)"
python3 probe_toolcalls.py --model dialectic-qwen3.5-9b --base "$BASE" 2>&1 | tee "$OUT/toolcalls_tuned.txt"
echo "[2/3] quality A/B (30 contexts each) — this is the slow part"
python3 eval_dialectic.py --contexts results/openrouter/contexts.jsonl \
  --model qwen3.5:9b --base "$BASE/v1" --out "$OUT/eval_baseline.jsonl" 2>&1 | tee "$OUT/eval_baseline.log" | tail -12
python3 eval_dialectic.py --contexts results/openrouter/contexts.jsonl \
  --model dialectic-qwen3.5-9b --base "$BASE/v1" --out "$OUT/eval_tuned.jsonl" 2>&1 | tee "$OUT/eval_tuned.log" | tail -12
echo "[3/3] tool-calling probe (baseline control)"
python3 probe_toolcalls.py --model qwen3.5:9b --base "$BASE" 2>&1 | tee "$OUT/toolcalls_baseline.txt"

echo
echo "=============================================="
echo "RESULT  ($OUT)"
echo "=============================================="
grep -A11 '^AGGREGATE' "$OUT"/*.log 2>/dev/null || cat "$OUT"/*.log | grep -A11 '^AGGREGATE'
grep -E 'PASS|FAIL|tool_calls' "$OUT/toolcalls_*.txt" | head
echo "details: $OUT/"
