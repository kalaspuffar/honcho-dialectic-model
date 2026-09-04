#!/usr/bin/env bash
# run_trial.sh — one-shot wrapper for Daniel's machine
# Usage:  bash run_trial.sh
#
# Knobs (optional env vars, each with a safe default):
#   CTX_MODEL   context-generator (default: deepseek/deepseek-chat)
#   ARMS        answer arms (default: deepseek,opus,sonnet,gemini-pro,gpt5,qwen3max)
#   MAX_USD     per-step wallet cap (default 5.00)
#   N_CTX       number of shared contexts (default 30)
#
# API key: loaded from environment or keys.env (auto-sourced below).
#
# Wallet safety: gen and run each print pre-spend estimate and ABORT if over cap.
#                run also checks running $ after every call (live cap).
#                contexts cached — re-run never re-pays for generation.
#
# Output: results/openrouter/ + tarball to copy back.

set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

if [ -f keys.env ]; then
  set -a
  . ./keys.env
  set +a
fi

CTX_MODEL="${CTX_MODEL:-deepseek/deepseek-chat}"
ARMS="${ARMS:-deepseek,opus,sonnet,gemini-pro,gpt5,qwen3max}"
MAX_USD="${MAX_USD:-5.00}"
N_CTX="${N_CTX:-30}"
CTXS="results/openrouter/contexts.jsonl"

echo ""
echo "=== 1/4 ESTIMATE (no network, no cost) ==="
python3 openrouter_trial.py estimate --arms "$ARMS" --n "$N_CTX"
echo ""
echo "=== 2/4 GENERATE CONTEXTS (${N_CTX} via ${CTX_MODEL}) ==="
needs_gen=1
if [ -f "$CTXS" ] && grep -q '"persona"' "$CTXS" 2>/dev/null; then
  echo "contexts.jsonl present with valid rows — cached, skipping generation."
  needs_gen=0
fi
if [ "$needs_gen" -eq 1 ]; then
  python3 openrouter_trial.py gen --context-model "$CTX_MODEL" --n "$N_CTX" --max-usd "$MAX_USD"
fi
echo ""
echo "=== 3/4 RUN ARMS (${ARMS}, cap \$${MAX_USD}) ==="
python3 openrouter_trial.py run --arms "$ARMS" --max-usd "$MAX_USD"
echo ""
echo "=== 4/4 COLLECT ==="
python3 openrouter_trial.py collect
echo ""
echo "DONE."
echo "Copy results/ tarball + blind-review.csv back to the host."
