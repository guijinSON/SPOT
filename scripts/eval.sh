#!/usr/bin/env bash
set -e

# Announce API keys (replace with your actual keys)
export OPENROUTER_API_KEY=""
export OPENAI_API_KEY=""
export HF_TOKEN=""

# Model settings
REVIEWER_MODEL="openrouter/anthropic/claude-opus-4"
# REVIEWER_MODEL="openrouter/google/gemini-2.0-flash-lite-001"
JUDGE_MODEL="gpt-4.1"
OUTPUT_DIR="results"

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Iterate over indexes 1, 2, 3
for INDEX in 0 1 2 3; do
  echo "=== Starting run $INDEX ==="
  python eval.py \
    --reviewer_model "$REVIEWER_MODEL" \
    --judge_model "$JUDGE_MODEL" \
    --index "$INDEX" \
    --hf_token "$HF_TOKEN" \
    --output_dir "$OUTPUT_DIR"
  echo "=== Finished run $INDEX ==="
done
