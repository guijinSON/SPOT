export OPENROUTER_API_KEY=

# MODEL_NAME="openrouter/google/gemini-2.5-pro-preview-03-25"
MODEL_NAME="openrouter/qwen/qwen-2.5-vl-72b-instruct"
# MODEL_NAME="anthropic/claude-3.7-sonnet:thinking"
JUDGE_NAME="openrouter/openai/gpt-4.1"
OUTPUT_PREFIX="results"

for idx in $(seq 0 1); do
  echo "Running idx=$idx..."
  python run_eval.py \
    --model_name "${MODEL_NAME}" \
    --judge_name "${JUDGE_NAME}" \
    --idx "${idx}" \
    --output_prefix "${OUTPUT_PREFIX}"
done

echo "All done!"
