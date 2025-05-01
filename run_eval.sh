export OPENROUTER_API_KEY=""

python run_eval.py \
  --model_name openrouter/openai/gpt-4.1-nano \
  --judge_name openrouter/openai/gpt-4.1 \
  --input errata_0429_v1.xlsx \
  --idx 1 \
  --output_prefix results
