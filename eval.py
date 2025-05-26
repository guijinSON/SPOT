#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import json
import litellm
from datasets import load_dataset
from prompts import llm_reviewer_template, llm_judge_template
from litellm import batch_completion, completion
from src import extract_response_dict, compute_all_metrics, clean_records
from tqdm import tqdm
import time

# Set your API keys (or load from environment securely)
os.environ['OPENROUTER_API_KEY'] = os.getenv('OPENROUTER_API_KEY')
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_API_KEY')

def generate_and_judge(paper_id: str, paper_content: str, error_annotation: str,
                       reviewer_model: str, judge_model: str,
                       llm_reviewer_template: str, llm_judge_template: str,
                       max_retries: int = 3) -> dict:
    """
    Generate review with reviewer_model and then judge it with judge_model.
    Returns a dict with paper_id, parsed flag, error flag, errors, matches, match_descriptions, and metrics.
    """
    # Step 1: Generate review
    review_qry = [
        {'role': 'system', 'content': llm_reviewer_template},
        {'role': 'user', 'content': clean_records(paper_content)}
    ]
    review_responses = []
    for attempt in range(max_retries + 1):
        try:
            resp = completion(model=reviewer_model, messages=review_qry, min_tokens=8)
            review_responses.append(resp)
            break
        except Exception as e:
            if attempt == max_retries:
                review_responses.append(None)

    parsed = False
    has_error = True
    errors = []
    try:
        parsed_data = extract_response_dict(review_responses[-1])
        parsed = parsed_data.get('parsed', False)
        has_error = bool(parsed_data.get('has_error', True))
        errors = parsed_data.get('errors', [])
    except Exception:
        pass

    record = {
        'paper_id': paper_id,
        'parsed': parsed,
        'has_error': has_error,
        'errors': errors,
        'review_response': review_responses[-1]
    }

    # Step 2: If parsed and errors exist, judge them
    if parsed and has_error and errors:
        payload = json.dumps({
            'annotations': error_annotation,
            'predictions': errors
        }, ensure_ascii=False)
        judge_qry = [
            {'role': 'system', 'content': llm_judge_template},
            {'role': 'user', 'content': payload}
        ]
        judge_resp = batch_completion(model=judge_model, messages=[judge_qry])[0]
        judge_data = extract_response_dict(judge_resp)
        matches = judge_data.get('matches', [])
        descriptions = [m.get('description', '') for m in matches]
        record.update({
            'matches': matches,
            'match_descriptions': descriptions,
            'judge_response': judge_resp
        })
    else:
        record.update({'matches': [], 'match_descriptions': []})
    return record


def process_dataset(df: pd.DataFrame, reviewer_model: str, judge_model: str,
                    llm_reviewer_template: str, llm_judge_template: str,
                    output_prefix: str, max_retries: int = 3):
    """
    Process each paper in df: generate review, judge, compute metrics, save results.
    Expects df has columns ['doi/arxiv_id', 'paper_content']
    """
    records = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Processing papers'):
        error_annotations = [
            {"location": loc, "description": desc}
            for loc, desc in zip(row.error_location, row.error_annotation)
        ]
        rec = generate_and_judge(
            paper_id=row['doi/arxiv_id'],
            paper_content=row['paper_content'],
            error_annotation=error_annotations,
            reviewer_model=reviewer_model,
            judge_model=judge_model,
            llm_reviewer_template=llm_reviewer_template,
            llm_judge_template=llm_judge_template,
            max_retries=max_retries
        )
        records.append(rec)
    out_df = pd.DataFrame(records)
    out_df = out_df.merge(df, left_on='paper_id', right_on='doi/arxiv_id', how='left')

    # Optionally compute metrics
    # metrics = compute_all_metrics(out_df)
    # pd.DataFrame([metrics]).to_csv(f"{output_prefix}_metrics.csv", index=False)

    # Save outputs
    resp_path = f"{output_prefix}_resp_df.csv"
    out_df.to_csv(resp_path, index=False)
    print(f"Saved responses to {resp_path}")
    return out_df


def main():
    parser = argparse.ArgumentParser(description="Generate and judge LLM reviews for papers.")
    parser.add_argument('--reviewer_model', type=str, required=True,
                        help="Model to use for generating reviews (e.g. openrouter/anthropic/claude-sonnet-4)")
    parser.add_argument('--judge_model', type=str, required=True,
                        help="Model to use for judging reviews (e.g. gpt-4.1)")
    parser.add_argument('--index', type=int, required=True,
                        help="Index number to differentiate this run for the same model")
    parser.add_argument('--hf_token', type=str, default=None,
                        help="Hugging Face token for dataset access (or set HF_TOKEN env variable)")
    parser.add_argument('--output_dir', type=str, default='results',
                        help="Directory to save results files")
    args = parser.parse_args()

    # Load dataset
    token = args.hf_token or os.getenv('HF_TOKEN')
    orig_df = load_dataset(
        'amphora/errata_0504_v0',
        split='train',
        token=token
    ).to_pandas()

    # Flatten grouped lists
    list_cols = ['paper_category', 'error_location', 'error_annotation']
    agg_dict = {
        col: (list if col in list_cols else 'first')
        for col in orig_df.columns
        if col != 'doi/arxiv_id'
    }
    flat_orig_df = orig_df.groupby('doi/arxiv_id', as_index=False).agg(agg_dict)

    # Build safe model name with index
    safe_name = args.reviewer_model.replace("/", "_").replace(".", "_")
    safe_name = f"{safe_name}_{args.index}"
    output_prefix = f"{args.output_dir}/{safe_name}"

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    process_dataset(
        flat_orig_df,
        args.reviewer_model,
        args.judge_model,
        llm_reviewer_template,
        llm_judge_template,
        output_prefix
    )

if __name__ == '__main__':
    main()
