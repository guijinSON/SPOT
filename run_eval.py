#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import json
import litellm
from prompts import llm_reviewer_template, llm_judge_template
from litellm import batch_completion
from src import extract_response_dict, compute_all_metrics

def main():
    parser = argparse.ArgumentParser(
        description="Review errata using LLMs and compute evaluation metrics"
    )
    parser.add_argument(
        '--model_name',
        type=str,
        required=True,
        help='Name of the reviewer LLM model'
    )
    parser.add_argument(
        '--judge_name',
        type=str,
        default='openrouter/openai/gpt-4.1',
        help='Name of the judge LLM model'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to the input Excel file'
    )
    parser.add_argument(
        '--output_prefix',
        type=str,
        default='results',
        help='Prefix for output files'
    )

    parser.add_argument(
        '--idx',
        type=str,
        default='0',
        help='index of files'
    )
    args = parser.parse_args()

    # Set API key for OpenRouter
    os.environ['OPENROUTER_API_KEY'] = "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"

    model_name = args.model_name
    judge_name = args.judge_name
    safe_model_name = model_name.replace("/", "_").replace(".", "_")

    # Load original data
    orig_df = pd.read_excel(args.input)
    list_cols = ['categories', 'location', 'annotation']
    agg_dict = {
        col: (list if col in list_cols else 'first')
        for col in orig_df.columns
        if col != 'doi/arxiv_id'
    }
    flat_orig_df = (
        orig_df
        .groupby('doi/arxiv_id', as_index=False)
        .agg(agg_dict)
    )

    # Prepare reviewer queries
    qrys = []
    papers = []
    for doi, paper in orig_df.groupby('doi/arxiv_id'):
        safe_doi = str(doi).replace('/', '_')
        path = f"data/{safe_doi}/metadata.json"
        try:
            with open(path, 'r') as fp:
                data = json.load(fp)
            qry = [
                {'role': 'system', 'content': llm_reviewer_template},
                {'role': 'user', 'content': data['content']},
            ]
            qrys.append(qry)
            papers.append(doi)
        except FileNotFoundError:
            print(f'{path} is missing!')

    print(f"Generating responses with {model_name}")
    responses = batch_completion(
        model=model_name,
        messages=qrys
    )

    parsed_results = [extract_response_dict(resp) for resp in responses]
    resp_df = pd.DataFrame({
        "doi/arxiv_id": papers,
        "parsed":    [r.get("parsed", False)        for r in parsed_results],
        "is_error":  [bool(r.get("has_error", True)) for r in parsed_results],
        "errors":    [r.get("errors", [])            for r in parsed_results],
    })
    resp_df = resp_df.merge(flat_orig_df, on=['doi/arxiv_id'])

    # Prepare judge queries for error cases
    error_cases = resp_df.loc[resp_df["parsed"] & resp_df["is_error"], :]
    qrys = []
    for _, row in error_cases.iterrows():
        annotations = [
            {"location": loc, "description": desc}
            for loc, desc in zip(row.location, row.annotation)
        ]
        predictions = row.errors
        payload = {
            "annotations": annotations,
            "predictions": predictions
        }
        user_content = json.dumps(payload, ensure_ascii=False, indent=2)
        qry = [
            {"role": "system", "content": llm_judge_template},
            {"role": "user",   "content": user_content},
        ]
        qrys.append(qry)

    print(f"Generating responses with {judge_name}")
    # Run judge model
    responses = batch_completion(
        model=judge_name,
        messages=qrys
    )
    parsed_results = [extract_response_dict(resp) for resp in responses]

    # Assemble judge results
    records = []
    for doi, res in zip(resp_df["doi/arxiv_id"], parsed_results):
        matches = res.get("matches", [])
        descriptions = [m.get("description", "") for m in matches]
        records.append({
            "doi/arxiv_id": doi,
            "matches": matches,
            "match_descriptions": descriptions
        })
    judge_df = pd.DataFrame.from_records(records)
    resp_df = resp_df.merge(judge_df, on="doi/arxiv_id", how="left")
    resp_df = resp_df[[
        'doi/arxiv_id', 'parsed', 'is_error', 'errors', 'title', 'pdf_path',
        'categories', 'location', 'retract/errata', 'journal/category',
        'source', 'annotation', 'matches', 'match_descriptions'
    ]]

    # Compute metrics
    metrics = compute_all_metrics(resp_df)
    print(f"Micro Precision: {metrics['precision_micro']:.3f}")
    print(f"Micro Recall:    {metrics['recall_micro']:.3f}")
    print(f"Macro Precision: {metrics['precision_macro']:.3f}")
    print(f"Macro Recall:    {metrics['recall_macro']:.3f}")
    print(f"PPR:             {metrics['PPR']:.3f}")

    # Export results and metrics
    resp_df.to_csv(f"{args.output_prefix}/{safe_model_name}_resp_df_{args.idx}.csv", index=False)
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(f"{args.output_prefix}/{safe_model_name}_metrics_{args.idx}.csv", index=False)
    print(f"Exported data to {args.output_prefix}_resp_df.csv and {args.output_prefix}_metrics.csv")

if __name__ == "__main__":
    main()
