import re
import json
from typing import Any, Dict
import pandas as pd

def extract_response_dict(response) -> Dict[str, Any]:
    """
    Safely extracts the JSON dict from an LLM response.
    
    1. Uses safe_parse to get the raw content string.
    2. If there's no content, returns a dict indicating the parse error.
    3. Otherwise, attempts to parse with parse_response_dict().
       — If that succeeds, returns the parsed dict.
       — If it fails, returns a dict with a 'parse_error' key and the raw content.
    """
    raw = safe_parse(response)
    if raw is None:
        return {
            "parse_error": "no content returned by safe_parse",
            "parsed": False
        }
    
    try:
        result = parse_response_dict(raw)
        # Indicate successful parse if you’d like
        result.setdefault("parsed", True)
        return result
    except ValueError as e:
        return {
            "parse_error": str(e),
            "parsed": False,
            "raw": raw
        }


def parse_response_dict(s: str) -> Dict[str, Any]:
    """
    Extracts and parses the JSON object between <response>...</response> tags
    in the input string `s`. Returns it as a Python dictionary.
    
    Raises:
        ValueError: if no <response> block is found or JSON is invalid.
    """
    pattern = r'<response>\s*(\{.*?\})\s*</response>'
    m = re.search(pattern, s, re.DOTALL)
    if not m:
        raise ValueError("No <response>...</response> block found")
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in <response> block: {e}") from e
        
def safe_parse(response):
    try:
        return response.choices[0].message.content
    except:
        return None


def compute_all_metrics(resp_df: pd.DataFrame):
    # 1) Work on a fresh copy
    df = resp_df.copy()

    # 2) Ensure 'matches' and 'errors' are always lists
    df["matches"] = df["matches"].apply(lambda x: x if isinstance(x, list) else [])
    df["errors"]  = df["errors"].apply(lambda x: x if isinstance(x, list) else [])

    # 3) Add all per-paper counts
    df = df.assign(
        k_i = df["annotation"].apply(len),
        TP_i = df["matches"].apply(len),
    )
    df["FP_i"] = df["errors"].apply(len) - df["TP_i"]
    df["FN_i"] = df["k_i"] - df["TP_i"]

    N = len(df)

    # 4) Micro-averaged Precision & Recall
    TP_total = df["TP_i"].sum()
    FP_total = df["FP_i"].sum()
    FN_total = df["FN_i"].sum()
    precision_micro = TP_total / (TP_total + FP_total) if (TP_total + FP_total) > 0 else 0.0
    recall_micro    = TP_total / (TP_total + FN_total) if (TP_total + FN_total) > 0 else 0.0

    # 5) Macro-averaged Precision & Recall (per-paper average)
    per_prec = df.apply(
        lambda row: row["TP_i"] / (row["TP_i"] + row["FP_i"])
        if (row["TP_i"] + row["FP_i"]) > 0 else 0.0,
        axis=1
    )
    per_rec = df.apply(
        lambda row: row["TP_i"] / (row["TP_i"] + row["FN_i"])
        if (row["TP_i"] + row["FN_i"]) > 0 else 0.0,
        axis=1
    )
    precision_macro = per_prec.mean()
    recall_macro    = per_rec.mean()

    # 6) Perfect-Paper Rate (PPR)
    perfect_mask = (df["TP_i"] == df["k_i"]) & (df["FP_i"] == 0)
    PPR = perfect_mask.sum() / N

    # 7) Collect per-paper stats
    per_paper = df[[
        "doi/arxiv_id", "k_i", "TP_i", "FP_i", "FN_i"
    ]].to_dict(orient="records")

    return {
        "N":                 N,
        "precision_micro":   precision_micro,
        "recall_micro":      recall_micro,
        "precision_macro":   precision_macro,
        "recall_macro":      recall_macro,
        "PPR":               PPR,
        "per_paper":         per_paper
    }
