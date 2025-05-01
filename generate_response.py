#!/usr/bin/env python3
"""
generate_response.py

Load data/{safe_id}/metadata.json (where safe_id = DOI with "/"→"_"),
run LLM reviewers on its "content" payload, and write back the responses.
"""

import argparse
import json
import os
import time
from pathlib import Path
import litellm
litellm._turn_on_debug()
from openai import OpenAI
from litellm import completion
from prompts import llm_reviewer_template

def main(doi: str):
    # 1) Build safe_id and metadata path
    safe_id = doi.replace("/", "_")
    meta_path = Path("data") / safe_id / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Could not find {meta_path}")

    # 2) Ensure API keys
    for key in ("OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if not os.getenv(key):
            raise EnvironmentError(f"{key} is required")

    # 3) Load existing metadata
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    content = metadata.get("content")
    if content is None:
        raise KeyError(f"{meta_path} has no 'content' field")

    # 4) Prepare common messages
    messages = [
        {"role": "system", "content": llm_reviewer_template},
        {"role": "user",   "content": content},
    ]

    # 5) OpenAI o3 reviewer
    print("🤖 Sending to OpenAI o3…")
    oa = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    try:
        resp_o3 = oa.chat.completions.create(
            model="o3", messages=messages
        )
        o3_answer = resp_o3.choices[0].message.content
        print("✅ o3 response received")
    except Exception as e:
        print(f"❌ o3 failed: {e}")
        o3_answer = None

    # 6) Gemini 2.5 reviewer with retries
    print("🤖 Sending to Gemini 2.5…")
    g25_answer = None
    for attempt in range(1, 6):
        try:
            resp_g25 = completion(
                model="openrouter/google/gemini-2.5-pro-preview-03-25",
                messages=messages
            )
            g25_answer = resp_g25.choices[0].message.content
            print(f"✅ Gemini response on attempt {attempt}")
            break
        except Exception as e:
            print(f"❌ Attempt {attempt} failed: {e}")
            if attempt < 5:
                time.sleep(2)

    # 7) Update metadata.json
    metadata["responses"] = {
        "openai_o3":  o3_answer,
        "gemini_2_5": g25_answer
    }
    meta_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )
    print(f"💾 Updated {meta_path.resolve()}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run LLM reviewers on metadata.json by DOI/id"
    )
    parser.add_argument(
        "doi",
        help=(
            "Original DOI (e.g. 10.1234/xyz123) or safe ID "
            "(slashes already replaced with '_')"
        )
    )
    args = parser.parse_args()
    main(args.doi)
