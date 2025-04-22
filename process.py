#!/usr/bin/env python3
"""
parse_and_review.py

Parse a scientific PDF, embed key figures/tables as base‑64 images,
call multiple LLM “reviewers,” and save every artifact in a tidy,
self‑contained folder:

    data/{safe_doi}/
    ├─ <original‑pdf>.pdf
    ├─ images/
    │   └─ <only‑images‑embedded‑in‑content>…
    └─ metadata.json

`metadata.json` contains:
    {
      "doi":             "...",
      "content":         [... list of dicts sent to the models ...],
      "responses":       { "openai_o3": "...", "gemini_2_5": "..." },
      "images":          ["images/figure_1.png", ...],
      "pdf":             "<original‑pdf>.pdf"
    }
"""

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import shutil
from pathlib import Path

from llama_cloud_services import LlamaParse
from openai import OpenAI
from litellm import completion
from prompts import llm_reviewer_template

# ──────────────────────────────────────────────────────────────────────────────
# Configuration (all keys are read from env for safety)
# ──────────────────────────────────────────────────────────────────────────────
LLAMA_API_KEY      = os.getenv("LLAMA_API_KEY")
OPENAI_API_KEY     = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

for key, name in [
    (LLAMA_API_KEY, "LLAMA_API_KEY"),
    (OPENAI_API_KEY, "OPENAI_API_KEY"),
    (OPENROUTER_API_KEY, "OPENROUTER_API_KEY"),
]:
    if not key:
        raise EnvironmentError(f"{name} environment variable is required")


# ──────────────────────────────────────────────────────────────────────────────
# Main async pipeline
# ──────────────────────────────────────────────────────────────────────────────
async def main(paper_path: str, doi: str) -> None:
    # 1. Prepare output directories ------------------------------------------------
    safe_doi = doi.replace("/", "_")
    paper_dir = Path("data") / safe_doi          # data/10.1234_abcd
    img_dir   = paper_dir / "images"
    paper_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(exist_ok=True)

    # Copy original PDF so it lives with everything else
    pdf_dst = paper_dir / Path(paper_path).name
    shutil.copy2(paper_path, pdf_dst)

    # 2. Parse PDF with LlamaParse --------------------------------------------------
    parser = LlamaParse(
        api_key=LLAMA_API_KEY,
        verbose=True,
        language="en",
        extract_layout=True,
        parse_mode="parse_page_with_agent",
    )
    result = await parser.aparse(paper_path)
    print("✅ Parsing complete")
    
    # Save every detected image to disk
    all_img_paths = await result.asave_all_images(img_dir)
    print(f"🖼 Saved {len(all_img_paths)} images to {img_dir}")
    
    # 3. Build chat‑ready `content` + track which images are used -------------------
    content: list[dict] = []
    used_img_names: set[str] = set()

    print("✍ Building content payload and embedding images...")
    for page in result.pages:
        # Add markdown for the page
        content.append({"type": "text", "text": page.md})

        # Embed only figures/tables (filter by filename)
        for img in page.images:
            if "picture" in img.name or "table" in img.name:
                img_file = img_dir / img.name
                raw      = img_file.read_bytes()
                b64      = base64.b64encode(raw).decode()
                mime, _  = mimetypes.guess_type(img.name)
                mime     = mime or "application/octet-stream"
                data_uri = f"data:{mime};base64,{b64}"

                content.append(
                    {"type": "image_url", "image_url": {"url": data_uri}}
                )
                used_img_names.add(img.name)

    # 4. Delete any saved images *not* referenced in `content` ----------------------
    for path in all_img_paths:
        if os.path.basename(path) not in used_img_names:     # compare filenames
            os.remove(path)                                  # works on str paths
    
    # list of kept images relative to paper_dir
    kept_img_relpaths = [
        os.path.relpath(os.path.join(img_dir, name), paper_dir)
        for name in used_img_names
]
    # 5. Call reviewer LLMs ---------------------------------------------------------
    print("🤖 Sending content to OpenAI o3 model...")
    oa_client = OpenAI(api_key=OPENAI_API_KEY)
    o3_resp = (
        oa_client.chat.completions.create(
            model="o3",
            messages=[
                {"role": "system", "content": llm_reviewer_template},
                {"role": "user",   "content": content},
            ],
        )
        .choices[0]
        .message.content
    )
    
    print("🤖 Sending content to Gemini 2.5 model...")
    g25_resp = (
        completion(
            model="openrouter/google/gemini-2.5-pro-preview-03-25",
            messages=[
                {"role": "system", "content": llm_reviewer_template},
                {"role": "user",   "content": content},
            ],
        )
        .choices[0]
        .message.content
    )

    # 6. Write one tidy metadata.json ----------------------------------------------
    print("💾 Writing metadata.json...")
    metadata = {
        "doi":       doi,
        "content":   content,
        "responses": {
            "openai_o3":   o3_resp,
            "gemini_2_5":  g25_resp,
        },
        "images": kept_img_relpaths,
        "pdf":    pdf_dst.name,
    }
    with open(paper_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Finished. All artifacts saved under:\n   {paper_dir.resolve()}")


# ──────────────────────────────────────────────────────────────────────────────
# CLI wrapper
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Parse a PDF, embed key images, call reviewer LLMs, and store "
            "every artifact in data/{safe_doi}/."
        )
    )
    parser.add_argument("paper_path", help="Path to the PDF file")
    parser.add_argument("doi",        help="DOI (e.g. 10.1000/xyz123)")
    args = parser.parse_args()

    asyncio.run(main(args.paper_path, args.doi))
