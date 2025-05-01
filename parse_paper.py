#!/usr/bin/env python3
"""
parse_and_review.py

Parse a scientific PDF, embed key figures/tables as base-64 images,
filter out non-manuscript artifacts, call LLM “reviewers,” and save every
artifact in a tidy, self-contained folder:

    data/{safe_doi}/
    ├─ <original-pdf>.pdf
    ├─ images/
    │   └─ <embedded images>…
    └─ metadata.json
"""

import argparse
import asyncio
import base64
import json
import mimetypes
import os
import shutil
import time
from pathlib import Path
from tqdm import tqdm

import litellm
from llama_cloud_services import LlamaParse
from openai import OpenAI
from litellm import completion
from prompts import llm_reviewer_template

litellm._turn_on_debug()

# ──────────────────────────────────────────────────────────────────────────────
# Environment variables
# ──────────────────────────────────────────────────────────────────────────────
for name in ("LLAMA_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
    if not os.getenv(name):
        raise EnvironmentError(f"{name} environment variable is required")

# ──────────────────────────────────────────────────────────────────────────────
def encode_data_uri(path: Path) -> str:
    """Read a file and return a data URI string."""
    raw = path.read_bytes()
    b64 = base64.b64encode(raw).decode()
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{b64}"

# ──────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ──────────────────────────────────────────────────────────────────────────────
async def main(paper_path: str, doi: str) -> None:
    safe_doi = doi.replace("/", "_")
    paper_dir = Path("data") / safe_doi
    img_dir = paper_dir / "images"
    paper_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(exist_ok=True)

    # Copy PDF
    pdf_dst = paper_dir / Path(paper_path).name
    shutil.copy2(paper_path, pdf_dst)

    # Parse PDF
    parser = LlamaParse(
        api_key=os.getenv("LLAMA_API_KEY"),
        verbose=True,
        language="en",
        extract_layout=True,
        parse_mode="parse_page_with_agent",
    )
    result = await parser.aparse(paper_path)
    all_img_paths = await result.asave_all_images(img_dir)
    
    # image_documents = await result.aget_image_documents(
    #     include_screenshot_images=True,
    #     include_object_images=True,
    #     image_download_dir=img_dir,
    # )

    content: list[dict] = []
    used_imgs: set[str] = set()

    # Process each page
    for page in tqdm(result.pages, desc="Pages"):
        # 1) Clean up OCR using LLM
        review_payload = [{"type": "text", "text": page.md}]
        for img in page.images:
            if img.type in ("layout_text", "layout_formula"):
                p = img_dir / img.name
                if not p.exists(): 
                    continue
                data_uri = encode_data_uri(p)
                review_payload.append({
                    "type": "image_url",
                    "image_url": {"url": data_uri}
                })

        llm_query = [
            {"role": "system", "content": (
                "You will be given an OCR output and associated image crops. "
                "Review and regenerate the OCR text with better accuracy. "
                "Return only the corrected text."
            )},
            {"role": "user", "content": review_payload}
        ]
        resp = completion(model="gpt-4.1", messages=llm_query)
        corrected = resp.choices[0].message.content
        content.append({"type": "text", "text": corrected})

        # 2) Embed only true manuscript figures (layout_picture)
        #    Use full-page screenshot + each layout_picture to classify
        full_page = next((i for i in page.images if i.type == "full_page_screenshot"), None)
        if full_page:
            fp_path = img_dir / full_page.name
            if fp_path.exists():
                fp_uri = encode_data_uri(fp_path)
                for img in page.images:
                    if img.type == "layout_picture":
                        pic_path = img_dir / img.name
                        if not pic_path.exists():
                            continue
                        pic_uri = encode_data_uri(pic_path)

                        # classification prompt
                        system_prompt = (
                            "You will be given a full-page screenshot and a cropped image. "
                            "Decide whether the cropped image is a labeled figure belonging to "
                            "the manuscript (e.g., Figure 1, Table 2) or an extraneous artifact "
                            "(e.g., banner, logo, advertisement). "
                            "If it is a manuscript figure, return [TRUE]; otherwise return [FALSE]. "
                            "Respond with exactly [TRUE] or [FALSE]."
                        )
                        messages = [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": [
                                {"type": "image_url", "image_url": {"url": fp_uri}},
                                {"type": "image_url", "image_url": {"url": pic_uri}}
                            ]}
                        ]
                        cls_resp = completion(model="gpt-4.1", messages=messages)
                        decision = cls_resp.choices[0].message.content.strip()

                        if "TRUE" in decision.upper():
                            # include in final content
                            content.append({
                                "type": "image_url",
                                "image_url": {"url": pic_uri}
                            })
                            used_imgs.add(img.name)
                        # else skip this image

    # 3) Clean up unused images
    for path in all_img_paths or []:
        if Path(path).name not in used_imgs:
            try:
                os.remove(path)
            except OSError:
                pass

    # 4) Write metadata.json
    metadata = {
        "doi":     doi,
        "content": content,
        # responses are added in a separate reviewer step if needed
        "images": [f"images/{n}" for n in used_imgs],
        "pdf":     pdf_dst.name,
    }
    with open(paper_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    print(f"✅ Finished. Artifacts saved under {paper_dir.resolve()}")

# ──────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Parse a PDF, filter images, clean OCR, and save artifacts."
    )
    parser.add_argument("paper_path", help="Path to the PDF file")
    parser.add_argument("doi",        help="DOI (e.g. 10.1000/xyz123)")
    args = parser.parse_args()
    asyncio.run(main(args.paper_path, args.doi))
