# SPOT

<font color="red"><b>Evaluation results—LLM generations and decisions—are available <a href="https://huggingface.co/datasets/amphora/spot-results">here</a>.</b></font>



## Table of Contents
1. [Annotation Platform (Streamlit)](#annotation-platform-streamlit)
2. [Processing Pipeline (OCR & Rresponse generation)](#processing-pipeline-process_paperpy)
3. [Ongoing Annotation Jobs](#ongoing-annotation-jobs)
---

## Annotation Platform (Streamlit)
A lightweight Streamlit app for labeling errors discussed on **PubPeer** or papers **withdrawn from arXiv**.
Contributors review randomly selected papers, answer guided questions, and append their work to `annotations.csv`.

### 🚀 Getting Started
#### Prerequisites
- Python ≥ 3.8
- [Streamlit](https://streamlit.io/)
- pandas

#### Installation
```bash
# 1 Clone the repo
git clone https://github.com/guijinSON/ai4s_r2.git
cd ai4s_r2

# 2 Install dependencies
pip install streamlit pandas
```

#### Dataset
`retracted_machine_filtered_final.csv` ships with the repository—no extra download required.

### 🎯 Usage
```bash
streamlit run streamlit_sample.py
```
1. **Shuffle Sample** → loads a random paper.
2. Fill in the **six** annotation questions (right panel).
3. **Save Annotation** → appends to `annotations.csv`.
4. Repeat until you have **3–5** completed rows.
5. **Submit** → send your `annotations.csv` back to the maintainer.

---

## Processing Pipeline (`process.py`)
The script:
1. Parses the PDF, extracting textual chunks.
2. Embeds key figures/tables as base‑64 PNGs.
3. Queries multiple "LLM reviewers" (OpenAI o3, Gemini 2.5, etc.).
4. Saves everything in a tidy, self‑contained folder:

```
 data/{safe_doi}/
 ├── <original>.pdf
 ├── images/
 │   └── figure_1.png …
 └── metadata.json
```
`metadata.json` Example:
```json
{
  "doi": "10.1234/abc123",
  "content": [...],
  "responses": {
      "openai_o3": "…",
      "gemini_2_5": "…"
  },
  "images": ["images/figure_1.png"],
  "pdf": "original.pdf"
}
```

### Requirements
- API access:
  - **LLAMA_API_KEY**
  - **OPENAI_API_KEY**
  - **OPENROUTER_API_KEY**

### Quick Start
```bash
# 1 Export your keys (replace with your own values!)
export LLAMA_API_KEY="<your‑llama‑key>"
export OPENAI_API_KEY="<your‑openai‑key>"
export OPENROUTER_API_KEY="<your‑openrouter‑key>"

# 2 Run the script
python process_paper.py 2401.06455v1.pdf arXiv:2401.06455
```
The processed artifacts land in `data/arXiv_2401.06455/`.

---

## Ongoing Annotation Jobs
- [Algebraic Geometry](https://docs.google.com/spreadsheets/d/1i4TRlCU8I4oV16pErHyhF38rqxOWWg6oXiaX7bCZ1Ew/edit?gid=0#gid=0)
- [Medicine & Health Sciences](https://docs.google.com/spreadsheets/d/1gV9LbCIdUtE0xAKUpY5teMiuyDyADGzBGY9zgR0BtKE/edit?gid=0#gid=0)
- [Interdisciplinary & General Science](https://docs.google.com/spreadsheets/d/18CIonCPL3AEi44lstzkQpwqLMoYYy6r1FQOWPtKnmjo/edit?gid=0#gid=0)

---
