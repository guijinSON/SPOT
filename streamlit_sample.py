import streamlit as st
import pandas as pd
import random
import os

# ——— PAGE CONFIG ———
st.set_page_config(page_title="Annotation Platform", layout="wide")

# ——— CONFIG ———
DATA_PATH = 'retracted_machine_filtered_final.csv'
ANNOTATIONS_PATH = 'annotations.csv'

# ——— LOAD DATA ———
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# ——— SESSION STATE FOR SHUFFLING ———
if 'idx' not in st.session_state:
    st.session_state.idx = random.randint(0, len(df) - 1)
if st.button('🔀 Shuffle Sample'):
    st.session_state.idx = random.randint(0, len(df) - 1)

row = df.iloc[st.session_state.idx]

# ——— LAYOUT: TWO COLUMNS ———
left_col, right_col = st.columns([2, 3])

# — Left: Paper Info and Summary ———
with left_col:
    st.markdown(f"### 📄 {row['title']}")
    st.markdown(f"**DOI/arXiv ID:** {row['doi/arxiv_id']}")
    st.markdown(f"**Source:** {row['source']}")
    st.markdown("**GPT‑4 Summary:**")
    # preserve line breaks in summary
    st.markdown(
        f"<div style='white-space: pre-wrap; line-height:1.5;'>{row['gpt4-summ']}</div>",
        unsafe_allow_html=True
    )

    # — Contextual Guidelines ———
    source_lower = str(row['source']).lower()
    if 'pubpeer' in source_lower:
        st.markdown("**PubPeer Search Guidelines:**")
        st.markdown(
            """
1. Search for the paper title or DOI on [PubPeer](https://pubpeer.com/).  
2. Review all comment threads.  
3. Look for any author responses.  
4. Assess whether the identified error(s) critically affect the paper’s conclusions.  
5. Confirm that the PDF version remains accessible.
            """
        )
    elif 'withdrarxiv' in source_lower or 'withdraw_arxiv' in source_lower or 'arxiv' in source_lower:
        st.markdown("**arXiv Withdrawal Guidelines:**")
        abs_url = f"https://arxiv.org/abs/{row['doi/arxiv_id'].strip()}"
        st.markdown(
            f"""
1. Visit the abstract page: [{abs_url}]({abs_url})  
2. Review all comments displayed.  
3. Assess whether the identified error(s) critically affect the paper’s conclusions.  
4. Confirm that the PDF version remains accessible.
"""
        )

# — Right: Annotation Questions ———
with right_col:
    error_ack = st.radio(
        "1. Have the authors publicly acknowledged this error?",
        ["Yes", "No", "Uncertain"]
    )
    is_severe = st.radio(
        "2. Is this error severe?",
        ["Yes", "No"]
    )
    accessible = st.radio(
        "3. Is the paper still accessible at its DOI/arXiv link?",
        ["Yes", "No"]
    )
    category = st.selectbox(
        "4. Which category best describes the error?",
        ["Figure duplication", "Data inconsistency", "Equation typo",
         "Methodological issue", "Other"]
    )
    if category == "Other":
        category = st.text_input("⮕ Please specify your own category:")
    error_count = st.number_input(
        "5. How many distinct errors are reported?",
        min_value=1, max_value=20, step=1
    )
    summary = st.text_area(
        "6. Summarize the error(s) in your own words:",
        height=120
    )

    # — Save Annotation ———
    if st.button('💾 Save Annotation'):
        new_row = {
            'title': row['title'],
            'doi/arxiv_id': row['doi/arxiv_id'],
            'source': row['source'],
            'gpt4-summ': row['gpt4-summ'],
            'author_ack': error_ack,
            'severe': is_severe,
            'accessible': accessible,
            'category': category,
            'error_count': error_count,
            'summary': summary
        }
        write_header = not os.path.exists(ANNOTATIONS_PATH)
        pd.DataFrame([new_row]).to_csv(
            ANNOTATIONS_PATH,
            mode='a',
            header=write_header,
            index=False
        )
        st.success("✅ Annotation saved.")

# ——— SEARCH GUIDELINES PANEL ———
with st.expander("🔍 General Search Guidelines", expanded=False):
    st.write("""
- **Start** at the paper’s PubPeer or journal page.  
- **Look** for any author replies or updates.  
- **Check** the PDF version for notes like “retracted” or footnotes.  
- **Search** the DOI on Google Scholar or the publisher site to confirm access.  
- **Note** any errata or corrigenda entries.
    """
)
