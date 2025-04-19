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
num_papers = len(df)

def get_completed():
    if os.path.exists(ANNOTATIONS_PATH):
        try:
            return len(pd.read_csv(ANNOTATIONS_PATH))
        except pd.errors.ParserError:
            with open(ANNOTATIONS_PATH, 'r', encoding='utf-8', errors='ignore') as f:
                return sum(1 for _ in f) - 1
    return 0

# ——— SIDEBAR: NAVIGATION & PROGRESS ———
st.sidebar.header("📋 Navigation & Progress")
completed = get_completed()
st.sidebar.write(f"Progress: **{completed}** annotated / **{num_papers}** total")
st.sidebar.progress(completed / num_papers if num_papers else 0)

titles = df['title'].tolist()
selection = st.sidebar.radio(
    "Select Paper:",
    options=titles,
    index=st.session_state.get('idx', 0)
)
if st.sidebar.button('Go'):
    st.session_state.idx = titles.index(selection)
if st.sidebar.button('🔀 Shuffle Sample'):
    st.session_state.idx = random.randint(0, num_papers - 1)

if 'idx' not in st.session_state:
    st.session_state.idx = 0

row = df.iloc[st.session_state.idx]

# ——— LAYOUT: TWO COLUMNS ———
left_col, right_col = st.columns([2, 3])

# — Left: Paper Info and Summary ———
with left_col:
    st.markdown(f"### 📄 {row['title']}")
    st.markdown(f"**DOI/arXiv ID:** {row['doi/arxiv_id']}")
    st.markdown(f"**Source:** {row['source']}")
    st.markdown("**GPT‑4 Summary:**")
    st.markdown(
        f"<div style='white-space: pre-wrap; line-height:1.5;'>{row['gpt4-summ']}</div>",
        unsafe_allow_html=True
    )

    source_lower = str(row['source']).lower()
    if 'pubpeer' in source_lower:
        st.markdown("**PubPeer Search Guidelines:**")
        guidelines = """
1. Search for the paper title or DOI on [PubPeer](https://pubpeer.com/).
2. Review all comment threads.
3. Look for any author responses.
4. Assess whether the identified error(s) critically affect the paper’s conclusions.
5. Confirm that the PDF version remains accessible.
"""
        st.code(guidelines, language='text')
    elif any(key in source_lower for key in ['withdrarxiv', 'withdraw_arxiv', 'arxiv']):
        st.markdown("**arXiv Withdrawal Guidelines:**")
        abs_url = f"https://arxiv.org/abs/{row['doi/arxiv_id'].strip()}"
        guidelines = f"""
1. Visit the abstract page: {abs_url}
2. Review all comments displayed.
3. Assess whether the identified error(s) critically affect the paper’s conclusions.
4. Confirm that the PDF version remains accessible.
"""
        st.code(guidelines, language='text')

# — Right: Annotation Section ———
with right_col:
    st.header("🔖 Annotation")

    error_count = st.number_input(
        "How many distinct errors are there in this paper?",
        min_value=1, max_value=20, step=1,
        key='error_count'
    )

    annotations = []
    for i in range(error_count):
        st.subheader(f"Error #{i+1}")
        error_ack = st.radio(
            f"1. Have the authors publicly acknowledged this error?",
            ["Yes", "No", "Uncertain"],
            key=f"ack_{i}" 
        )
        self_contained = st.radio(
            f"2. Can this error be identified by examining only the paper’s content?",
            ["Yes", "No"],
            key=f"self_{i}" 
        )
        is_severe = st.radio(
            f"3. Is this error severe?",
            ["Yes", "No"],
            key=f"sev_{i}" 
        )
        accessible = st.radio(
            f"4. Is the paper still accessible at its DOI/arXiv link?",
            ["Yes", "No"],
            key=f"acc_{i}" 
        )
        categories = [
            "Figure duplication", "Data inconsistency", "Equation typo",
            "Methodological issue", "Other"
        ]
        if error_count > 1:
            chosen = st.multiselect(
                f"5. Select all categories for error #{i+1}:",
                options=categories,
                key=f"cat_{i}" 
            )
        else:
            chosen = st.selectbox(
                f"5. Category for error #{i+1}:",
                options=categories,
                key=f"cat_{i}" 
            )
        if isinstance(chosen, list) and "Other" in chosen:
            other_text = st.text_input(f"⮕ Specify category for error #{i+1}:", key=f"oth_{i}")
            chosen = [c for c in chosen if c != "Other"] + [other_text]
        elif chosen == "Other":
            chosen = st.text_input(f"⮕ Specify category for error #{i+1}:", key=f"oth_{i}")

        summary = st.text_area(
            f"7. Summarize error #{i+1}:",
            height=120,
            key=f"sum_{i}"
        )
        annotations.append({
            'title': row['title'],
            'doi/arxiv_id': row['doi/arxiv_id'],
            'source': row['source'],
            'gpt4-summ': row['gpt4-summ'],
            'author_ack': error_ack,
            'self_contained': self_contained,
            'severe': is_severe,
            'accessible': accessible,
            'categories': chosen,
            'error_count': error_count,
            'summary': summary
        })

    if st.button('💾 Save All Annotations'):
        write_header = not os.path.exists(ANNOTATIONS_PATH)
        pd.DataFrame(annotations).to_csv(
            ANNOTATIONS_PATH,
            mode='a',
            header=write_header,
            index=False
        )
        st.success("✅ All annotations saved.")
