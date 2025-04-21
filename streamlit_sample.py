import streamlit as st
import pandas as pd
import random
import os

# ——— PAGE CONFIG ———
st.set_page_config(page_title="Annotation Platform", layout="wide")

# ——— CONFIG ———
DATA_PATH = 'retracted_machine_filtered_final.csv'
ANNOTATIONS_PATH = 'annotations.csv'
FLAGS_PATH = 'flags.csv'

# ——— LOAD DATA ———
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df_full = load_data()

# ——— FILTER BY ASSIGNEE ———
assignees = sorted(df_full['assigned_to'].dropna().unique())
selected_assignee = st.sidebar.selectbox("Select Assignee:", assignees)
# filter and reset index for positional indexing
df = df_full[df_full['assigned_to'] == selected_assignee].reset_index(drop=True)
num_papers = len(df)

# ——— STATE: CURRENT ROW ———
if 'idx' not in st.session_state:
    st.session_state.idx = 0

# ——— SIDEBAR: NAVIGATION: Previous/Next & DOI Jump ———
if st.sidebar.button('← Previous'):
    st.session_state.idx = max(0, st.session_state.idx - 1)
if st.sidebar.button('Next →'):
    st.session_state.idx = min(num_papers - 1, st.session_state.idx + 1)

doi_input = st.sidebar.text_input("Jump to DOI:", "")
if st.sidebar.button("Go to DOI"):
    dois = df['doi/arxiv_id'].tolist()
    if doi_input in dois:
        st.session_state.idx = dois.index(doi_input)
    else:
        st.sidebar.error("DOI not found for this assignee.")

# ——— SIDEBAR: PROGRESS ———
st.sidebar.header("📋 Navigation & Progress")
completed = 0
if os.path.exists(ANNOTATIONS_PATH):
    try:
        completed = len(pd.read_csv(ANNOTATIONS_PATH))
    except pd.errors.ParserError:
        with open(ANNOTATIONS_PATH, 'r', encoding='utf-8', errors='ignore') as f:
            completed = sum(1 for _ in f) - 1
st.sidebar.write(f"Progress: **{completed}** annotated / **{num_papers}** total")
st.sidebar.progress(completed / num_papers if num_papers else 0)

# ——— PAPER SELECTION (for manual override) ———
titles = df['title'].tolist()
selection = st.sidebar.radio(
    "Select Paper:",
    options=titles,
    index=st.session_state.idx
)
if st.sidebar.button('Go'):
    st.session_state.idx = titles.index(selection)
if st.sidebar.button('🔀 Shuffle Sample'):
    st.session_state.idx = random.randint(0, num_papers - 1)

row = df.iloc[st.session_state.idx]

# ——— TOP: DYNAMIC GUIDELINES ———
source_lower = str(row['source']).lower()
if 'pubpeer' in source_lower:
    st.markdown("**PubPeer Search Guidelines:**")
    guidelines = """
1. Search for the DOI on [PubPeer](https://pubpeer.com/).
2. Read all comment threads.
3. Check whether the authors responded.
4. Assess whether the identified error(s) critically affect the paper’s conclusions.
5. Confirm that the PDF version remains accessible.
"""
    st.code(guidelines, language='text')
elif any(key in source_lower for key in ['withdrarxiv', 'withdraw_arxiv', 'arxiv']):
    st.markdown("**arXiv Withdrawal Guidelines:**")
    abs_url = f"https://arxiv.org/abs/{row['doi/arxiv_id'].strip()}"
    guidelines = f"""
1. Visit the abstract page: {abs_url}
2. Read the comment.
3. Assess whether the identified error(s) critically affect the paper’s conclusions.
4. Confirm that the PDF version remains accessible.
"""
    st.code(guidelines, language='text')

# ——— LAYOUT: TWO COLUMNS ———
left_col, right_col = st.columns([2, 3])

# — Left: Paper Info and Summary ———
with left_col:
    st.markdown(f"### 📄 {row['title']}")
    st.markdown(f"**DOI/arXiv ID:** {row['doi/arxiv_id']}")
    st.markdown(f"**Assigned To:** {selected_assignee}")
    st.markdown(f"**Source:** {row['source']}")
    st.markdown("**GPT‑4 Summary:**")
    st.markdown(
        f"<div style='white-space: pre-wrap; line-height:1.5;'>{row['gpt4-summ']}</div>",
        unsafe_allow_html=True
    )

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
            ["Yes", "No"],
            key=f"ack_{i}" 
        )
        self_contained = st.radio(
            f"2. Can this error be identified by examining only the paper’s content? (Is the error self-contained?)",
            ["Yes", "No"],
            key=f"self_{i}" 
        )
        is_severe = st.radio(
            f"3. Is the error critical enough to (partially) challenge or doubt the findings of the paper?",
            ["Yes", "No"],
            key=f"sev_{i}" 
        )
        # File-handling: error version or reconstruction
        error_version_accessible = st.radio(
            f"4. Is the version containing this error still accessible?",
            ["Yes", "No"],
            key=f"acc_{i}"
        )
        error_pdf_path = ""
        if error_version_accessible == "Yes":
            pdf_file = st.file_uploader(
                f"📄 Upload the original PDF (error version) for error #{i+1}:",
                type=["pdf"],
                key=f"file_error_{i}"
            )
            if pdf_file is not None:
                os.makedirs("uploaded_pdfs", exist_ok=True)
                safe_id = row['doi/arxiv_id'].replace('/', '_')
                file_path = f"uploaded_pdfs/{safe_id}_error{i+1}.pdf"
                with open(file_path, "wb") as f:
                    f.write(pdf_file.getbuffer())
                error_pdf_path = file_path
        else:
            recon_possible = st.radio(
                f"4b. Can this error be reconstructed from a newer version?",
                ["Yes", "No"],
                key=f"recon_{i}"
            )
            if recon_possible == "Yes":
                pdf_file = st.file_uploader(
                    f"📄 Upload the newer PDF (reconstructed) for error #{i+1}:",
                    type=["pdf"],
                    key=f"file_recon_{i}"
                )
                if pdf_file is not None:
                    os.makedirs("uploaded_pdfs", exist_ok=True)
                    safe_id = row['doi/arxiv_id'].replace('/', '_')
                    file_path = f"uploaded_pdfs/{safe_id}_recon{i+1}.pdf"
                    with open(file_path, "wb") as f:
                        f.write(pdf_file.getbuffer())
                    error_pdf_path = file_path

        categories = [
            "Figure duplication", "Data inconsistency", "Figure-Text Inconsistency", "proof/equation",
            "statistical reporting error", "reagent identity error", "Other"
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
            f"7. Copy paste the sentence the author recognizes error #{i+1}:",
            height=120,
            key=f"sum_{i}"
        )

        annotations.append({
            'title': row['title'],
            'doi/arxiv_id': row['doi/arxiv_id'],
            'assigned_to': selected_assignee,
            'author_ack': error_ack,
            'self_contained': self_contained,
            'severe': is_severe,
            'error_version_accessible': error_version_accessible,
            'error_pdf_path': error_pdf_path,
            'categories': chosen,
            'error_count': error_count,
            'summary': summary
        })

    # ——— Flag Issue Button ———
    if st.button('⚑ Flag Issue'):
        flag_entry = {'doi/arxiv_id': row['doi/arxiv_id'], 'assigned_to': selected_assignee}
        write_flag_header = not os.path.exists(FLAGS_PATH)
        pd.DataFrame([flag_entry]).to_csv(FLAGS_PATH, mode='a', header=write_flag_header, index=False)
        st.success("🚩 Instance flagged for review.")

    # ——— Save Annotations Button ———
    if st.button('💾 Save All Annotations'):
        write_header = not os.path.exists(ANNOTATIONS_PATH)
        pd.DataFrame(annotations).to_csv(
            ANNOTATIONS_PATH,
            mode='a',
            header=write_header,
            index=False
        )
        st.success("✅ All annotations saved.")
