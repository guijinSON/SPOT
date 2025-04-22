# Streamlit Annotation Platform — full self‑contained script
# -----------------------------------------------------------
# Features
# • Sidebar “Show papers” filter:  Not yet | Annotated | Flagged
# • Assignee selector & manual picker always respect that filter
# • Annotation editor enabled ONLY for “Not yet” items
# • Uses st.rerun() (no deprecated calls)

import streamlit as st
import pandas as pd
import random
import os

# ——— PAGE CONFIG ———
st.set_page_config(page_title="Annotation Platform", layout="wide")

# ——— CONFIG ———
DATA_PATH        = 'retracted_machine_filtered_final.csv'
ANNOTATIONS_PATH = 'annotations.csv'
FLAGS_PATH       = 'flags.csv'
UPLOAD_DIR       = 'uploaded_pdfs'
DOI_COL          = 'doi/arxiv_id'          # single source of truth

# ——— UTILS ———
def safe_doi(doi: str) -> str:
    return doi

def read_first_col(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    try:
        series = pd.read_csv(path, usecols=[0], dtype=str).iloc[:, 0]
        return series.str.strip().tolist()
    except Exception:
        with open(path, 'r', encoding='utf-8', errors='ignore') as fh:
            return [ln.split(',')[0].strip() for ln in fh.read().splitlines()[1:] if ln]

# ——— LOAD EXISTING LISTS ———
annotated_dois = read_first_col(ANNOTATIONS_PATH)
flagged_dois   = read_first_col(FLAGS_PATH)
excluded_dois  = set(annotated_dois) | set(flagged_dois)

# ——— LOAD MASTER DATAFRAME ———
@st.cache_data(show_spinner=False)
def load_master():
    df = pd.read_csv(DATA_PATH, dtype=str)
    df[DOI_COL] = df[DOI_COL].str.strip()
    return df

df_master = load_master()

# ——— SIDEBAR: STATUS FILTER ———
status_choice = st.sidebar.selectbox(
    "Show papers",
    ("Not yet", "Annotated", "Flagged"),
    index=0
)

if status_choice == "Not yet":
    base_df = df_master[~df_master[DOI_COL].isin(excluded_dois)]
elif status_choice == "Annotated":
    base_df = df_master[df_master[DOI_COL].isin(annotated_dois)]
else:  # "Flagged"
    base_df = df_master[df_master[DOI_COL].isin(flagged_dois)]

# ——— SIDEBAR: ASSIGNEE FILTER ———
assignees = sorted(base_df['assigned_to'].dropna().unique())
if not assignees:
    st.error("No papers match the current filters.")
    st.stop()

selected_assignee = st.sidebar.selectbox("Select Assignee", assignees)
papers = base_df[base_df['assigned_to'] == selected_assignee].reset_index(drop=True)
num_papers = len(papers)
if num_papers == 0:
    st.warning("No papers for this assignee under current filter.")
    st.stop()

# ——— SESSION STATE: CURRENT INDEX ———
if 'idx' not in st.session_state:
    st.session_state.idx = 0
st.session_state.idx = min(st.session_state.idx, num_papers - 1)

# ——— SIDEBAR NAVIGATION ———
nav_prev, nav_next = st.sidebar.columns(2)
if nav_prev.button("← Previous", use_container_width=True):
    st.session_state.idx = max(0, st.session_state.idx - 1)
if nav_next.button("Next →", use_container_width=True):
    st.session_state.idx = min(num_papers - 1, st.session_state.idx + 1)

doi_input = st.sidebar.text_input("Jump to DOI")
if st.sidebar.button("Go", use_container_width=True):
    idxs = papers.index[papers[DOI_COL] == doi_input.strip()]
    if not idxs.empty:
        st.session_state.idx = int(idxs[0])
    else:
        st.sidebar.error("DOI not found in current view.")

# ——— SIDEBAR PROGRESS FOR ASSIGNEE ———
tot_for_assignee = df_master[df_master['assigned_to'] == selected_assignee]
done_for_assignee = tot_for_assignee[tot_for_assignee[DOI_COL].isin(annotated_dois)]
st.sidebar.markdown("### 📊 Progress")
st.sidebar.write(f"Annotated: **{len(done_for_assignee)}**")
st.sidebar.write(f"Visible in list: **{num_papers}**")

# ——— SIDEBAR MANUAL PICKER ———
titles = papers['title'].tolist()
manual_pick = st.sidebar.radio("Quick select", titles, index=st.session_state.idx)
if st.sidebar.button("Load", use_container_width=True):
    st.session_state.idx = titles.index(manual_pick)
if st.sidebar.button("🔀 Random", use_container_width=True):
    st.session_state.idx = random.randint(0, num_papers - 1)

# ——— CURRENT PAPER ROW ———
row = papers.iloc[st.session_state.idx]

# ——— DISPLAY GUIDELINES ———
source_lower = str(row['source']).lower()
with st.expander("Guidelines", expanded=True):
    if 'pubpeer' in source_lower:
        st.code("""1· Search the DOI at https://pubpeer.com/
2· Read the entire thread
3· Note any author responses
4· Decide if error(s) critically affect conclusions
5· Confirm the PDF is still available""")
    elif any(k in source_lower for k in ['withdraw', 'arxiv']):
        abs_url = f"https://arxiv.org/abs/{row[DOI_COL]}"
        st.code(f"""1· Visit {abs_url}
2· Read the withdrawal comment
3· Decide if error(s) critically affect conclusions
4· Confirm the PDF is still available""")
    else:
        st.write("No specific guideline for this source.")

# ——— MAIN LAYOUT ———
left, right = st.columns([2, 3])

with left:
    st.markdown(f"## 📄 {row['title']}")
    st.write(f"**DOI / arXiv ID:** {row[DOI_COL]}")
    st.write(f"**Source:** {row['source']}")
    st.write(f"**Assignee:** {row['assigned_to']}")
    st.markdown("#### GPT‑4 Summary")
    st.markdown(f"<div style='white-space:pre-wrap'>{row.get('gpt4-summ', '')}</div>",
                unsafe_allow_html=True)

with right:
    st.header("🔖 Annotation")

    if status_choice != "Not yet":
        st.info(f"This paper is already **{status_choice.lower()}**. "
                "Editing disabled.")
        st.stop()

    # ——— ANNOTATION FORM ———
    error_count = st.number_input("Number of distinct errors", 1, 20, 1)

    if not os.path.exists(UPLOAD_DIR):
        os.makedirs(UPLOAD_DIR, exist_ok=True)

    categories = [
        "Figure duplication", "Data inconsistency", "Figure‑Text inconsistency",
        "Equation / proof", "Statistical reporting", "Reagent identity", "Other"
    ]

    annotations = []
    for i in range(int(error_count)):
        st.subheader(f"Error {i+1}")
        ack  = st.radio("1. Author acknowledged?",  ("Yes", "No"), key=f"ack_{i}")
        selfc= st.radio("2. Self‑contained (detectable from paper alone)?", ("Yes", "No"), key=f"self_{i}")
        sev  = st.radio("3. Severe (undermines conclusions)?", ("Yes", "No"), key=f"sev_{i}")
        accessible = st.radio("4. Error version still accessible?", ("Yes", "No"), key=f"acc_{i}")

        pdf_path = ""
        if accessible == "Yes":
            pdf = st.file_uploader("Upload error‑version PDF", type=["pdf"], key=f"pdf_err_{i}")
            if pdf is not None:
                pdf_path = os.path.join(UPLOAD_DIR, f"{safe_doi(row[DOI_COL])}_err{i+1}.pdf")
                with open(pdf_path, 'wb') as fh:
                    fh.write(pdf.getbuffer())
        else:
            recon = st.radio("4b. Reconstructable from newer version?", ("Yes", "No"), key=f"recon_{i}")
            if recon == "Yes":
                pdf = st.file_uploader("Upload reconstructed PDF", type=["pdf"], key=f"pdf_rec_{i}")
                if pdf is not None:
                    pdf_path = os.path.join(UPLOAD_DIR, f"{safe_doi(row[DOI_COL])}_recon{i+1}.pdf")
                    with open(pdf_path, 'wb') as fh:
                        fh.write(pdf.getbuffer())

        cat_sel = st.multiselect("5. Category", categories, key=f"cat_{i}")
        if "Other" in cat_sel:
            txt = st.text_input("⮕ Specify other", key=f"other_{i}")
            cat_sel = [c for c in cat_sel if c != "Other"] + ([txt] if txt else [])

        loc   = st.text_input("6. Section / Figure / Table number", key=f"loc_{i}")
        quote = st.text_area("7. Paste sentence where author recognises error (if any)", key=f"quote_{i}")

        annotations.append({
            'title'                  : row['title'],
            DOI_COL                  : row[DOI_COL],
            'assigned_to'            : selected_assignee,
            'author_ack'             : ack,
            'self_contained'         : selfc,
            'severe'                 : sev,
            'error_version_accessible': accessible,
            'pdf_path'               : pdf_path,
            'categories'             : ';'.join(cat_sel),
            'location'               : loc,
            'quote'                  : quote
        })

    # ——— SAVE OR FLAG ———
    st.divider()
    col_save, col_flag = st.columns(2)

    with col_save:
        if st.button("💾 Save annotations",
                     key=f"save_{row[DOI_COL]}",
                     use_container_width=True):
    
            df_out = pd.DataFrame(annotations)
    
            # —— reorder so DOI / arXiv ID is the FIRST column ——
            cols = [DOI_COL] + [c for c in df_out.columns if c != DOI_COL]
            df_out = df_out[cols]
    
            header = not os.path.exists(ANNOTATIONS_PATH)
            df_out.to_csv(ANNOTATIONS_PATH, mode='a',
                          header=header, index=False)
    
            st.success("Saved! Moving to next paper…")
            st.rerun()


    with col_flag:
        if st.button("🚩 Flag this paper", key=f"flag_{row[DOI_COL]}",use_container_width=True):
            header = not os.path.exists(FLAGS_PATH)
            pd.DataFrame([{DOI_COL: row[DOI_COL], 'assigned_to': selected_assignee}]).to_csv(
                FLAGS_PATH, mode='a', header=header, index=False)
            st.warning("Paper flagged — it will disappear from current view.")
            st.rerun()
