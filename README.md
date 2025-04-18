# Annotation Platform

A simple Streamlit app for annotating errors in academic papers (PubPeer or arXiv withdrawals). Contributors can load random samples, answer guided questions, and save results to `annotations.csv`.

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- [Streamlit](https://streamlit.io/)
- pandas

### Installation

1. **Clone this repository**
   ```bash
   git clone https://github.com/guijinSON/ai4s_r2.git
   cd ai4s_r2
   ```

2. **Install dependencies**
   ```bash
   pip install streamlit pandas
   ```

3. **Dataset**

   The file `retracted_machine_filtered_final.csv` is already included in this repository.

---

## 🎯 Usage

1. **Run the app**
   ```bash
   streamlit run streamlit_sample.py
   ```

2. **Annotate**
   - Click **Shuffle Sample** to load a random paper.
   - Answer the six annotation questions on the right panel.
   - Click **Save Annotation** to append your answers to `annotations.csv`.

3. **Repeat** until you have completed **3–5 annotations**.

4. **Submit**
   - Upload or send your generated `annotations.csv` file back to the project maintainer.



