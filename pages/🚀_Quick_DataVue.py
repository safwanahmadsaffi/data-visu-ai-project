# 🚀 Quick_DataVue.py
# Streamlit EDA app that avoids ydata-profiling and re-uses existing helpers.

import base64
import io
import os
import tempfile

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import streamlit as st

# ──────────────────────────────────────────────
# 🔧 1. HELPER FUNCTIONS  (your existing code + a few extras)
# ──────────────────────────────────────────────

def load_css(css_file: str = "style.css"):
    """
    Load a local CSS file (if present) to style the app.
    """
    if os.path.exists(css_file):
        with open(css_file, "r") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

def load_example_data(dataset_name):
    """
    Ship with two built-in demo datasets so users can play without uploading.
    """
    if dataset_name == "Titanic":
        return sns.load_dataset("titanic")
    elif dataset_name == "Iris":
        return sns.load_dataset("iris")
    return pd.DataFrame()

def get_download_link(file_name, link_text):
    """
    Base64-encode a local file so the user can download the generated report.
    """
    with open(file_name, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    href = f'<a href="data:file/html;base64,{b64}" download="{file_name}">{link_text}</a>'
    return href

def generate_summary_report(df: pd.DataFrame, file_name="summary_report.html"):
    """
    Create a very lightweight HTML summary (describe table only) for download.
    """
    html = df.describe(include="all").to_html(classes="table table-striped", border=0)
    with open(file_name, "w") as f:
        f.write("<h2>Summary Statistics</h2>\n" + html)
    return file_name

def auto_eda():
    """
    Main UI + logic; placed in its own function so you can call it
    from Multi-Page Streamlit setups if you wish.
    """
    # ── Sidebar UI ───────────────────────────────
    st.sidebar.image("https://via.placeholder.com/150x150.png?text=DataVue", width=150)
    st.sidebar.title("DataVue")
    st.sidebar.subheader("Auto EDA Tool")

    # Upload or pick demo data
    st.sidebar.markdown("### Load data")
    demo_choice = st.sidebar.selectbox("Or choose a demo dataset", ["None", "Titanic", "Iris"])
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

    # Decide which dataframe to use
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        data_source = f"Uploaded file • `{uploaded_file.name}`"
    elif demo_choice != "None":
        df = load_example_data(demo_choice)
        data_source = f"Demo dataset • **{demo_choice}**"
    else:
        st.info("⬅️  Upload a CSV or select a demo dataset to get started.")
        return

    # ── Main Page ────────────────────────────────
    st.title("🚀 Quick DataVue – Exploratory Data Analysis Dashboard")
    st.caption(f"Data source: {data_source}")

    # 1. Quick glance
    st.header("🔎 Preview & Shape")
    st.write(df.head())
    st.write(f"Shape: **{df.shape[0]} rows × {df.shape[1]} columns**")

    # 2. Summary stats
    st.header("📊 Summary Statistics")
    st.dataframe(df.describe(include="all"))

    # 3. Missing-value scan
    st.header("🧱 Missing Values")
    na_counts = df.isna().sum().sort_values(ascending=False)
    st.bar_chart(na_counts)

    # 4. Correlation heatmap (numeric only)
    st.header("📈 Correlation Heatmap")
    num_df = df.select_dtypes(include=["int64", "float64"])
    if num_df.empty:
        st.warning("No numeric columns available for correlation.")
    else:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # 5. Distributions / value counts
    st.header("📌 Column Distributions")
    column = st.selectbox("Pick a column", df.columns)
    if pd.api.types.is_numeric_dtype(df[column]):
        fig2, ax2 = plt.subplots()
        sns.histplot(df[column].dropna(), kde=True, ax=ax2)
        st.pyplot(fig2)
    else:
        st.bar_chart(df[column].value_counts())

    # 6. Download tiny HTML summary
    st.header("📥 Download Report")
    if st.button("Generate & Download"):
        html_path = generate_summary_report(df)
        st.success("Report generated!")
        st.markdown(get_download_link(html_path, "⬇️ Download summary_report.html"), unsafe_allow_html=True)

# ──────────────────────────────────────────────
# 🚀 APP ENTRYPOINT
# ──────────────────────────────────────────────
def main():
    load_css()        # Optional custom styles
    auto_eda()        # Build EDA UI

if __name__ == "__main__":
    main()
