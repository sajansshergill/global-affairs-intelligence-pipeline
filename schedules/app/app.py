"""
app.py — GARIP Streamlit application entrypoint.

Three pages:
  01_qa_interface.py       Policymaker Q&A with cited answers
  02_compliance_dashboard  Jurisdiction-level stats and conflict signals
  03_pipeline_health       Ingestion runs, SLA tracking, quality scorecard

Run:
  streamlit run app/app.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st

st.set_page_config(
    page_title="GARIP — Global Affairs Regulatory Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.title("⚖️ GARIP")
    st.caption("Global Affairs Regulatory Intelligence Pipeline")
    st.divider()

    st.markdown("**Navigate**")
    st.page_link("pages/01_qa_interface.py",         label="💬 Policymaker Q&A")
    st.page_link("pages/02_compliance_dashboard.py", label="📊 Compliance Dashboard")
    st.page_link("pages/03_pipeline_health.py",      label="🔧 Pipeline Health")
    st.divider()

    # Live corpus stats from DuckDB
    st.markdown("**Corpus**")
    try:
        from storage.duckdb_loader import DuckDBLoader
        db_path = os.getenv("DUCKDB_PATH", "./data/garip.duckdb")
        with DuckDBLoader(db_path=db_path) as loader:
            counts = loader.get_total_counts()
        st.metric("Regulations", counts.get("total_regulations", 0))
        st.metric("Chunks",      counts.get("total_chunks", 0))
        st.metric("Runs",        counts.get("total_runs", 0))
    except Exception:
        st.metric("Regulations", "—")
        st.metric("Chunks",      "—")
        st.caption("⚠️ Run the pipeline first.")

    st.divider()
    st.caption("Sajan Shergill · M.S. Data Science · Pace University")

# ------------------------------------------------------------------
# Home page
# ------------------------------------------------------------------

st.title("⚖️ Global Affairs Regulatory Intelligence Pipeline")
st.subheader("End-to-end ETL + RAG system for cross-jurisdiction compliance intelligence")

st.markdown("""
GARIP ingests regulatory documents from **5 public sources** across EU, US, and UK,
transforms them through a structured ETL pipeline, and serves a policymaker Q&A
interface powered by the Claude API.
""")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 🔄 Data Sources")
    st.markdown("""
| Source | Jurisdiction | Content |
|---|---|---|
| EUR-Lex | 🇪🇺 EU | Regulations, Directives |
| Congress.gov | 🇺🇸 US | Federal legislation |
| FTC | 🇺🇸 US | Enforcement actions |
| Regulations.gov | 🇺🇸 US | Federal rulemaking |
| ICO | 🇬🇧 UK | GDPR decisions |
""")

with col2:
    st.markdown("### 🏗️ Stack")
    st.info("DuckDB → BigQuery\nPinecone → Vertex AI Vector Search\nDagster → Cloud Composer\nStreamlit → Looker")

st.divider()

st.markdown("### ⚡ Pipeline")
st.code(
    "Connectors → ETL → DuckDB + Pinecone → Hybrid RAG → Claude API → Streamlit",
    language=None,
)