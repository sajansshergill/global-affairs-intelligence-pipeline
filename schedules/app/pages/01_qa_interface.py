"""
01_qa_interface.py — Policymaker Q&A interface.

Ask natural language questions about the regulatory corpus.
Answers are grounded in retrieved chunks with inline citations.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import streamlit as st

st.set_page_config(
    page_title="Q&A — GARIP",
    page_icon="💬",
    layout="wide",
)

# ------------------------------------------------------------------
# Load pipeline (cached — only initializes once per session)
# ------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading retrieval pipeline...")
def load_pipeline():
    try:
        from storage.vector_loader   import VectorLoader
        from storage.duckdb_loader   import DuckDBLoader
        from retrieval.hybrid_retriever import HybridRetriever
        from retrieval.query_rewriter   import QueryRewriter
        from api.llm_interface          import LLMInterface

        backend  = os.getenv("VECTOR_BACKEND", "chromadb")
        db_path  = os.getenv("DUCKDB_PATH", "./data/garip.duckdb")

        vector_loader = VectorLoader(backend=backend)
        rewriter      = QueryRewriter(mode="full")
        retriever     = HybridRetriever(
            vector_loader=vector_loader,
            top_k=6,
            use_reranker=True,
        )

        with DuckDBLoader(db_path=db_path) as loader:
            retriever.build_bm25_from_duckdb(loader)

        llm = LLMInterface(
            hybrid_retriever=retriever,
            query_rewriter=rewriter,
        )
        return llm, None

    except Exception as exc:
        return None, str(exc)


llm_interface, load_error = load_pipeline()

# ------------------------------------------------------------------
# Page header
# ------------------------------------------------------------------

st.title("💬 Policymaker Q&A")
st.caption(
    "Ask questions about global regulatory and compliance topics. "
    "Answers are grounded in the GARIP corpus with inline citations."
)

if load_error:
    st.warning(f"⚠️ Pipeline not fully loaded: {load_error}. Running in demo mode.")

# ------------------------------------------------------------------
# Controls
# ------------------------------------------------------------------

col_query, col_filter = st.columns([3, 1])

with col_filter:
    jurisdiction_options = ["Auto-detect", "EU", "US", "UK", "DE", "FR"]
    selected = st.selectbox(
        "Jurisdiction",
        jurisdiction_options,
        help="Filter retrieval to a specific jurisdiction.",
    )
    jurisdiction_filter = None if selected == "Auto-detect" else selected

with col_query:
    query = st.text_input(
        "Your question",
        placeholder="e.g. What are the GDPR breach notification requirements?",
    )

# Example query buttons
EXAMPLES = [
    "What are GDPR Article 17 right to erasure obligations?",
    "How does the EU AI Act regulate high-risk AI systems?",
    "What FTC enforcement actions targeted data privacy in 2023?",
    "How do EU and US cross-border data transfer rules conflict?",
    "What are GDPR penalties for data breaches?",
]

st.caption("**Examples:**")
cols = st.columns(len(EXAMPLES))
for col, example in zip(cols, EXAMPLES):
    with col:
        if st.button(example[:45] + "…", use_container_width=True, key=example):
            query = example

# ------------------------------------------------------------------
# Answer generation
# ------------------------------------------------------------------

if query:
    st.divider()

    with st.spinner("Retrieving documents and generating answer…"):
        if llm_interface:
            response = llm_interface.answer(
                question=query,
                jurisdiction_filter=jurisdiction_filter,
            )
        else:
            # Demo mode placeholder
            from api.llm_interface import QAResponse
            response = QAResponse(
                question=query,
                answer="Demo mode — set ANTHROPIC_API_KEY and run the pipeline.",
                key_takeaway="Configure your environment variables to enable live answers.",
                limitations="Pipeline not configured.",
                citations=[],
                retrieved_chunks=[],
                model="demo",
                detected_jurisdiction=jurisdiction_filter,
            )

    # Jurisdiction detection badge
    if response.detected_jurisdiction:
        st.info(f"🌍 Jurisdiction detected: **{response.detected_jurisdiction}**")

    # Answer
    st.subheader("📋 Answer")
    st.markdown(response.answer)

    if response.key_takeaway:
        st.success(f"**Key Takeaway:** {response.key_takeaway}")

    if response.limitations:
        st.warning(f"**Limitations:** {response.limitations}")

    # Citations
    if response.citations:
        st.divider()
        st.subheader("📚 Sources")
        for c in response.citations:
            with st.expander(
                f"[{c.index}] {c.regulation_title[:70]} | {c.jurisdiction}",
                expanded=False,
            ):
                col_meta, col_link = st.columns([3, 1])
                with col_meta:
                    st.markdown(f"**Type:** {c.regulation_type}")
                    if c.article_ref:
                        st.markdown(f"**Article:** {c.article_ref}")
                    if c.effective_date:
                        st.markdown(f"**Date:** {c.effective_date}")
                    st.markdown(f"**Excerpt:** *{c.text_excerpt}…*")
                with col_link:
                    if c.source_url:
                        st.markdown(f"[🔗 Source]({c.source_url})")

    # Retrieved chunks debug panel
    if response.retrieved_chunks:
        with st.expander(
            f"🔍 Retrieved chunks ({len(response.retrieved_chunks)})",
            expanded=False,
        ):
            for i, chunk in enumerate(response.retrieved_chunks):
                meta = chunk.get("metadata", {})
                st.markdown(
                    f"**[{i+1}]** `{chunk.get('chunk_id','')[:12]}` | "
                    f"{meta.get('jurisdiction','')} | "
                    f"score={chunk.get('score', 0):.4f} | "
                    f"{chunk.get('retrieval_method','')}"
                )
                st.caption(chunk.get("text", "")[:200])

# ------------------------------------------------------------------
# Session chat history
# ------------------------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

if query and "response" in dir():
    st.session_state.history.append({
        "question":     query,
        "answer":       response.answer,
        "jurisdiction": response.detected_jurisdiction,
    })

if st.session_state.history:
    st.divider()
    with st.expander("📜 Session history", expanded=False):
        for entry in reversed(st.session_state.history[-5:]):
            st.markdown(f"**Q:** {entry['question']}")
            st.markdown(f"**A:** {entry['answer'][:200]}…")
            st.divider()