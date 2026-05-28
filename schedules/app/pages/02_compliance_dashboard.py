"""
02_compliance_dashboard.py — Jurisdiction-level compliance dashboard.

Shows:
  - KPI row: total regulations, versions, chunks, amendments
  - Regulation count by jurisdiction (bar + pie charts)
  - Regulation type × jurisdiction heatmap
  - Regulatory timeline by year
  - Cross-jurisdiction conflict signals (expandable cards)
  - Amendment history table
  - Filterable regulation browser
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Compliance Dashboard — GARIP",
    page_icon="📊",
    layout="wide",
)

st.title("📊 Compliance Dashboard")
st.caption(
    "Jurisdiction-level regulatory coverage, conflict signals, and amendment history."
)

# ------------------------------------------------------------------
# Load all data from DuckDB
# ------------------------------------------------------------------

@st.cache_data(ttl=300)
def load_data():
    try:
        from storage.duckdb_loader import DuckDBLoader
        db_path = os.getenv("DUCKDB_PATH", "./data/garip.duckdb")
        with DuckDBLoader(db_path=db_path) as loader:
            counts      = loader.get_total_counts()
            jur_stats   = loader.get_jurisdiction_stats()
            amendments  = loader.get_amendment_history()
            regulations = loader.get_regulations(limit=500)
            conflicts   = loader.query(
                "SELECT * FROM conflict_signals ORDER BY severity DESC LIMIT 50"
            )
        return counts, jur_stats, amendments, regulations, conflicts, None
    except Exception as exc:
        return {}, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), str(exc)


counts, jur_stats, amendments, regulations, conflicts, error = load_data()

if error:
    st.warning(f"⚠️ Could not load data: {error}")
    st.info(
        "Run the pipeline first:\n"
        "`dagster dev -f orchestration/dagster_pipeline.py`"
    )
    st.stop()

# ------------------------------------------------------------------
# KPI row
# ------------------------------------------------------------------

c1, c2, c3, c4 = st.columns(4)
c1.metric("📁 Regulations",   counts.get("total_regulations", 0))
c2.metric("🔄 Versions",      counts.get("total_versions",    0))
c3.metric("🧩 Chunks",        counts.get("total_chunks",      0))
c4.metric("✏️ Amendments",    len(amendments) if not amendments.empty else 0)

st.divider()

# ------------------------------------------------------------------
# Jurisdiction breakdown — bar + table
# ------------------------------------------------------------------

st.subheader("🌍 Regulations by Jurisdiction")

if not jur_stats.empty:
    try:
        import plotly.express as px

        col_bar, col_table = st.columns([1, 1])

        with col_bar:
            fig_bar = px.bar(
                jur_stats,
                x="jurisdiction",
                y="regulation_count",
                color="jurisdiction",
                title="Regulation Count by Jurisdiction",
                labels={
                    "regulation_count": "Count",
                    "jurisdiction":     "Jurisdiction",
                },
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_bar.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_table:
            show_cols = [
                c for c in [
                    "jurisdiction", "jurisdiction_name",
                    "regulation_count", "type_diversity",
                    "amendment_count", "latest_date",
                ]
                if c in jur_stats.columns
            ]
            st.dataframe(
                jur_stats[show_cols],
                use_container_width=True,
                hide_index=True,
            )

    except ImportError:
        st.dataframe(jur_stats, use_container_width=True, hide_index=True)
else:
    st.info("No jurisdiction data yet — run the ingestion pipeline.")

st.divider()

# ------------------------------------------------------------------
# Regulation type breakdown — pie + heatmap
# ------------------------------------------------------------------

st.subheader("📂 Regulation Type Distribution")

if not regulations.empty and "regulation_type" in regulations.columns:
    try:
        import plotly.express as px

        col_pie, col_heat = st.columns([1, 1])

        with col_pie:
            type_counts = (
                regulations["regulation_type"]
                .value_counts()
                .reset_index()
            )
            type_counts.columns = ["regulation_type", "count"]

            fig_pie = px.pie(
                type_counts.head(8),
                names="regulation_type",
                values="count",
                title="Regulation Type Share",
                color_discrete_sequence=px.colors.qualitative.Pastel,
            )
            fig_pie.update_layout(height=350)
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_heat:
            pivot = (
                regulations
                .groupby(["jurisdiction", "regulation_type"])
                .size()
                .reset_index(name="count")
            )
            if len(pivot) > 1:
                fig_heat = px.density_heatmap(
                    pivot,
                    x="jurisdiction",
                    y="regulation_type",
                    z="count",
                    title="Jurisdiction × Type Coverage",
                    color_continuous_scale="Blues",
                )
                fig_heat.update_layout(height=350)
                st.plotly_chart(fig_heat, use_container_width=True)

    except ImportError:
        st.bar_chart(regulations["regulation_type"].value_counts())

st.divider()

# ------------------------------------------------------------------
# Regulatory timeline
# ------------------------------------------------------------------

st.subheader("📅 Regulatory Timeline")

if not regulations.empty and "effective_date" in regulations.columns:
    dated = regulations.dropna(subset=["effective_date"]).copy()
    if not dated.empty:
        try:
            import plotly.express as px
            dated["effective_date"] = pd.to_datetime(
                dated["effective_date"], errors="coerce"
            )
            dated["year"] = dated["effective_date"].dt.year
            timeline = (
                dated
                .groupby(["year", "jurisdiction"])
                .size()
                .reset_index(name="count")
            )
            fig_line = px.line(
                timeline,
                x="year",
                y="count",
                color="jurisdiction",
                markers=True,
                title="Regulations by Year and Jurisdiction",
            )
            fig_line.update_layout(height=320)
            st.plotly_chart(fig_line, use_container_width=True)
        except Exception as exc:
            st.warning(f"Timeline error: {exc}")

st.divider()

# ------------------------------------------------------------------
# Cross-jurisdiction conflict signals
# ------------------------------------------------------------------

st.subheader("⚡ Cross-Jurisdiction Conflict Signals")

SEVERITY_ICON = {"high": "🔴", "medium": "🟡", "low": "🟢"}

if conflicts.empty:
    st.info(
        "No conflict signals yet. "
        "Run `ConflictDetector.detect_all()` after corpus is populated."
    )
else:
    for _, row in conflicts.iterrows():
        icon     = SEVERITY_ICON.get(str(row.get("severity", "low")), "⚪")
        jur_a    = row.get("jurisdiction_a", "")
        jur_b    = row.get("jurisdiction_b", "")
        topic    = row.get("topic", "Unknown topic")
        summary  = row.get("conflict_summary", "")
        severity = str(row.get("severity", "")).capitalize()
        detected = str(row.get("detected_at", ""))[:10]

        with st.expander(
            f"{icon} {topic} | {jur_a} ↔ {jur_b}",
            expanded=(row.get("severity") == "high"),
        ):
            st.markdown(f"**Summary:** {summary}")
            col_s, col_d = st.columns([1, 2])
            with col_s:
                st.caption(f"Severity: **{severity}**")
            with col_d:
                st.caption(f"Detected: {detected}")

st.divider()

# ------------------------------------------------------------------
# Amendment history
# ------------------------------------------------------------------

st.subheader("🔄 Amendment History")

if not amendments.empty:
    show_cols = [
        c for c in [
            "regulation_id", "title", "jurisdiction",
            "version_id", "effective_date", "ingested_at",
        ]
        if c in amendments.columns
    ]
    st.dataframe(
        amendments[show_cols],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No amendments detected yet.")

st.divider()

# ------------------------------------------------------------------
# Regulation browser with filters
# ------------------------------------------------------------------

st.subheader("📋 Regulation Browser")

if not regulations.empty:
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        filter_jur = st.multiselect(
            "Filter by jurisdiction",
            options=sorted(regulations["jurisdiction"].dropna().unique()),
        )
    with col_f2:
        filter_type = st.multiselect(
            "Filter by type",
            options=(
                sorted(regulations["regulation_type"].dropna().unique())
                if "regulation_type" in regulations.columns
                else []
            ),
        )

    filtered = regulations.copy()
    if filter_jur:
        filtered = filtered[filtered["jurisdiction"].isin(filter_jur)]
    if filter_type and "regulation_type" in filtered.columns:
        filtered = filtered[filtered["regulation_type"].isin(filter_type)]

    show_cols = [
        c for c in [
            "title", "jurisdiction", "regulation_type",
            "effective_date", "version_id", "source_url",
        ]
        if c in filtered.columns
    ]

    st.dataframe(
        filtered[show_cols].head(200),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        f"Showing {min(200, len(filtered))} of {len(filtered)} regulations"
    )
else:
    st.info("No regulations loaded yet.")