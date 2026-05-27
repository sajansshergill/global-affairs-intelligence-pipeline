"""
03_pipeline_health.py — Pipeline observability and data quality scorecard.

Shows:
  - KPI summary (total runs, rows, SLA breaches, drift events)
  - Rows ingested per source (bar chart)
  - Null rate per source (bar chart with threshold line)
  - SLA compliance scatter timeline
  - Schema drift event log
  - Full run log table with status formatting
  - Data quality scorecard from JSONL health log
  - Manual pipeline trigger (dev mode)
"""

import json
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import pandas as pd
import streamlit as st

st.set_page_config(
    page_title="Pipeline Health — GARIP",
    page_icon="🔧",
    layout="wide",
)

st.title("🔧 Pipeline Health Monitor")
st.caption("Observability for ingestion runs, data quality checks, and SLA compliance.")

# ------------------------------------------------------------------
# Load data
# ------------------------------------------------------------------

@st.cache_data(ttl=60)
def load_health():
    try:
        from storage.duckdb_loader import DuckDBLoader
        db_path = os.getenv("DUCKDB_PATH", "./data/garip.duckdb")
        with DuckDBLoader(db_path=db_path) as loader:
            recent  = loader.get_pipeline_health()
            all_runs = loader.query(
                "SELECT * FROM pipeline_health ORDER BY run_at DESC LIMIT 200"
            )
        return recent, all_runs, None
    except Exception as exc:
        return pd.DataFrame(), pd.DataFrame(), str(exc)


recent, all_runs, error = load_health()

if error:
    st.warning(f"⚠️ Health data unavailable: {error}")
    st.info("Run the ingestion pipeline to populate health metrics.")
    st.stop()

# ------------------------------------------------------------------
# KPI row
# ------------------------------------------------------------------

if not all_runs.empty:
    total_runs    = len(all_runs)
    sla_breaches  = int((~all_runs["sla_met"].astype(bool)).sum()) if "sla_met" in all_runs.columns else 0
    drift_events  = int(all_runs["schema_drift"].astype(bool).sum()) if "schema_drift" in all_runs.columns else 0
    total_rows    = int(all_runs["rows_ingested"].sum()) if "rows_ingested" in all_runs.columns else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔁 Total Runs",    total_runs)
    c2.metric("📥 Rows Ingested", f"{total_rows:,}")
    c3.metric("⚠️ SLA Breaches",  sla_breaches)
    c4.metric("🔀 Schema Drift",  drift_events)
    st.divider()

# ------------------------------------------------------------------
# Per-source charts (last 7 days)
# ------------------------------------------------------------------

st.subheader("📊 Source Summary — Last 7 Days")

if not recent.empty:
    try:
        import plotly.express as px

        col_rows, col_null = st.columns(2)

        with col_rows:
            if "total_rows" in recent.columns:
                fig = px.bar(
                    recent,
                    x="source",
                    y="total_rows",
                    color="source",
                    title="Total Rows Ingested by Source",
                    labels={"total_rows": "Rows", "source": "Source"},
                )
                fig.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig, use_container_width=True)

        with col_null:
            if "avg_null_rate_pct" in recent.columns:
                fig2 = px.bar(
                    recent,
                    x="source",
                    y="avg_null_rate_pct",
                    color="source",
                    title="Avg Null Rate % by Source",
                    labels={"avg_null_rate_pct": "Null Rate %", "source": "Source"},
                )
                fig2.add_hline(
                    y=10,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="10% threshold",
                )
                fig2.update_layout(showlegend=False, height=300)
                st.plotly_chart(fig2, use_container_width=True)

    except ImportError:
        st.dataframe(recent, use_container_width=True, hide_index=True)
else:
    st.info("No runs in the last 7 days.")

st.divider()

# ------------------------------------------------------------------
# SLA timeline scatter
# ------------------------------------------------------------------

st.subheader("⏱️ SLA Compliance Timeline")

if not all_runs.empty and "run_at" in all_runs.columns and "sla_met" in all_runs.columns:
    try:
        import plotly.express as px
        df_sla = all_runs.tail(100).copy()
        df_sla["run_at"]     = pd.to_datetime(df_sla["run_at"])
        df_sla["sla_status"] = df_sla["sla_met"].map(
            {True: "Met", False: "Breached", 1: "Met", 0: "Breached"}
        )
        fig_sla = px.scatter(
            df_sla,
            x="run_at",
            y="source",
            color="sla_status",
            symbol="sla_status",
            color_discrete_map={"Met": "#2ecc71", "Breached": "#e74c3c"},
            title="SLA Status per Run",
        )
        fig_sla.update_layout(height=300)
        st.plotly_chart(fig_sla, use_container_width=True)
    except Exception as exc:
        st.warning(f"SLA chart error: {exc}")

st.divider()

# ------------------------------------------------------------------
# Schema drift log
# ------------------------------------------------------------------

st.subheader("🔀 Schema Drift Events")

if not all_runs.empty and "schema_drift" in all_runs.columns:
    drift = all_runs[all_runs["schema_drift"].astype(bool)]
    if drift.empty:
        st.success("✅ No schema drift detected")
    else:
        st.warning(f"⚠️ {len(drift)} drift event(s)")
        cols = [c for c in ["run_id", "source", "run_at", "rows_ingested", "errors"]
                if c in drift.columns]
        st.dataframe(drift[cols], use_container_width=True, hide_index=True)

st.divider()

# ------------------------------------------------------------------
# Full run log
# ------------------------------------------------------------------

st.subheader("📋 Full Run Log")

if not all_runs.empty:
    col_f, _ = st.columns([1, 3])
    with col_f:
        src_filter = st.multiselect(
            "Filter by source",
            options=sorted(all_runs["source"].dropna().unique())
            if "source" in all_runs.columns else [],
        )

    display = all_runs if not src_filter else all_runs[all_runs["source"].isin(src_filter)]

    show_cols = [
        c for c in [
            "run_id", "source", "run_at", "rows_ingested",
            "duplicate_count", "null_rate", "schema_drift",
            "sla_met", "quality_passed",
        ]
        if c in display.columns
    ]
    styled = display[show_cols].copy()

    if "null_rate"     in styled.columns:
        styled["null_rate"]     = styled["null_rate"].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else ""
        )
    if "sla_met"       in styled.columns:
        styled["sla_met"]       = styled["sla_met"].map(
            {True: "✅", False: "❌", 1: "✅", 0: "❌"}
        )
    if "schema_drift"  in styled.columns:
        styled["schema_drift"]  = styled["schema_drift"].map(
            {True: "⚠️", False: "✅", 1: "⚠️", 0: "✅"}
        )
    if "quality_passed" in styled.columns:
        styled["quality_passed"] = styled["quality_passed"].map(
            {True: "✅", False: "❌", 1: "✅", 0: "❌", None: "—"}
        )

    st.dataframe(styled.head(50), use_container_width=True, hide_index=True)
    st.caption(f"Showing {min(50, len(display))} of {len(display)} runs")

st.divider()

# ------------------------------------------------------------------
# Quality scorecard from JSONL
# ------------------------------------------------------------------

st.subheader("✅ Latest Quality Scorecard")

jsonl_path = os.getenv("HEALTH_JSONL_PATH", "./data/raw/pipeline_health.jsonl")

if os.path.exists(jsonl_path):
    entries = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                entries.append(json.loads(line.strip()))
            except Exception:
                pass
    if entries:
        latest = entries[-1]
        st.json({
            "source":        latest.get("source"),
            "run_id":        latest.get("run_id"),
            "run_at":        latest.get("run_at"),
            "rows_ingested": latest.get("rows_ingested"),
            "null_rate":     f"{latest.get('null_rate', 0):.1%}",
            "schema_drift":  latest.get("schema_drift"),
            "sla_met":       latest.get("sla_met"),
            "errors":        latest.get("errors", []),
        })
    else:
        st.info("No entries in health log yet.")
else:
    st.info(f"Health log not found at `{jsonl_path}`. Run the pipeline first.")

st.divider()

# ------------------------------------------------------------------
# Manual trigger (dev mode)
# ------------------------------------------------------------------

st.subheader("🚀 Manual Pipeline Trigger")
st.caption("Trigger an ingestion run directly from the UI — development mode only.")

col_t1, col_t2, col_t3 = st.columns(3)
with col_t1:
    trigger_source = st.selectbox(
        "Source",
        ["all", "eurlex", "congress", "ftc", "regulations_gov", "ico"],
    )
with col_t2:
    trigger_limit = st.number_input(
        "Records per source", min_value=5, max_value=500, value=20
    )
with col_t3:
    st.write("")
    st.write("")
    if st.button("▶ Run", type="primary", use_container_width=True):
        with st.spinner(f"Running {trigger_source}…"):
            try:
                from ingestion.eurlex_connector      import EURLexConnector
                from ingestion.congress_connector    import CongressConnector
                from ingestion.ftc_connector         import FTCConnector
                from ingestion.regulations_connector import RegulationsGovConnector
                from ingestion.ico_connector         import ICOConnector
                import pandas as pd

                connector_map = {
                    "eurlex":          EURLexConnector,
                    "congress":        CongressConnector,
                    "ftc":             FTCConnector,
                    "regulations_gov": RegulationsGovConnector,
                    "ico":             ICOConnector,
                }
                to_run = (
                    list(connector_map.keys())
                    if trigger_source == "all"
                    else [trigger_source]
                )
                dfs = []
                for name in to_run:
                    df = connector_map[name]().run(limit=trigger_limit)
                    dfs.append(df)
                merged = pd.concat(dfs, ignore_index=True)
                st.success(f"✅ Ingested {len(merged)} records from {trigger_source}")
                st.cache_data.clear()
            except Exception as exc:
                st.error(f"❌ Ingestion failed: {exc}")