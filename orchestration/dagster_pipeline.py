"""
dagster_pipeline.py — Dagster job definitions and schedules for GARIP.

Defines the full pipeline as Dagster software-defined assets:

  raw_regulations          → ingest from all 5 connectors
  validated_regulations    → quality checks
  deduplicated_regulations → dedup + amendment detection
  tagged_regulations       → NER enrichment
  duckdb_regulations       → load into DuckDB (BigQuery equivalent)
  vector_store_chunks      → chunk + embed + upsert to vector store

Two jobs:
  garip_full_pipeline      → all 6 assets end-to-end
  garip_ingestion_only     → ingestion + validation only (no storage write)

Two schedules:
  daily  → 6 AM UTC every day
  weekly → 2 AM UTC every Sunday

GCP equivalent: Cloud Composer (Airflow) DAG with same task graph.

Run locally:
  dagster dev -f orchestration/dagster_pipeline.py
  Open: http://localhost:3000
"""

import logging
import os

logger = logging.getLogger(__name__)

try:
    from dagster import (
        asset,
        define_asset_job,
        AssetIn,
        AssetSelection,
        Definitions,
        MetadataValue,
        Output,
        ScheduleDefinition,
        get_dagster_logger,
    )
    HAS_DAGSTER = True
except ImportError:
    HAS_DAGSTER = False
    logger.warning(
        "dagster not installed — orchestration layer unavailable. "
        "Run: pip install dagster dagster-webserver"
    )


if HAS_DAGSTER:

    # ------------------------------------------------------------------
    # Asset 1: Raw ingestion
    # ------------------------------------------------------------------

    @asset(
        description="Ingest regulatory documents from all 5 configured sources.",
        compute_kind="python",
        group_name="ingestion",
    )
    def raw_regulations():
        """
        Runs all GARIP connectors and returns merged DataFrame.
        Connector selection and record limit controlled via env vars:
          GARIP_SOURCES      comma-separated list (default: all)
          GARIP_INGEST_LIMIT records per source   (default: 100)
        """
        log = get_dagster_logger()

        from ingestion.eurlex_connector      import EURLexConnector
        from ingestion.congress_connector    import CongressConnector
        from ingestion.ftc_connector         import FTCConnector
        from ingestion.regulations_connector import RegulationsGovConnector
        from ingestion.ico_connector         import ICOConnector

        import pandas as pd

        limit       = int(os.getenv("GARIP_INGEST_LIMIT", "100"))
        sources_env = os.getenv("GARIP_SOURCES", "")
        raw_dir     = os.getenv("RAW_DATA_DIR", "./data/raw")

        connector_map = {
            "eurlex":           EURLexConnector,
            "congress":         CongressConnector,
            "ftc":              FTCConnector,
            "regulations_gov":  RegulationsGovConnector,
            "ico":              ICOConnector,
        }

        selected = (
            [s.strip() for s in sources_env.split(",") if s.strip()]
            if sources_env
            else list(connector_map.keys())
        )

        dfs = []
        for name in selected:
            cls = connector_map.get(name)
            if not cls:
                log.warning(f"Unknown source: {name} — skipping")
                continue
            try:
                df = cls(raw_data_dir=raw_dir).run(limit=limit)
                dfs.append(df)
                log.info(f"{name}: {len(df)} rows ingested")
            except Exception as exc:
                log.error(f"{name}: FAILED — {exc}")

        if not dfs:
            raise ValueError("No data ingested from any source")

        merged = pd.concat(dfs, ignore_index=True)
        before = len(merged)
        merged = merged.drop_duplicates(subset=["source_hash"])
        cross_dupes = before - len(merged)

        log.info(
            f"Ingestion complete: {len(merged)} rows "
            f"(cross-source dupes dropped: {cross_dupes})"
        )

        return Output(
            value=merged,
            metadata={
                "num_records":   MetadataValue.int(len(merged)),
                "sources_run":   MetadataValue.text(str(selected)),
                "cross_dupes":   MetadataValue.int(cross_dupes),
                "jurisdictions": MetadataValue.text(
                    str(merged["jurisdiction"].value_counts().to_dict())
                    if not merged.empty else "{}"
                ),
            },
        )

    # ------------------------------------------------------------------
    # Asset 2: Quality validation
    # ------------------------------------------------------------------

    @asset(
        ins={"raw_regulations": AssetIn()},
        description="Run data quality checks — fails asset if critical checks fail.",
        compute_kind="python",
        group_name="etl",
    )
    def validated_regulations(raw_regulations):
        log = get_dagster_logger()
        from extraction.quality_checks import QualityChecker

        if raw_regulations.empty:
            raise ValueError("No records to validate")

        checker = QualityChecker(source="merged")
        suite   = checker.run(raw_regulations)

        log.info(suite.summary())

        if not suite.passed:
            failures = [f.message for f in suite.critical_failures]
            raise ValueError(f"Quality validation failed: {failures}")

        return Output(
            value=raw_regulations,
            metadata={
                "checks_passed":      MetadataValue.int(
                    sum(1 for c in suite.checks if c.passed)
                ),
                "total_checks":       MetadataValue.int(len(suite.checks)),
                "critical_failures":  MetadataValue.int(len(suite.critical_failures)),
                "warnings":           MetadataValue.int(len(suite.warnings)),
            },
        )

    # ------------------------------------------------------------------
    # Asset 3: Deduplication + amendment detection
    # ------------------------------------------------------------------

    @asset(
        ins={"validated_regulations": AssetIn()},
        description="Deduplicate records and detect amendments (version_id bumps).",
        compute_kind="python",
        group_name="etl",
    )
    def deduplicated_regulations(validated_regulations):
        log = get_dagster_logger()
        from extraction.deduplicator import Deduplicator
        from storage.duckdb_loader   import DuckDBLoader

        deduper = Deduplicator()

        # Load existing records for cross-batch amendment detection
        existing_df = None
        try:
            db_path = os.getenv("DUCKDB_PATH", "./data/garip.duckdb")
            with DuckDBLoader(db_path=db_path) as loader:
                existing_df = loader.get_regulations(limit=10_000)
        except Exception as exc:
            log.warning(f"Could not load existing records for dedup: {exc}")

        df, report = deduper.deduplicate(validated_regulations, existing_df=existing_df)

        log.info(
            f"Dedup: {report.input_rows} → {report.output_rows} "
            f"(exact={report.exact_duplicates_dropped} "
            f"amendments={report.amendments_detected})"
        )

        return Output(
            value=df,
            metadata={
                "input_rows":    MetadataValue.int(report.input_rows),
                "output_rows":   MetadataValue.int(report.output_rows),
                "exact_dropped": MetadataValue.int(report.exact_duplicates_dropped),
                "amendments":    MetadataValue.int(report.amendments_detected),
            },
        )

    # ------------------------------------------------------------------
    # Asset 4: NER tagging
    # ------------------------------------------------------------------

    @asset(
        ins={"deduplicated_regulations": AssetIn()},
        description="Enrich records with NER tags — jurisdiction, dates, article citations.",
        compute_kind="python",
        group_name="etl",
    )
    def tagged_regulations(deduplicated_regulations):
        log = get_dagster_logger()
        from etl.ner_tagger import NERTagger

        tagger = NERTagger(use_spacy=True)
        df     = tagger.tag_dataframe(deduplicated_regulations)

        log.info(f"NER tagging complete: {len(df)} records enriched")

        return Output(
            value=df,
            metadata={"num_tagged": MetadataValue.int(len(df))},
        )

    # ------------------------------------------------------------------
    # Asset 5: DuckDB load
    # ------------------------------------------------------------------

    @asset(
        ins={"tagged_regulations": AssetIn()},
        description="Load structured metadata into DuckDB (BigQuery equivalent).",
        compute_kind="duckdb",
        group_name="storage",
    )
    def duckdb_regulations(tagged_regulations):
        log = get_dagster_logger()
        from storage.duckdb_loader import DuckDBLoader

        db_path = os.getenv("DUCKDB_PATH", "./data/garip.duckdb")
        with DuckDBLoader(db_path=db_path) as loader:
            rows_loaded = loader.load_regulations(tagged_regulations)
            counts      = loader.get_total_counts()

        log.info(f"DuckDB load complete: {rows_loaded} rows")

        return Output(
            value=rows_loaded,
            metadata={
                "rows_loaded":        MetadataValue.int(rows_loaded),
                "total_regulations":  MetadataValue.int(counts.get("total_regulations", 0)),
                "total_versions":     MetadataValue.int(counts.get("total_versions", 0)),
            },
        )

    # ------------------------------------------------------------------
    # Asset 6: Vector store chunking + embedding
    # ------------------------------------------------------------------

    @asset(
        ins={"tagged_regulations": AssetIn()},
        description="Chunk documents and upsert embeddings to the vector store.",
        compute_kind="python",
        group_name="storage",
    )
    def vector_store_chunks(tagged_regulations):
        log = get_dagster_logger()
        from extraction.chunker    import Chunker
        from storage.vector_loader import VectorLoader
        from storage.duckdb_loader import DuckDBLoader

        backend = os.getenv("VECTOR_BACKEND", "chromadb")
        db_path = os.getenv("DUCKDB_PATH", "./data/garip.duckdb")

        chunker      = Chunker()
        all_chunks   = chunker.chunk_dataframe(tagged_regulations)
        vector_loader = VectorLoader(backend=backend)

        with DuckDBLoader(db_path=db_path) as duckdb_loader:
            duckdb_loader.load_chunks(all_chunks)
            upserted = vector_loader.upsert_chunks(
                all_chunks, duckdb_loader=duckdb_loader
            )

        log.info(f"Vector store upsert: {upserted}/{len(all_chunks)} chunks")

        return Output(
            value=upserted,
            metadata={
                "total_chunks": MetadataValue.int(len(all_chunks)),
                "upserted":     MetadataValue.int(upserted),
                "backend":      MetadataValue.text(backend),
            },
        )

    # ------------------------------------------------------------------
    # Jobs
    # ------------------------------------------------------------------

    full_pipeline_job = define_asset_job(
        name="garip_full_pipeline",
        selection=AssetSelection.all(),
        description="Full GARIP pipeline: ingest → validate → dedup → tag → store",
    )

    ingestion_only_job = define_asset_job(
        name="garip_ingestion_only",
        selection=AssetSelection.assets(
            "raw_regulations",
            "validated_regulations",
        ),
        description="Ingestion and validation only — no storage write",
    )

    # ------------------------------------------------------------------
    # Schedules
    # ------------------------------------------------------------------

    daily_schedule = ScheduleDefinition(
        name="garip_daily",
        job=full_pipeline_job,
        cron_schedule="0 6 * * *",       # 6 AM UTC every day
        description="Run full GARIP pipeline daily at 6 AM UTC",
    )

    weekly_schedule = ScheduleDefinition(
        name="garip_weekly",
        job=full_pipeline_job,
        cron_schedule="0 2 * * 0",       # 2 AM UTC every Sunday
        description="Full pipeline run every Sunday at 2 AM UTC",
    )

    # ------------------------------------------------------------------
    # Definitions — entry point for `dagster dev`
    # ------------------------------------------------------------------

    defs = Definitions(
        assets=[
            raw_regulations,
            validated_regulations,
            deduplicated_regulations,
            tagged_regulations,
            duckdb_regulations,
            vector_store_chunks,
        ],
        jobs=[
            full_pipeline_job,
            ingestion_only_job,
        ],
        schedules=[
            daily_schedule,
            weekly_schedule,
        ],
    )