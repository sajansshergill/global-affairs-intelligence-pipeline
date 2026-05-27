"""
duckdb_loader.py — Loads structured regulatory metadata into DuckDB.

DuckDB is the local BigQuery equivalent in GARIP's stack.
Handles schema initialization, regulation upserts, chunk loading,
pipeline health logging, and dashboard query utilities.

GCP equivalent: BigQuery client with dataset garip_regulations.
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

SCHEMA_PATH = Path(__file__).parent / "schema.sql"
DEFAULT_DB_PATH = "./data/garip.duckdb"


class DuckDBLoader:
    """
    Manages the GARIP DuckDB database.

    Usage:
        with DuckDBLoader() as loader:
            loader.load_regulations(df)
            loader.load_chunks(chunks)
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        self.db_path = db_path
        self._conn = None
        self._init_schema()

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    def _get_conn(self):
        if self._conn is None:
            try:
                import duckdb
                Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
                self._conn = duckdb.connect(self.db_path)
                logger.info(f"DuckDB connected: {self.db_path}")
            except ImportError:
                raise RuntimeError("duckdb not installed — run: pip install duckdb")
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ------------------------------------------------------------------
    # Schema init
    # ------------------------------------------------------------------

    def _init_schema(self):
        """Run schema.sql to create tables and views if they don't exist."""
        if not SCHEMA_PATH.exists():
            logger.warning(f"schema.sql not found at {SCHEMA_PATH}")
            return
        conn = self._get_conn()
        sql = SCHEMA_PATH.read_text()
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                try:
                    conn.execute(stmt)
                except Exception as exc:
                    logger.debug(f"Schema stmt skipped (likely already exists): {exc}")
        logger.info("DuckDB schema initialized")

    # ------------------------------------------------------------------
    # Regulation loading
    # ------------------------------------------------------------------

    def load_regulations(self, df: pd.DataFrame) -> int:
        """
        Upsert a regulations DataFrame into the regulations table.
        Returns number of rows written.
        """
        if df.empty:
            logger.warning("load_regulations called with empty DataFrame")
            return 0

        conn = self._get_conn()

        # Columns that exist in the schema
        schema_cols = [
            "regulation_id", "version_id", "source_hash", "title",
            "jurisdiction", "regulation_type", "effective_date",
            "source_url", "raw_text", "ingested_at", "source_name",
            "article_citations", "fine_amounts", "named_entities_count",
        ]
        load_cols = [c for c in schema_cols if c in df.columns]
        df_load = df[load_cols].copy()

        # Type coercions
        if "version_id" in df_load.columns:
            df_load["version_id"] = df_load["version_id"].fillna(1).astype(int)
        if "effective_date" in df_load.columns:
            df_load["effective_date"] = pd.to_datetime(
                df_load["effective_date"], errors="coerce"
            ).dt.date

        conn.register("_new_regs", df_load)
        conn.execute("""
            INSERT INTO regulations
            SELECT * FROM _new_regs
            ON CONFLICT (regulation_id, version_id) DO UPDATE SET
                source_hash  = excluded.source_hash,
                title        = excluded.title,
                raw_text     = excluded.raw_text,
                ingested_at  = excluded.ingested_at
        """)
        conn.unregister("_new_regs")

        logger.info(f"Loaded {len(df_load)} regulations into DuckDB")
        return len(df_load)

    # ------------------------------------------------------------------
    # Chunk loading
    # ------------------------------------------------------------------

    def load_chunks(self, chunks: list) -> int:
        """
        Insert chunk metadata into the chunks table.
        Accepts a list of Chunk objects from extraction/chunker.py.
        """
        if not chunks:
            return 0

        conn = self._get_conn()
        rows = []
        for c in chunks:
            rows.append({
                "chunk_id":       c.chunk_id,
                "regulation_id":  c.regulation_id,
                "version_id":     c.extra.get("version_id", 1),
                "chunk_index":    c.chunk_index,
                "total_chunks":   c.total_chunks,
                "text":           c.text,
                "token_estimate": c.token_estimate,
                "jurisdiction":   c.jurisdiction,
                "regulation_type":c.regulation_type,
                "effective_date": c.effective_date,
                "source_url":     c.source_url,
                "page_number":    c.page_number,
                "article_ref":    c.article_ref,
                "section_ref":    c.section_ref,
                "chapter_ref":    c.chapter_ref,
                "block_type":     c.block_type,
                "embedded_at":    None,
                "vector_store":   None,
            })

        df = pd.DataFrame(rows)
        conn.register("_new_chunks", df)
        conn.execute("""
            INSERT INTO chunks SELECT * FROM _new_chunks
            ON CONFLICT (chunk_id) DO NOTHING
        """)
        conn.unregister("_new_chunks")

        logger.info(f"Loaded {len(rows)} chunks into DuckDB")
        return len(rows)

    def mark_chunks_embedded(
        self, chunk_ids: list[str], vector_store: str = "pinecone"
    ) -> None:
        """Mark chunks as embedded after vector store upsert."""
        if not chunk_ids:
            return
        conn = self._get_conn()
        now = datetime.now(timezone.utc).isoformat()
        ids_str = ", ".join(f"'{cid}'" for cid in chunk_ids)
        conn.execute(f"""
            UPDATE chunks
            SET embedded_at  = '{now}',
                vector_store = '{vector_store}'
            WHERE chunk_id IN ({ids_str})
        """)

    # ------------------------------------------------------------------
    # Pipeline health
    # ------------------------------------------------------------------

    def log_pipeline_health(self, health: dict) -> None:
        """Insert a single pipeline health record."""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO pipeline_health (
                run_id, source, run_at, rows_ingested, duplicate_count,
                null_rate, schema_drift, sla_met, elapsed_seconds, errors,
                quality_passed, quality_critical_failures, quality_warnings
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (run_id) DO NOTHING
        """, [
            health.get("run_id"),
            health.get("source"),
            health.get("run_at", datetime.now(timezone.utc).isoformat()),
            health.get("rows_ingested", 0),
            health.get("duplicate_count", 0),
            health.get("null_rate", 0.0),
            health.get("schema_drift", False),
            health.get("sla_met", True),
            health.get("elapsed_seconds"),
            json.dumps(health.get("errors", [])),
            health.get("quality_passed"),
            health.get("quality_critical_failures"),
            health.get("quality_warnings"),
        ])

    # ------------------------------------------------------------------
    # Query utilities (used by Streamlit dashboard)
    # ------------------------------------------------------------------

    def query(self, sql: str, params: list | None = None) -> pd.DataFrame:
        """Execute any SQL and return a DataFrame."""
        conn = self._get_conn()
        return conn.execute(sql, params or []).df()

    def get_regulations(
        self,
        jurisdiction: str | None = None,
        regulation_type: str | None = None,
        limit: int = 100,
        latest_only: bool = True,
    ) -> pd.DataFrame:
        table = "regulations_latest" if latest_only else "regulations"
        clauses, params = [], []
        if jurisdiction:
            clauses.append("jurisdiction = ?")
            params.append(jurisdiction)
        if regulation_type:
            clauses.append("regulation_type = ?")
            params.append(regulation_type)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"SELECT * FROM {table} {where} ORDER BY effective_date DESC LIMIT {limit}"
        return self.query(sql, params or None)

    def get_jurisdiction_stats(self) -> pd.DataFrame:
        return self.query("SELECT * FROM jurisdiction_stats")

    def get_pipeline_health(self) -> pd.DataFrame:
        return self.query("SELECT * FROM pipeline_health_recent")

    def get_amendment_history(self) -> pd.DataFrame:
        return self.query("SELECT * FROM amendment_history ORDER BY regulation_id, version_id")

    def get_total_counts(self) -> dict:
        conn = self._get_conn()
        row = conn.execute("""
            SELECT
                (SELECT COUNT(DISTINCT regulation_id) FROM regulations) AS total_regulations,
                (SELECT COUNT(*)                       FROM regulations) AS total_versions,
                (SELECT COUNT(*)                       FROM chunks)      AS total_chunks,
                (SELECT COUNT(*)                       FROM pipeline_health) AS total_runs
        """).fetchone()
        return {
            "total_regulations": row[0],
            "total_versions":    row[1],
            "total_chunks":      row[2],
            "total_runs":        row[3],
        }