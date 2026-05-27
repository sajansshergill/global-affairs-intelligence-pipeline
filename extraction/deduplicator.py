"""
deduplicator.py — Deduplication and versioning for GARIP ETL layer.

Implements two deduplication strategies:
  1. Exact deduplication: source_hash match → drop duplicate
  2. Amendment detection: same regulation_id, different source_hash
     → increment version_id, keep both (audit trail)

Also provides cross-source deduplication (same regulation from
EUR-Lex and regulations.gov) using fuzzy title matching.
"""

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False
    logger.warning("rapidfuzz not installed — fuzzy dedup disabled")


@dataclass
class DeduplicationReport:
    """Summary of what the deduplicator did."""
    input_rows: int
    exact_duplicates_dropped: int
    amendments_detected: int
    cross_source_duplicates_dropped: int
    output_rows: int
    version_bumps: list[dict]    # [{regulation_id, old_hash, new_hash, new_version}]


class Deduplicator:
    """
    Handles deduplication and versioning of regulatory records.

    Flow:
      1. Exact dedup by source_hash (identical content)
      2. Amendment detection by regulation_id — same doc, new content
      3. Cross-source fuzzy dedup (optional, requires rapidfuzz)
      4. Version assignment

    Designed to work with both in-memory DataFrames (during ETL)
    and against the DuckDB regulations table (incremental loads).
    """

    def __init__(
        self,
        fuzzy_threshold: float = 0.92,
        enable_fuzzy: bool = True,
    ):
        self.fuzzy_threshold = fuzzy_threshold
        self.enable_fuzzy = enable_fuzzy and HAS_RAPIDFUZZ

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def deduplicate(
        self,
        df: pd.DataFrame,
        existing_df: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, DeduplicationReport]:
        """
        Deduplicate a batch DataFrame, optionally against existing records.

        Args:
            df:          New records to process (must have source_hash,
                         regulation_id, title columns).
            existing_df: Already-stored records (from DuckDB). When provided,
                         cross-batch dedup and amendment detection run against
                         this set.

        Returns:
            (deduplicated_df, DeduplicationReport)
        """
        input_rows = len(df)
        version_bumps: list[dict] = []

        # Step 1: Within-batch exact dedup
        df, exact_dropped = self._exact_dedup(df)

        # Step 2: Amendment detection and versioning
        if existing_df is not None and not existing_df.empty:
            df, amendments = self._detect_amendments(df, existing_df)
            version_bumps.extend(amendments)
        else:
            amendments = []

        # Step 3: Cross-source fuzzy dedup
        cross_dropped = 0
        if self.enable_fuzzy and existing_df is not None and not existing_df.empty:
            df, cross_dropped = self._fuzzy_cross_source_dedup(df, existing_df)

        # Step 4: Assign version_id to new records
        df = self._assign_versions(df, existing_df)

        report = DeduplicationReport(
            input_rows=input_rows,
            exact_duplicates_dropped=exact_dropped,
            amendments_detected=len(amendments),
            cross_source_duplicates_dropped=cross_dropped,
            output_rows=len(df),
            version_bumps=version_bumps,
        )

        logger.info(
            f"Dedup complete: {input_rows} in → {len(df)} out "
            f"(exact={exact_dropped}, amendments={len(amendments)}, "
            f"cross-source={cross_dropped})"
        )

        return df, report

    # ------------------------------------------------------------------
    # Step 1: Exact deduplication
    # ------------------------------------------------------------------

    def _exact_dedup(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        """Drop records with identical source_hash within the batch."""
        before = len(df)
        df = df.drop_duplicates(subset=["source_hash"], keep="first")
        dropped = before - len(df)
        if dropped:
            logger.debug(f"Exact dedup dropped {dropped} rows")
        return df, dropped

    # ------------------------------------------------------------------
    # Step 2: Amendment detection
    # ------------------------------------------------------------------

    def _detect_amendments(
        self,
        new_df: pd.DataFrame,
        existing_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, list[dict]]:
        """
        Detect when a regulation_id appears in both new and existing records
        but with a different source_hash — indicating an amendment.

        For amendments: mark the new record with incremented version_id.
        For exact matches (same hash): drop the new record (already stored).
        """
        amendments: list[dict] = []

        existing_index: dict[str, dict] = {}
        if "regulation_id" in existing_df.columns and "source_hash" in existing_df.columns:
            for _, row in existing_df.iterrows():
                rid = row["regulation_id"]
                if rid and rid not in existing_index:
                    existing_index[rid] = {
                        "source_hash": row.get("source_hash"),
                        "version_id": row.get("version_id", 1),
                    }

        rows_to_drop: list[int] = []

        for idx, row in new_df.iterrows():
            rid = row.get("regulation_id")
            if not rid or rid not in existing_index:
                continue

            existing = existing_index[rid]

            if row.get("source_hash") == existing["source_hash"]:
                # Exact duplicate — drop
                rows_to_drop.append(idx)
            else:
                # Amendment — bump version
                new_version = int(existing["version_id"] or 1) + 1
                new_df.at[idx, "version_id"] = new_version
                amendments.append({
                    "regulation_id": rid,
                    "old_hash": existing["source_hash"],
                    "new_hash": row.get("source_hash"),
                    "new_version": new_version,
                    "detected_at": datetime.now(timezone.utc).isoformat(),
                })
                logger.info(
                    f"Amendment detected: {rid} → version {new_version}"
                )

        new_df = new_df.drop(index=rows_to_drop)
        return new_df, amendments

    # ------------------------------------------------------------------
    # Step 3: Cross-source fuzzy deduplication
    # ------------------------------------------------------------------

    def _fuzzy_cross_source_dedup(
        self,
        new_df: pd.DataFrame,
        existing_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, int]:
        """
        Drop new records whose title fuzzy-matches an existing record
        from a different source — prevents the same regulation appearing
        twice (e.g. from EUR-Lex and regulations.gov).

        Uses token_sort_ratio to handle minor title variations.
        """
        if not HAS_RAPIDFUZZ or "title" not in existing_df.columns:
            return new_df, 0

        existing_titles = existing_df["title"].dropna().tolist()
        rows_to_drop: list[int] = []

        for idx, row in new_df.iterrows():
            new_title = row.get("title", "")
            if not new_title:
                continue

            # Skip if same source — cross-source only
            new_source = row.get("source_name", "")

            for ex_title in existing_titles:
                score = fuzz.token_sort_ratio(new_title, ex_title) / 100.0
                if score >= self.fuzzy_threshold:
                    logger.debug(
                        f"Fuzzy match ({score:.2f}): "
                        f"'{new_title[:60]}' ~ '{ex_title[:60]}'"
                    )
                    rows_to_drop.append(idx)
                    break

        new_df = new_df.drop(index=rows_to_drop)
        return new_df, len(rows_to_drop)

    # ------------------------------------------------------------------
    # Step 4: Version assignment
    # ------------------------------------------------------------------

    def _assign_versions(
        self,
        df: pd.DataFrame,
        existing_df: pd.DataFrame | None,
    ) -> pd.DataFrame:
        """
        Ensure every record has a valid version_id.
        New records without a prior version get version_id = 1.
        """
        if "version_id" not in df.columns:
            df["version_id"] = 1
        else:
            df["version_id"] = df["version_id"].fillna(1).astype(int)
        return df

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def compute_source_hash(text: str) -> str:
        """Canonical content hash used throughout GARIP."""
        return hashlib.sha256(
            text.encode("utf-8", errors="replace")
        ).hexdigest()[:32]

    @staticmethod
    def compute_regulation_id(source_url: str) -> str:
        """Stable ID derived from source URL."""
        return hashlib.sha256(
            source_url.encode("utf-8", errors="replace")
        ).hexdigest()[:32]


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    # Demo: deduplicate a sample DataFrame
    sample = pd.DataFrame([
        {
            "regulation_id": "abc123",
            "source_hash": "hash_a",
            "title": "EU AI Act",
            "jurisdiction": "EU",
            "regulation_type": "Regulation",
            "effective_date": "2024-01-01",
            "source_url": "https://eur-lex.europa.eu/ai-act",
            "source_name": "eurlex",
            "version_id": 1,
        },
        {
            "regulation_id": "abc123",   # same reg_id
            "source_hash": "hash_a",     # same hash → exact dup
            "title": "EU AI Act",
            "jurisdiction": "EU",
            "regulation_type": "Regulation",
            "effective_date": "2024-01-01",
            "source_url": "https://eur-lex.europa.eu/ai-act",
            "source_name": "eurlex",
            "version_id": 1,
        },
        {
            "regulation_id": "def456",
            "source_hash": "hash_b",
            "title": "UK GDPR Enforcement Notice",
            "jurisdiction": "UK",
            "regulation_type": "Enforcement Action",
            "effective_date": "2024-02-15",
            "source_url": "https://ico.org.uk/enforcement/notice-1",
            "source_name": "ico",
            "version_id": 1,
        },
    ])

    deduper = Deduplicator()
    result_df, report = deduper.deduplicate(sample)
    print(f"Input: {report.input_rows} → Output: {report.output_rows}")
    print(f"Exact dropped: {report.exact_duplicates_dropped}")
    print(f"Amendments: {report.amendments_detected}")
    print(result_df[["regulation_id", "title", "version_id"]].to_string())