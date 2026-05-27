"""
quality_checks.py — Data quality validation suite for GARIP ETL layer.

Implements a Great Expectations-style validation framework adapted for
regulatory document pipelines. Falls back to native pandas checks if
great_expectations is not installed.

Checks cover:
  - Schema completeness (required columns present)
  - Null rates per column (thresholds per field criticality)
  - Value set validation (jurisdictions, regulation types)
  - Referential integrity (regulation_id format, URL format)
  - Date validity and range checks
  - Duplicate detection (source_hash uniqueness)
  - Pipeline SLA (row count minimums)
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Validation config
# ------------------------------------------------------------------

REQUIRED_COLUMNS = [
    "regulation_id", "source_hash", "title", "jurisdiction",
    "regulation_type", "effective_date", "source_url",
    "raw_text", "version_id", "ingested_at",
]

# Max allowed null rate per column (0.0 = must be fully populated)
NULL_RATE_THRESHOLDS = {
    "regulation_id": 0.0,
    "source_hash": 0.0,
    "title": 0.05,
    "jurisdiction": 0.0,
    "regulation_type": 0.10,
    "effective_date": 0.30,   # dates often missing in enforcement actions
    "source_url": 0.0,
    "raw_text": 0.05,
    "version_id": 0.0,
    "ingested_at": 0.0,
}

VALID_JURISDICTIONS = {
    "EU", "US", "UK", "DE", "FR", "CA", "AU", "UNKNOWN",
}

VALID_REGULATION_TYPES = {
    "Regulation", "Directive", "Decision", "Enforcement Action",
    "Monetary Penalty", "Proposed Rule", "Final Rule", "Notice",
    "Guidance", "Undertaking", "Reprimand", "Warning",
    "House Bill", "Senate Bill", "House Joint Resolution",
    "Senate Joint Resolution", "Legislation", "Legal Act",
    "Legal Document", "GDPR Enforcement Decision",
    "Privacy Enforcement", "Data Protection", "Consumer Protection",
    "Antitrust Action", "Competition Enforcement",
    "Advertising Enforcement", "Merger Review",
    "Federal Notice", "Supporting Material", "Public Comment",
    "Other Federal Document", "Criminal Prosecution",
}

URL_PATTERN = re.compile(r"^https?://[^\s]+$")
DATE_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")
HASH_PATTERN = re.compile(r"^[a-f0-9]{24,64}$")

# Minimum rows expected from each source per run
MIN_ROW_THRESHOLDS = {
    "eurlex": 5,
    "congress": 5,
    "ftc": 3,
    "regulations_gov": 5,
    "ico": 3,
    "default": 1,
}


# ------------------------------------------------------------------
# Result types
# ------------------------------------------------------------------

@dataclass
class CheckResult:
    """Result of a single validation check."""
    check_name: str
    passed: bool
    severity: str           # "critical" | "warning" | "info"
    message: str
    details: dict = field(default_factory=dict)


@dataclass
class ValidationSuite:
    """Aggregated results from all checks on a DataFrame."""
    source: str
    run_at: str
    total_rows: int
    checks: list[CheckResult]

    @property
    def passed(self) -> bool:
        return all(c.passed for c in self.checks if c.severity == "critical")

    @property
    def critical_failures(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "critical"]

    @property
    def warnings(self) -> list[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "warning"]

    def summary(self) -> str:
        total = len(self.checks)
        passed = sum(1 for c in self.checks if c.passed)
        return (
            f"Validation [{self.source}]: {passed}/{total} checks passed | "
            f"Critical failures: {len(self.critical_failures)} | "
            f"Warnings: {len(self.warnings)}"
        )

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "run_at": self.run_at,
            "total_rows": self.total_rows,
            "overall_passed": self.passed,
            "critical_failures": len(self.critical_failures),
            "warnings": len(self.warnings),
            "checks": [
                {
                    "name": c.check_name,
                    "passed": c.passed,
                    "severity": c.severity,
                    "message": c.message,
                }
                for c in self.checks
            ],
        }


# ------------------------------------------------------------------
# Validator
# ------------------------------------------------------------------

class QualityChecker:
    """
    Runs a suite of data quality checks on a GARIP regulations DataFrame.

    Designed to run:
      - After each connector ingestion (raw check)
      - After ETL transform (enriched check)
      - Before vector store load (pre-embed check)

    Each check returns a CheckResult. Critical failures block downstream
    processing; warnings are logged but non-blocking.
    """

    def __init__(self, source: str = "unknown"):
        self.source = source

    def run(self, df: pd.DataFrame) -> ValidationSuite:
        """Run all checks and return a ValidationSuite."""
        checks: list[CheckResult] = []

        checks.append(self._check_not_empty(df))
        checks.append(self._check_schema(df))
        checks.extend(self._check_null_rates(df))
        checks.append(self._check_duplicate_hashes(df))
        checks.append(self._check_jurisdiction_values(df))
        checks.append(self._check_regulation_type_values(df))
        checks.append(self._check_url_format(df))
        checks.append(self._check_date_format(df))
        checks.append(self._check_hash_format(df))
        checks.append(self._check_min_rows(df))
        checks.append(self._check_raw_text_length(df))

        suite = ValidationSuite(
            source=self.source,
            run_at=datetime.now(timezone.utc).isoformat(),
            total_rows=len(df),
            checks=checks,
        )

        logger.info(suite.summary())
        for failure in suite.critical_failures:
            logger.error(f"CRITICAL: {failure.check_name} — {failure.message}")
        for warning in suite.warnings:
            logger.warning(f"WARNING: {warning.check_name} — {warning.message}")

        return suite

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_not_empty(self, df: pd.DataFrame) -> CheckResult:
        passed = len(df) > 0
        return CheckResult(
            check_name="not_empty",
            passed=passed,
            severity="critical",
            message="DataFrame has rows" if passed else "DataFrame is empty — no data ingested",
            details={"row_count": len(df)},
        )

    def _check_schema(self, df: pd.DataFrame) -> CheckResult:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        passed = len(missing) == 0
        return CheckResult(
            check_name="schema_completeness",
            passed=passed,
            severity="critical",
            message=(
                "All required columns present"
                if passed
                else f"Missing columns: {missing}"
            ),
            details={"missing_columns": missing},
        )

    def _check_null_rates(self, df: pd.DataFrame) -> list[CheckResult]:
        results: list[CheckResult] = []
        for col, threshold in NULL_RATE_THRESHOLDS.items():
            if col not in df.columns:
                continue
            null_rate = df[col].isnull().mean()
            passed = null_rate <= threshold
            severity = "critical" if threshold == 0.0 else "warning"
            results.append(CheckResult(
                check_name=f"null_rate_{col}",
                passed=passed,
                severity=severity,
                message=(
                    f"{col} null rate {null_rate:.1%} ≤ {threshold:.0%}"
                    if passed
                    else f"{col} null rate {null_rate:.1%} exceeds threshold {threshold:.0%}"
                ),
                details={"null_rate": round(null_rate, 4), "threshold": threshold},
            ))
        return results

    def _check_duplicate_hashes(self, df: pd.DataFrame) -> CheckResult:
        if "source_hash" not in df.columns:
            return CheckResult("duplicate_source_hash", True, "info", "Column not present")
        dup_count = df["source_hash"].duplicated().sum()
        passed = dup_count == 0
        return CheckResult(
            check_name="duplicate_source_hash",
            passed=passed,
            severity="warning",
            message=(
                "No duplicate source_hash values"
                if passed
                else f"{dup_count} duplicate source_hash values detected"
            ),
            details={"duplicate_count": int(dup_count)},
        )

    def _check_jurisdiction_values(self, df: pd.DataFrame) -> CheckResult:
        if "jurisdiction" not in df.columns:
            return CheckResult("valid_jurisdictions", True, "info", "Column not present")
        invalid = df["jurisdiction"].dropna()
        invalid = invalid[~invalid.isin(VALID_JURISDICTIONS)]
        passed = len(invalid) == 0
        return CheckResult(
            check_name="valid_jurisdictions",
            passed=passed,
            severity="warning",
            message=(
                "All jurisdiction values are valid"
                if passed
                else f"{len(invalid)} invalid jurisdiction values: {invalid.unique().tolist()}"
            ),
            details={"invalid_values": invalid.unique().tolist()},
        )

    def _check_regulation_type_values(self, df: pd.DataFrame) -> CheckResult:
        if "regulation_type" not in df.columns:
            return CheckResult("valid_regulation_types", True, "info", "Column not present")
        invalid = df["regulation_type"].dropna()
        invalid = invalid[~invalid.isin(VALID_REGULATION_TYPES)]
        passed = len(invalid) == 0
        return CheckResult(
            check_name="valid_regulation_types",
            passed=passed,
            severity="info",
            message=(
                "All regulation_type values are recognized"
                if passed
                else f"{len(invalid)} unrecognized regulation types: {invalid.unique().tolist()[:5]}"
            ),
            details={"unrecognized": invalid.unique().tolist()},
        )

    def _check_url_format(self, df: pd.DataFrame) -> CheckResult:
        if "source_url" not in df.columns:
            return CheckResult("url_format", True, "info", "Column not present")
        non_null = df["source_url"].dropna()
        invalid_count = (~non_null.str.match(URL_PATTERN.pattern)).sum()
        passed = invalid_count == 0
        return CheckResult(
            check_name="url_format",
            passed=passed,
            severity="warning",
            message=(
                "All source_url values are valid URLs"
                if passed
                else f"{invalid_count} source_url values fail URL format check"
            ),
            details={"invalid_count": int(invalid_count)},
        )

    def _check_date_format(self, df: pd.DataFrame) -> CheckResult:
        if "effective_date" not in df.columns:
            return CheckResult("date_format", True, "info", "Column not present")
        non_null = df["effective_date"].dropna().astype(str)
        invalid = non_null[~non_null.str.match(DATE_PATTERN.pattern)]
        passed = len(invalid) == 0
        return CheckResult(
            check_name="effective_date_format",
            passed=passed,
            severity="warning",
            message=(
                "All effective_date values match YYYY-MM-DD"
                if passed
                else f"{len(invalid)} effective_date values fail format check"
            ),
            details={"invalid_samples": invalid.head(3).tolist()},
        )

    def _check_hash_format(self, df: pd.DataFrame) -> CheckResult:
        if "source_hash" not in df.columns:
            return CheckResult("hash_format", True, "info", "Column not present")
        non_null = df["source_hash"].dropna().astype(str)
        invalid = non_null[~non_null.str.match(HASH_PATTERN.pattern)]
        passed = len(invalid) == 0
        return CheckResult(
            check_name="source_hash_format",
            passed=passed,
            severity="critical",
            message=(
                "All source_hash values are valid hex strings"
                if passed
                else f"{len(invalid)} source_hash values fail format check"
            ),
        )

    def _check_min_rows(self, df: pd.DataFrame) -> CheckResult:
        min_rows = MIN_ROW_THRESHOLDS.get(self.source, MIN_ROW_THRESHOLDS["default"])
        passed = len(df) >= min_rows
        return CheckResult(
            check_name="minimum_row_count",
            passed=passed,
            severity="warning",
            message=(
                f"Row count {len(df)} meets minimum {min_rows}"
                if passed
                else f"Row count {len(df)} below minimum {min_rows} for source '{self.source}'"
            ),
            details={"row_count": len(df), "min_required": min_rows},
        )

    def _check_raw_text_length(self, df: pd.DataFrame) -> CheckResult:
        if "raw_text" not in df.columns:
            return CheckResult("raw_text_length", True, "info", "Column not present")
        non_null = df["raw_text"].dropna()
        too_short = (non_null.str.len() < 20).sum()
        passed = too_short == 0
        return CheckResult(
            check_name="raw_text_min_length",
            passed=passed,
            severity="warning",
            message=(
                "All raw_text values have ≥20 characters"
                if passed
                else f"{too_short} raw_text values are suspiciously short (<20 chars)"
            ),
            details={"too_short_count": int(too_short)},
        )