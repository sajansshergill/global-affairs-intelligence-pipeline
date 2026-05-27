"""
test_quality_checks.py — Tests for extraction/quality_checks.py.

Covers:
  - Valid DataFrame passes all critical checks
  - Empty DataFrame fails critical
  - Missing required column fails schema check
  - Null regulation_id fails null rate check
  - Duplicate source_hash triggers warning
  - Invalid jurisdiction triggers warning
  - Bad URL format triggers warning
  - Bad date format triggers warning
  - Short raw_text triggers warning
  - suite.summary() returns a string
  - suite.to_dict() has required keys
  - Below minimum row count triggers warning

Run: pytest tests/test_quality_checks.py -v
"""

import pytest
import pandas as pd
from extraction.quality_checks import QualityChecker, ValidationSuite


# ------------------------------------------------------------------
# Fixture: valid DataFrame
# ------------------------------------------------------------------

def make_valid_df(n: int = 5) -> pd.DataFrame:
    return pd.DataFrame([
        {
            "regulation_id":   f"reg_{i:03d}{'a' * 28}",   # 32-char hex-like
            "version_id":      1,
            "source_hash":     f"{i:01d}" + "a" * 31,       # 32-char
            "title":           f"Test Regulation {i}",
            "jurisdiction":    "EU",
            "regulation_type": "Regulation",
            "effective_date":  "2024-01-01",
            "source_url":      f"https://eur-lex.europa.eu/doc/{i}",
            "raw_text":        f"This is valid regulatory text for regulation {i}. " * 3,
            "ingested_at":     "2024-06-01T00:00:00Z",
            "source_name":     "eurlex",
        }
        for i in range(n)
    ])


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------

class TestQualityChecker:

    def test_valid_df_passes(self):
        checker = QualityChecker(source="eurlex")
        suite   = checker.run(make_valid_df(10))
        assert suite.passed, (
            f"Expected pass but got: "
            f"{[f.message for f in suite.critical_failures]}"
        )

    def test_empty_df_fails_critical(self):
        checker = QualityChecker(source="eurlex")
        suite   = checker.run(pd.DataFrame())
        assert not suite.passed
        names   = [c.check_name for c in suite.critical_failures]
        assert "not_empty" in names

    def test_missing_column_fails_schema(self):
        checker = QualityChecker(source="eurlex")
        df      = make_valid_df(5).drop(columns=["regulation_id"])
        suite   = checker.run(df)
        assert not suite.passed
        names   = [c.check_name for c in suite.critical_failures]
        assert "schema_completeness" in names

    def test_null_regulation_id_fails(self):
        checker = QualityChecker(source="eurlex")
        df      = make_valid_df(5)
        df["regulation_id"] = None
        suite   = checker.run(df)
        assert not suite.passed

    def test_duplicate_source_hash_warning(self):
        checker = QualityChecker(source="eurlex")
        df      = make_valid_df(5)
        df["source_hash"] = "a" * 32   # all identical
        suite   = checker.run(df)
        dup     = next((c for c in suite.checks if c.check_name == "duplicate_source_hash"), None)
        assert dup is not None
        assert not dup.passed

    def test_invalid_jurisdiction_warning(self):
        checker = QualityChecker(source="eurlex")
        df      = make_valid_df(5)
        df.loc[0, "jurisdiction"] = "INVALID"
        suite   = checker.run(df)
        jur     = next((c for c in suite.checks if c.check_name == "valid_jurisdictions"), None)
        assert jur is not None
        assert not jur.passed
        assert jur.severity == "warning"

    def test_bad_url_format_warning(self):
        checker = QualityChecker(source="eurlex")
        df      = make_valid_df(5)
        df.loc[0, "source_url"] = "not-a-valid-url"
        suite   = checker.run(df)
        url     = next((c for c in suite.checks if c.check_name == "url_format"), None)
        assert url is not None
        assert not url.passed

    def test_bad_date_format_warning(self):
        checker = QualityChecker(source="eurlex")
        df      = make_valid_df(5)
        df.loc[0, "effective_date"] = "March 15 2024"
        suite   = checker.run(df)
        date    = next((c for c in suite.checks if c.check_name == "effective_date_format"), None)
        assert date is not None
        assert not date.passed

    def test_short_raw_text_warning(self):
        checker = QualityChecker(source="eurlex")
        df      = make_valid_df(5)
        df.loc[0, "raw_text"] = "short"
        suite   = checker.run(df)
        txt     = next((c for c in suite.checks if c.check_name == "raw_text_min_length"), None)
        assert txt is not None
        assert not txt.passed

    def test_summary_returns_string(self):
        checker = QualityChecker(source="test")
        suite   = checker.run(make_valid_df(3))
        summary = suite.summary()
        assert isinstance(summary, str)
        assert "test" in summary

    def test_to_dict_has_required_keys(self):
        checker = QualityChecker(source="test")
        suite   = checker.run(make_valid_df(3))
        d       = suite.to_dict()
        for key in ["source", "checks", "overall_passed", "total_rows"]:
            assert key in d, f"Missing key: {key}"

    def test_below_min_rows_warning(self):
        checker = QualityChecker(source="eurlex")   # min = 5
        suite   = checker.run(make_valid_df(2))
        min_chk = next(
            (c for c in suite.checks if c.check_name == "minimum_row_count"), None
        )
        assert min_chk is not None
        assert not min_chk.passed
        assert min_chk.severity == "warning"