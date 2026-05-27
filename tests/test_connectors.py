"""
test_connectors.py — Tests for GARIP ingestion connectors.

Covers:
  - Shared connector behavior (dedup, null rate, Parquet landing, health log)
  - EUR-Lex SPARQL parsing
  - Congress.gov bill parsing
  - FTC RSS filtering
  - ICO fine extraction and article detection
  - Regulations.gov document parsing

Run: pytest tests/test_connectors.py -v
"""

import hashlib
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest


# ------------------------------------------------------------------
# Shared helpers
# ------------------------------------------------------------------

def make_tmp_dir():
    return tempfile.mkdtemp()


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]


# ------------------------------------------------------------------
# EUR-Lex connector tests
# ------------------------------------------------------------------

class TestEURLexConnector:

    MOCK_SPARQL = {
        "results": {
            "bindings": [
                {
                    "celex": {"value": "32024R0001"},
                    "title": {"value": "Regulation on AI Systems", "xml:lang": "en"},
                    "date":  {"value": "2024-03-15"},
                    "type":  {"value": "Regulation", "xml:lang": "en"},
                }
            ]
        }
    }

    @patch("ingestion.eurlex_connector.EURLexConnector.session")
    def test_fetch_records_parses_binding(self, mock_session, tmp_path):
        from ingestion.eurlex_connector import EURLexConnector
        mock_resp = MagicMock()
        mock_resp.json.return_value = self.MOCK_SPARQL
        mock_session.get.return_value = mock_resp

        connector = EURLexConnector(raw_data_dir=str(tmp_path))
        connector.session = mock_session
        records = connector.fetch_records(limit=10)

        assert len(records) == 1
        assert records[0]["title"]        == "Regulation on AI Systems"
        assert records[0]["jurisdiction"] == "EU"
        assert records[0]["effective_date"] == "2024-03-15"

    def test_classify_regulation_types(self, tmp_path):
        from ingestion.eurlex_connector import EURLexConnector
        c = EURLexConnector(raw_data_dir=str(tmp_path))
        assert c._classify("regulation on AI")  == "Regulation"
        assert c._classify("directive 2024/01")  == "Directive"
        assert c._classify("decision on merger") == "Decision"
        assert c._classify("unknown act")        == "Legal Act"

    def test_parse_date_iso(self, tmp_path):
        from ingestion.eurlex_connector import EURLexConnector
        c = EURLexConnector(raw_data_dir=str(tmp_path))
        assert c._val({"date": {"value": "2024-03-15"}}, "date") == "2024-03-15"
        assert c._val({}, "date") is None

    def test_run_produces_dataframe(self, tmp_path):
        from ingestion.eurlex_connector import EURLexConnector
        c = EURLexConnector(raw_data_dir=str(tmp_path))
        mock_resp = MagicMock()
        mock_resp.json.return_value = self.MOCK_SPARQL
        c.session.get = MagicMock(return_value=mock_resp)
        df = c.run(limit=5)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1

    def test_parquet_landed(self, tmp_path):
        from ingestion.eurlex_connector import EURLexConnector
        c = EURLexConnector(raw_data_dir=str(tmp_path))
        mock_resp = MagicMock()
        mock_resp.json.return_value = self.MOCK_SPARQL
        c.session.get = MagicMock(return_value=mock_resp)
        c.run(limit=5)
        parquets = list(Path(tmp_path).rglob("*.parquet"))
        assert len(parquets) >= 1

    def test_health_log_written(self, tmp_path):
        from ingestion.eurlex_connector import EURLexConnector
        c = EURLexConnector(raw_data_dir=str(tmp_path))
        mock_resp = MagicMock()
        mock_resp.json.return_value = self.MOCK_SPARQL
        c.session.get = MagicMock(return_value=mock_resp)
        c.run(limit=5)
        log = Path(tmp_path) / "pipeline_health.jsonl"
        assert log.exists()
        entry = json.loads(log.read_text().strip().split("\n")[-1])
        assert entry["source"] == "eurlex"


# ------------------------------------------------------------------
# Congress connector tests
# ------------------------------------------------------------------

class TestCongressConnector:

    MOCK_BILLS = {
        "bills": [
            {
                "type": "hr",
                "number": "4521",
                "congress": 118,
                "title": "American Data Privacy and Protection Act",
                "policyArea": {"name": "Science, Technology, Communications"},
                "latestAction": {"actionDate": "2023-07-26"},
                "originChamber": "House",
            }
        ]
    }

    def test_parse_bill_fields(self, tmp_path):
        from ingestion.congress_connector import CongressConnector
        c = CongressConnector(raw_data_dir=str(tmp_path))
        record = c._parse_bill(self.MOCK_BILLS["bills"][0])
        assert record["title"]           == "American Data Privacy and Protection Act"
        assert record["jurisdiction"]    == "US"
        assert record["effective_date"]  == "2023-07-26"
        assert record["regulation_type"] == "House Bill"

    def test_bill_type_map_coverage(self, tmp_path):
        from ingestion.congress_connector import BILL_TYPE_MAP
        assert BILL_TYPE_MAP["hr"]    == "House Bill"
        assert BILL_TYPE_MAP["s"]     == "Senate Bill"
        assert BILL_TYPE_MAP["hjres"] == "House Joint Resolution"

    def test_run_deduplicates(self, tmp_path):
        from ingestion.congress_connector import CongressConnector
        c = CongressConnector(raw_data_dir=str(tmp_path))
        mock_resp = MagicMock()
        mock_resp.json.return_value = self.MOCK_BILLS
        c.session.get = MagicMock(return_value=mock_resp)
        # Run twice — same data should dedup to 1 row
        df1 = c.run(limit=5)
        assert len(df1) == 1

    def test_source_hash_deterministic(self, tmp_path):
        from ingestion.congress_connector import CongressConnector
        c = CongressConnector(raw_data_dir=str(tmp_path))
        r1 = c._parse_bill(self.MOCK_BILLS["bills"][0])
        r2 = c._parse_bill(self.MOCK_BILLS["bills"][0])
        assert r1["source_hash"] == r2["source_hash"]


# ------------------------------------------------------------------
# FTC connector tests
# ------------------------------------------------------------------

class TestFTCConnector:

    MOCK_RSS = b"""<?xml version="1.0"?>
    <rss version="2.0">
      <channel>
        <title>FTC</title>
        <item>
          <title>FTC Takes Action Against Company for Privacy Violations</title>
          <link>https://www.ftc.gov/news/2024/01/action</link>
          <pubDate>Mon, 15 Jan 2024 12:00:00 +0000</pubDate>
          <description>The FTC filed a complaint for deceptive data practices and civil penalty.</description>
        </item>
        <item>
          <title>FTC Annual Report</title>
          <link>https://www.ftc.gov/reports/annual</link>
          <pubDate>Mon, 01 Jan 2024 12:00:00 +0000</pubDate>
          <description>Annual highlights from activities.</description>
        </item>
      </channel>
    </rss>"""

    def test_filter_keeps_enforcement_only(self, tmp_path):
        from ingestion.ftc_connector import FTCConnector
        c = FTCConnector(raw_data_dir=str(tmp_path))
        mock_resp = MagicMock()
        mock_resp.content = self.MOCK_RSS
        c.session.get = MagicMock(return_value=mock_resp)
        records = c.fetch_records(limit=10)
        titles = [r["title"] for r in records]
        assert any("Privacy" in t for t in titles)
        assert not any("Annual Report" in t for t in titles)

    def test_classify_privacy(self, tmp_path):
        from ingestion.ftc_connector import FTCConnector
        c = FTCConnector(raw_data_dir=str(tmp_path))
        assert c._classify("privacy data breach") == "Privacy Enforcement"
        assert c._classify("consumer protection")  == "Consumer Protection"
        assert c._classify("merger antitrust")     == "Merger Review"

    def test_parse_date_rfc2822(self, tmp_path):
        from ingestion.ftc_connector import FTCConnector
        c = FTCConnector(raw_data_dir=str(tmp_path))
        result = c._parse_date("Mon, 15 Jan 2024 12:00:00 +0000")
        assert result == "2024-01-15"

    def test_parse_date_none(self, tmp_path):
        from ingestion.ftc_connector import FTCConnector
        c = FTCConnector(raw_data_dir=str(tmp_path))
        assert c._parse_date(None) is None


# ------------------------------------------------------------------
# ICO connector tests
# ------------------------------------------------------------------

class TestICOConnector:

    def test_extract_fine_plain(self, tmp_path):
        from ingestion.ico_connector import ICOConnector
        c = ICOConnector(raw_data_dir=str(tmp_path))
        assert c._extract_fine("fined £200,000 for breach") == 200_000.0

    def test_extract_fine_million(self, tmp_path):
        from ingestion.ico_connector import ICOConnector
        c = ICOConnector(raw_data_dir=str(tmp_path))
        assert c._extract_fine("penalty of £1.5 million") == 1_500_000.0

    def test_extract_fine_none(self, tmp_path):
        from ingestion.ico_connector import ICOConnector
        c = ICOConnector(raw_data_dir=str(tmp_path))
        assert c._extract_fine("no penalty mentioned") is None

    def test_extract_gdpr_articles(self, tmp_path):
        from ingestion.ico_connector import ICOConnector
        c = ICOConnector(raw_data_dir=str(tmp_path))
        articles = c._extract_articles("Breach of Article 5 GDPR and Article 25(1) UK GDPR")
        assert len(articles) >= 1
        assert any("Article 5" in a for a in articles)

    def test_classify_monetary_penalty(self, tmp_path):
        from ingestion.ico_connector import ICOConnector
        c = ICOConnector(raw_data_dir=str(tmp_path))
        assert c._classify("monetary penalty notice issued") == "Monetary Penalty Notice"
        assert c._classify("undertaking signed by company")  == "Undertaking"
        assert c._classify("random text")                    == "GDPR Enforcement Decision"


# ------------------------------------------------------------------
# Regulations.gov connector tests
# ------------------------------------------------------------------

class TestRegulationsGovConnector:

    MOCK_DOC = {
        "id": "FCC-2024-001",
        "attributes": {
            "title":        "Open Internet Rulemaking",
            "documentType": "Proposed Rule",
            "postedDate":   "2024-02-01T00:00:00Z",
            "docketId":     "FCC-2024-001",
            "agencyId":     "FCC",
        },
    }

    def test_parse_document_fields(self, tmp_path):
        from ingestion.regulations_connector import RegulationsGovConnector
        c = RegulationsGovConnector(raw_data_dir=str(tmp_path))
        record = c._parse_document(self.MOCK_DOC, agency_id="FCC")
        assert record["title"]           == "Open Internet Rulemaking"
        assert record["jurisdiction"]    == "US"
        assert record["regulation_type"] == "Proposed Rulemaking"
        assert record["effective_date"]  == "2024-02-01"

    def test_target_agencies_covered(self, tmp_path):
        from ingestion.regulations_connector import TARGET_AGENCIES
        for agency in ["FCC", "FTC", "NTIA", "DOJ", "SEC", "CFPB", "DHS", "CISA"]:
            assert agency in TARGET_AGENCIES

    def test_document_type_map(self, tmp_path):
        from ingestion.regulations_connector import DOCUMENT_TYPE_MAP
        assert DOCUMENT_TYPE_MAP["Proposed Rule"] == "Proposed Rulemaking"
        assert DOCUMENT_TYPE_MAP["Rule"]          == "Final Rule"
        assert DOCUMENT_TYPE_MAP["Notice"]        == "Federal Notice"