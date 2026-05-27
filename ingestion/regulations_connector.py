"""
regulations_connector.py — Regulations.gov API v4 connector (standalone).

Fetches US federal rulemaking documents across 8 high-relevance agencies.
No base class — fully self-contained.
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

REGULATIONS_GOV_BASE = "https://api.regulations.gov/v4"

DOCUMENT_TYPE_MAP = {
    "Proposed Rule": "Proposed Rulemaking",
    "Rule":          "Final Rule",
    "Notice":        "Federal Notice",
    "Supporting & Related Material": "Supporting Material",
    "Public Submission": "Public Comment",
    "Other":         "Other Federal Document",
}

TARGET_AGENCIES = ["FCC", "FTC", "NTIA", "DOJ", "SEC", "CFPB", "DHS", "CISA"]

REQUIRED_COLUMNS = [
    "regulation_id", "version_id", "source_hash", "title",
    "jurisdiction", "regulation_type", "effective_date",
    "source_url", "raw_text", "ingested_at",
]


class RegulationsGovConnector:
    """
    Self-contained Regulations.gov connector.
    Distributes requests across 8 tech-policy-relevant agencies.
    Free API key at https://api.data.gov/signup/
    Set REGULATIONS_GOV_API_KEY in your .env — falls back to DEMO_KEY.
    """

    source_name = "regulations_gov"

    def __init__(
        self,
        agencies: list[str] | None = None,
        posted_date_from: str = "2022-01-01",
        raw_data_dir: str = "./data/raw",
        request_delay: float = 1.5,
        timeout: int = 30,
    ):
        self.agencies = agencies or TARGET_AGENCIES
        self.posted_date_from = posted_date_from
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self.timeout = timeout
        self.api_key = os.getenv("REGULATIONS_GOV_API_KEY", "DEMO_KEY")

        if self.api_key == "DEMO_KEY":
            logger.warning("Using DEMO_KEY for regulations.gov — rate limited. "
                           "Get a free key at https://api.data.gov/signup/")

        self.session = requests.Session()
        retry = Retry(total=3, backoff_factor=2, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("https://", HTTPAdapter(max_retries=retry))

        self._health = {
            "source": self.source_name,
            "run_id": self._run_id(),
            "run_at": datetime.now(timezone.utc).isoformat(),
            "rows_ingested": 0,
            "duplicate_count": 0,
            "null_rate": 0.0,
            "schema_drift": False,
            "sla_met": True,
            "errors": [],
        }
        self._t0 = time.time()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def run(self, limit: int = 50) -> pd.DataFrame:
        logger.info(f"Regulations.gov ingestion — agencies={self.agencies} limit={limit}")
        try:
            records = self.fetch_records(limit)
        except Exception as exc:
            self._health["errors"].append(str(exc))
            self._health["sla_met"] = False
            raise

        df = self._to_dataframe(records)
        df, dupes = self._dedup(df)

        self._health["rows_ingested"] = len(df)
        self._health["duplicate_count"] = dupes
        self._health["null_rate"] = self._null_rate(df)
        self._health["sla_met"] = (time.time() - self._t0) < 300

        self._land_parquet(df)
        self._log_health()
        logger.info(f"Regulations.gov done — {len(df)} rows, {dupes} dupes dropped")
        return df

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch_records(self, limit: int) -> list[dict]:
        per_agency = max(1, limit // len(self.agencies))
        all_records: list[dict] = []

        for agency in self.agencies:
            try:
                records = self._fetch_agency(agency, limit=per_agency)
                all_records.extend(records)
                logger.info(f"Agency {agency} → {len(records)} records")
            except Exception as exc:
                logger.warning(f"Agency {agency} failed: {exc}")

        logger.info(f"Regulations.gov total: {len(all_records)} records")
        return all_records[:limit]

    def _fetch_agency(self, agency_id: str, limit: int) -> list[dict]:
        url = f"{REGULATIONS_GOV_BASE}/documents"
        params = {
            "api_key": self.api_key,
            "filter[agencyId]": agency_id,
            "filter[postedDate][ge]": self.posted_date_from,
            "page[size]": min(limit, 25),
            "page[number]": 1,
            "sort": "-postedDate",
        }
        time.sleep(self.request_delay)
        resp = self.session.get(url, params=params, timeout=self.timeout)
        resp.raise_for_status()
        documents = resp.json().get("data", [])
        return [self._parse_document(doc, agency_id) for doc in documents]

    def _parse_document(self, doc: dict, agency_id: str) -> dict:
        attrs = doc.get("attributes", {})
        doc_id = doc.get("id", "")
        title = attrs.get("title", f"Regulations.gov Document {doc_id}")
        doc_type_raw = attrs.get("documentType", "Other")
        posted_date = attrs.get("postedDate", "")
        docket_id = attrs.get("docketId", "")
        regulation_type = DOCUMENT_TYPE_MAP.get(doc_type_raw, doc_type_raw)
        effective_date = posted_date[:10] if posted_date else None
        source_url = f"https://www.regulations.gov/document/{doc_id}"
        raw_text = f"{title} {regulation_type} {agency_id} docket {docket_id}"

        return {
            "title": title,
            "jurisdiction": "US",
            "regulation_type": regulation_type,
            "effective_date": effective_date,
            "source_url": source_url,
            "raw_text": raw_text,
            "version_id": 1,
            "source_name": self.source_name,
            "agency_id": agency_id,
            "docket_id": docket_id,
            "regulation_id": self._hash(source_url),
            "source_hash": self._hash(raw_text),
            "ingested_at": datetime.now(timezone.utc).isoformat(),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_dataframe(self, records: list[dict]) -> pd.DataFrame:
        df = pd.DataFrame(records)
        for col in REQUIRED_COLUMNS:
            if col not in df.columns:
                df[col] = None
                self._health["schema_drift"] = True
        return df[REQUIRED_COLUMNS]

    def _dedup(self, df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        before = len(df)
        df = df.drop_duplicates(subset=["source_hash"])
        return df, before - len(df)

    def _null_rate(self, df: pd.DataFrame) -> float:
        cols = ["title", "jurisdiction", "regulation_type", "effective_date", "source_url"]
        present = [c for c in cols if c in df.columns]
        return float(df[present].isnull().mean().mean()) if present and not df.empty else 0.0

    def _land_parquet(self, df: pd.DataFrame) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = self.raw_data_dir / self.source_name / f"{ts}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        logger.info(f"Parquet: {path}")

    def _log_health(self) -> None:
        path = self.raw_data_dir / "pipeline_health.jsonl"
        with open(path, "a") as f:
            f.write(json.dumps(self._health) + "\n")

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]

    @staticmethod
    def _run_id() -> str:
        return hashlib.md5(
            f"regulations_gov{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:12]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--date-from", default="2022-01-01")
    parser.add_argument("--agencies", nargs="+", default=None)
    args = parser.parse_args()
    df = RegulationsGovConnector(
        agencies=args.agencies,
        posted_date_from=args.date_from,
    ).run(limit=args.limit)
    print(df[["title", "jurisdiction", "regulation_type", "effective_date"]].to_string())