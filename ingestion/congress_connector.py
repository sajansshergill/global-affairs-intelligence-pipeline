"""
congress_connector.py — Congress.gov REST API connector (standalone).

Fetches US federal legislation via the Congress.gov API v3.
No base class — all retry, hashing, Parquet landing, and health
logging logic is self-contained.
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

CONGRESS_API_BASE = "https://api.congress.gov/v3"

BILL_TYPE_MAP = {
    "hr":      "House Bill",
    "s":       "Senate Bill",
    "hjres":   "House Joint Resolution",
    "sjres":   "Senate Joint Resolution",
    "hconres": "House Concurrent Resolution",
    "sconres": "Senate Concurrent Resolution",
    "hres":    "House Simple Resolution",
    "sres":    "Senate Simple Resolution",
}

REQUIRED_COLUMNS = [
    "regulation_id", "version_id", "source_hash", "title",
    "jurisdiction", "regulation_type", "effective_date",
    "source_url", "raw_text", "ingested_at",
]


class CongressConnector:
    """
    Self-contained Congress.gov connector.
    Fetches US federal bills via the public REST API.
    Free API key at https://api.congress.gov/sign-up/
    Set CONGRESS_API_KEY in your .env — falls back to DEMO_KEY.
    """

    source_name = "congress"

    def __init__(
        self,
        congress: int = 118,
        raw_data_dir: str = "./data/raw",
        request_delay: float = 1.5,
        timeout: int = 30,
    ):
        self.congress = congress
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self.timeout = timeout
        self.api_key = os.getenv("CONGRESS_API_KEY", "DEMO_KEY")

        if self.api_key == "DEMO_KEY":
            logger.warning("Using DEMO_KEY — rate limited to 40 req/hr. "
                           "Get a free key at https://api.congress.gov/sign-up/")

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
        logger.info(f"Congress.gov ingestion — congress={self.congress} limit={limit}")
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
        logger.info(f"Congress done — {len(df)} rows, {dupes} dupes dropped")
        return df

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch_records(self, limit: int) -> list[dict]:
        records: list[dict] = []
        offset = 0
        page_size = min(limit, 20)

        while len(records) < limit:
            batch = self._fetch_page(offset=offset, page_size=page_size)
            if not batch:
                break
            records.extend(batch)
            offset += page_size
            page_size = min(limit - len(records), 20)

        logger.info(f"Congress.gov fetched {len(records)} records")
        return records[:limit]

    def _fetch_page(self, offset: int, page_size: int) -> list[dict]:
        url = f"{CONGRESS_API_BASE}/bill/{self.congress}"
        params = {
            "api_key": self.api_key,
            "format": "json",
            "limit": page_size,
            "offset": offset,
            "sort": "updateDate+desc",
        }
        time.sleep(self.request_delay)
        try:
            resp = self.session.get(url, params=params, timeout=self.timeout)
            resp.raise_for_status()
            bills = resp.json().get("bills", [])
            return [self._parse_bill(b) for b in bills]
        except Exception as exc:
            logger.error(f"Congress page failed at offset={offset}: {exc}")
            return []

    def _parse_bill(self, bill: dict) -> dict:
        bill_type = bill.get("type", "").lower()
        bill_number = bill.get("number", "")
        congress_num = bill.get("congress", self.congress)
        title = bill.get("title", f"Bill {bill_type.upper()} {bill_number}")
        policy_area = bill.get("policyArea", {})
        policy_name = policy_area.get("name", "") if isinstance(policy_area, dict) else ""
        latest_action = bill.get("latestAction", {})
        action_date = latest_action.get("actionDate") if isinstance(latest_action, dict) else None
        origin_chamber = bill.get("originChamber", "")
        regulation_type = BILL_TYPE_MAP.get(bill_type, "Legislation")
        chamber_slug = "house" if origin_chamber == "House" else "senate"
        source_url = (
            f"https://www.congress.gov/bill/{congress_num}th-congress/"
            f"{chamber_slug}-bill/{bill_number}"
        )
        raw_text = f"{title} {policy_name} {regulation_type} Congress {congress_num}"

        return {
            "title": title,
            "jurisdiction": "US",
            "regulation_type": regulation_type,
            "effective_date": action_date,
            "source_url": source_url,
            "raw_text": raw_text,
            "version_id": 1,
            "source_name": self.source_name,
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
            f"congress{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:12]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--congress", type=int, default=118)
    args = parser.parse_args()
    df = CongressConnector(congress=args.congress).run(limit=args.limit)
    print(df[["title", "jurisdiction", "regulation_type", "effective_date"]].to_string())