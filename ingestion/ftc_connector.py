"""
ftc_connector.py — FTC RSS + HTML scraper connector (standalone).

Fetches Federal Trade Commission enforcement actions from FTC's
public RSS feeds. No base class — fully self-contained.
"""

import hashlib
import json
import logging
import time
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

FTC_RSS_FEEDS = [
    "https://www.ftc.gov/feeds/press-release-and-consumer-alert.xml",
    "https://www.ftc.gov/feeds/enforcement-actions.xml",
]

ENFORCEMENT_KEYWORDS = [
    "enforcement", "complaint", "settlement", "consent order",
    "civil penalty", "injunction", "violation", "charges",
    "data breach", "privacy", "deceptive", "unfair",
]

TOPIC_TYPE_MAP = {
    "privacy":     "Privacy Enforcement",
    "data":        "Data Protection",
    "merger":      "Merger Review",
    "antitrust":   "Antitrust Action",
    "consumer":    "Consumer Protection",
    "advertising": "Advertising Enforcement",
    "competition": "Competition Enforcement",
}

REQUIRED_COLUMNS = [
    "regulation_id", "version_id", "source_hash", "title",
    "jurisdiction", "regulation_type", "effective_date",
    "source_url", "raw_text", "ingested_at",
]


class FTCConnector:
    """
    Self-contained FTC connector.
    Parses RSS feeds and filters for enforcement actions — no API key required.
    """

    source_name = "ftc"

    def __init__(
        self,
        fetch_full_text: bool = False,
        raw_data_dir: str = "./data/raw",
        request_delay: float = 1.0,
        timeout: int = 30,
    ):
        self.fetch_full_text = fetch_full_text
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.request_delay = request_delay
        self.timeout = timeout

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
        logger.info(f"FTC ingestion — limit={limit}")
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
        logger.info(f"FTC done — {len(df)} rows, {dupes} dupes dropped")
        return df

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch_records(self, limit: int) -> list[dict]:
        all_items: list[dict] = []

        for feed_url in FTC_RSS_FEEDS:
            try:
                items = self._parse_feed(feed_url)
                all_items.extend(items)
                logger.info(f"Feed {feed_url} → {len(items)} items")
            except Exception as exc:
                logger.warning(f"Feed failed: {feed_url} — {exc}")

        enforcement = self._filter_enforcement(all_items)

        # Deduplicate by URL
        seen: set[str] = set()
        unique: list[dict] = []
        for item in enforcement:
            url = item.get("source_url", "")
            if url not in seen:
                seen.add(url)
                unique.append(item)

        if self.fetch_full_text:
            unique = [self._enrich(item) for item in unique[:limit]]

        logger.info(f"FTC: {len(all_items)} total → {len(unique)} unique enforcement items")
        return unique[:limit]

    def _parse_feed(self, feed_url: str) -> list[dict]:
        time.sleep(self.request_delay)
        resp = self.session.get(feed_url, timeout=self.timeout)
        resp.raise_for_status()
        root = ET.fromstring(resp.content)
        channel = root.find("channel")
        if channel is None:
            return []

        items = []
        for item_el in channel.findall("item"):
            title = self._text_el(item_el, "title")
            link = self._text_el(item_el, "link")
            pub_date = self._text_el(item_el, "pubDate")
            description = self._text_el(item_el, "description") or ""
            effective_date = self._parse_date(pub_date)
            raw_text = f"{title} {description}"

            items.append({
                "title": title,
                "source_url": link,
                "effective_date": effective_date,
                "raw_text": raw_text,
                "description": description,
                "jurisdiction": "US",
                "source_name": self.source_name,
                "version_id": 1,
                "regulation_id": self._hash(link or ""),
                "source_hash": self._hash(raw_text),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            })
        return items

    def _filter_enforcement(self, items: list[dict]) -> list[dict]:
        filtered = []
        for item in items:
            text = f"{item.get('title', '')} {item.get('description', '')}".lower()
            if any(kw in text for kw in ENFORCEMENT_KEYWORDS):
                item["regulation_type"] = self._classify(text)
                filtered.append(item)
        return filtered

    def _enrich(self, item: dict) -> dict:
        url = item.get("source_url", "")
        if not url:
            return item
        try:
            time.sleep(self.request_delay)
            resp = self.session.get(url, timeout=self.timeout)
            html = resp.text
            clean = re.sub(r"<[^>]+>", " ", html)
            clean = re.sub(r"\s+", " ", clean).strip()
            item["raw_text"] = clean[:10_000]
            item["source_hash"] = self._hash(item["raw_text"])
        except Exception as exc:
            logger.warning(f"Full text fetch failed for {url}: {exc}")
        return item

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
    def _classify(text: str) -> str:
        for kw, label in TOPIC_TYPE_MAP.items():
            if kw in text:
                return label
        return "Enforcement Action"

    @staticmethod
    def _text_el(el: ET.Element, tag: str) -> str | None:
        child = el.find(tag)
        return child.text.strip() if child is not None and child.text else None

    @staticmethod
    def _parse_date(date_str: str | None) -> str | None:
        if not date_str:
            return None
        try:
            return parsedate_to_datetime(date_str).strftime("%Y-%m-%d")
        except Exception:
            return date_str[:10] if date_str else None

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]

    @staticmethod
    def _run_id() -> str:
        return hashlib.md5(
            f"ftc{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:12]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--full-text", action="store_true")
    args = parser.parse_args()
    df = FTCConnector(fetch_full_text=args.full_text).run(limit=args.limit)
    print(df[["title", "jurisdiction", "regulation_type", "effective_date"]].to_string())