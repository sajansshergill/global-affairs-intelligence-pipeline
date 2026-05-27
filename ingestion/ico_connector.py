"""
ico_connector.py — UK ICO enforcement decisions connector (standalone).

Fetches GDPR enforcement decisions from the Information Commissioner's
Office via RSS feed with HTML scrape fallback. No base class — fully
self-contained.
"""

import hashlib
import json
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import urljoin

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

ICO_BASE_URL = "https://ico.org.uk"
ICO_RSS_URL = "https://ico.org.uk/feed/?post_type=enforcement"
ICO_ENFORCEMENT_URL = "https://ico.org.uk/action-weve-taken/enforcement/"

ENFORCEMENT_TYPE_MAP = {
    "monetary penalty": "Monetary Penalty Notice",
    "enforcement notice": "Enforcement Notice",
    "undertaking": "Undertaking",
    "reprimand": "Reprimand",
    "warning": "Warning",
    "prosecution": "Criminal Prosecution",
}

GDPR_ARTICLE_PATTERN = re.compile(
    r"(Article\s+\d+[a-z]?(?:\(\d+\))?(?:\s+(?:UK\s+)?GDPR)?)", re.IGNORECASE
)

FINE_PATTERN = re.compile(
    r"£([\d,]+(?:\.\d+)?)\s*(?:million|m)?", re.IGNORECASE
)

REQUIRED_COLUMNS = [
    "regulation_id", "version_id", "source_hash", "title",
    "jurisdiction", "regulation_type", "effective_date",
    "source_url", "raw_text", "ingested_at",
]


class ICOConnector:
    """
    Self-contained ICO connector.
    Fetches UK GDPR enforcement decisions — no API key required.
    """

    source_name = "ico"

    def __init__(
        self,
        raw_data_dir: str = "./data/raw",
        request_delay: float = 1.0,
        timeout: int = 30,
    ):
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
        logger.info(f"ICO ingestion — limit={limit}")
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
        logger.info(f"ICO done — {len(df)} rows, {dupes} dupes dropped")
        return df

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch_records(self, limit: int) -> list[dict]:
        records = self._fetch_via_rss(limit)
        if not records:
            logger.info("RSS empty — falling back to HTML scrape")
            records = self._fetch_via_html(limit)
        logger.info(f"ICO returned {len(records)} records")
        return records[:limit]

    def _fetch_via_rss(self, limit: int) -> list[dict]:
        try:
            time.sleep(self.request_delay)
            resp = self.session.get(ICO_RSS_URL, timeout=self.timeout)
            resp.raise_for_status()
            root = ET.fromstring(resp.content)
        except Exception as exc:
            logger.warning(f"ICO RSS failed: {exc}")
            return []

        channel = root.find("channel")
        if channel is None:
            return []

        items = []
        for el in channel.findall("item")[:limit]:
            title = self._text_el(el, "title")
            link = self._text_el(el, "link")
            pub_date = self._text_el(el, "pubDate")
            description = self._text_el(el, "description") or ""
            effective_date = self._parse_date(pub_date)
            raw_text = f"{title} {description}"

            items.append({
                "title": title,
                "jurisdiction": "UK",
                "regulation_type": self._classify(f"{title} {description}".lower()),
                "effective_date": effective_date,
                "source_url": link,
                "raw_text": raw_text,
                "version_id": 1,
                "source_name": self.source_name,
                "fine_amount_gbp": self._extract_fine(description),
                "gdpr_articles": "|".join(self._extract_articles(description)),
                "regulation_id": self._hash(link or ""),
                "source_hash": self._hash(raw_text),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            })
        return items

    def _fetch_via_html(self, limit: int) -> list[dict]:
        try:
            time.sleep(self.request_delay)
            resp = self.session.get(ICO_ENFORCEMENT_URL, timeout=self.timeout)
            html = resp.text
        except Exception as exc:
            logger.error(f"ICO HTML scrape failed: {exc}")
            return []

        link_pattern = re.compile(
            r'<a\s+href="(/action-weve-taken/enforcement/[^"]+)"[^>]*>\s*([^<]+)\s*</a>',
            re.IGNORECASE,
        )
        records = []
        for path, title in link_pattern.findall(html)[:limit]:
            title = re.sub(r"\s+", " ", title).strip()
            if not title or len(title) < 5:
                continue
            source_url = urljoin(ICO_BASE_URL, path)
            raw_text = f"{title} GDPR enforcement ICO UK"
            records.append({
                "title": title,
                "jurisdiction": "UK",
                "regulation_type": self._classify(title.lower()),
                "effective_date": None,
                "source_url": source_url,
                "raw_text": raw_text,
                "version_id": 1,
                "source_name": self.source_name,
                "regulation_id": self._hash(source_url),
                "source_hash": self._hash(raw_text),
                "ingested_at": datetime.now(timezone.utc).isoformat(),
            })
        return records

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
        for kw, label in ENFORCEMENT_TYPE_MAP.items():
            if kw in text:
                return label
        return "GDPR Enforcement Decision"

    @staticmethod
    def _extract_fine(text: str) -> float | None:
        match = FINE_PATTERN.search(text)
        if not match:
            return None
        try:
            amount = float(match.group(1).replace(",", ""))
            if "million" in text[match.start():match.end() + 10].lower():
                amount *= 1_000_000
            return amount
        except ValueError:
            return None

    @staticmethod
    def _extract_articles(text: str) -> list[str]:
        return list(dict.fromkeys(GDPR_ARTICLE_PATTERN.findall(text)))

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
            f"ico{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:12]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    df = ICOConnector().run(limit=args.limit)
    print(df[["title", "jurisdiction", "regulation_type", "effective_date"]].to_string())