"""
eurlex_connector.py — EUR-Lex SPARQL connector (standalone).

Fetches EU Regulations, Directives, and Decisions from the EUR-Lex
publications endpoint. No base class — retry, hashing, Parquet landing,
and health logging are all self-contained.
"""

import hashlib
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

EURLEX_SPARQL_ENDPOINT = "https://publications.europa.eu/webapi/rdf/sparql"

SPARQL_QUERY = """
PREFIX cdm: <http://publications.europa.eu/ontology/cdm#>
PREFIX skos: <http://www.w3.org/2004/02/skos/core#>
PREFIX dc: <http://purl.org/dc/elements/1.1/>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>

SELECT DISTINCT ?work ?title ?date ?type ?celex
WHERE {{
  ?work cdm:work_date_document ?date ;
        cdm:resource_legal_id_celex ?celex .
  OPTIONAL {{ ?work dc:title ?title . FILTER(lang(?title) = "en") }}
  OPTIONAL {{ ?work cdm:work_has_resource-type ?typeNode .
              ?typeNode skos:prefLabel ?type . FILTER(lang(?type) = "en") }}
  FILTER(?date >= "{date_from}"^^xsd:date)
  FILTER(STRSTARTS(STR(?celex), "3"))
}}
ORDER BY DESC(?date)
LIMIT {limit}
"""

REGULATION_TYPE_MAP = {
    "regulation": "Regulation",
    "directive": "Directive",
    "decision": "Decision",
    "recommendation": "Recommendation",
}

REQUIRED_COLUMNS = [
    "regulation_id", "version_id", "source_hash", "title",
    "jurisdiction", "regulation_type", "effective_date",
    "source_url", "raw_text", "ingested_at",
]


class EURLexConnector:
    """
    Self-contained EUR-Lex connector.
    Fetches EU legislative acts via SPARQL — no API key required.
    """

    session = None
    source_name = "eurlex"
    BASE_DOC_URL = "https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:{celex}"

    def __init__(
        self,
        date_from: str = "2020-01-01",
        raw_data_dir: str = "./data/raw",
        request_delay: float = 1.0,
        timeout: int = 30,
    ):
        self.date_from = date_from
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
        logger.info(f"EUR-Lex ingestion — date_from={self.date_from} limit={limit}")
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
        logger.info(f"EUR-Lex done — {len(df)} rows, {dupes} dupes dropped")
        return df

    # ------------------------------------------------------------------
    # Fetch
    # ------------------------------------------------------------------

    def fetch_records(self, limit: int) -> list[dict]:
        query = SPARQL_QUERY.format(date_from=self.date_from, limit=limit)
        time.sleep(self.request_delay)
        resp = self.session.get(
            EURLEX_SPARQL_ENDPOINT,
            params={"query": query, "format": "application/sparql-results+json"},
            timeout=self.timeout,
        )
        resp.raise_for_status()
        bindings = resp.json().get("results", {}).get("bindings", [])
        logger.info(f"EUR-Lex SPARQL returned {len(bindings)} bindings")
        return [self._parse_binding(b) for b in bindings]

    def _parse_binding(self, b: dict) -> dict:
        celex = self._val(b, "celex")
        title = self._val(b, "title") or f"EUR-Lex {celex}"
        raw_type = self._val(b, "type") or ""
        date_str = self._val(b, "date")
        source_url = self.BASE_DOC_URL.format(celex=celex) if celex else ""
        raw_text = f"{title} {celex} {raw_type}"

        return {
            "title": title,
            "jurisdiction": "EU",
            "regulation_type": self._classify(raw_type),
            "effective_date": (date_str or "")[:10] or None,
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
    def _classify(raw: str) -> str:
        low = raw.lower()
        for kw, label in REGULATION_TYPE_MAP.items():
            if kw in low:
                return label
        return "Legal Act"

    @staticmethod
    def _val(b: dict, key: str) -> str | None:
        node = b.get(key, {})
        return node.get("value") if node else None

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]

    @staticmethod
    def _run_id() -> str:
        return hashlib.md5(
            f"eurlex{datetime.now(timezone.utc).isoformat()}".encode()
        ).hexdigest()[:12]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--date-from", default="2022-01-01")
    args = parser.parse_args()
    df = EURLexConnector(date_from=args.date_from).run(limit=args.limit)
    print(df[["title", "jurisdiction", "regulation_type", "effective_date"]].to_string())