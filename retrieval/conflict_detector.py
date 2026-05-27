"""
conflict_detector.py — Cross-jurisdiction regulatory conflict detection.

Identifies cases where two jurisdictions have contradictory rules on
the same regulatory topic — a core Global Affairs use case.

Real-world conflicts GARIP detects:
  - EU GDPR Art. 44 (data localisation) vs US CLOUD Act (compelled disclosure)
  - EU AI Act prohibited practices vs US permissive federal AI policy
  - GDPR right to erasure vs US litigation hold obligations
  - EU ePrivacy consent requirements vs US opt-out tracking rules

Two detection strategies:
  1. SQL patterns  — deterministic, based on known hardcoded conflicts
  2. LLM analysis  — Claude compares retrieved chunks for novel conflicts
"""

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL         = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")

# ------------------------------------------------------------------
# LLM prompt
# ------------------------------------------------------------------

CONFLICT_PROMPT = """You are a regulatory compliance expert.
I will give you two regulatory excerpts from different jurisdictions on the same topic.
Identify whether they conflict and describe it precisely.

Jurisdiction A ({jur_a}):
{text_a}

Jurisdiction B ({jur_b}):
{text_b}

Topic: {topic}

Output ONLY valid JSON — no preamble:
{{
  "conflict_detected": true | false,
  "severity": "high" | "medium" | "low" | null,
  "conflict_summary": "One sentence or null",
  "details": "2-3 sentences explaining the conflict in legal terms",
  "recommendation": "What a compliance team should do"
}}"""

# ------------------------------------------------------------------
# Known conflict patterns (SQL-based, always evaluated)
# ------------------------------------------------------------------

KNOWN_PATTERNS = [
    {
        "topic":          "Data Cross-Border Transfers",
        "jurisdiction_a": "EU",
        "jurisdiction_b": "US",
        "severity":       "high",
        "summary": (
            "EU GDPR Article 44 restricts cross-border data transfers while the "
            "US CLOUD Act compels US companies to provide data stored abroad to US authorities."
        ),
        "query_a": (
            "SELECT * FROM regulations_latest "
            "WHERE jurisdiction='EU' AND raw_text ILIKE '%cross-border%transfer%' LIMIT 3"
        ),
        "query_b": (
            "SELECT * FROM regulations_latest "
            "WHERE jurisdiction='US' AND (raw_text ILIKE '%CLOUD Act%' "
            "OR raw_text ILIKE '%subpoena%data%') LIMIT 3"
        ),
    },
    {
        "topic":          "AI System Regulation",
        "jurisdiction_a": "EU",
        "jurisdiction_b": "US",
        "severity":       "medium",
        "summary": (
            "EU AI Act imposes strict prohibitions on high-risk AI systems while "
            "the US takes a sector-specific approach with no equivalent federal law."
        ),
        "query_a": (
            "SELECT * FROM regulations_latest "
            "WHERE jurisdiction='EU' AND raw_text ILIKE '%AI Act%' LIMIT 3"
        ),
        "query_b": (
            "SELECT * FROM regulations_latest "
            "WHERE jurisdiction='US' AND raw_text ILIKE '%artificial intelligence%' LIMIT 3"
        ),
    },
    {
        "topic":          "Right to Erasure",
        "jurisdiction_a": "EU",
        "jurisdiction_b": "US",
        "severity":       "medium",
        "summary": (
            "GDPR Article 17 grants individuals a right to erasure while US litigation "
            "hold requirements may legally compel organisations to retain the same data."
        ),
        "query_a": (
            "SELECT * FROM regulations_latest "
            "WHERE jurisdiction='EU' AND raw_text ILIKE '%right to erasure%' LIMIT 3"
        ),
        "query_b": (
            "SELECT * FROM regulations_latest "
            "WHERE jurisdiction='US' AND raw_text ILIKE '%legal hold%' LIMIT 3"
        ),
    },
    {
        "topic":          "Cookie and Tracking Consent",
        "jurisdiction_a": "EU",
        "jurisdiction_b": "US",
        "severity":       "low",
        "summary": (
            "EU ePrivacy rules require explicit opt-in consent for tracking cookies "
            "while US federal law lacks equivalent consent requirements."
        ),
        "query_a": (
            "SELECT * FROM regulations_latest "
            "WHERE jurisdiction='EU' AND raw_text ILIKE '%consent%cookie%' LIMIT 3"
        ),
        "query_b": (
            "SELECT * FROM regulations_latest "
            "WHERE jurisdiction='US' AND raw_text ILIKE '%tracking%' LIMIT 3"
        ),
    },
]


@dataclass
class ConflictSignal:
    """A detected regulatory conflict between two jurisdictions."""
    signal_id:        str
    jurisdiction_a:   str
    jurisdiction_b:   str
    regulation_id_a:  str
    regulation_id_b:  str
    topic:            str
    conflict_summary: str
    severity:         str       # "high" | "medium" | "low"
    details:          str
    recommendation:   str
    detected_at:      str
    detection_method: str       # "sql" | "llm"


class ConflictDetector:
    """
    Detects cross-jurisdiction regulatory conflicts.

    Usage:
        detector = ConflictDetector(duckdb_loader=loader, mode="sql")
        signals  = detector.detect_all()
        detector.save_to_duckdb(signals)

        # LLM-based novel conflict detection
        detector = ConflictDetector(
            duckdb_loader=loader,
            hybrid_retriever=retriever,
            mode="llm",
        )
        signals = detector.detect_for_topic("biometric data processing", ["EU", "US"])
    """

    def __init__(
        self,
        duckdb_loader=None,
        hybrid_retriever=None,
        mode: str = "sql",
    ):
        self.duckdb_loader    = duckdb_loader
        self.hybrid_retriever = hybrid_retriever
        self.mode             = mode
        self._client          = None

        if mode == "llm" and ANTHROPIC_API_KEY:
            self._init_client()

    def _init_client(self) -> None:
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info("ConflictDetector: Anthropic client ready")
        except ImportError:
            logger.warning("anthropic not installed — LLM detection disabled")
            self.mode = "sql"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect_all(self) -> list[ConflictSignal]:
        """Run all detection strategies and return deduplicated signals."""
        signals: list[ConflictSignal] = []
        signals.extend(self._detect_sql_patterns())

        if self.mode == "llm" and self._client and self.hybrid_retriever:
            signals.extend(self._detect_llm_conflicts())

        # Deduplicate by topic + jurisdiction pair
        seen: set[str] = set()
        unique: list[ConflictSignal] = []
        for s in signals:
            key = f"{s.topic}::{s.jurisdiction_a}::{s.jurisdiction_b}"
            if key not in seen:
                seen.add(key)
                unique.append(s)

        logger.info(f"Conflict detection complete: {len(unique)} signals")
        return unique

    def detect_for_topic(
        self,
        topic: str,
        jurisdictions: list[str] | None = None,
    ) -> list[ConflictSignal]:
        """
        LLM-based detection for a specific regulatory topic.
        Retrieves relevant chunks for each jurisdiction pair and
        asks Claude to identify conflicts.
        """
        jurisdictions = jurisdictions or ["EU", "US"]
        signals: list[ConflictSignal] = []

        if not self.hybrid_retriever:
            logger.warning("No retriever — cannot do topic-based detection")
            return signals

        for jur_a, jur_b in combinations(jurisdictions, 2):
            results_a = self.hybrid_retriever.retrieve(
                query=topic, top_k=3,
                filter_metadata={"jurisdiction": jur_a},
            )
            results_b = self.hybrid_retriever.retrieve(
                query=topic, top_k=3,
                filter_metadata={"jurisdiction": jur_b},
            )

            if results_a and results_b:
                signal = self._llm_compare(
                    topic=topic,
                    jur_a=jur_a,
                    text_a="\n".join(r["text"] for r in results_a[:2]),
                    reg_id_a=results_a[0]["metadata"].get("regulation_id", ""),
                    jur_b=jur_b,
                    text_b="\n".join(r["text"] for r in results_b[:2]),
                    reg_id_b=results_b[0]["metadata"].get("regulation_id", ""),
                )
                if signal:
                    signals.append(signal)

        return signals

    def save_to_duckdb(self, signals: list[ConflictSignal]) -> int:
        """Persist conflict signals to DuckDB conflict_signals table."""
        if not self.duckdb_loader or not signals:
            return 0

        import pandas as pd
        rows = [
            {
                "signal_id":        s.signal_id,
                "jurisdiction_a":   s.jurisdiction_a,
                "jurisdiction_b":   s.jurisdiction_b,
                "regulation_id_a":  s.regulation_id_a,
                "regulation_id_b":  s.regulation_id_b,
                "topic":            s.topic,
                "conflict_summary": s.conflict_summary,
                "severity":         s.severity,
                "detected_at":      s.detected_at,
            }
            for s in signals
        ]
        df = pd.DataFrame(rows)
        conn = self.duckdb_loader._get_conn()
        conn.register("_conflict_signals", df)
        conn.execute("""
            INSERT INTO conflict_signals SELECT * FROM _conflict_signals
            ON CONFLICT (signal_id) DO NOTHING
        """)
        conn.unregister("_conflict_signals")
        logger.info(f"Saved {len(signals)} conflict signals to DuckDB")
        return len(signals)

    # ------------------------------------------------------------------
    # SQL-based detection
    # ------------------------------------------------------------------

    def _detect_sql_patterns(self) -> list[ConflictSignal]:
        signals: list[ConflictSignal] = []

        for pattern in KNOWN_PATTERNS:
            reg_id_a = reg_id_b = "known_pattern"

            if self.duckdb_loader:
                try:
                    df_a = self.duckdb_loader.query(pattern["query_a"])
                    df_b = self.duckdb_loader.query(pattern["query_b"])
                    if df_a.empty or df_b.empty:
                        continue
                    reg_id_a = df_a.iloc[0].get("regulation_id", "")
                    reg_id_b = df_b.iloc[0].get("regulation_id", "")
                except Exception as exc:
                    logger.debug(f"SQL pattern query failed: {exc}")

            signal_id = hashlib.sha256(
                f"{pattern['topic']}::{pattern['jurisdiction_a']}::{pattern['jurisdiction_b']}".encode()
            ).hexdigest()[:24]

            signals.append(ConflictSignal(
                signal_id=signal_id,
                jurisdiction_a=pattern["jurisdiction_a"],
                jurisdiction_b=pattern["jurisdiction_b"],
                regulation_id_a=reg_id_a,
                regulation_id_b=reg_id_b,
                topic=pattern["topic"],
                conflict_summary=pattern["summary"],
                severity=pattern["severity"],
                details="",
                recommendation="Consult legal counsel to assess compliance obligations in both jurisdictions.",
                detected_at=datetime.now(timezone.utc).isoformat(),
                detection_method="sql",
            ))

        return signals

    # ------------------------------------------------------------------
    # LLM-based detection
    # ------------------------------------------------------------------

    def _detect_llm_conflicts(self) -> list[ConflictSignal]:
        topics = [
            "biometric data collection and processing",
            "algorithmic decision making and automated profiling",
            "government access to private data",
            "data breach notification requirements",
        ]
        signals: list[ConflictSignal] = []
        for topic in topics:
            try:
                signals.extend(self.detect_for_topic(topic))
            except Exception as exc:
                logger.warning(f"LLM detection failed for '{topic}': {exc}")
        return signals

    def _llm_compare(
        self,
        topic: str,
        jur_a: str, text_a: str, reg_id_a: str,
        jur_b: str, text_b: str, reg_id_b: str,
    ) -> ConflictSignal | None:
        if not self._client:
            return None

        prompt = CONFLICT_PROMPT.format(
            jur_a=jur_a, text_a=text_a[:1500],
            jur_b=jur_b, text_b=text_b[:1500],
            topic=topic,
        )
        try:
            msg = self._client.messages.create(
                model=LLM_MODEL,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw    = msg.content[0].text.strip()
            raw    = re.sub(r"```json|```", "", raw).strip()
            result = json.loads(raw)

            if not result.get("conflict_detected"):
                return None

            signal_id = hashlib.sha256(
                f"{topic}::{jur_a}::{jur_b}".encode()
            ).hexdigest()[:24]

            return ConflictSignal(
                signal_id=signal_id,
                jurisdiction_a=jur_a,
                jurisdiction_b=jur_b,
                regulation_id_a=reg_id_a,
                regulation_id_b=reg_id_b,
                topic=topic,
                conflict_summary=result.get("conflict_summary", ""),
                severity=result.get("severity", "medium"),
                details=result.get("details", ""),
                recommendation=result.get("recommendation", ""),
                detected_at=datetime.now(timezone.utc).isoformat(),
                detection_method="llm",
            )
        except Exception as exc:
            logger.warning(f"LLM compare failed {jur_a}/{jur_b} '{topic}': {exc}")
            return None