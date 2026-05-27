"""
query_rewriter.py — LLM-assisted query rewriting and expansion.

Transforms natural language policymaker questions into optimized
retrieval queries using three techniques:

  1. Query expansion   — alternative phrasings using legal synonyms
  2. Jurisdiction detection — infers EU / US / UK from query signals
  3. HyDE              — generates a hypothetical document excerpt
                         and embeds that instead of the raw query
                         (Hypothetical Document Embedding, Gao et al. 2022)

Two modes:
  full  → calls Claude API (production)
  fast  → regex-only (offline / testing, no API call)
"""

import json
import logging
import os
import re

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL         = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")

# ------------------------------------------------------------------
# Jurisdiction keyword signals
# ------------------------------------------------------------------

JURISDICTION_SIGNALS: dict[str, list[str]] = {
    "EU": [
        "eu", "european", "gdpr", "dsa", "dma", "ai act",
        "celex", "eur-lex", "regulation (eu)", "directive",
    ],
    "UK": [
        "uk", "united kingdom", "ico", "uk gdpr",
        "data protection act", "fca", "ofcom", "cma",
    ],
    "US": [
        "us", "united states", "ftc", "fcc", "cfpb", "sec",
        "doj", "ntia", "congress", "federal register",
        "c.f.r.", "u.s.c.", "regulations.gov",
    ],
    "DE": ["germany", "german", "bfdi", "bundestag"],
    "FR": ["france", "french", "cnil"],
}

# ------------------------------------------------------------------
# System prompt for LLM rewrite
# ------------------------------------------------------------------

REWRITE_SYSTEM_PROMPT = """You are a regulatory intelligence assistant.
Given a user query, output ONLY a valid JSON object — no preamble, no markdown fences.

JSON schema:
{
  "original_query": "...",
  "expanded_queries": ["variant 1", "variant 2", "variant 3"],
  "detected_jurisdiction": "EU" | "US" | "UK" | "UNKNOWN",
  "detected_regulation_type": "GDPR" | "Enforcement Action" | "Proposed Rule" | null,
  "temporal_constraint": "recent" | "2024" | null,
  "key_legal_terms": ["term1", "term2"],
  "hyde_passage": "A 2-3 sentence hypothetical regulatory document excerpt that would answer this query."
}

Rules:
- expanded_queries: 2-3 alternatives using legal synonyms and formal terminology
- hyde_passage: written AS IF it were an excerpt from a real regulatory document
- Only set detected_jurisdiction if clearly indicated in the query
- Output ONLY the JSON object"""


class QueryRewriter:
    """
    Rewrites policymaker queries for optimal regulatory document retrieval.

    Usage:
        rewriter = QueryRewriter(mode="full")   # uses Claude API
        rewriter = QueryRewriter(mode="fast")   # regex only, no API

        result  = rewriter.rewrite("What are GDPR breach notification rules?")
        queries = rewriter.get_retrieval_queries("GDPR breach notification")
    """

    def __init__(self, mode: str = "full"):
        self.mode = mode
        self._client = None

        if mode == "full":
            if ANTHROPIC_API_KEY:
                self._init_client()
            else:
                logger.warning("ANTHROPIC_API_KEY not set — QueryRewriter falling back to fast mode")
                self.mode = "fast"

    def _init_client(self) -> None:
        try:
            import anthropic
            self._client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info("QueryRewriter: Anthropic client ready")
        except ImportError:
            logger.warning("anthropic not installed — falling back to fast mode")
            self.mode = "fast"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rewrite(self, query: str) -> dict:
        """
        Rewrite a query into an optimized retrieval form.

        Returns dict with keys:
          original_query, expanded_queries, detected_jurisdiction,
          detected_regulation_type, temporal_constraint,
          key_legal_terms, hyde_passage, rewrite_method
        """
        if self.mode == "full" and self._client:
            return self._rewrite_llm(query)
        return self._rewrite_fast(query)

    def get_retrieval_queries(self, query: str) -> list[str]:
        """
        Returns all queries to use for retrieval:
        original + expanded variants + HyDE passage (if available).
        """
        result  = self.rewrite(query)
        queries = [result["original_query"]] + result.get("expanded_queries", [])
        if result.get("hyde_passage"):
            queries.append(result["hyde_passage"])
        return queries

    def extract_jurisdiction(self, query: str) -> str | None:
        """Fast regex jurisdiction extraction — no LLM required."""
        q = query.lower()
        for jur, signals in JURISDICTION_SIGNALS.items():
            if any(sig in q for sig in signals):
                return jur
        return None

    # ------------------------------------------------------------------
    # LLM rewrite
    # ------------------------------------------------------------------

    def _rewrite_llm(self, query: str) -> dict:
        try:
            message = self._client.messages.create(
                model=LLM_MODEL,
                max_tokens=512,
                messages=[{
                    "role":    "user",
                    "content": f"{REWRITE_SYSTEM_PROMPT}\n\nQuery: {query}",
                }],
            )
            raw = message.content[0].text.strip()
            raw = re.sub(r"```json|```", "", raw).strip()
            result = json.loads(raw)
            result["rewrite_method"] = "llm"
            return result
        except Exception as exc:
            logger.warning(f"LLM rewrite failed: {exc} — falling back to fast mode")
            return self._rewrite_fast(query)

    # ------------------------------------------------------------------
    # Fast (regex) rewrite
    # ------------------------------------------------------------------

    def _rewrite_fast(self, query: str) -> dict:
        """Lightweight regex-based query analysis — no API call."""
        jurisdiction  = self.extract_jurisdiction(query)
        key_terms     = self._extract_legal_terms(query)
        reg_type      = self._detect_regulation_type(query)
        temporal      = self._detect_temporal(query)

        expanded: list[str] = []
        if jurisdiction:
            expanded.append(f"{jurisdiction} regulation: {query}")
        if key_terms:
            expanded.append(f"{' '.join(key_terms[:3])} {query}")

        return {
            "original_query":         query,
            "expanded_queries":       expanded[:3],
            "detected_jurisdiction":  jurisdiction or "UNKNOWN",
            "detected_regulation_type": reg_type,
            "temporal_constraint":    temporal,
            "key_legal_terms":        key_terms,
            "hyde_passage":           None,
            "rewrite_method":         "fast",
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_legal_terms(query: str) -> list[str]:
        patterns = [
            r"\bArticle\s+\d+[a-z]?\b",
            r"\bSection\s+\d+\b",
            r"\bGDPR\b", r"\bDSA\b", r"\bDMA\b", r"\bAI Act\b",
            r"\bCCPA\b", r"\bPIPEDA\b",
            r"\bCELEX\s*[\d\w]+",
        ]
        found = []
        for p in patterns:
            found.extend(re.findall(p, query, re.IGNORECASE))
        return list(dict.fromkeys(found))

    @staticmethod
    def _detect_regulation_type(query: str) -> str | None:
        lower = query.lower()
        if any(t in lower for t in ["enforcement", "fine", "penalty", "violation", "complaint"]):
            return "Enforcement Action"
        if any(t in lower for t in ["proposed", "rulemaking", "comment period", "nprm"]):
            return "Proposed Rule"
        if any(t in lower for t in ["directive", "regulation (eu)", "gdpr", "dsa", "dma"]):
            return "Regulation"
        return None

    @staticmethod
    def _detect_temporal(query: str) -> str | None:
        lower = query.lower()
        if any(t in lower for t in ["recent", "latest", "new", "current", "2024", "2025"]):
            return "recent"
        match = re.search(r"\b(20\d{2})\b", query)
        return match.group(1) if match else None