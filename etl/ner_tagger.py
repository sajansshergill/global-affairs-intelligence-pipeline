"""
ner_tagger.py — NER tagging for GARIP ETL layer.

Extracts structured metadata from regulatory text:
  - Jurisdiction (EU, US, UK, etc.)
  - Regulation type (Directive, Regulation, Enforcement Action, etc.)
  - Effective / publication dates
  - Cited GDPR / legislative articles
  - Named entities (organizations, geographic entities)

Uses spaCy for NER with regex fallbacks — no API key required.
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Jurisdiction signals
# ------------------------------------------------------------------

JURISDICTION_PATTERNS: list[tuple[str, list[str]]] = [
    ("EU", [
        r"\bEU\b", r"\bEuropean Union\b", r"\bEuropean Parliament\b",
        r"\bEUR-Lex\b", r"\bCELEX\b", r"\bGDPR\b", r"\bDSA\b", r"\bDMA\b",
        r"\bAI Act\b", r"Regulation \(EU\)", r"Directive \d{4}/\d+/EU",
    ]),
    ("UK", [
        r"\bUK\b", r"\bUnited Kingdom\b", r"\bICO\b",
        r"\bUK GDPR\b", r"\bData Protection Act\b", r"\bFCA\b",
        r"\bOfcom\b", r"\bCMA\b",
    ]),
    ("US", [
        r"\bUSA?\b", r"\bUnited States\b", r"\bFTC\b", r"\bFCC\b",
        r"\bCFPB\b", r"\bSEC\b", r"\bDOJ\b", r"\bNTIA\b",
        r"\bCongress\b", r"\bFederal Register\b", r"\bC\.F\.R\.\b",
        r"\bU\.S\.C\.\b", r"regulations\.gov",
    ]),
    ("DE", [r"\bGermany\b", r"\bBundestag\b", r"\bBundesrat\b", r"\bBfDI\b"]),
    ("FR", [r"\bFrance\b", r"\bCNIL\b", r"\bConseil d'État\b"]),
    ("CA", [r"\bCanada\b", r"\bPIPEDA\b", r"\bOPC\b", r"\bPrivacy Commissioner\b"]),
    ("AU", [r"\bAustralia\b", r"\bOAIC\b", r"\bAPP\b", r"\bACCC\b"]),
]

# ------------------------------------------------------------------
# Regulation type signals
# ------------------------------------------------------------------

REGULATION_TYPE_PATTERNS: list[tuple[str, list[str]]] = [
    ("Regulation", [r"\bRegulation \(EU\)", r"\bFinal Rule\b", r"\bfederal regulation\b"]),
    ("Directive", [r"\bDirective \d{4}/\d+", r"\bEU Directive\b"]),
    ("Enforcement Action", [
        r"\bcomplaint\b", r"\bsettlement\b", r"\bconsent order\b",
        r"\bcivil penalty\b", r"\benforcement action\b",
    ]),
    ("Monetary Penalty", [r"\bmonetary penalty\b", r"\bfine of\b", r"£[\d,]+", r"\$[\d,]+ million"]),
    ("Proposed Rule", [r"\bProposed Rule\b", r"\bNPRM\b", r"\bNotice of Proposed Rulemaking\b"]),
    ("Guidance", [r"\bguidance\b", r"\bguidelines?\b", r"\bframework\b"]),
    ("Decision", [r"\bdecision\b", r"\bruling\b", r"\badjudication\b"]),
    ("Notice", [r"\bFederal Notice\b", r"\bfederal register notice\b"]),
]

# ------------------------------------------------------------------
# Date patterns
# ------------------------------------------------------------------

DATE_PATTERNS = [
    # ISO: 2024-03-15
    (re.compile(r"\b(\d{4}-\d{2}-\d{2})\b"), "%Y-%m-%d"),
    # Long form: 15 March 2024 / March 15, 2024
    (
        re.compile(
            r"\b(\d{1,2}\s+(?:January|February|March|April|May|June|July|"
            r"August|September|October|November|December)\s+\d{4})\b",
            re.IGNORECASE,
        ),
        "%d %B %Y",
    ),
    (
        re.compile(
            r"\b((?:January|February|March|April|May|June|July|"
            r"August|September|October|November|December)\s+\d{1,2},?\s+\d{4})\b",
            re.IGNORECASE,
        ),
        "%B %d %Y",
    ),
    # Short: 03/15/2024 or 15/03/2024
    (re.compile(r"\b(\d{2}/\d{2}/\d{4})\b"), None),  # ambiguous — kept as string
]

# Legislative article citations
ARTICLE_CITATION_PATTERN = re.compile(
    r"(Article\s+\d+[a-z]?(?:\(\d+\))?(?:\s+(?:of\s+)?(?:the\s+)?(?:GDPR|UK\s+GDPR|DSA|DMA|AI\s+Act))?)",
    re.IGNORECASE,
)

# Fine / penalty amount extraction
FINE_PATTERN = re.compile(
    r"(?:£|€|\$|USD|EUR|GBP)\s*([\d,]+(?:\.\d+)?)\s*(?:million|billion|m|bn)?",
    re.IGNORECASE,
)


@dataclass
class TaggedDocument:
    """NER output for a single document."""
    text: str
    jurisdiction: str
    jurisdiction_confidence: float
    regulation_type: str
    effective_date: str | None
    dates_found: list[str]
    article_citations: list[str]
    fine_amounts: list[float]
    named_entities: list[dict]          # [{text, label, start, end}]
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class NERTagger:
    """
    Extracts structured regulatory metadata from text.

    Two-pass strategy:
      1. Fast regex pass — jurisdiction, type, dates, citations, fines
      2. spaCy NER pass (if available) — ORG, GPE, LAW, DATE entities

    Designed to run on raw_text from connectors OR on individual
    DocumentBlock.text from the PDF parser.
    """

    def __init__(self, use_spacy: bool = True, spacy_model: str = "en_core_web_sm"):
        self._nlp = None
        if use_spacy:
            self._load_spacy(spacy_model)

    def _load_spacy(self, model: str):
        try:
            import spacy
            self._nlp = spacy.load(model)
            logger.info(f"spaCy model loaded: {model}")
        except (ImportError, OSError) as exc:
            logger.warning(f"spaCy unavailable ({exc}) — using regex only")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tag(self, text: str, existing_metadata: dict | None = None) -> TaggedDocument:
        """
        Tag a single text string with regulatory metadata.

        Args:
            text:              The document or chunk text to tag.
            existing_metadata: Pre-existing fields (e.g. from connector) to
                               incorporate — connector values take precedence.

        Returns:
            TaggedDocument with all extracted fields.
        """
        existing_metadata = existing_metadata or {}

        # Pass 1: Regex extraction
        jurisdiction, j_conf = self._extract_jurisdiction(text)
        reg_type = self._extract_regulation_type(text)
        dates = self._extract_dates(text)
        effective_date = dates[0] if dates else None
        articles = self._extract_article_citations(text)
        fines = self._extract_fines(text)

        # Pass 2: spaCy NER
        named_entities = self._extract_named_entities(text)

        # Connector values override regex inferences
        jurisdiction = existing_metadata.get("jurisdiction") or jurisdiction
        reg_type = existing_metadata.get("regulation_type") or reg_type
        effective_date = existing_metadata.get("effective_date") or effective_date

        return TaggedDocument(
            text=text,
            jurisdiction=jurisdiction,
            jurisdiction_confidence=j_conf,
            regulation_type=reg_type,
            effective_date=effective_date,
            dates_found=dates,
            article_citations=list(dict.fromkeys(articles)),  # dedupe, preserve order
            fine_amounts=fines,
            named_entities=named_entities,
            raw_metadata=existing_metadata,
        )

    def tag_dataframe(self, df) -> "pd.DataFrame":
        """
        Apply NER tagging to a DataFrame with a 'raw_text' column.
        Fills in / enriches: jurisdiction, regulation_type, effective_date,
        article_citations, fine_amounts.
        """
        import pandas as pd

        results = []
        for _, row in df.iterrows():
            existing = row.to_dict()
            tagged = self.tag(row.get("raw_text", ""), existing_metadata=existing)
            results.append({
                **existing,
                "jurisdiction": tagged.jurisdiction,
                "regulation_type": tagged.regulation_type,
                "effective_date": tagged.effective_date,
                "article_citations": "|".join(tagged.article_citations),
                "fine_amounts": str(tagged.fine_amounts),
                "named_entities_count": len(tagged.named_entities),
            })

        return pd.DataFrame(results)

    # ------------------------------------------------------------------
    # Extraction helpers
    # ------------------------------------------------------------------

    def _extract_jurisdiction(self, text: str) -> tuple[str, float]:
        """Return (jurisdiction_code, confidence_score)."""
        scores: dict[str, int] = {}
        for jur, patterns in JURISDICTION_PATTERNS:
            count = sum(
                len(re.findall(p, text, re.IGNORECASE)) for p in patterns
            )
            if count:
                scores[jur] = count

        if not scores:
            return "UNKNOWN", 0.0

        best = max(scores, key=lambda k: scores[k])
        total = sum(scores.values())
        confidence = scores[best] / total if total else 0.0
        return best, round(confidence, 3)

    def _extract_regulation_type(self, text: str) -> str:
        text_lower = text.lower()
        for reg_type, patterns in REGULATION_TYPE_PATTERNS:
            for pattern in patterns:
                if re.search(pattern, text, re.IGNORECASE):
                    return reg_type
        return "Legal Document"

    def _extract_dates(self, text: str) -> list[str]:
        """Return list of date strings found, in order of appearance."""
        found: list[str] = []
        for pattern, fmt in DATE_PATTERNS:
            for match in pattern.finditer(text):
                raw = match.group(1).replace(",", "")
                if fmt:
                    try:
                        dt = datetime.strptime(raw, fmt)
                        found.append(dt.strftime("%Y-%m-%d"))
                        continue
                    except ValueError:
                        pass
                # Ambiguous or failed parse — keep raw
                found.append(raw)

        # Deduplicate preserving order
        seen = set()
        deduped = []
        for d in found:
            if d not in seen:
                seen.add(d)
                deduped.append(d)
        return deduped

    def _extract_article_citations(self, text: str) -> list[str]:
        return ARTICLE_CITATION_PATTERN.findall(text)

    def _extract_fines(self, text: str) -> list[float]:
        fines: list[float] = []
        for match in FINE_PATTERN.finditer(text):
            raw = match.group(1).replace(",", "")
            try:
                amount = float(raw)
                suffix = text[match.end():match.end() + 10].lower()
                if "billion" in suffix or "bn" in suffix:
                    amount *= 1_000_000_000
                elif "million" in suffix or " m" in suffix:
                    amount *= 1_000_000
                fines.append(amount)
            except ValueError:
                pass
        return fines

    def _extract_named_entities(self, text: str) -> list[dict]:
        """Run spaCy NER if available, else return empty list."""
        if not self._nlp:
            return []
        # Cap text length to avoid spaCy memory issues on huge docs
        doc = self._nlp(text[:50_000])
        return [
            {
                "text": ent.text,
                "label": ent.label_,
                "start": ent.start_char,
                "end": ent.end_char,
            }
            for ent in doc.ents
            if ent.label_ in {"ORG", "GPE", "LAW", "DATE", "PERSON", "NORP"}
        ]


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run NER tagger on text")
    parser.add_argument("--text", type=str, help="Text to tag")
    parser.add_argument("--file", type=str, help="File containing text to tag")
    args = parser.parse_args()

    if args.file:
        with open(args.file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        text = (
            "The FTC filed a complaint against Meta Platforms Inc. on 15 January 2024 "
            "for violations of Article 5 GDPR and Article 25 UK GDPR, imposing a "
            "monetary penalty of £12.5 million under the Data Protection Act 2018."
        )

    tagger = NERTagger()
    result = tagger.tag(text)

    print(f"Jurisdiction:    {result.jurisdiction} (confidence={result.jurisdiction_confidence})")
    print(f"Regulation type: {result.regulation_type}")
    print(f"Effective date:  {result.effective_date}")
    print(f"Dates found:     {result.dates_found}")
    print(f"Article refs:    {result.article_citations}")
    print(f"Fines (£/$/€):   {result.fine_amounts}")
    print(f"Named entities:  {result.named_entities[:5]}")