"""
pdf_parser.py — Structure-aware PDF extraction for GARIP ETL layer.

Preserves hierarchical document structure (Articles → Clauses → Subsections)
and attaches per-chunk metadata required for jurisdiction-aware RAG retrieval.

Supports both pdfplumber (layout-aware) and pymupdf (fast fallback).
"""

import re
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

logger = logging.getLogger(__name__)

# Regex patterns for regulatory document structure
ARTICLE_PATTERN = re.compile(
    r"^(Article|ARTICLE|Art\.?)\s+(\d+[a-z]?)\b", re.MULTILINE
)
SECTION_PATTERN = re.compile(
    r"^(Section|SECTION|§)\s+(\d+[\.\d]*)\b", re.MULTILINE
)
CHAPTER_PATTERN = re.compile(
    r"^(Chapter|CHAPTER)\s+([IVXLCDM]+|\d+)\b", re.MULTILINE
)
RECITAL_PATTERN = re.compile(r"^\((\d+)\)\s+", re.MULTILINE)


@dataclass
class DocumentBlock:
    """A single structural unit extracted from a regulatory PDF."""
    text: str
    page_number: int
    block_type: str          # "article", "section", "chapter", "recital", "body"
    block_id: str            # e.g. "article_5", "section_3_2"
    article_ref: str | None = None
    section_ref: str | None = None
    chapter_ref: str | None = None
    char_start: int = 0
    char_end: int = 0
    metadata: dict = field(default_factory=dict)


@dataclass
class ParsedDocument:
    """Full parsed output from a single PDF."""
    source_path: str
    total_pages: int
    total_blocks: int
    blocks: list[DocumentBlock]
    extraction_method: str   # "pdfplumber" | "pymupdf" | "fallback"
    raw_text: str            # Full concatenated text
    parse_errors: list[str] = field(default_factory=list)


class PDFParser:
    """
    Structure-aware PDF parser for regulatory documents.

    Strategy:
      1. Attempt pdfplumber (best layout preservation)
      2. Fall back to pymupdf (faster, less layout-aware)
      3. Final fallback: raw byte extraction

    Post-extraction, applies regex-based structural tagging to
    identify Articles, Sections, Chapters, and Recitals — preserving
    the hierarchy needed for metadata-rich chunking downstream.
    """

    def __init__(self, prefer_pymupdf: bool = False):
        self.prefer_pymupdf = prefer_pymupdf
        self._check_dependencies()

    def _check_dependencies(self):
        self._has_pdfplumber = False
        self._has_pymupdf = False
        try:
            import pdfplumber  # noqa: F401
            self._has_pdfplumber = True
        except ImportError:
            logger.warning("pdfplumber not installed — falling back to pymupdf")
        try:
            import fitz  # noqa: F401
            self._has_pymupdf = True
        except ImportError:
            logger.warning("pymupdf not installed")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, pdf_path: str | Path) -> ParsedDocument:
        """
        Parse a PDF and return a structured ParsedDocument.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            ParsedDocument with per-block structural metadata.
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Parsing PDF: {pdf_path.name}")

        # Try extraction methods in priority order
        pages_text, method, errors = self._extract_text(pdf_path)

        if not pages_text:
            return ParsedDocument(
                source_path=str(pdf_path),
                total_pages=0,
                total_blocks=0,
                blocks=[],
                extraction_method=method,
                raw_text="",
                parse_errors=errors + ["No text extracted"],
            )

        raw_text = "\n".join(pages_text)
        blocks = list(self._tag_structure(pages_text))

        logger.info(
            f"Parsed {pdf_path.name}: {len(pages_text)} pages, "
            f"{len(blocks)} blocks via {method}"
        )

        return ParsedDocument(
            source_path=str(pdf_path),
            total_pages=len(pages_text),
            total_blocks=len(blocks),
            blocks=blocks,
            extraction_method=method,
            raw_text=raw_text,
            parse_errors=errors,
        )

    def parse_url(self, url: str, session=None) -> ParsedDocument:
        """
        Download a PDF from URL and parse it.
        Useful for ICO and EUR-Lex direct PDF links.
        """
        import io
        import requests
        import tempfile

        logger.info(f"Downloading PDF: {url}")
        s = session or requests.Session()
        resp = s.get(url, timeout=30)
        resp.raise_for_status()

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(resp.content)
            tmp_path = Path(tmp.name)

        try:
            return self.parse(tmp_path)
        finally:
            tmp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Extraction methods
    # ------------------------------------------------------------------

    def _extract_text(
        self, pdf_path: Path
    ) -> tuple[list[str], str, list[str]]:
        """
        Try extractors in order, return (pages_text, method, errors).
        pages_text: list of strings, one per page.
        """
        errors: list[str] = []

        if not self.prefer_pymupdf and self._has_pdfplumber:
            try:
                pages = self._extract_pdfplumber(pdf_path)
                return pages, "pdfplumber", errors
            except Exception as exc:
                errors.append(f"pdfplumber: {exc}")
                logger.warning(f"pdfplumber failed: {exc}")

        if self._has_pymupdf:
            try:
                pages = self._extract_pymupdf(pdf_path)
                return pages, "pymupdf", errors
            except Exception as exc:
                errors.append(f"pymupdf: {exc}")
                logger.warning(f"pymupdf failed: {exc}")

        # Last resort: try pypdf
        try:
            pages = self._extract_pypdf(pdf_path)
            return pages, "fallback", errors
        except Exception as exc:
            errors.append(f"fallback: {exc}")

        return [], "none", errors

    def _extract_pdfplumber(self, pdf_path: Path) -> list[str]:
        import pdfplumber
        pages = []
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text = page.extract_text(x_tolerance=2, y_tolerance=2) or ""
                pages.append(text)
        return pages

    def _extract_pymupdf(self, pdf_path: Path) -> list[str]:
        import fitz
        doc = fitz.open(str(pdf_path))
        pages = []
        for page in doc:
            text = page.get_text("text")
            pages.append(text)
        doc.close()
        return pages

    def _extract_pypdf(self, pdf_path: Path) -> list[str]:
        try:
            from pypdf import PdfReader
        except ImportError:
            from PyPDF2 import PdfReader  # older alias
        reader = PdfReader(str(pdf_path))
        return [
            page.extract_text() or "" for page in reader.pages
        ]

    # ------------------------------------------------------------------
    # Structural tagging
    # ------------------------------------------------------------------

    def _tag_structure(
        self, pages_text: list[str]
    ) -> Iterator[DocumentBlock]:
        """
        Walk through page text and yield DocumentBlocks with structural tags.

        Tracks current Article / Section / Chapter context so every block
        carries its full structural lineage as metadata.
        """
        current_chapter: str | None = None
        current_article: str | None = None
        current_section: str | None = None
        block_idx = 0

        for page_num, page_text in enumerate(pages_text, start=1):
            if not page_text.strip():
                continue

            # Split page into paragraphs
            paragraphs = [p.strip() for p in page_text.split("\n\n") if p.strip()]

            for para in paragraphs:
                block_type, ref = self._classify_paragraph(para)

                # Update structural context
                if block_type == "chapter":
                    current_chapter = ref
                    current_article = None
                    current_section = None
                elif block_type == "article":
                    current_article = ref
                    current_section = None
                elif block_type == "section":
                    current_section = ref

                block_id = self._make_block_id(
                    block_type, ref, block_idx
                )

                yield DocumentBlock(
                    text=para,
                    page_number=page_num,
                    block_type=block_type,
                    block_id=block_id,
                    article_ref=current_article,
                    section_ref=current_section,
                    chapter_ref=current_chapter,
                    metadata={
                        "page": page_num,
                        "article": current_article,
                        "section": current_section,
                        "chapter": current_chapter,
                    },
                )
                block_idx += 1

    def _classify_paragraph(self, text: str) -> tuple[str, str | None]:
        """Classify a paragraph as article/section/chapter/recital/body."""
        first_line = text.split("\n")[0].strip()

        m = CHAPTER_PATTERN.match(first_line)
        if m:
            return "chapter", f"chapter_{m.group(2)}"

        m = ARTICLE_PATTERN.match(first_line)
        if m:
            return "article", f"article_{m.group(2)}"

        m = SECTION_PATTERN.match(first_line)
        if m:
            return "section", f"section_{m.group(2).replace('.', '_')}"

        m = RECITAL_PATTERN.match(text)
        if m:
            return "recital", f"recital_{m.group(1)}"

        return "body", None

    @staticmethod
    def _make_block_id(block_type: str, ref: str | None, idx: int) -> str:
        if ref:
            return ref
        return f"{block_type}_{idx}"


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Parse a regulatory PDF")
    parser.add_argument("pdf", help="Path to PDF file")
    parser.add_argument("--json", action="store_true", help="Output JSON")
    args = parser.parse_args()

    p = PDFParser()
    doc = p.parse(args.pdf)

    if args.json:
        output = {
            "source_path": doc.source_path,
            "total_pages": doc.total_pages,
            "total_blocks": doc.total_blocks,
            "extraction_method": doc.extraction_method,
            "blocks": [
                {
                    "block_id": b.block_id,
                    "block_type": b.block_type,
                    "page_number": b.page_number,
                    "article_ref": b.article_ref,
                    "text_preview": b.text[:120],
                }
                for b in doc.blocks
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"Pages: {doc.total_pages} | Blocks: {doc.total_blocks} | Method: {doc.extraction_method}")
        for b in doc.blocks[:5]:
            print(f"  [{b.block_type:8s}] {b.block_id:20s} p.{b.page_number} — {b.text[:80]!r}")