"""
test_chunker.py — Tests for extraction/chunker.py.

Covers:
  - Text mode chunking (sliding window)
  - Block mode chunking (structure-aware)
  - Unique chunk IDs
  - total_chunks backfill
  - Metadata propagation
  - Article boundary respect
  - Overlap continuity
  - to_metadata_dict() keys
  - chunk_dataframe() across multiple records
  - Empty text handling

Run: pytest tests/test_chunker.py -v
"""

import pytest
import pandas as pd
from extraction.chunker import Chunker, Chunk


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

BASE_META = {
    "regulation_id":   "test_reg_001",
    "title":           "Test Regulation on AI",
    "jurisdiction":    "EU",
    "regulation_type": "Regulation",
    "effective_date":  "2024-01-01",
    "source_url":      "https://eur-lex.europa.eu/test",
}


def make_block(
    text: str,
    page: int = 1,
    block_type: str = "body",
    article: str | None = None,
):
    """Helper to create a minimal DocumentBlock-like object."""
    from types import SimpleNamespace
    return SimpleNamespace(
        text=text,
        page_number=page,
        block_type=block_type,
        block_id=f"{block_type}_test",
        article_ref=article,
        section_ref=None,
        chapter_ref=None,
    )


# ------------------------------------------------------------------
# Text mode
# ------------------------------------------------------------------

class TestTextModeChunking:

    def test_produces_multiple_chunks(self):
        chunker = Chunker(chunk_tokens=50, overlap_tokens=5)
        text    = " ".join(["word"] * 500)
        chunks  = chunker.chunk_document(raw_text=text, regulation_metadata=BASE_META)
        assert len(chunks) > 1

    def test_chunk_ids_are_unique(self):
        chunker = Chunker(chunk_tokens=50, overlap_tokens=5)
        text    = " ".join([f"word{i}" for i in range(500)])
        chunks  = chunker.chunk_document(raw_text=text, regulation_metadata=BASE_META)
        ids     = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_total_chunks_backfilled(self):
        chunker = Chunker(chunk_tokens=50, overlap_tokens=5)
        text    = " ".join(["word"] * 300)
        chunks  = chunker.chunk_document(raw_text=text, regulation_metadata=BASE_META)
        total   = len(chunks)
        for c in chunks:
            assert c.total_chunks == total

    def test_metadata_propagated_to_every_chunk(self):
        chunker = Chunker()
        text    = "EU regulation on data privacy. " * 50
        chunks  = chunker.chunk_document(raw_text=text, regulation_metadata=BASE_META)
        for chunk in chunks:
            assert chunk.jurisdiction    == "EU"
            assert chunk.regulation_id  == "test_reg_001"
            assert chunk.source_url     == "https://eur-lex.europa.eu/test"

    def test_empty_text_returns_no_chunks(self):
        chunker = Chunker()
        chunks  = chunker.chunk_document(raw_text="", regulation_metadata=BASE_META)
        assert len(chunks) == 0

    def test_overlap_creates_content_continuity(self):
        chunker = Chunker(chunk_tokens=20, overlap_tokens=5)
        text    = " ".join([f"token{i}" for i in range(200)])
        chunks  = chunker.chunk_document(raw_text=text, regulation_metadata=BASE_META)
        if len(chunks) > 1:
            tail_words = set(chunks[0].text.split()[-3:])
            head_words = set(chunks[1].text.split()[:6])
            assert len(tail_words & head_words) > 0


# ------------------------------------------------------------------
# Block mode
# ------------------------------------------------------------------

class TestBlockModeChunking:

    def test_block_mode_produces_chunks(self):
        chunker = Chunker(chunk_tokens=30, overlap_tokens=3)
        blocks  = [
            make_block("Preamble text about the regulation scope.", block_type="body"),
            make_block("Article 1 This article defines key obligations.", block_type="article", article="article_1"),
            make_block("Article 2 This article covers enforcement powers.", block_type="article", article="article_2"),
        ]
        chunks = chunker.chunk_document(raw_text="", regulation_metadata=BASE_META, blocks=blocks)
        assert len(chunks) >= 1

    def test_article_ref_attached_to_block_chunks(self):
        chunker = Chunker()
        blocks  = [
            make_block(
                "Article 5 Right to erasure text. " * 10,
                block_type="article",
                article="article_5",
            )
        ]
        chunks = chunker.chunk_document(raw_text="", regulation_metadata=BASE_META, blocks=blocks)
        for c in chunks:
            assert c.article_ref == "article_5"

    def test_total_chunks_backfilled_in_block_mode(self):
        chunker = Chunker(chunk_tokens=20, overlap_tokens=2)
        blocks  = [make_block(f"Block text number {i}. " * 5) for i in range(10)]
        chunks  = chunker.chunk_document(raw_text="", regulation_metadata=BASE_META, blocks=blocks)
        total   = len(chunks)
        for c in chunks:
            assert c.total_chunks == total


# ------------------------------------------------------------------
# Metadata dict
# ------------------------------------------------------------------

class TestMetadataDict:

    def test_to_metadata_dict_has_required_keys(self):
        chunker = Chunker()
        text    = "Regulatory text for testing metadata. " * 30
        chunks  = chunker.chunk_document(raw_text=text, regulation_metadata=BASE_META)
        meta    = chunks[0].to_metadata_dict()
        required = [
            "chunk_id", "regulation_id", "jurisdiction",
            "regulation_type", "chunk_index", "total_chunks",
            "source_url", "block_type",
        ]
        for key in required:
            assert key in meta, f"Missing key: {key}"

    def test_to_metadata_dict_title_truncated(self):
        chunker   = Chunker()
        long_meta = {**BASE_META, "title": "A" * 300}
        chunks    = chunker.chunk_document(raw_text="Some text. " * 30, regulation_metadata=long_meta)
        meta      = chunks[0].to_metadata_dict()
        assert len(meta["regulation_title"]) <= 200


# ------------------------------------------------------------------
# chunk_dataframe
# ------------------------------------------------------------------

class TestChunkDataframe:

    def test_chunk_dataframe_multiple_records(self):
        chunker = Chunker(chunk_tokens=100, overlap_tokens=10)
        df = pd.DataFrame([
            {**BASE_META, "raw_text": "EU GDPR regulation text. " * 30},
            {**BASE_META, "regulation_id": "reg_002", "raw_text": "US FTC enforcement text. " * 30},
        ])
        chunks  = chunker.chunk_dataframe(df)
        reg_ids = {c.regulation_id for c in chunks}
        assert "test_reg_001" in reg_ids
        assert "reg_002"      in reg_ids

    def test_chunk_dataframe_returns_flat_list(self):
        chunker = Chunker()
        df = pd.DataFrame([
            {**BASE_META, "raw_text": "Text A. " * 50},
            {**BASE_META, "regulation_id": "reg_b", "raw_text": "Text B. " * 50},
        ])
        chunks = chunker.chunk_dataframe(df)
        assert isinstance(chunks, list)
        assert all(isinstance(c, Chunk) for c in chunks)