"""
chunker.py — Metadata-preserving chunker for GARIP ETL layer.

Converts parsed DocumentBlocks into fixed-size chunks suitable for
embedding and vector store ingestion. Each chunk carries its full
structural lineage (jurisdiction, article_ref, regulation_type, etc.)
as metadata — enabling jurisdiction-aware filtered retrieval in RAG.

Strategy:
  - Target 512 tokens per chunk, 10% overlap (~51 tokens)
  - Respect structural boundaries (never split mid-Article if avoidable)
  - Attach regulation-level metadata to every chunk
  - Produce a stable chunk_id (hash of content + source + position)
"""

import hashlib
import logging
from dataclasses import dataclass, field
from typing import Iterator

logger = logging.getLogger(__name__)

# Approximate chars-per-token for English regulatory text
# (more conservative than 4 chars/token to avoid over-chunking)
CHARS_PER_TOKEN = 3.8

DEFAULT_CHUNK_TOKENS = 512
DEFAULT_OVERLAP_TOKENS = 51  # 10% of 512


@dataclass
class Chunk:
    """
    A single embeddable text chunk with full regulatory metadata.

    All metadata fields are stored flat so they can be passed directly
    to Pinecone / ChromaDB as filterable metadata dicts.
    """
    chunk_id: str
    text: str
    token_estimate: int

    # Structural lineage
    source_url: str
    regulation_id: str
    regulation_title: str
    jurisdiction: str
    regulation_type: str
    effective_date: str | None

    # Intra-document position
    chunk_index: int
    total_chunks: int          # filled in after all chunks are collected
    page_number: int | None
    article_ref: str | None
    section_ref: str | None
    chapter_ref: str | None
    block_type: str            # article / section / body / recital

    # Extra metadata (article citations, fines, etc.)
    extra: dict = field(default_factory=dict)

    def to_metadata_dict(self) -> dict:
        """Flat dict suitable for Pinecone / ChromaDB metadata."""
        return {
            "chunk_id": self.chunk_id,
            "source_url": self.source_url,
            "regulation_id": self.regulation_id,
            "regulation_title": self.regulation_title[:200],  # metadata size limit
            "jurisdiction": self.jurisdiction,
            "regulation_type": self.regulation_type,
            "effective_date": self.effective_date or "",
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
            "page_number": self.page_number or 0,
            "article_ref": self.article_ref or "",
            "section_ref": self.section_ref or "",
            "chapter_ref": self.chapter_ref or "",
            "block_type": self.block_type,
            **{k: str(v) for k, v in self.extra.items()},
        }


class Chunker:
    """
    Converts raw text or DocumentBlock lists into Chunk objects.

    Two modes:
      1. block_mode: takes a list[DocumentBlock] from PDFParser — respects
         structural boundaries, preferred for parsed PDFs.
      2. text_mode: takes a plain string — used for connector raw_text
         where no structural parse is available.

    In both modes, overlap is applied by re-including the tail of the
    previous chunk at the start of the next.
    """

    def __init__(
        self,
        chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
        overlap_tokens: int = DEFAULT_OVERLAP_TOKENS,
    ):
        self.chunk_tokens = chunk_tokens
        self.overlap_tokens = overlap_tokens
        self.chunk_chars = int(chunk_tokens * CHARS_PER_TOKEN)
        self.overlap_chars = int(overlap_tokens * CHARS_PER_TOKEN)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(
        self,
        raw_text: str,
        regulation_metadata: dict,
        blocks=None,
    ) -> list[Chunk]:
        """
        Chunk a document into overlapping text windows.

        Args:
            raw_text:            Full document text (fallback if no blocks).
            regulation_metadata: Dict with regulation_id, title, jurisdiction,
                                 regulation_type, effective_date, source_url.
            blocks:              Optional list[DocumentBlock] from PDFParser.
                                 If provided, uses block-aware chunking.

        Returns:
            List of Chunk objects with full metadata.
        """
        if blocks:
            chunks = list(self._chunk_blocks(blocks, regulation_metadata))
        else:
            chunks = list(self._chunk_text(raw_text, regulation_metadata))

        # Backfill total_chunks now that we know the count
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total

        logger.info(
            f"Chunked '{regulation_metadata.get('title', '?')[:50]}': "
            f"{total} chunks from {'blocks' if blocks else 'raw text'}"
        )
        return chunks

    def chunk_dataframe(self, df, blocks_map: dict | None = None) -> list[Chunk]:
        """
        Chunk all records in a DataFrame.

        Args:
            df:         DataFrame with at minimum: regulation_id, raw_text,
                        title, jurisdiction, regulation_type, effective_date,
                        source_url columns.
            blocks_map: Optional dict mapping regulation_id → list[DocumentBlock]
                        for documents that have been PDF-parsed.

        Returns:
            Flat list of Chunk objects across all records.
        """
        all_chunks: list[Chunk] = []
        blocks_map = blocks_map or {}

        for _, row in df.iterrows():
            meta = {
                "regulation_id": row.get("regulation_id", ""),
                "title": row.get("title", "Untitled"),
                "jurisdiction": row.get("jurisdiction", "UNKNOWN"),
                "regulation_type": row.get("regulation_type", ""),
                "effective_date": row.get("effective_date"),
                "source_url": row.get("source_url", ""),
            }
            blocks = blocks_map.get(meta["regulation_id"])
            chunks = self.chunk_document(
                raw_text=row.get("raw_text", ""),
                regulation_metadata=meta,
                blocks=blocks,
            )
            all_chunks.extend(chunks)

        logger.info(f"Total chunks produced: {len(all_chunks)}")
        return all_chunks

    # ------------------------------------------------------------------
    # Block-aware chunking
    # ------------------------------------------------------------------

    def _chunk_blocks(self, blocks, regulation_metadata: dict) -> Iterator[Chunk]:
        """
        Chunk a list of DocumentBlocks, respecting structural boundaries.

        Fills a buffer until it reaches chunk_chars, then yields. On Article
        boundaries, always flushes the buffer first to keep articles intact.
        """
        buffer_text: list[str] = []
        buffer_chars = 0
        buffer_meta: dict = {}
        chunk_index = 0
        prev_tail = ""  # overlap carry-over from previous chunk

        def flush(carry_tail: str = "") -> Iterator[Chunk]:
            nonlocal chunk_index
            if not buffer_text:
                return
            text = carry_tail + " ".join(buffer_text)
            yield self._make_chunk(
                text=text,
                chunk_index=chunk_index,
                regulation_metadata=regulation_metadata,
                block_metadata=buffer_meta,
            )
            chunk_index += 1

        for block in blocks:
            block_text = block.text.strip()
            if not block_text:
                continue

            is_article_boundary = block.block_type in {"article", "chapter"}
            will_overflow = (buffer_chars + len(block_text)) > self.chunk_chars

            if (is_article_boundary or will_overflow) and buffer_text:
                # Capture tail for overlap before flushing
                current_text = " ".join(buffer_text)
                tail = current_text[-self.overlap_chars:] if self.overlap_chars else ""

                yield from flush(prev_tail)
                buffer_text = []
                buffer_chars = 0
                buffer_meta = {}
                prev_tail = tail

            # Update buffer metadata with current block's structural context
            buffer_meta = {
                "page_number": block.page_number,
                "article_ref": block.article_ref,
                "section_ref": block.section_ref,
                "chapter_ref": block.chapter_ref,
                "block_type": block.block_type,
            }
            buffer_text.append(block_text)
            buffer_chars += len(block_text)

        # Flush remaining buffer
        if buffer_text:
            yield from flush(prev_tail)

    # ------------------------------------------------------------------
    # Plain text chunking (sliding window)
    # ------------------------------------------------------------------

    def _chunk_text(self, text: str, regulation_metadata: dict) -> Iterator[Chunk]:
        """
        Sliding window chunker for plain text.
        Tries to split at sentence / paragraph boundaries.
        """
        if not text.strip():
            return

        step = self.chunk_chars - self.overlap_chars
        start = 0
        chunk_index = 0

        while start < len(text):
            end = start + self.chunk_chars

            # Prefer splitting at paragraph then sentence boundary
            if end < len(text):
                para_break = text.rfind("\n\n", start, end)
                sent_break = text.rfind(". ", start, end)

                if para_break > start + step // 2:
                    end = para_break
                elif sent_break > start + step // 2:
                    end = sent_break + 1  # include the period

            chunk_text = text[start:end].strip()
            if chunk_text:
                yield self._make_chunk(
                    text=chunk_text,
                    chunk_index=chunk_index,
                    regulation_metadata=regulation_metadata,
                    block_metadata={"block_type": "body"},
                )
                chunk_index += 1

            start = end - self.overlap_chars
            if start >= len(text):
                break

    # ------------------------------------------------------------------
    # Chunk factory
    # ------------------------------------------------------------------

    def _make_chunk(
        self,
        text: str,
        chunk_index: int,
        regulation_metadata: dict,
        block_metadata: dict,
    ) -> Chunk:
        reg_id = regulation_metadata.get("regulation_id", "")
        chunk_id = self._make_chunk_id(reg_id, chunk_index, text)
        token_estimate = max(1, len(text) // int(CHARS_PER_TOKEN))

        return Chunk(
            chunk_id=chunk_id,
            text=text,
            token_estimate=token_estimate,
            source_url=regulation_metadata.get("source_url", ""),
            regulation_id=reg_id,
            regulation_title=regulation_metadata.get("title", ""),
            jurisdiction=regulation_metadata.get("jurisdiction", "UNKNOWN"),
            regulation_type=regulation_metadata.get("regulation_type", ""),
            effective_date=regulation_metadata.get("effective_date"),
            chunk_index=chunk_index,
            total_chunks=0,  # backfilled after all chunks collected
            page_number=block_metadata.get("page_number"),
            article_ref=block_metadata.get("article_ref"),
            section_ref=block_metadata.get("section_ref"),
            chapter_ref=block_metadata.get("chapter_ref"),
            block_type=block_metadata.get("block_type", "body"),
        )

    @staticmethod
    def _make_chunk_id(regulation_id: str, chunk_index: int, text: str) -> str:
        payload = f"{regulation_id}::{chunk_index}::{text[:64]}"
        return hashlib.sha256(payload.encode()).hexdigest()[:24]


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chunk a text file")
    parser.add_argument("--file", required=True, help="Path to text file")
    parser.add_argument("--chunk-tokens", type=int, default=512)
    parser.add_argument("--overlap-tokens", type=int, default=51)
    args = parser.parse_args()

    with open(args.file) as f:
        text = f.read()

    chunker = Chunker(
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
    )
    meta = {
        "regulation_id": "demo",
        "title": "Demo Document",
        "jurisdiction": "EU",
        "regulation_type": "Regulation",
        "effective_date": None,
        "source_url": "https://example.com",
    }
    chunks = chunker.chunk_document(raw_text=text, regulation_metadata=meta)

    print(f"Produced {len(chunks)} chunks")
    for c in chunks[:3]:
        print(f"\n[{c.chunk_index}] id={c.chunk_id} tokens≈{c.token_estimate}")
        print(f"  article={c.article_ref} section={c.section_ref}")
        print(f"  text: {c.text[:120]!r}...")