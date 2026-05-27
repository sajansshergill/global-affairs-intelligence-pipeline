"""
hybrid_retriever.py — Hybrid BM25 + dense retrieval with cross-encoder reranking.

Three-stage pipeline:
  Stage 1a: Dense retrieval   — semantic similarity via vector store
  Stage 1b: BM25 retrieval    — exact keyword match (catches legal terms)
  Stage 2:  RRF merge         — Reciprocal Rank Fusion of both result sets
  Stage 3:  Cross-encoder     — reranks merged candidates for final top-k

Why hybrid matters for regulatory text:
  Exact legal terms like "Article 13 DSA", "Section 230", "CELEX 32024R0001"
  must be retrievable even when semantically distant from the query.
  BM25 catches these; dense retrieval catches paraphrases. RRF combines both.

GCP equivalent: Vertex AI Vector Search + BigQuery Full-Text Search
"""

import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)


class HybridRetriever:
    """
    Retrieves relevant regulatory chunks using hybrid BM25 + dense search.

    Architecture:
        Query
          ├─► Dense retrieval (vector store)   top_k × 3
          └─► BM25 retrieval  (in-memory)      top_k × 3
                    │
                    ▼
            Reciprocal Rank Fusion
                    │
                    ▼
            Cross-Encoder Reranker
                    │
                    ▼
              Final top_k results

    Usage:
        retriever = HybridRetriever(vector_loader=loader)
        retriever.build_bm25_index(chunks)
        results = retriever.retrieve("GDPR right to erasure", top_k=5)
    """

    def __init__(
        self,
        vector_loader,
        chunks: list | None = None,
        top_k: int = 5,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
        rrf_k: int = 60,
        use_reranker: bool = True,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.vector_loader = vector_loader
        self.top_k = top_k
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight
        self.rrf_k = rrf_k
        self.use_reranker = use_reranker

        self._bm25 = None
        self._bm25_chunks: list[dict] = []
        self._reranker = None

        if chunks:
            self.build_bm25_index(chunks)
        if use_reranker:
            self._init_reranker(reranker_model)

    # ------------------------------------------------------------------
    # BM25 index
    # ------------------------------------------------------------------

    def build_bm25_index(self, chunks: list) -> None:
        """
        Build in-memory BM25 index from a list of Chunk objects or dicts.
        Call after ingestion or on app startup.
        """
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank-bm25 not installed — sparse retrieval disabled. "
                           "Run: pip install rank-bm25")
            return

        self._bm25_chunks = [
            {
                "chunk_id": getattr(c, "chunk_id", c.get("chunk_id", "")),
                "text":     getattr(c, "text",     c.get("text", "")),
                "metadata": (
                    c.to_metadata_dict()
                    if hasattr(c, "to_metadata_dict")
                    else c.get("metadata", {})
                ),
            }
            for c in chunks
        ]

        tokenized = [self._tokenize(c["text"]) for c in self._bm25_chunks]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(f"BM25 index built: {len(self._bm25_chunks)} documents")

    def build_bm25_from_duckdb(
        self, duckdb_loader, jurisdiction: str | None = None
    ) -> None:
        """Load chunks from DuckDB and build BM25 index."""
        sql = "SELECT chunk_id, text, jurisdiction, regulation_type, article_ref FROM chunks"
        params = None
        if jurisdiction:
            sql += " WHERE jurisdiction = ?"
            params = [jurisdiction]
        df = duckdb_loader.query(sql, params)
        self.build_bm25_index(df.to_dict("records"))

    # ------------------------------------------------------------------
    # Reranker
    # ------------------------------------------------------------------

    def _init_reranker(self, model_name: str) -> None:
        try:
            from sentence_transformers import CrossEncoder
            self._reranker = CrossEncoder(model_name)
            logger.info(f"Cross-encoder loaded: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not installed — reranking disabled. "
                           "Run: pip install sentence-transformers")
            self.use_reranker = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> list[dict]:
        """
        Full hybrid retrieval pipeline: dense + BM25 → RRF → rerank.

        Args:
            query:           Natural language query string.
            top_k:           Number of final results (overrides init value).
            filter_metadata: Metadata filter for dense retrieval
                             e.g. {"jurisdiction": "EU"}

        Returns:
            List of result dicts sorted by relevance:
            [{chunk_id, text, score, metadata, retrieval_method}]
        """
        k = top_k or self.top_k
        candidate_k = k * 3

        dense_results  = self._dense_retrieve(query, candidate_k, filter_metadata)
        sparse_results = self._sparse_retrieve(query, candidate_k)
        fused          = self._reciprocal_rank_fusion(dense_results, sparse_results)

        if self.use_reranker and self._reranker and fused:
            fused = self._rerank(query, fused, top_k=k)
        else:
            fused = fused[:k]

        logger.info(
            f"Retrieval: '{query[:50]}' → "
            f"dense={len(dense_results)} sparse={len(sparse_results)} "
            f"final={len(fused)}"
        )
        return fused

    def retrieve_with_jurisdiction_filter(
        self, query: str, jurisdiction: str, top_k: int | None = None
    ) -> list[dict]:
        """Convenience wrapper that hard-filters by jurisdiction."""
        return self.retrieve(
            query=query,
            top_k=top_k,
            filter_metadata={"jurisdiction": jurisdiction},
        )

    # ------------------------------------------------------------------
    # Stage 1a: Dense retrieval
    # ------------------------------------------------------------------

    def _dense_retrieve(
        self,
        query: str,
        top_k: int,
        filter_metadata: dict | None,
    ) -> list[dict]:
        try:
            results = self.vector_loader.query(
                query_text=query,
                top_k=top_k,
                filter_metadata=filter_metadata,
            )
            for r in results:
                r["retrieval_method"] = "dense"
            return results
        except Exception as exc:
            logger.error(f"Dense retrieval failed: {exc}")
            return []

    # ------------------------------------------------------------------
    # Stage 1b: BM25 sparse retrieval
    # ------------------------------------------------------------------

    def _sparse_retrieve(self, query: str, top_k: int) -> list[dict]:
        if not self._bm25 or not self._bm25_chunks:
            return []

        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]

        results = []
        for idx, score in indexed:
            if score <= 0:
                continue
            chunk = self._bm25_chunks[idx]
            results.append({
                "chunk_id":         chunk["chunk_id"],
                "text":             chunk["text"],
                "score":            float(score),
                "metadata":         chunk.get("metadata", {}),
                "retrieval_method": "bm25",
            })
        return results

    # ------------------------------------------------------------------
    # Stage 2: Reciprocal Rank Fusion
    # ------------------------------------------------------------------

    def _reciprocal_rank_fusion(
        self,
        dense_results:  list[dict],
        sparse_results: list[dict],
    ) -> list[dict]:
        """
        RRF score = Σ weight_i / (k + rank_i)
        k=60 dampens the sensitivity to exact rank position.
        """
        rrf_scores: dict[str, float] = defaultdict(float)
        chunk_data: dict[str, dict]  = {}

        for rank, result in enumerate(dense_results):
            cid = result["chunk_id"]
            rrf_scores[cid] += self.dense_weight / (self.rrf_k + rank + 1)
            if cid not in chunk_data:
                chunk_data[cid] = result

        for rank, result in enumerate(sparse_results):
            cid = result["chunk_id"]
            rrf_scores[cid] += self.sparse_weight / (self.rrf_k + rank + 1)
            if cid not in chunk_data:
                chunk_data[cid] = result

        sorted_ids = sorted(rrf_scores, key=lambda c: rrf_scores[c], reverse=True)

        fused = []
        for cid in sorted_ids:
            item = dict(chunk_data[cid])
            item["rrf_score"] = round(rrf_scores[cid], 6)
            item["score"]     = item["rrf_score"]
            fused.append(item)
        return fused

    # ------------------------------------------------------------------
    # Stage 3: Cross-encoder reranking
    # ------------------------------------------------------------------

    def _rerank(self, query: str, candidates: list[dict], top_k: int) -> list[dict]:
        """Score (query, passage) pairs and re-sort by rerank score."""
        pairs = [(query, c["text"]) for c in candidates]
        try:
            scores = self._reranker.predict(pairs)
            for candidate, score in zip(candidates, scores):
                candidate["rerank_score"] = float(score)
                candidate["score"]        = float(score)
            candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        except Exception as exc:
            logger.warning(f"Reranking failed: {exc} — using RRF scores")
        return candidates[:top_k]

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase whitespace tokenizer — preserves legal term structure."""
        return re.findall(r"\b\w+\b", text.lower())