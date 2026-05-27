"""
test_retrieval.py — Tests for retrieval layer.

Covers:
  - QueryRewriter (fast mode): jurisdiction detection, field presence,
    legal term extraction, temporal detection, retrieval query list
  - HybridRetriever: BM25 index build, dense retrieval call,
    RRF deduplication and scoring, top-k limit
  - ConflictDetector: known patterns structure, signal ID uniqueness,
    detect_all return type

Run: pytest tests/test_retrieval.py -v
"""

import pytest
from unittest.mock import MagicMock


# ------------------------------------------------------------------
# QueryRewriter tests
# ------------------------------------------------------------------

class TestQueryRewriter:

    def test_eu_jurisdiction_detected(self):
        from retrieval.query_rewriter import QueryRewriter
        r      = QueryRewriter(mode="fast")
        result = r.rewrite("What are the GDPR enforcement requirements?")
        assert result["detected_jurisdiction"] == "EU"

    def test_us_jurisdiction_detected(self):
        from retrieval.query_rewriter import QueryRewriter
        r      = QueryRewriter(mode="fast")
        result = r.rewrite("What FTC actions targeted data brokers?")
        assert result["detected_jurisdiction"] == "US"

    def test_uk_jurisdiction_detected(self):
        from retrieval.query_rewriter import QueryRewriter
        r      = QueryRewriter(mode="fast")
        result = r.rewrite("What ICO penalties were issued in 2023?")
        assert result["detected_jurisdiction"] == "UK"

    def test_required_fields_present(self):
        from retrieval.query_rewriter import QueryRewriter
        r      = QueryRewriter(mode="fast")
        result = r.rewrite("What is the AI Act?")
        for field in [
            "original_query", "expanded_queries", "detected_jurisdiction",
            "detected_regulation_type", "temporal_constraint",
            "key_legal_terms", "rewrite_method",
        ]:
            assert field in result, f"Missing field: {field}"

    def test_legal_term_article_extracted(self):
        from retrieval.query_rewriter import QueryRewriter
        r     = QueryRewriter(mode="fast")
        terms = r._extract_legal_terms("Does Article 5 GDPR apply here?")
        assert any("Article 5" in t for t in terms)

    def test_temporal_recent_detected(self):
        from retrieval.query_rewriter import QueryRewriter
        assert QueryRewriter._detect_temporal("What are the latest GDPR updates?") == "recent"

    def test_temporal_year_detected(self):
        from retrieval.query_rewriter import QueryRewriter
        assert QueryRewriter._detect_temporal("What happened in 2023?") == "2023"

    def test_get_retrieval_queries_includes_original(self):
        from retrieval.query_rewriter import QueryRewriter
        r       = QueryRewriter(mode="fast")
        query   = "GDPR data breach notification rules"
        queries = r.get_retrieval_queries(query)
        assert isinstance(queries, list)
        assert queries[0] == query

    def test_fast_mode_rewrite_method_label(self):
        from retrieval.query_rewriter import QueryRewriter
        r      = QueryRewriter(mode="fast")
        result = r.rewrite("Any query here")
        assert result["rewrite_method"] == "fast"

    def test_no_api_key_falls_back_to_fast(self):
        import os
        from retrieval.query_rewriter import QueryRewriter
        original = os.environ.pop("ANTHROPIC_API_KEY", None)
        r = QueryRewriter(mode="full")
        assert r.mode == "fast"
        if original:
            os.environ["ANTHROPIC_API_KEY"] = original


# ------------------------------------------------------------------
# HybridRetriever tests
# ------------------------------------------------------------------

def make_mock_chunks(n: int = 20) -> list[dict]:
    return [
        {
            "chunk_id": f"chunk_{i:03d}",
            "text":     f"Regulatory text {i} about GDPR data privacy compliance obligations.",
            "metadata": {"jurisdiction": "EU"},
        }
        for i in range(n)
    ]


def make_mock_vector_loader(results=None):
    mock = MagicMock()
    mock.query.return_value = results or [
        {
            "chunk_id":         "chunk_001",
            "text":             "GDPR Article 5 requires lawful processing.",
            "score":            0.92,
            "metadata":         {"jurisdiction": "EU"},
            "retrieval_method": "dense",
        }
    ]
    return mock


class TestHybridRetriever:

    def test_bm25_index_built(self):
        from retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(
            vector_loader=make_mock_vector_loader(),
            chunks=make_mock_chunks(10),
            use_reranker=False,
        )
        assert retriever._bm25 is not None
        assert len(retriever._bm25_chunks) == 10

    def test_dense_retrieval_called(self):
        from retrieval.hybrid_retriever import HybridRetriever
        mock_loader = make_mock_vector_loader()
        retriever   = HybridRetriever(
            vector_loader=mock_loader,
            use_reranker=False,
            top_k=3,
        )
        retriever.retrieve("GDPR data processing")
        mock_loader.query.assert_called()

    def test_sparse_retrieve_returns_results(self):
        from retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(
            vector_loader=make_mock_vector_loader(),
            chunks=make_mock_chunks(20),
            use_reranker=False,
        )
        results = retriever._sparse_retrieve("GDPR privacy compliance", top_k=5)
        assert isinstance(results, list)

    def test_rrf_deduplicates(self):
        from retrieval.hybrid_retriever import HybridRetriever
        retriever = HybridRetriever(
            vector_loader=make_mock_vector_loader(),
            use_reranker=False,
        )
        dense  = [
            {"chunk_id": "a", "text": "text a", "score": 0.9, "metadata": {}, "retrieval_method": "dense"},
            {"chunk_id": "b", "text": "text b", "score": 0.8, "metadata": {}, "retrieval_method": "dense"},
        ]
        sparse = [
            {"chunk_id": "b", "text": "text b", "score": 5.0, "metadata": {}, "retrieval_method": "bm25"},
            {"chunk_id": "c", "text": "text c", "score": 4.0, "metadata": {}, "retrieval_method": "bm25"},
        ]
        fused = retriever._reciprocal_rank_fusion(dense, sparse)
        ids   = [f["chunk_id"] for f in fused]
        assert len(ids) == len(set(ids))

    def test_retrieve_respects_top_k(self):
        from retrieval.hybrid_retriever import HybridRetriever
        mock_results = [
            {"chunk_id": f"c{i}", "text": f"text {i}", "score": 1 - i * 0.05, "metadata": {}}
            for i in range(10)
        ]
        retriever = HybridRetriever(
            vector_loader=make_mock_vector_loader(mock_results),
            use_reranker=False,
            top_k=3,
        )
        results = retriever.retrieve("test query")
        assert len(results) <= 3

    def test_tokenize_lowercases(self):
        from retrieval.hybrid_retriever import HybridRetriever
        tokens = HybridRetriever._tokenize("Article 5 GDPR requires lawful processing.")
        assert "article" in tokens
        assert "gdpr"    in tokens
        assert "."       not in tokens


# ------------------------------------------------------------------
# ConflictDetector tests
# ------------------------------------------------------------------

class TestConflictDetector:

    def test_detect_all_returns_list(self):
        from retrieval.conflict_detector import ConflictDetector
        detector = ConflictDetector(duckdb_loader=None, mode="sql")
        signals  = detector.detect_all()
        assert isinstance(signals, list)

    def test_signal_ids_unique(self):
        from retrieval.conflict_detector import ConflictDetector
        detector = ConflictDetector(duckdb_loader=None, mode="sql")
        signals  = detector.detect_all()
        ids      = [s.signal_id for s in signals]
        assert len(ids) == len(set(ids))

    def test_known_patterns_have_required_fields(self):
        from retrieval.conflict_detector import KNOWN_PATTERNS
        for p in KNOWN_PATTERNS:
            assert "topic"          in p
            assert "jurisdiction_a" in p
            assert "jurisdiction_b" in p
            assert "severity"       in p
            assert "summary"        in p

    def test_sql_patterns_without_db(self):
        from retrieval.conflict_detector import ConflictDetector
        detector = ConflictDetector(duckdb_loader=None, mode="sql")
        signals  = detector._detect_sql_patterns()
        # Without DB, known_pattern is used as reg_id placeholder — signals still created
        assert isinstance(signals, list)