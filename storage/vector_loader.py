"""
vector_loader.py — Loads chunk embeddings into Pinecone or ChromaDB.

Handles batch embedding, upsert, metadata attachment, and query.
Supports two backends selectable via VECTOR_BACKEND env var:
  - "pinecone"  → production (cloud, filterable metadata)
  - "chromadb"  → local development (no API key required)

GCP equivalent: Vertex AI Vector Search (Matching Engine)
"""

import logging
import os
from typing import Literal

logger = logging.getLogger(__name__)

EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX   = os.getenv("PINECONE_INDEX", "garip-regulations")
PINECONE_ENV     = os.getenv("PINECONE_ENV", "us-east-1")
CHROMA_PERSIST   = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
BATCH_SIZE       = 50
EMBED_DIM        = 1536   # text-embedding-3-small


class VectorLoader:
    """
    Embeds chunks and upserts them into a vector store backend.

    Usage:
        loader = VectorLoader(backend="chromadb")
        loader.upsert_chunks(chunks)
        results = loader.query("GDPR right to erasure", top_k=5)
    """

    def __init__(
        self,
        backend: Literal["pinecone", "chromadb"] = "chromadb",
        embedding_model: str = EMBEDDING_MODEL,
    ):
        self.backend = backend
        self.embedding_model = embedding_model
        self._embedder = self._init_embedder()
        self._store    = self._init_store()

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------

    def _init_embedder(self):
        if "text-embedding" in self.embedding_model:
            return _OpenAIEmbedder(self.embedding_model)
        return _SentenceTransformerEmbedder(self.embedding_model)

    def _init_store(self):
        if self.backend == "pinecone":
            return _PineconeStore()
        return _ChromaDBStore()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def upsert_chunks(self, chunks: list, duckdb_loader=None) -> int:
        """
        Embed and upsert a list of Chunk objects.

        Args:
            chunks:        List of Chunk objects from extraction/chunker.py.
            duckdb_loader: Optional DuckDBLoader — marks chunks as embedded
                           in the chunks table after successful upsert.

        Returns:
            Number of chunks successfully upserted.
        """
        if not chunks:
            return 0

        total_upserted = 0
        embedded_ids: list[str] = []

        for i in range(0, len(chunks), BATCH_SIZE):
            batch = chunks[i:i + BATCH_SIZE]
            texts = [c.text for c in batch]

            try:
                embeddings = self._embedder.embed(texts)
            except Exception as exc:
                logger.error(f"Embedding batch {i // BATCH_SIZE} failed: {exc}")
                continue

            vectors = [
                {
                    "id":       chunk.chunk_id,
                    "values":   embedding,
                    "metadata": chunk.to_metadata_dict(),
                    "text":     chunk.text,
                }
                for chunk, embedding in zip(batch, embeddings)
            ]

            try:
                self._store.upsert(vectors)
                total_upserted += len(vectors)
                embedded_ids.extend(c.chunk_id for c in batch)
                logger.info(
                    f"Batch {i // BATCH_SIZE + 1}: upserted {len(vectors)} "
                    f"({total_upserted}/{len(chunks)} total)"
                )
            except Exception as exc:
                logger.error(f"Store upsert failed batch {i // BATCH_SIZE}: {exc}")

        if duckdb_loader and embedded_ids:
            duckdb_loader.mark_chunks_embedded(embedded_ids, vector_store=self.backend)

        logger.info(f"Upsert complete: {total_upserted}/{len(chunks)} chunks")
        return total_upserted

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        filter_metadata: dict | None = None,
    ) -> list[dict]:
        """
        Embed a query and retrieve top-k similar chunks.

        Returns list of dicts: [{chunk_id, text, score, metadata}]
        """
        try:
            embedding = self._embedder.embed([query_text])[0]
        except Exception as exc:
            logger.error(f"Query embedding failed: {exc}")
            return []
        return self._store.query(embedding, top_k=top_k, filter_metadata=filter_metadata)

    def delete_by_regulation(self, regulation_id: str) -> int:
        """Delete all chunks belonging to a regulation_id."""
        return self._store.delete_by_filter({"regulation_id": regulation_id})


# ------------------------------------------------------------------
# Embedder backends
# ------------------------------------------------------------------

class _OpenAIEmbedder:
    def __init__(self, model: str):
        self.model = model
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        except ImportError:
            raise RuntimeError("openai not installed — run: pip install openai")

    def embed(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embeddings.create(model=self.model, input=texts)
        return [item.embedding for item in resp.data]


class _SentenceTransformerEmbedder:
    def __init__(self, model: str):
        try:
            from sentence_transformers import SentenceTransformer
            self.model = SentenceTransformer(model)
        except ImportError:
            raise RuntimeError(
                "sentence-transformers not installed — "
                "run: pip install sentence-transformers"
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=False).tolist()


# ------------------------------------------------------------------
# Vector store backends
# ------------------------------------------------------------------

class _PineconeStore:
    def __init__(self):
        try:
            from pinecone import Pinecone, ServerlessSpec
            pc = Pinecone(api_key=PINECONE_API_KEY)
            existing = [idx.name for idx in pc.list_indexes()]
            if PINECONE_INDEX not in existing:
                pc.create_index(
                    name=PINECONE_INDEX,
                    dimension=EMBED_DIM,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region=PINECONE_ENV),
                )
            self.index = pc.Index(PINECONE_INDEX)
            logger.info(f"Pinecone index ready: {PINECONE_INDEX}")
        except ImportError:
            raise RuntimeError("pinecone-client not installed — run: pip install pinecone-client")

    def upsert(self, vectors: list[dict]) -> None:
        self.index.upsert(
            vectors=[(v["id"], v["values"], v["metadata"]) for v in vectors]
        )

    def query(
        self, embedding: list[float], top_k: int, filter_metadata: dict | None
    ) -> list[dict]:
        kwargs = {"vector": embedding, "top_k": top_k, "include_metadata": True}
        if filter_metadata:
            kwargs["filter"] = filter_metadata
        resp = self.index.query(**kwargs)
        return [
            {
                "chunk_id": m["id"],
                "score":    m["score"],
                "metadata": m.get("metadata", {}),
                "text":     m.get("metadata", {}).get("text", ""),
            }
            for m in resp.get("matches", [])
        ]

    def delete_by_filter(self, filter_metadata: dict) -> int:
        self.index.delete(filter=filter_metadata)
        return -1


class _ChromaDBStore:
    COLLECTION = "garip_regulations"

    def __init__(self):
        try:
            import chromadb
            self.client = chromadb.PersistentClient(path=CHROMA_PERSIST)
            self.collection = self.client.get_or_create_collection(
                name=self.COLLECTION,
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"ChromaDB collection ready: {self.COLLECTION}")
        except ImportError:
            raise RuntimeError("chromadb not installed — run: pip install chromadb")

    def upsert(self, vectors: list[dict]) -> None:
        self.collection.upsert(
            ids=[v["id"] for v in vectors],
            embeddings=[v["values"] for v in vectors],
            documents=[v.get("text", "") for v in vectors],
            metadatas=[v["metadata"] for v in vectors],
        )

    def query(
        self, embedding: list[float], top_k: int, filter_metadata: dict | None
    ) -> list[dict]:
        kwargs = {
            "query_embeddings": [embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if filter_metadata:
            kwargs["where"] = {k: {"$eq": v} for k, v in filter_metadata.items()}
        resp = self.collection.query(**kwargs)
        results = []
        for cid, doc, meta, dist in zip(
            resp["ids"][0],
            resp["documents"][0],
            resp["metadatas"][0],
            resp["distances"][0],
        ):
            results.append({
                "chunk_id": cid,
                "score":    round(1 - dist, 4),
                "metadata": meta,
                "text":     doc,
            })
        return results

    def delete_by_filter(self, filter_metadata: dict) -> int:
        where = {k: {"$eq": v} for k, v in filter_metadata.items()}
        results = self.collection.get(where=where, include=[])
        ids = results.get("ids", [])
        if ids:
            self.collection.delete(ids=ids)
        return len(ids)