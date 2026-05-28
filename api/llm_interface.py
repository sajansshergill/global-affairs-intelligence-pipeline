"""
llm_interface.py — Claude API integration and citation builder for GARIP.

Full RAG orchestration in one class:
  1. Rewrite query        → QueryRewriter detects jurisdiction, expands terms
  2. Retrieve chunks      → HybridRetriever (BM25 + dense + rerank)
  3. HyDE second pass     → retrieve with hypothetical document passage too
  4. Build prompt         → numbered source list with chunk metadata
  5. Call Claude API      → grounded answer with inline [1], [2] citations
  6. Parse response       → Answer / Key Takeaway / Limitations sections
  7. Return QAResponse    → structured object with Citation list attached

GCP equivalent: Vertex AI + Gemini API with Vertex AI Vector Search retrieval
"""

import logging
import os
import re
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY  = os.getenv("ANTHROPIC_API_KEY", "")
LLM_MODEL          = os.getenv("LLM_MODEL", "claude-sonnet-4-20250514")
MAX_CONTEXT_CHUNKS = 6
MAX_TOKENS         = 1024

# ------------------------------------------------------------------
# System prompt
# ------------------------------------------------------------------

QA_SYSTEM_PROMPT = """You are a regulatory intelligence advisor for a Global Affairs team
at a major technology company. You provide accurate, cited answers about global regulatory
and compliance topics based on the provided regulatory document excerpts.

Guidelines:
- Answer ONLY based on the provided context. Do not use general knowledge.
- Every factual claim must be supported by a cited source.
- Use citation markers [1], [2], etc. matching the numbered sources below.
- If context is insufficient, say so explicitly.
- Be precise about jurisdictions — do not conflate EU GDPR with UK GDPR.
- Highlight contradictions between jurisdictions when relevant.
- Keep answers concise and policy-actionable.

Format your response exactly as:
Answer: [your response with inline citations]
Key Takeaway: [one sentence summary]
Limitations: [caveats about what the context doesn't cover]"""


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------

@dataclass
class Citation:
    """A source citation attached to a QA answer."""
    index:           int
    chunk_id:        str
    regulation_title: str
    jurisdiction:    str
    regulation_type: str
    effective_date:  str | None
    article_ref:     str | None
    source_url:      str
    text_excerpt:    str      # first ~150 chars of the chunk


@dataclass
class QAResponse:
    """Full response from LLMInterface including answer + citations."""
    question:             str
    answer:               str
    key_takeaway:         str
    limitations:          str
    citations:            list[Citation]
    retrieved_chunks:     list[dict]
    model:                str
    detected_jurisdiction: str | None
    tokens_used:          int = 0

    def format_citations(self) -> str:
        """Return a formatted reference list string."""
        lines = []
        for c in self.citations:
            date    = f" ({c.effective_date})" if c.effective_date else ""
            article = f" — {c.article_ref}"    if c.article_ref    else ""
            lines.append(
                f"[{c.index}] {c.regulation_title}{article} | "
                f"{c.jurisdiction}{date} | {c.source_url}"
            )
        return "\n".join(lines)


# ------------------------------------------------------------------
# Main interface
# ------------------------------------------------------------------

class LLMInterface:
    """
    Orchestrates the full RAG pipeline for policymaker Q&A.

    Usage:
        llm = LLMInterface(
            hybrid_retriever=retriever,
            query_rewriter=rewriter,
        )
        response = llm.answer("What are GDPR breach notification rules?")
        print(response.answer)
        print(response.format_citations())
    """

    def __init__(
        self,
        hybrid_retriever,
        query_rewriter,
        model:              str = LLM_MODEL,
        max_context_chunks: int = MAX_CONTEXT_CHUNKS,
    ):
        self.retriever          = hybrid_retriever
        self.rewriter           = query_rewriter
        self.model              = model
        self.max_context_chunks = max_context_chunks
        self._client            = self._init_client()

    def _init_client(self):
        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set — LLMInterface in demo mode")
            return None
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            logger.info(f"LLMInterface: Anthropic client ready ({self.model})")
            return client
        except ImportError:
            logger.error("anthropic not installed — run: pip install anthropic")
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def answer(
        self,
        question:            str,
        jurisdiction_filter: str | None = None,
    ) -> QAResponse:
        """
        Answer a policymaker question using hybrid RAG + Claude.

        Args:
            question:            Natural language question.
            jurisdiction_filter: Optional hard filter e.g. "EU", "US", "UK".
                                 Auto-detected from query if None.

        Returns:
            QAResponse with answer, key takeaway, limitations, and citations.
        """
        # Step 1: Query rewrite
        rewrite = self.rewriter.rewrite(question)
        detected = (
            jurisdiction_filter
            or (
                rewrite["detected_jurisdiction"]
                if rewrite["detected_jurisdiction"] != "UNKNOWN"
                else None
            )
        )

        # Step 2: Retrieval
        filter_meta = {"jurisdiction": detected} if detected else None
        chunks      = self.retriever.retrieve(
            query=question,
            top_k=self.max_context_chunks,
            filter_metadata=filter_meta,
        )

        # Step 3: HyDE second-pass retrieval (merge if available)
        if rewrite.get("hyde_passage"):
            hyde_chunks = self.retriever.retrieve(
                query=rewrite["hyde_passage"],
                top_k=self.max_context_chunks // 2,
                filter_metadata=filter_meta,
            )
            chunks = self._merge_chunks(chunks, hyde_chunks, self.max_context_chunks)

        if not chunks:
            return self._empty_response(question, detected)

        # Step 4: Build prompt + citation list
        prompt, citations = self._build_prompt(question, chunks)

        # Step 5: Call LLM
        raw = self._call_llm(prompt)

        # Step 6: Parse response
        answer_text, takeaway, limitations = self._parse_response(raw)

        return QAResponse(
            question=question,
            answer=answer_text,
            key_takeaway=takeaway,
            limitations=limitations,
            citations=citations,
            retrieved_chunks=chunks,
            model=self.model,
            detected_jurisdiction=detected,
        )

    def stream_answer(
        self,
        question:            str,
        jurisdiction_filter: str | None = None,
    ):
        """
        Streaming version of answer() for Streamlit real-time display.
        Yields text tokens as they arrive from the Claude API.
        """
        if not self._client:
            yield "⚠️ API client not configured. Set ANTHROPIC_API_KEY."
            return

        rewrite  = self.rewriter.rewrite(question)
        detected = jurisdiction_filter or (
            rewrite["detected_jurisdiction"]
            if rewrite["detected_jurisdiction"] != "UNKNOWN"
            else None
        )

        filter_meta = {"jurisdiction": detected} if detected else None
        chunks      = self.retriever.retrieve(
            query=question,
            top_k=self.max_context_chunks,
            filter_metadata=filter_meta,
        )

        if not chunks:
            yield "No relevant regulatory documents found for this query."
            return

        prompt, _ = self._build_prompt(question, chunks)

        with self._client.messages.stream(
            model=self.model,
            max_tokens=MAX_TOKENS,
            system=QA_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        ) as stream:
            for text in stream.text_stream:
                yield text

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_prompt(
        self,
        question: str,
        chunks:   list[dict],
    ) -> tuple[str, list[Citation]]:
        """
        Build a numbered-source context prompt and citation list.
        Each source is numbered [1], [2], … matching the LLM's inline citations.
        """
        citations:     list[Citation] = []
        context_lines: list[str]      = []

        for i, chunk in enumerate(chunks, start=1):
            meta  = chunk.get("metadata", {})
            title = meta.get("regulation_title", "Unknown Regulation")
            jur   = meta.get("jurisdiction",     "")
            rtype = meta.get("regulation_type",  "")
            date  = meta.get("effective_date",   "")
            art   = meta.get("article_ref",      "")
            url   = meta.get("source_url",       "")
            cid   = chunk.get("chunk_id",        "")
            text  = chunk.get("text",            "")

            header = f"[{i}] {title}"
            if jur:   header += f" | {jur}"
            if art:   header += f" | {art}"
            if date:  header += f" | {date}"

            context_lines.append(f"{header}\n{text[:800]}")

            citations.append(Citation(
                index=i,
                chunk_id=cid,
                regulation_title=title,
                jurisdiction=jur,
                regulation_type=rtype,
                effective_date=date or None,
                article_ref=art or None,
                source_url=url,
                text_excerpt=text[:150],
            ))

        context_block = "\n\n---\n\n".join(context_lines)

        prompt = (
            f"REGULATORY SOURCES:\n{context_block}\n\n"
            f"---\n\n"
            f"QUESTION: {question}\n\n"
            f"Using ONLY the sources above, provide a cited answer. "
            f"Use [1], [2], etc. to cite sources inline."
        )

        return prompt, citations

    # ------------------------------------------------------------------
    # LLM call
    # ------------------------------------------------------------------

    def _call_llm(self, prompt: str) -> str:
        if not self._client:
            return (
                "Answer: [Demo mode — set ANTHROPIC_API_KEY to enable live answers.]\n"
                "Key Takeaway: Configure environment variables and run the pipeline.\n"
                "Limitations: API client not configured."
            )
        try:
            message = self._client.messages.create(
                model=self.model,
                max_tokens=MAX_TOKENS,
                system=QA_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            return message.content[0].text
        except Exception as exc:
            logger.error(f"LLM call failed: {exc}")
            return (
                f"Answer: Error generating response — {exc}\n"
                f"Key Takeaway: N/A\n"
                f"Limitations: N/A"
            )

    # ------------------------------------------------------------------
    # Response parsing
    # ------------------------------------------------------------------

    def _parse_response(self, raw: str) -> tuple[str, str, str]:
        """Extract Answer / Key Takeaway / Limitations from LLM response."""
        answer      = raw
        takeaway    = ""
        limitations = ""

        ans_match = re.search(
            r"Answer:\s*(.+?)(?:\nKey Takeaway:|$)", raw, re.DOTALL
        )
        kt_match  = re.search(
            r"Key Takeaway:\s*(.+?)(?:\nLimitations:|$)", raw, re.DOTALL
        )
        lim_match = re.search(
            r"Limitations:\s*(.+?)$", raw, re.DOTALL
        )

        if ans_match:  answer      = ans_match.group(1).strip()
        if kt_match:   takeaway    = kt_match.group(1).strip()
        if lim_match:  limitations = lim_match.group(1).strip()

        return answer, takeaway, limitations

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def _merge_chunks(
        self,
        primary:   list[dict],
        secondary: list[dict],
        limit:     int,
    ) -> list[dict]:
        """Merge two chunk lists, deduplicate by chunk_id, keep top limit."""
        seen:   set[str]  = set()
        merged: list[dict] = []
        for chunk in primary + secondary:
            cid = chunk.get("chunk_id", "")
            if cid not in seen:
                seen.add(cid)
                merged.append(chunk)
        return merged[:limit]

    def _empty_response(
        self,
        question:     str,
        jurisdiction: str | None,
    ) -> QAResponse:
        scope = f" for jurisdiction '{jurisdiction}'" if jurisdiction else ""
        return QAResponse(
            question=question,
            answer=(
                f"No relevant regulatory documents found in the corpus{scope} "
                f"for this query."
            ),
            key_takeaway="Expand the query or ingest more regulatory documents.",
            limitations="The corpus may not contain documents relevant to this question.",
            citations=[],
            retrieved_chunks=[],
            model=self.model,
            detected_jurisdiction=jurisdiction,
        )