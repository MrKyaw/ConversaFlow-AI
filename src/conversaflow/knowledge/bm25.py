"""BM25 retrieval over tenant-scoped knowledge, with citations.

BM25 remains an excellent baseline for FAQ retrieval: no embedding service, no
cold-start, fully deterministic, and the score is interpretable. The retrieval
interface is intentionally narrow so a vector or hybrid retriever can be added
without touching the dialogue layer.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from ..models import KnowledgeDocumentIn
from ..nlu.text import content_tokens

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


@dataclass
class Chunk:
    document_id: str
    chunk_id: str
    title: str
    text: str
    tags: tuple[str, ...]
    source: str | None
    tokens: list[str] = field(default_factory=list)

    @property
    def searchable(self) -> str:
        return f"{self.title} {' '.join(self.tags)} {self.text}"


@dataclass(frozen=True)
class SearchHit:
    document_id: str
    title: str
    text: str
    score: float
    source: str | None
    tags: tuple[str, ...]


class KnowledgeBase:
    def __init__(self, max_chars: int = 600, k1: float = 1.5, b: float = 0.75) -> None:
        self.max_chars = max_chars
        self.k1 = k1
        self.b = b
        self._documents: dict[str, dict[str, KnowledgeDocumentIn]] = {}
        self._chunks: dict[str, list[Chunk]] = {}
        self._idf: dict[str, dict[str, float]] = {}
        self._average_length: dict[str, float] = {}
        self._lock = threading.Lock()

    def ingest(self, tenant_id: str, documents: Iterable[KnowledgeDocumentIn]) -> int:
        with self._lock:
            tenant_docs = self._documents.setdefault(tenant_id, {})
            count = 0
            for document in documents:
                tenant_docs[document.id] = document
                count += 1
            self._reindex(tenant_id)
            return count

    def delete(self, tenant_id: str, document_id: str) -> bool:
        with self._lock:
            tenant_docs = self._documents.get(tenant_id, {})
            removed = tenant_docs.pop(document_id, None) is not None
            if removed:
                self._reindex(tenant_id)
            return removed

    def search(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> list[SearchHit]:
        chunks = self._chunks.get(tenant_id, [])
        tokens = content_tokens(query)
        if not chunks or not tokens:
            return []
        idf = self._idf.get(tenant_id, {})
        average_length = self._average_length.get(tenant_id, 1.0) or 1.0
        query_counts = Counter(tokens)
        scored: list[tuple[float, Chunk]] = []
        for chunk in chunks:
            frequencies = Counter(chunk.tokens)
            length = len(chunk.tokens) or 1
            score = 0.0
            for term, query_frequency in query_counts.items():
                term_frequency = frequencies.get(term, 0)
                if term_frequency == 0:
                    continue
                weight = idf.get(term, 0.0)
                denominator = term_frequency + self.k1 * (
                    1 - self.b + self.b * length / average_length
                )
                score += query_frequency * weight * (term_frequency * (self.k1 + 1) / denominator)
            if score > min_score:
                scored.append((score, chunk))
        scored.sort(key=lambda item: (-item[0], item[1].chunk_id))
        return [
            SearchHit(
                document_id=chunk.document_id,
                title=chunk.title,
                text=chunk.text,
                score=round(score, 4),
                source=chunk.source,
                tags=chunk.tags,
            )
            for score, chunk in scored[:top_k]
        ]

    def count_documents(self, tenant_id: str | None = None) -> int:
        if tenant_id is not None:
            return len(self._documents.get(tenant_id, {}))
        return sum(len(documents) for documents in self._documents.values())

    def count_chunks(self, tenant_id: str | None = None) -> int:
        if tenant_id is not None:
            return len(self._chunks.get(tenant_id, []))
        return sum(len(chunks) for chunks in self._chunks.values())

    def document_ids(self, tenant_id: str) -> list[str]:
        return sorted(self._documents.get(tenant_id, {}))

    def _reindex(self, tenant_id: str) -> None:
        documents = self._documents.get(tenant_id, {})
        chunks: list[Chunk] = []
        for document in documents.values():
            for index, text in enumerate(self._split(document.text)):
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        chunk_id=f"{document.id}#{index}",
                        title=document.title,
                        text=text,
                        tags=tuple(document.tags),
                        source=document.source,
                        tokens=content_tokens(
                            f"{document.title} {document.title} {' '.join(document.tags)} {text}"
                        ),
                    )
                )
        document_frequency: Counter[str] = Counter()
        for chunk in chunks:
            document_frequency.update(set(chunk.tokens))
        total = len(chunks)
        average_length = sum(len(chunk.tokens) for chunk in chunks) / total if total else 0.0
        self._chunks[tenant_id] = chunks
        self._average_length[tenant_id] = average_length
        self._idf[tenant_id] = {
            term: math.log(1 + (total - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def _split(self, text: str) -> list[str]:
        paragraphs = [part.strip() for part in re.split(r"\n\s*\n", text) if part.strip()]
        chunks: list[str] = []
        for paragraph in paragraphs or [text.strip()]:
            if len(paragraph) <= self.max_chars:
                chunks.append(paragraph)
                continue
            current = ""
            for sentence in _SENTENCE_RE.split(paragraph):
                if len(current) + len(sentence) + 1 > self.max_chars and current:
                    chunks.append(current.strip())
                    current = ""
                current += sentence + " "
            if current.strip():
                chunks.append(current.strip())
        return chunks or [text.strip()]
