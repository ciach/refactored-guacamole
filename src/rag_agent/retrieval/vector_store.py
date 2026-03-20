from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Protocol

from rag_agent.ingest.embedder import embed_text
from rag_agent.models import SourceChunk


class VectorStore(Protocol):
    async def search(self, query: str, k: int) -> list[SourceChunk]: ...


@dataclass
class InMemoryVectorStore:
    seed_chunks: list[SourceChunk] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._chunks = list(self.seed_chunks)
        self._embeddings = {chunk.chunk_id: embed_text(chunk.text) for chunk in self._chunks}

    async def search(self, query: str, k: int) -> list[SourceChunk]:
        query_embedding = embed_text(query)
        scored: list[SourceChunk] = []
        for chunk in self._chunks:
            score = _cosine_similarity(query_embedding, self._embeddings[chunk.chunk_id])
            scored.append(
                SourceChunk(
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    title=chunk.title,
                    text=chunk.text,
                    source_path=chunk.source_path,
                    score=score,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    dot = sum(l * r for l, r in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left)) or 1.0
    right_norm = math.sqrt(sum(value * value for value in right)) or 1.0
    return dot / (left_norm * right_norm)
