from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rag_agent.models import SourceChunk


class VectorStore(Protocol):
    async def upsert(self, chunks: list[SourceChunk]) -> None: ...

    async def search(self, query: str, k: int = 8) -> list[SourceChunk]: ...


@dataclass
class InMemoryVectorStore:
    chunks: list[SourceChunk] = field(default_factory=list)

    async def upsert(self, chunks: list[SourceChunk]) -> None:
        existing = {chunk.chunk_id: chunk for chunk in self.chunks}
        for chunk in chunks:
            existing[chunk.chunk_id] = chunk
        self.chunks = list(existing.values())

    async def search(self, query: str, k: int = 8) -> list[SourceChunk]:
        terms = {term.lower() for term in query.split() if term}
        scored: list[SourceChunk] = []
        for chunk in self.chunks:
            haystack = f"{chunk.title} {chunk.text}".lower()
            overlap = sum(1 for term in terms if term in haystack)
            if overlap == 0:
                continue
            scored.append(chunk.model_copy(update={"score": float(overlap)}))
        scored.sort(key=lambda chunk: chunk.score, reverse=True)
        return scored[:k]
