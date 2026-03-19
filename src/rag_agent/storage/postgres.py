from __future__ import annotations

from collections.abc import Sequence

from rag_agent.models import SourceChunk
from rag_agent.retrieval.vector_store import InMemoryVectorStore


class PostgresVectorStore(InMemoryVectorStore):
    """Placeholder pgvector-backed store.

    The in-memory implementation keeps the developer loop lightweight while preserving the
    interface expected by the agent. Swap the internals for SQLAlchemy + pgvector queries when
    wiring a live database.
    """

    def __init__(self, dsn: str, embedding_model: str, api_key: str) -> None:
        super().__init__()
        self.dsn = dsn
        self.embedding_model = embedding_model
        self.api_key = api_key

    async def bootstrap(self) -> None:
        return None

    async def seed(self, chunks: Sequence[SourceChunk]) -> None:
        await self.upsert(list(chunks))
