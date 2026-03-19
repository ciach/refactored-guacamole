from __future__ import annotations

from collections.abc import Sequence

from rag_agent.models import GraphFact
from rag_agent.retrieval.graph_store import InMemoryGraphStore


class Neo4jGraphStore(InMemoryGraphStore):
    """Placeholder Neo4j-backed store with an in-memory fallback implementation."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        super().__init__()
        self.uri = uri
        self.user = user
        self.password = password

    async def bootstrap(self) -> None:
        return None

    async def seed(self, facts: Sequence[GraphFact]) -> None:
        await self.upsert(list(facts))
