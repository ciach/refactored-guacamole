from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rag_agent.models import GraphFact


class GraphStore(Protocol):
    async def upsert(self, facts: list[GraphFact]) -> None: ...

    async def search(self, query: str, k: int = 8) -> list[GraphFact]: ...


@dataclass
class InMemoryGraphStore:
    facts: list[GraphFact] = field(default_factory=list)

    async def upsert(self, facts: list[GraphFact]) -> None:
        existing = {fact.fact_id: fact for fact in self.facts}
        for fact in facts:
            existing[fact.fact_id] = fact
        self.facts = list(existing.values())

    async def search(self, query: str, k: int = 8) -> list[GraphFact]:
        terms = {term.lower() for term in query.split() if term}
        scored: list[GraphFact] = []
        for fact in self.facts:
            haystack = " ".join(
                [
                    fact.subject,
                    fact.predicate,
                    fact.obj,
                    fact.evidence or "",
                    fact.timestamp or "",
                ]
            ).lower()
            overlap = sum(1 for term in terms if term in haystack)
            if overlap == 0:
                continue
            scored.append(fact.model_copy(update={"score": float(overlap)}))
        scored.sort(key=lambda fact: fact.score, reverse=True)
        return scored[:k]
