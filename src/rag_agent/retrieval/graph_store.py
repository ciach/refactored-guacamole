from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from rag_agent.models import GraphFact


class GraphStore(Protocol):
    async def search(self, query: str, k: int) -> list[GraphFact]: ...


@dataclass
class InMemoryGraphStore:
    seed_facts: list[GraphFact] = field(default_factory=list)

    async def search(self, query: str, k: int) -> list[GraphFact]:
        tokens = set(query.lower().split())
        scored: list[GraphFact] = []
        for fact in self.seed_facts:
            haystack = " ".join(
                part for part in [fact.subject, fact.predicate, fact.obj, fact.evidence or ""] if part
            ).lower()
            score = float(sum(token in haystack for token in tokens))
            if score > 0:
                scored.append(
                    GraphFact(
                        fact_id=fact.fact_id,
                        subject=fact.subject,
                        predicate=fact.predicate,
                        obj=fact.obj,
                        evidence=fact.evidence,
                        timestamp=fact.timestamp,
                        score=score,
                    )
                )
        scored.sort(key=lambda item: item.score, reverse=True)
        return scored[:k]
