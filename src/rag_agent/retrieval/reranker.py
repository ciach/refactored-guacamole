from __future__ import annotations

from rag_agent.models import GraphFact, SourceChunk


def rerank(vector_hits: list[SourceChunk], graph_hits: list[GraphFact]) -> tuple[list[SourceChunk], list[GraphFact]]:
    return sorted(vector_hits, key=lambda hit: hit.score, reverse=True), sorted(
        graph_hits, key=lambda hit: hit.score, reverse=True
    )
