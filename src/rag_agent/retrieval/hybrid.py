from __future__ import annotations

from rag_agent.models import GraphFact, SourceChunk
from rag_agent.retrieval.graph_store import GraphStore
from rag_agent.retrieval.vector_store import VectorStore


async def hybrid_search(
    *,
    query: str,
    vector_store: VectorStore,
    graph_store: GraphStore,
    k: int = 12,
) -> dict[str, list[SourceChunk] | list[GraphFact]]:
    vector_k = max(4, k // 2)
    graph_k = max(4, k // 2)
    vector_hits = await vector_store.search(query=query, k=vector_k)
    graph_hits = await graph_store.search(query=query, k=graph_k)
    return {"vector_hits": vector_hits, "graph_hits": graph_hits}
