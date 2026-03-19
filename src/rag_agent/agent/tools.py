from __future__ import annotations

from rag_agent.compat import RunContext

from rag_agent.agent.deps import AgentDeps
from rag_agent.models import GraphFact, SourceChunk
from rag_agent.retrieval.hybrid import hybrid_search


async def vector_search_tool(
    ctx: RunContext[AgentDeps], query: str, k: int | None = None
) -> list[SourceChunk]:
    limit = k or ctx.deps.settings.top_k_vector
    return await ctx.deps.vector_store.search(query=query, k=limit)


async def graph_search_tool(
    ctx: RunContext[AgentDeps], query: str, k: int | None = None
) -> list[GraphFact]:
    limit = k or ctx.deps.settings.top_k_graph
    return await ctx.deps.graph_store.search(query=query, k=limit)


async def hybrid_search_tool(
    ctx: RunContext[AgentDeps], query: str, k: int | None = None
) -> dict[str, list[dict[str, object]]]:
    limit = k or ctx.deps.settings.top_k_hybrid
    result = await hybrid_search(
        query=query,
        vector_store=ctx.deps.vector_store,
        graph_store=ctx.deps.graph_store,
        k=limit,
    )
    return {
        "vector_hits": [item.model_dump() for item in result["vector_hits"]],
        "graph_hits": [item.model_dump() for item in result["graph_hits"]],
    }
