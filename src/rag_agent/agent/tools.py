from __future__ import annotations

from typing import Any

from rag_agent.agent.deps import AgentDeps
from rag_agent.models import GraphFact, SourceChunk

try:  # pragma: no cover - exercised when PydanticAI is installed.
    from pydantic_ai import RunContext
except Exception:  # pragma: no cover
    RunContext = Any


async def vector_search_tool(ctx: RunContext, query: str, k: int = 8) -> list[SourceChunk]:
    deps: AgentDeps = ctx.deps if hasattr(ctx, "deps") else ctx
    return await deps.vector_store.search(query=query, k=k)


async def graph_search_tool(ctx: RunContext, query: str, k: int = 8) -> list[GraphFact]:
    deps: AgentDeps = ctx.deps if hasattr(ctx, "deps") else ctx
    return await deps.graph_store.search(query=query, k=k)


async def hybrid_search_tool(ctx: RunContext, query: str, k: int = 12) -> dict[str, list[dict[str, Any]]]:
    deps: AgentDeps = ctx.deps if hasattr(ctx, "deps") else ctx
    vector_hits = await deps.vector_store.search(query=query, k=max(4, k // 2))
    graph_hits = await deps.graph_store.search(query=query, k=max(4, k // 2))
    return {
        "vector_hits": [hit.model_dump() for hit in vector_hits],
        "graph_hits": [hit.model_dump() for hit in graph_hits],
    }
