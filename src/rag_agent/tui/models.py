from __future__ import annotations

from typing import Literal

from rag_agent._compat import BaseModel, Field
from rag_agent.models import Citation, GraphFact, SourceChunk

StrategyMode = Literal["auto", "vector", "graph", "hybrid"]


class QueryConfigSnapshot(BaseModel):
    default_model: str
    fast_model: str
    embedding_model: str
    retrieval_strategy: StrategyMode
    top_k_vector: int
    top_k_graph: int
    top_k_hybrid: int


class SessionOverrides(BaseModel):
    default_model: str | None = None
    fast_model: str | None = None
    retrieval_strategy: StrategyMode = "auto"
    top_k_vector: int | None = None
    top_k_graph: int | None = None
    top_k_hybrid: int | None = None


class QueryRunResult(BaseModel):
    question: str
    answer: str
    reasoning_summary: str
    used_strategy: Literal["vector", "graph", "hybrid"]
    config: QueryConfigSnapshot
    citations: list[Citation] = Field(default_factory=list)
    vector_hits: list[SourceChunk] = Field(default_factory=list)
    graph_hits: list[GraphFact] = Field(default_factory=list)
    trace: list[str] = Field(default_factory=list)
    error: str | None = None
