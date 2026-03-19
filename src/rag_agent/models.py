from __future__ import annotations

from typing import Any, Literal

from rag_agent.compat import BaseModel, Field


class SourceChunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    text: str
    score: float
    source_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class GraphFact(BaseModel):
    fact_id: str
    subject: str
    predicate: str
    obj: str
    evidence: str | None = None
    score: float = 0.0
    timestamp: str | None = None


class Citation(BaseModel):
    source_type: Literal["vector", "graph"]
    ref_id: str
    title: str
    snippet: str


class AgentAnswer(BaseModel):
    answer: str
    reasoning_summary: str
    used_strategy: Literal["vector", "graph", "hybrid"]
    citations: list[Citation] = Field(default_factory=list)


class ExtractedRelation(BaseModel):
    subject: str
    predicate: str
    obj: str
    evidence: str


class ExtractionResult(BaseModel):
    entities: list[str] = Field(default_factory=list)
    relations: list[ExtractedRelation] = Field(default_factory=list)
