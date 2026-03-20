from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Literal

from rag_agent.agent.deps import AgentDeps
from rag_agent.agent.tools import graph_search_tool, hybrid_search_tool, vector_search_tool
from rag_agent.config import Settings
from rag_agent.llm.models import build_openai_text_model
from rag_agent.models import AgentAnswer, Citation, GraphFact, SourceChunk
from rag_agent.prompts import SYSTEM_PROMPT

StrategyName = Literal["vector", "graph", "hybrid"]
StrategyOverride = Literal["auto", "vector", "graph", "hybrid"] | None

VECTOR_HINTS = {"what is", "summarize", "describe", "mention", "explain"}
GRAPH_HINTS = {"relationship", "connected", "depends on", "before", "after", "timeline", "who works with"}
HYBRID_HINTS = {"compare", "analyze", "why", "impact", "tradeoff"}


@dataclass
class RouteDecision:
    strategy: StrategyName
    reasoning: str


@dataclass
class QueryExecution:
    answer: AgentAnswer
    vector_hits: list[SourceChunk]
    graph_hits: list[GraphFact]
    model_name: str
    fast_model_name: str
    trace: list[str]


def route_query(question: str) -> RouteDecision:
    normalized = question.lower()
    if any(hint in normalized for hint in HYBRID_HINTS):
        return RouteDecision("hybrid", "Used hybrid_search because the query asks for analysis, comparison, or tradeoffs.")
    if any(hint in normalized for hint in GRAPH_HINTS):
        return RouteDecision("graph", "Used graph_search because the query focuses on relationships or chronology.")
    if any(hint in normalized for hint in VECTOR_HINTS):
        return RouteDecision("vector", "Used vector_search because the query is descriptive or topical.")
    return RouteDecision("hybrid", "Used hybrid_search because the query is ambiguous and benefits from both evidence types.")


def resolve_strategy(question: str, strategy_override: StrategyOverride = None) -> RouteDecision:
    if strategy_override in {"vector", "graph", "hybrid"}:
        return RouteDecision(strategy_override, f"Used {strategy_override}_search because the strategy was overridden for this run.")
    return route_query(question)


def _chunk_to_citation(chunk: SourceChunk) -> Citation:
    return Citation(
        source_type="vector",
        ref_id=chunk.chunk_id,
        title=chunk.title,
        snippet=chunk.text[:180],
    )


def _fact_to_citation(fact: GraphFact) -> Citation:
    snippet = fact.evidence or f"{fact.subject} {fact.predicate} {fact.obj}"
    return Citation(
        source_type="graph",
        ref_id=fact.fact_id,
        title=f"{fact.subject} {fact.predicate} {fact.obj}",
        snippet=snippet[:180],
    )


def _compose_answer(question: str, vector_hits: list[SourceChunk], graph_hits: list[GraphFact], strategy: StrategyName, reasoning: str) -> AgentAnswer:
    evidence_parts = [chunk.text for chunk in vector_hits[:2]] + [
        f"{fact.subject} {fact.predicate} {fact.obj}: {fact.evidence or ''}".strip()
        for fact in graph_hits[:2]
    ]
    answer = " ".join(evidence_parts) if evidence_parts else f"No evidence found for: {question}"
    citations = [_chunk_to_citation(chunk) for chunk in vector_hits]
    citations.extend(_fact_to_citation(fact) for fact in graph_hits)
    return AgentAnswer(answer=answer, reasoning_summary=reasoning, used_strategy=strategy, citations=citations)


@dataclass
class OfflineRagAgent:
    model_name: str

    async def run(self, question: str, deps: AgentDeps) -> SimpleNamespace:
        execution = await execute_query(question, deps)
        return SimpleNamespace(output=execution.answer)


def build_agent(settings: Settings) -> object:
    model = build_openai_text_model(settings, role="default")
    try:  # pragma: no cover - exercised when dependencies are installed.
        from pydantic_ai import Agent

        return Agent(
            model=model,
            deps_type=AgentDeps,
            output_type=AgentAnswer,
            system_prompt=SYSTEM_PROMPT,
            tools=[vector_search_tool, graph_search_tool, hybrid_search_tool],
        )
    except Exception:
        return OfflineRagAgent(model_name=settings.resolve_text_model("default"))


async def execute_query(question: str, deps: AgentDeps, strategy_override: StrategyOverride = None) -> QueryExecution:
    decision = resolve_strategy(question, strategy_override)
    ctx = SimpleNamespace(deps=deps)
    model_name = deps.settings.resolve_text_model("default")
    fast_model_name = deps.settings.resolve_text_model("fast")
    trace = [
        f"default_model={model_name}",
        f"fast_model={fast_model_name}",
        f"strategy={decision.strategy}",
        f"strategy_override={strategy_override or 'auto'}",
    ]

    if decision.strategy == "vector":
        vector_hits = await vector_search_tool(ctx, question, deps.settings.top_k_vector)
        graph_hits: list[GraphFact] = []
    elif decision.strategy == "graph":
        vector_hits = []
        graph_hits = await graph_search_tool(ctx, question, deps.settings.top_k_graph)
    else:
        payload = await hybrid_search_tool(ctx, question, deps.settings.top_k_hybrid)
        vector_hits = [SourceChunk(**hit) for hit in payload["vector_hits"]]
        graph_hits = [GraphFact(**hit) for hit in payload["graph_hits"]]

    trace.append(f"vector_hits={len(vector_hits)}")
    trace.append(f"graph_hits={len(graph_hits)}")
    answer = _compose_answer(question, vector_hits, graph_hits, decision.strategy, decision.reasoning)
    return QueryExecution(
        answer=answer,
        vector_hits=vector_hits,
        graph_hits=graph_hits,
        model_name=model_name,
        fast_model_name=fast_model_name,
        trace=trace,
    )


async def run_query(question: str, deps: AgentDeps, strategy_override: StrategyOverride = None) -> AgentAnswer:
    execution = await execute_query(question, deps, strategy_override=strategy_override)
    return execution.answer
