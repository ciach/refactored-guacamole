from __future__ import annotations

from typing import Literal

from rag_agent.compat import Agent, OpenAIResponsesModel

from rag_agent.agent.deps import AgentDeps
from rag_agent.agent.tools import graph_search_tool, hybrid_search_tool, vector_search_tool
from rag_agent.models import AgentAnswer
from rag_agent.prompts import SYSTEM_PROMPT

Strategy = Literal["vector", "graph", "hybrid"]

VECTOR_HINTS = {"what is", "summarize", "describe", "mention", "explain"}
GRAPH_HINTS = {
    "relationship",
    "relationships",
    "connected",
    "depends on",
    "before",
    "after",
    "timeline",
    "who works with whom",
}
HYBRID_HINTS = {"compare", "analysis", "analyze", "why", "impact", "tradeoff", "tradeoff"}


def choose_strategy(question: str) -> Strategy:
    normalized = question.lower()
    if any(hint in normalized for hint in HYBRID_HINTS):
        return "hybrid"
    if any(hint in normalized for hint in GRAPH_HINTS):
        return "graph"
    if any(hint in normalized for hint in VECTOR_HINTS):
        return "vector"
    return "hybrid"


rag_agent = Agent(
    model=OpenAIResponsesModel("gpt-5"),
    deps_type=AgentDeps,
    output_type=AgentAnswer,
    system_prompt=SYSTEM_PROMPT,
    tools=[vector_search_tool, graph_search_tool, hybrid_search_tool],
)
