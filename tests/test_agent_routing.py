import asyncio

from rag_agent.agent.agent import route_query, run_query
from rag_agent.bootstrap import build_deps


def test_route_query_prefers_expected_tool_family() -> None:
    assert route_query("Explain the architecture").strategy == "vector"
    assert route_query("What relationship connects CLI and Rich?").strategy == "graph"
    assert route_query("Compare the impact of vector and graph retrieval").strategy == "hybrid"


def test_answer_includes_citations() -> None:
    answer = asyncio.run(run_query("Compare vector and graph retrieval", deps=build_deps()))
    assert answer.used_strategy == "hybrid"
    assert answer.citations
    assert {citation.source_type for citation in answer.citations} == {"vector", "graph"}
