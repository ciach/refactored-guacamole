from rag_agent.agent.agent import choose_strategy
from rag_agent.cli import _compose_answer
from rag_agent.models import GraphFact, SourceChunk


def test_choose_strategy_prefers_vector_for_descriptive_queries() -> None:
    assert choose_strategy("Explain the ingestion pipeline") == "vector"


def test_choose_strategy_prefers_graph_for_relationship_queries() -> None:
    assert choose_strategy("What relationship exists before and after deployment?") == "graph"


def test_choose_strategy_prefers_hybrid_for_analysis_queries() -> None:
    assert choose_strategy("Compare the impact of hybrid retrieval") == "hybrid"


def test_answer_includes_citations() -> None:
    answer = _compose_answer(
        "What does the system use?",
        used_strategy="hybrid",
        vector_hits=[
            SourceChunk(
                chunk_id="chunk-1",
                doc_id="doc-1",
                title="Architecture",
                text="One PydanticAI agent uses vector, graph, and hybrid tools.",
                score=2.0,
            )
        ],
        graph_hits=[
            GraphFact(
                fact_id="fact-1",
                subject="agent",
                predicate="uses",
                obj="retrieval tools",
                evidence="The agent calls retrieval tools based on the query.",
            )
        ],
    )

    assert answer.citations
    assert {citation.source_type for citation in answer.citations} == {"vector", "graph"}
