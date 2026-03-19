import asyncio

from rag_agent.models import GraphFact
from rag_agent.retrieval.graph_store import InMemoryGraphStore


def test_graph_search_returns_expected_relation() -> None:
    store = InMemoryGraphStore(
        facts=[
            GraphFact(
                fact_id="fact-1",
                subject="pipeline",
                predicate="writes to",
                obj="vector store",
                evidence="The ingestion pipeline writes embeddings to the vector store.",
            ),
            GraphFact(
                fact_id="fact-2",
                subject="cli",
                predicate="renders",
                obj="rich tables",
                evidence="The CLI renders citations.",
            ),
        ]
    )

    result = asyncio.run(store.search("pipeline writes vector", k=1))

    assert [fact.fact_id for fact in result] == ["fact-1"]
