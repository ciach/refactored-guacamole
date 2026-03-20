import asyncio

from rag_agent.models import GraphFact
from rag_agent.retrieval.graph_store import InMemoryGraphStore


def test_graph_search_returns_expected_relation() -> None:
    store = InMemoryGraphStore(
        seed_facts=[
            GraphFact(fact_id="1", subject="CLI", predicate="uses", obj="Rich", evidence="CLI uses Rich"),
            GraphFact(fact_id="2", subject="Vector", predicate="uses", obj="pgvector", evidence="Vector uses pgvector"),
        ]
    )

    results = asyncio.run(store.search("Which component uses Rich?", k=1))

    assert results[0].subject == "CLI"
    assert results[0].obj == "Rich"
