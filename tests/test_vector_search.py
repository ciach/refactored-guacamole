import asyncio

from rag_agent.models import SourceChunk
from rag_agent.retrieval.vector_store import InMemoryVectorStore


def test_vector_search_returns_best_matching_chunk() -> None:
    store = InMemoryVectorStore(
        seed_chunks=[
            SourceChunk(chunk_id="1", doc_id="a", title="CLI", text="Rich panels improve CLI debugging."),
            SourceChunk(chunk_id="2", doc_id="b", title="Graph", text="Neo4j stores entities and relations."),
        ]
    )

    results = asyncio.run(store.search("CLI debugging with Rich", k=1))

    assert results[0].chunk_id == "1"
    assert results[0].score >= 0.0
