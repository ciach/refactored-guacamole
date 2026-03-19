import asyncio

from rag_agent.models import SourceChunk
from rag_agent.retrieval.vector_store import InMemoryVectorStore


def test_vector_search_returns_expected_chunk() -> None:
    store = InMemoryVectorStore(
        chunks=[
            SourceChunk(
                chunk_id="a",
                doc_id="doc-1",
                title="Architecture",
                text="One agent with explicit retrieval tools.",
                score=0.0,
            ),
            SourceChunk(
                chunk_id="b",
                doc_id="doc-2",
                title="Unrelated",
                text="Completely different topic.",
                score=0.0,
            ),
        ]
    )

    result = asyncio.run(store.search("retrieval tools", k=1))

    assert [chunk.chunk_id for chunk in result] == ["a"]
