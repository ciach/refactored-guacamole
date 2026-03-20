from __future__ import annotations

from rag_agent.agent.deps import AgentDeps
from rag_agent.config import Settings, settings as default_settings
from rag_agent.models import GraphFact, SourceChunk
from rag_agent.retrieval.graph_store import InMemoryGraphStore
from rag_agent.retrieval.vector_store import InMemoryVectorStore


SAMPLE_CHUNKS = [
    SourceChunk(
        chunk_id="chunk-1",
        doc_id="doc-architecture",
        title="Architecture Overview",
        text=(
            "The first version uses one PydanticAI agent with explicit vector, graph, and hybrid retrieval tools. "
            "The interface is CLI-first with Rich panels and citations."
        ),
        source_path="README.md",
    ),
    SourceChunk(
        chunk_id="chunk-2",
        doc_id="doc-ingestion",
        title="Ingestion Pipeline",
        text=(
            "Ingestion loads markdown, text, and PDFs, chunks documents into 400 to 800 token windows, "
            "embeds them, extracts entities and relations, and writes both vector and graph stores."
        ),
        source_path="docs/ingestion.md",
    ),
]

SAMPLE_FACTS = [
    GraphFact(
        fact_id="fact-1",
        subject="vector_search",
        predicate="complements",
        obj="graph_search",
        evidence="Hybrid search combines raw semantic evidence with graph relationships.",
    ),
    GraphFact(
        fact_id="fact-2",
        subject="CLI",
        predicate="uses",
        obj="Rich",
        evidence="The operator interface is intentionally CLI-first with Rich for traces and debugging.",
    ),
]


def build_deps(settings: Settings | None = None) -> AgentDeps:
    active_settings = settings or default_settings
    vector_store = InMemoryVectorStore(seed_chunks=SAMPLE_CHUNKS)
    graph_store = InMemoryGraphStore(seed_facts=SAMPLE_FACTS)
    return AgentDeps(settings=active_settings, vector_store=vector_store, graph_store=graph_store)
