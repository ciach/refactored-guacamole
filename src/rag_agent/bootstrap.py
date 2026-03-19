from __future__ import annotations

from rag_agent.agent.deps import AgentDeps
from rag_agent.config import get_settings
from rag_agent.models import GraphFact, SourceChunk
from rag_agent.storage.neo4j import Neo4jGraphStore
from rag_agent.storage.postgres import PostgresVectorStore


async def build_deps() -> AgentDeps:
    settings = get_settings()
    vector_store = PostgresVectorStore(
        dsn=settings.postgres_dsn,
        embedding_model=settings.openai_embedding_model,
        api_key=settings.openai_api_key,
    )
    graph_store = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password,
    )

    await vector_store.bootstrap()
    await graph_store.bootstrap()

    if not vector_store.chunks:
        await vector_store.seed(
            [
                SourceChunk(
                    chunk_id="chunk-1",
                    doc_id="doc-architecture",
                    title="Recommended v1 architecture",
                    text=(
                        "The first version should keep one PydanticAI agent with explicit "
                        "vector_search, graph_search, and hybrid_search tools plus a Rich CLI."
                    ),
                    score=1.0,
                    source_path="README.md",
                ),
                SourceChunk(
                    chunk_id="chunk-2",
                    doc_id="doc-ingest",
                    title="Ingestion pipeline",
                    text=(
                        "Ingestion loads documents, chunks them, embeds them, extracts entities "
                        "and relations, and writes both vector and graph stores."
                    ),
                    score=1.0,
                    source_path="README.md",
                ),
            ]
        )
    if not graph_store.facts:
        await graph_store.seed(
            [
                GraphFact(
                    fact_id="fact-1",
                    subject="hybrid_search",
                    predicate="combines",
                    obj="vector hits and graph hits",
                    evidence="Hybrid retrieval merges raw chunk evidence with graph relationships.",
                ),
                GraphFact(
                    fact_id="fact-2",
                    subject="ingestion pipeline",
                    predicate="writes to",
                    obj="vector store and graph store",
                    evidence="The pipeline persists embeddings and extracted relations.",
                ),
            ]
        )

    return AgentDeps(settings=settings, vector_store=vector_store, graph_store=graph_store)
