from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from rag_agent.ingest.chunker import chunk_text
from rag_agent.ingest.entity_extractor import extract_entities_and_relations
from rag_agent.ingest.loader import load_document
from rag_agent.models import GraphFact, SourceChunk
from rag_agent.retrieval.graph_store import GraphStore
from rag_agent.retrieval.vector_store import VectorStore


async def ingest_path(path: Path, *, vector_store: VectorStore, graph_store: GraphStore) -> None:
    title, text = load_document(path)
    chunks = chunk_text(text)
    source_chunks: list[SourceChunk] = []
    graph_facts: list[GraphFact] = []
    doc_id = path.stem

    for index, chunk in enumerate(chunks):
        chunk_id = f"{doc_id}-{index}"
        source_chunks.append(
            SourceChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                title=title,
                text=chunk.text,
                score=0.0,
                source_path=str(path),
                metadata={"start": chunk.start, "end": chunk.end},
            )
        )
        extraction = extract_entities_and_relations(chunk.text)
        for relation in extraction.relations:
            graph_facts.append(
                GraphFact(
                    fact_id=str(uuid4()),
                    subject=relation.subject,
                    predicate=relation.predicate,
                    obj=relation.obj,
                    evidence=relation.evidence,
                )
            )

    await vector_store.upsert(source_chunks)
    await graph_store.upsert(graph_facts)
