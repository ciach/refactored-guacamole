from __future__ import annotations

from pathlib import Path

from rag_agent.ingest.chunker import chunk_text
from rag_agent.ingest.embedder import embed_text
from rag_agent.ingest.entity_extractor import extract_entities_and_relations
from rag_agent.ingest.loader import load_document
from rag_agent.models import GraphFact, SourceChunk


class IngestionPipeline:
    def ingest_path(self, path: Path) -> tuple[list[SourceChunk], list[GraphFact]]:
        text = load_document(path)
        chunks = chunk_text(text, chunk_size=80, overlap=10)
        source_chunks: list[SourceChunk] = []
        graph_facts: list[GraphFact] = []
        for index, chunk in enumerate(chunks):
            _ = embed_text(chunk)
            chunk_id = f"{path.stem}-chunk-{index}"
            source_chunks.append(
                SourceChunk(
                    chunk_id=chunk_id,
                    doc_id=path.stem,
                    title=path.stem.replace("_", " ").title(),
                    text=chunk,
                    source_path=str(path),
                )
            )
            extraction = extract_entities_and_relations(chunk)
            for rel_index, relation in enumerate(extraction.relations):
                graph_facts.append(
                    GraphFact(
                        fact_id=f"{chunk_id}-fact-{rel_index}",
                        subject=relation.subject,
                        predicate=relation.predicate,
                        obj=relation.obj,
                        evidence=relation.evidence,
                    )
                )
        return source_chunks, graph_facts
