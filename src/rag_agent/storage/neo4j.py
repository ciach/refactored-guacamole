from __future__ import annotations

NEO4J_SCHEMA_OVERVIEW = {
    "nodes": ["Entity {name, type}", "Document {doc_id, title}", "Chunk {chunk_id}"],
    "edges": [
        "(Entity)-[:RELATES_TO]->(Entity)",
        "(Chunk)-[:MENTIONS]->(Entity)",
        "(Chunk)-[:FROM_DOCUMENT]->(Document)",
    ],
}
