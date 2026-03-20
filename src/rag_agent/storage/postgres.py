from __future__ import annotations

PGVECTOR_SCHEMA_SQL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    title TEXT NOT NULL,
    text TEXT NOT NULL,
    source_path TEXT,
    embedding vector(3072)
);
""".strip()
