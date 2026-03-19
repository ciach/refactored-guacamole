from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    text: str
    start: int
    end: int


def chunk_text(text: str, *, chunk_size: int = 600, overlap: int = 100) -> list[Chunk]:
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")
    words = text.split()
    if not words:
        return []
    chunks: list[Chunk] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(Chunk(text=" ".join(words[start:end]), start=start, end=end))
        if end == len(words):
            break
        start = end - overlap
    return chunks
