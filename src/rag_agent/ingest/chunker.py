from __future__ import annotations

from collections.abc import Iterable


def chunk_text(text: str, *, chunk_size: int = 80, overlap: int = 10) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and < chunk_size")

    tokens = text.split()
    if not tokens:
        return []

    chunks: list[str] = []
    step = chunk_size - overlap
    for start in range(0, len(tokens), step):
        window = tokens[start : start + chunk_size]
        if not window:
            break
        chunks.append(" ".join(window))
        if start + chunk_size >= len(tokens):
            break
    return chunks


def chunk_many(texts: Iterable[str], *, chunk_size: int = 80, overlap: int = 10) -> list[str]:
    chunks: list[str] = []
    for text in texts:
        chunks.extend(chunk_text(text, chunk_size=chunk_size, overlap=overlap))
    return chunks
