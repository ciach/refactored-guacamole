from __future__ import annotations

import math
from collections import Counter


def embed_text(text: str, dimensions: int = 16) -> list[float]:
    """Deterministic fallback embedding for tests and offline development."""

    counts = Counter(token.lower() for token in text.split())
    vector = [0.0] * dimensions
    for token, count in counts.items():
        index = hash(token) % dimensions
        vector[index] += float(count)
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]
