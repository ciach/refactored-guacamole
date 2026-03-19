from __future__ import annotations

import hashlib


def embed_text(text: str, *, dimensions: int = 12) -> list[float]:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return [byte / 255 for byte in digest[:dimensions]]
