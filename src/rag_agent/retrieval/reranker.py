from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TypeVar

T = TypeVar("T")


def rerank_by_score(items: Iterable[T], *, key: Callable[[T], float]) -> list[T]:
    return sorted(items, key=key, reverse=True)
