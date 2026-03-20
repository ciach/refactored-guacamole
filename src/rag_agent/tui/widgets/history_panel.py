from __future__ import annotations

from rag_agent.tui.models import QueryRunResult


def history_label(result: QueryRunResult) -> str:
    status = "error" if result.error else result.used_strategy
    return f"[{status}] {result.question[:60]}"
