from __future__ import annotations

from rag_agent.tui.models import QueryRunResult


def format_answer(result: QueryRunResult) -> str:
    if result.error:
        return f"Error: {result.error}"
    return f"Strategy: {result.used_strategy}\n\n{result.answer}\n\nReasoning: {result.reasoning_summary}"
