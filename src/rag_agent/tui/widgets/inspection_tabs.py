from __future__ import annotations

from rag_agent.tui.models import QueryRunResult


def citation_rows(result: QueryRunResult) -> list[tuple[str, str, str]]:
    return [(item.source_type, item.title, item.snippet) for item in result.citations]


def vector_rows(result: QueryRunResult) -> list[tuple[str, str, str, str]]:
    return [(item.chunk_id, item.title, f"{item.score:.3f}", item.text) for item in result.vector_hits]


def graph_rows(result: QueryRunResult) -> list[tuple[str, str, str, str, str]]:
    return [
        (item.fact_id, item.subject, item.predicate, item.obj, item.evidence or "")
        for item in result.graph_hits
    ]


def trace_lines(result: QueryRunResult) -> str:
    return "\n".join(result.trace)
