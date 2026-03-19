from __future__ import annotations

import asyncio
from typing import Any

from rag_agent.compat import Console, JSON, Panel, Table, Typer

from rag_agent.agent.agent import choose_strategy
from rag_agent.bootstrap import build_deps
from rag_agent.models import AgentAnswer, Citation, GraphFact, SourceChunk

app = Typer(help="CLI for the rag-agent project.")
console = Console()


def _build_citations(*, vector_hits: list[SourceChunk], graph_hits: list[GraphFact]) -> list[Citation]:
    citations: list[Citation] = []
    for chunk in vector_hits:
        citations.append(
            Citation(
                source_type="vector",
                ref_id=chunk.chunk_id,
                title=chunk.title,
                snippet=chunk.text[:180],
            )
        )
    for fact in graph_hits:
        citations.append(
            Citation(
                source_type="graph",
                ref_id=fact.fact_id,
                title=f"{fact.subject} {fact.predicate} {fact.obj}",
                snippet=(fact.evidence or "")[:180],
            )
        )
    return citations


def _compose_answer(
    question: str,
    *,
    used_strategy: str,
    vector_hits: list[SourceChunk],
    graph_hits: list[GraphFact],
) -> AgentAnswer:
    evidence_lines: list[str] = []
    if vector_hits:
        evidence_lines.append(
            "Vector evidence: " + "; ".join(chunk.text for chunk in vector_hits[:2])
        )
    if graph_hits:
        evidence_lines.append(
            "Graph evidence: "
            + "; ".join(
                f"{fact.subject} {fact.predicate} {fact.obj}" for fact in graph_hits[:2]
            )
        )
    answer = (
        f"Question: {question}\n\n"
        + ("\n".join(evidence_lines) if evidence_lines else "No evidence matched the query.")
    )
    reasoning_summary = (
        f"Selected {used_strategy} because the query pattern matched the v1 routing policy and "
        "the CLI keeps the evidence visible for debugging."
    )
    return AgentAnswer(
        answer=answer,
        reasoning_summary=reasoning_summary,
        used_strategy=used_strategy,
        citations=_build_citations(vector_hits=vector_hits, graph_hits=graph_hits),
    )


async def run_query(question: str, *, debug: bool = False) -> AgentAnswer:
    deps = await build_deps()
    strategy = choose_strategy(question)
    vector_hits: list[SourceChunk] = []
    graph_hits: list[GraphFact] = []

    if strategy in {"vector", "hybrid"}:
        vector_hits = await deps.vector_store.search(question, k=deps.settings.top_k_vector)
    if strategy in {"graph", "hybrid"}:
        graph_hits = await deps.graph_store.search(question, k=deps.settings.top_k_graph)

    result = _compose_answer(
        question,
        used_strategy=strategy,
        vector_hits=vector_hits,
        graph_hits=graph_hits,
    )

    if debug:
        debug_payload: dict[str, Any] = {
            "strategy": strategy,
            "vector_hits": [chunk.model_dump() for chunk in vector_hits],
            "graph_hits": [fact.model_dump() for fact in graph_hits],
        }
        console.print(Panel(JSON.from_data(debug_payload), title="Retrieval Debug"))

    console.print(Panel(result.answer, title=f"Answer [{result.used_strategy}]"))
    console.print(Panel(result.reasoning_summary, title="Routing Reason"))

    table = Table(title="Citations")
    table.add_column("Type")
    table.add_column("Title")
    table.add_column("Snippet")

    for citation in result.citations:
        table.add_row(citation.source_type, citation.title, citation.snippet)

    console.print(table)
    return result


@app.command()
def ask(question: str, debug: bool = False) -> None:
    """Run a single query against the sample retrieval stack."""

    asyncio.run(run_query(question, debug=debug))
