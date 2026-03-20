from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Callable

try:  # pragma: no cover - exercised when Rich is installed.
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
except Exception:  # pragma: no cover
    Console = None
    Panel = None
    Table = None

try:  # pragma: no cover - exercised when Typer is installed.
    import typer
except Exception:  # pragma: no cover
    typer = None

from rag_agent.models import AgentAnswer
from rag_agent.tui.app import launch_tui
from rag_agent.tui.services import run_query_with_trace_sync

console = Console() if Console is not None else None


class _FallbackApp:
    def __init__(self) -> None:
        self._commands: dict[str, Callable[..., None]] = {}

    def command(self, name: str | None = None, **_kwargs: Any):
        def decorator(func: Callable[..., None]) -> Callable[..., None]:
            self._commands[name or func.__name__] = func
            return func

        return decorator

    def __call__(self) -> None:
        parser = argparse.ArgumentParser(prog="rag-agent", description="CLI-first RAG agent")
        subparsers = parser.add_subparsers(dest="command", required=True)
        ask_parser = subparsers.add_parser("ask", help="Ask one grounded question")
        ask_parser.add_argument("question")
        ask_parser.add_argument("--as-json", action="store_true")
        tui_parser = subparsers.add_parser("tui", help="Launch the Textual TUI")
        tui_parser.set_defaults(_noop=True)
        args = parser.parse_args(sys.argv[1:])
        command = self._commands[args.command]
        kwargs = vars(args)
        kwargs.pop("command", None)
        kwargs.pop("_noop", None)
        command(**kwargs)


def _render_answer(output: AgentAnswer) -> None:
    if console is None or Panel is None or Table is None:
        print(f"Answer [{output.used_strategy}]: {output.answer}")
        print(f"Routing Reason: {output.reasoning_summary}")
        for citation in output.citations:
            print(f"- {citation.source_type}: {citation.title} :: {citation.snippet}")
        return

    console.print(Panel(output.answer, title=f"Answer [{output.used_strategy}]"))
    console.print(Panel(output.reasoning_summary, title="Routing Reason"))

    table = Table(title="Citations")
    table.add_column("Type")
    table.add_column("Title")
    table.add_column("Snippet")
    for citation in output.citations:
        table.add_row(citation.source_type, citation.title, citation.snippet[:120])
    console.print(table)


app = typer.Typer(help="CLI-first RAG agent") if typer is not None else _FallbackApp()


@app.command()
def ask(question: str, as_json: bool = False) -> None:
    """Ask one grounded question using the shared query execution service."""

    result = run_query_with_trace_sync(question=question)
    output = AgentAnswer(
        answer=result.answer,
        reasoning_summary=result.reasoning_summary,
        used_strategy=result.used_strategy,
        citations=result.citations,
    )
    if as_json:
        print(json.dumps(result.model_dump(), indent=2))
        return
    _render_answer(output)


@app.command()
def tui() -> None:
    """Launch the Textual terminal UI."""

    launch_tui()
