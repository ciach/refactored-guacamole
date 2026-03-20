from __future__ import annotations

from typing import Any

from rag_agent.tui.models import QueryRunResult, SessionOverrides
from rag_agent.tui.services import run_query_with_trace_sync
from rag_agent.tui.state import TuiSessionState
from rag_agent.tui.widgets.answer_panel import format_answer
from rag_agent.tui.widgets.config_panel import format_config_snapshot
from rag_agent.tui.widgets.history_panel import history_label
from rag_agent.tui.widgets.inspection_tabs import citation_rows, graph_rows, trace_lines, vector_rows
from rag_agent.tui.widgets.query_editor import DEFAULT_QUERY

try:  # pragma: no cover - requires Textual.
    from textual.app import App, ComposeResult
    from textual.containers import Horizontal, Vertical
    from textual.widgets import Button, DataTable, Footer, Header, Input, ListItem, ListView, Static, TabPane, TabbedContent, TextArea
except Exception:  # pragma: no cover
    App = None
    ComposeResult = Any
    Horizontal = Vertical = object
    Button = DataTable = Footer = Header = Input = ListItem = ListView = Static = TabPane = TabbedContent = TextArea = object


class MissingTextualApp:
    def run(self) -> None:
        raise RuntimeError("Textual is not installed. Run `uv sync` to use the TUI.")


def build_tui_app() -> object:
    if App is None:
        return MissingTextualApp()

    class RagTuiApp(App[None]):
        BINDINGS = [("ctrl+enter", "run_query", "Run"), ("r", "run_query", "Rerun"), ("q", "quit", "Quit")]

        def __init__(self) -> None:
            super().__init__()
            self.state = TuiSessionState(current_query=DEFAULT_QUERY, overrides=SessionOverrides())

        def compose(self) -> ComposeResult:
            yield Header()
            with Horizontal():
                with Vertical(id="left-pane"):
                    yield Static("Query")
                    yield TextArea(DEFAULT_QUERY, id="query-editor")
                    yield Button("Run", id="run-query", variant="primary")
                    yield Static("History")
                    yield ListView(id="history-list")
                with Vertical(id="right-pane"):
                    yield Static("Config", id="config-title")
                    yield Input(placeholder="Default model override", id="default-model")
                    yield Input(placeholder="Fast model override", id="fast-model")
                    yield Input(value="auto", placeholder="Strategy override", id="strategy-override")
                    yield Input(placeholder="Top-k vector", id="top-k-vector")
                    yield Input(placeholder="Top-k graph", id="top-k-graph")
                    yield Input(placeholder="Top-k hybrid", id="top-k-hybrid")
                    yield Static("", id="config-panel")
                    yield Static("", id="answer-panel")
                    with TabbedContent(id="inspection-tabs"):
                        with TabPane("Citations", id="citations-tab"):
                            yield DataTable(id="citations-table")
                        with TabPane("Vector Hits", id="vector-tab"):
                            yield DataTable(id="vector-table")
                        with TabPane("Graph Hits", id="graph-tab"):
                            yield DataTable(id="graph-table")
                        with TabPane("Trace", id="trace-tab"):
                            yield Static("", id="trace-panel")
            yield Footer()

        def on_mount(self) -> None:
            self._setup_tables()
            self._render_result(
                QueryRunResult(
                    question=DEFAULT_QUERY,
                    answer="Run a query to inspect answer, citations, and retrieval traces.",
                    reasoning_summary="",
                    used_strategy="hybrid",
                    config=run_query_with_trace_sync(DEFAULT_QUERY).config,
                    citations=[],
                    vector_hits=[],
                    graph_hits=[],
                    trace=["idle"],
                    error=None,
                )
            )

        def _setup_tables(self) -> None:
            citations = self.query_one("#citations-table", DataTable)
            citations.add_columns("Type", "Title", "Snippet")
            vector_table = self.query_one("#vector-table", DataTable)
            vector_table.add_columns("Chunk", "Title", "Score", "Text")
            graph_table = self.query_one("#graph-table", DataTable)
            graph_table.add_columns("Fact", "Subject", "Predicate", "Object", "Evidence")

        def _read_overrides(self) -> SessionOverrides:
            def _value(widget_id: str) -> str:
                return self.query_one(widget_id, Input).value.strip()

            return SessionOverrides(
                default_model=_value("#default-model") or None,
                fast_model=_value("#fast-model") or None,
                retrieval_strategy=(_value("#strategy-override") or "auto"),
                top_k_vector=int(_value("#top-k-vector")) if _value("#top-k-vector") else None,
                top_k_graph=int(_value("#top-k-graph")) if _value("#top-k-graph") else None,
                top_k_hybrid=int(_value("#top-k-hybrid")) if _value("#top-k-hybrid") else None,
            )

        def _render_result(self, result: QueryRunResult) -> None:
            self.state.current_result = result
            self.state.history.insert(0, result)
            self.query_one("#config-panel", Static).update(format_config_snapshot(result.config))
            self.query_one("#answer-panel", Static).update(format_answer(result))
            self.query_one("#trace-panel", Static).update(trace_lines(result))

            citations = self.query_one("#citations-table", DataTable)
            citations.clear(columns=False)
            for row in citation_rows(result):
                citations.add_row(*row)

            vector_table = self.query_one("#vector-table", DataTable)
            vector_table.clear(columns=False)
            for row in vector_rows(result):
                vector_table.add_row(*row)

            graph_table = self.query_one("#graph-table", DataTable)
            graph_table.clear(columns=False)
            for row in graph_rows(result):
                graph_table.add_row(*row)

            history = self.query_one("#history-list", ListView)
            history.clear()
            for item in self.state.history:
                history.append(ListItem(Static(history_label(item))))

        def action_run_query(self) -> None:
            query = self.query_one("#query-editor", TextArea).text
            overrides = self._read_overrides()
            self.state.status = "running"
            result = run_query_with_trace_sync(query, settings_overrides=overrides)
            self.state.status = "error" if result.error else "success"
            self._render_result(result)

        def on_button_pressed(self, event: Button.Pressed) -> None:
            if event.button.id == "run-query":
                self.action_run_query()

    return RagTuiApp()


def launch_tui() -> None:
    app = build_tui_app()
    app.run()
