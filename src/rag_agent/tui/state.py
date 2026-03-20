from __future__ import annotations

from dataclasses import dataclass, field

from rag_agent.tui.models import QueryRunResult, SessionOverrides


@dataclass
class TuiSessionState:
    current_query: str = ""
    overrides: SessionOverrides = field(default_factory=SessionOverrides)
    history: list[QueryRunResult] = field(default_factory=list)
    current_result: QueryRunResult | None = None
    status: str = "idle"
