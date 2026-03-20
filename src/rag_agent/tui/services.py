from __future__ import annotations

import asyncio
from typing import Any

from rag_agent.agent.agent import execute_query
from rag_agent.bootstrap import build_deps
from rag_agent.config import Settings, settings as default_settings
from rag_agent.tui.models import QueryConfigSnapshot, QueryRunResult, SessionOverrides


def _settings_kwargs(settings: Settings) -> dict[str, Any]:
    return {
        "openai_api_key": settings.openai_api_key,
        "openai_model": settings.openai_model,
        "openai_model_default": settings.openai_model_default,
        "openai_model_fast": settings.openai_model_fast,
        "openai_embedding_model": settings.openai_embedding_model,
        "postgres_dsn": settings.postgres_dsn,
        "neo4j_uri": settings.neo4j_uri,
        "neo4j_user": settings.neo4j_user,
        "neo4j_password": settings.neo4j_password,
        "top_k_vector": settings.top_k_vector,
        "top_k_graph": settings.top_k_graph,
        "top_k_hybrid": settings.top_k_hybrid,
    }


def apply_settings_overrides(base_settings: Settings | None = None, settings_overrides: dict[str, Any] | None = None) -> Settings:
    merged = _settings_kwargs(base_settings or default_settings)
    if settings_overrides:
        translated = {
            "openai_model_default": settings_overrides.get("default_model") or settings_overrides.get("openai_model_default"),
            "openai_model_fast": settings_overrides.get("fast_model") or settings_overrides.get("openai_model_fast"),
            "top_k_vector": settings_overrides.get("top_k_vector"),
            "top_k_graph": settings_overrides.get("top_k_graph"),
            "top_k_hybrid": settings_overrides.get("top_k_hybrid"),
        }
        passthrough = {
            key: value
            for key, value in settings_overrides.items()
            if key not in {"default_model", "fast_model", "retrieval_strategy", "openai_model_default", "openai_model_fast"}
        }
        merged.update({key: value for key, value in {**passthrough, **translated}.items() if value is not None})
    return Settings(**merged)


def _snapshot(settings: Settings, strategy: str) -> QueryConfigSnapshot:
    return QueryConfigSnapshot(
        default_model=settings.resolve_text_model("default"),
        fast_model=settings.resolve_text_model("fast"),
        embedding_model=settings.openai_embedding_model,
        retrieval_strategy=strategy,
        top_k_vector=int(settings.top_k_vector),
        top_k_graph=int(settings.top_k_graph),
        top_k_hybrid=int(settings.top_k_hybrid),
    )


async def run_query_with_trace(
    question: str,
    settings_overrides: dict[str, Any] | SessionOverrides | None = None,
    strategy_override: str | None = None,
    base_settings: Settings | None = None,
) -> QueryRunResult:
    if isinstance(settings_overrides, SessionOverrides):
        overrides_dict = settings_overrides.model_dump()
    else:
        overrides_dict = settings_overrides

    resolved_settings = apply_settings_overrides(base_settings=base_settings, settings_overrides=overrides_dict)
    active_strategy = strategy_override or (overrides_dict or {}).get("retrieval_strategy") or "auto"

    try:
        deps = build_deps(resolved_settings)
        execution = await execute_query(question, deps, strategy_override=active_strategy)
        return QueryRunResult(
            question=question,
            answer=execution.answer.answer,
            reasoning_summary=execution.answer.reasoning_summary,
            used_strategy=execution.answer.used_strategy,
            config=_snapshot(resolved_settings, active_strategy),
            citations=execution.answer.citations,
            vector_hits=execution.vector_hits,
            graph_hits=execution.graph_hits,
            trace=execution.trace,
        )
    except Exception as exc:
        trace = [f"error={exc}"]
        return QueryRunResult(
            question=question,
            answer="",
            reasoning_summary="",
            used_strategy="hybrid",
            config=_snapshot(resolved_settings, active_strategy),
            citations=[],
            vector_hits=[],
            graph_hits=[],
            trace=trace,
            error=str(exc),
        )


def run_query_with_trace_sync(
    question: str,
    settings_overrides: dict[str, Any] | SessionOverrides | None = None,
    strategy_override: str | None = None,
    base_settings: Settings | None = None,
) -> QueryRunResult:
    return asyncio.run(
        run_query_with_trace(
            question=question,
            settings_overrides=settings_overrides,
            strategy_override=strategy_override,
            base_settings=base_settings,
        )
    )
