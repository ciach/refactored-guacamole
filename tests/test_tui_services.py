from rag_agent.tui.models import SessionOverrides
from rag_agent.tui.services import apply_settings_overrides, run_query_with_trace_sync


def test_run_query_with_trace_returns_structured_result() -> None:
    result = run_query_with_trace_sync("Compare vector and graph retrieval")

    assert result.answer
    assert result.used_strategy == "hybrid"
    assert result.citations
    assert result.vector_hits
    assert result.graph_hits
    assert result.trace


def test_settings_overrides_are_applied_to_snapshot() -> None:
    overrides = SessionOverrides(default_model="gpt-5.1", fast_model="gpt-5-mini", retrieval_strategy="vector", top_k_vector=3)

    result = run_query_with_trace_sync("Explain the architecture", settings_overrides=overrides)

    assert result.config.default_model == "gpt-5.1"
    assert result.config.fast_model == "gpt-5-mini"
    assert result.config.retrieval_strategy == "vector"
    assert result.config.top_k_vector == 3


def test_apply_settings_overrides_keeps_default_fast_fallback() -> None:
    settings = apply_settings_overrides(settings_overrides={"openai_model_default": "gpt-5.1"})

    assert settings.resolve_text_model("default") == "gpt-5.1"
    assert settings.resolve_text_model("fast") == "gpt-5.1"
