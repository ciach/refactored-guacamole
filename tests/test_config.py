from rag_agent.config import Settings


def test_default_model_overrides_legacy_alias() -> None:
    settings = Settings(openai_model="gpt-5-legacy", openai_model_default="gpt-5", openai_model_fast=None)

    assert settings.resolve_text_model("default") == "gpt-5"
    assert settings.resolve_text_model("fast") == "gpt-5"


def test_legacy_model_still_populates_default_and_fast_roles() -> None:
    settings = Settings(openai_model="gpt-5-legacy", openai_model_default=None, openai_model_fast=None)

    assert settings.resolve_text_model("default") == "gpt-5-legacy"
    assert settings.resolve_text_model("fast") == "gpt-5-legacy"


def test_fast_model_can_be_configured_independently() -> None:
    settings = Settings(openai_model_default="gpt-5", openai_model_fast="gpt-5-mini")

    assert settings.resolve_text_model("default") == "gpt-5"
    assert settings.resolve_text_model("fast") == "gpt-5-mini"
