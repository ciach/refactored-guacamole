from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from rag_agent.config import Settings

ModelRole = Literal["default", "fast"]


@dataclass(frozen=True)
class OfflineOpenAIModel:
    model_name: str
    role: ModelRole


def resolve_model_name(settings: Settings, role: ModelRole = "default") -> str:
    return settings.resolve_text_model(role)


def build_openai_text_model(settings: Settings, role: ModelRole = "default") -> object:
    model_name = resolve_model_name(settings, role)
    try:  # pragma: no cover - exercised when dependencies are installed.
        from pydantic_ai.models.openai import OpenAIResponsesModel

        return OpenAIResponsesModel(model_name)
    except Exception:
        return OfflineOpenAIModel(model_name=model_name, role=role)
