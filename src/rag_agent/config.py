from __future__ import annotations

from typing import Literal

from rag_agent._compat import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = "test-key"
    openai_model: str | None = None
    openai_model_default: str | None = None
    openai_model_fast: str | None = None
    openai_embedding_model: str = "text-embedding-3-large"

    postgres_dsn: str = "postgresql+psycopg://user:pass@localhost:5432/rag_agent"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    top_k_vector: int = 8
    top_k_graph: int = 8
    top_k_hybrid: int = 12

    def resolve_text_model(self, role: Literal["default", "fast"] = "default") -> str:
        default_model = self.openai_model_default or self.openai_model or "gpt-5"
        if role == "fast":
            return self.openai_model_fast or default_model
        return default_model


settings = Settings()
