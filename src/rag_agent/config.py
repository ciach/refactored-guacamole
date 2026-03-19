from __future__ import annotations

from rag_agent._compat import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    openai_api_key: str = "test-key"
    openai_model: str = "gpt-5"
    openai_embedding_model: str = "text-embedding-3-large"

    postgres_dsn: str = "postgresql+psycopg://user:pass@localhost:5432/rag_agent"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"

    top_k_vector: int = 8
    top_k_graph: int = 8
    top_k_hybrid: int = 12


settings = Settings()
