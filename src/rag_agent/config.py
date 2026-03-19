from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from rag_agent.compat import SettingsBase


@dataclass
class Settings(SettingsBase):
    openai_api_key: str = ""
    openai_model: str = "gpt-5"
    openai_embedding_model: str = "text-embedding-3-large"

    postgres_dsn: str = "postgresql+psycopg://user:pass@localhost:5432/rag_agent"
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "yourpassword"

    top_k_vector: int = 8
    top_k_graph: int = 8
    top_k_hybrid: int = 12


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings.from_env()
