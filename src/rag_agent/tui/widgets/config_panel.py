from __future__ import annotations

from rag_agent.tui.models import QueryConfigSnapshot


def format_config_snapshot(config: QueryConfigSnapshot) -> str:
    return (
        f"default={config.default_model}\n"
        f"fast={config.fast_model}\n"
        f"embedding={config.embedding_model}\n"
        f"strategy={config.retrieval_strategy}\n"
        f"top_k(vector/graph/hybrid)={config.top_k_vector}/{config.top_k_graph}/{config.top_k_hybrid}"
    )
