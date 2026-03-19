from dataclasses import dataclass

from rag_agent.config import Settings
from rag_agent.retrieval.graph_store import GraphStore
from rag_agent.retrieval.vector_store import VectorStore


@dataclass
class AgentDeps:
    settings: Settings
    vector_store: VectorStore
    graph_store: GraphStore
