# RAG Agent

A CLI-first retrieval-augmented generation scaffold built around **OpenAI**, **PydanticAI**, **Rich**, and **uv**.

This repository intentionally starts with **one agent and three explicit retrieval tools**:

- `vector_search`
- `graph_search`
- `hybrid_search`

The implementation keeps the first version debuggable instead of overengineering multi-agent orchestration.

## Architecture

```text
rag_agent/
├── pyproject.toml
├── README.md
├── src/rag_agent/
│   ├── agent/
│   ├── ingest/
│   ├── retrieval/
│   ├── storage/
│   ├── bootstrap.py
│   ├── cli.py
│   ├── config.py
│   ├── logging.py
│   ├── main.py
│   ├── models.py
│   └── prompts.py
└── tests/
```

## Why this v1 shape

- **uv** owns project metadata and environment management.
- **PydanticAI** owns typed agent wiring, tool registration, and structured outputs when installed.
- **Rich** makes the CLI inspectable with strategy, traces, and citations.
- **OpenAI** powers both answer generation and embedding/extraction integration points.
- **Postgres + pgvector** and **Neo4j** are represented as storage adapters, but tests use in-memory stores so the repository works without external services.

## Quick start

1. Create `.env` from `.env.example`.
2. Install dependencies with `uv sync`.
3. Run the CLI:

```bash
uv run python -m rag_agent.main ask "What changed in the design?"
```

4. Run tests:

```bash
uv run pytest
```

## Offline-friendly note

The execution environment used for this task did not allow fetching packages from PyPI, so the project includes graceful fallbacks that keep the code importable and testable even before `uv sync` succeeds. Once dependencies are available, the OpenAI/PydanticAI/Rich integrations are used automatically.
