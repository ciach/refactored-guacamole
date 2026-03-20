# RAG Agent

A CLI-first retrieval-augmented generation scaffold built around **OpenAI**, **PydanticAI**, **Rich**, **Textual**, and **uv**.

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
│   ├── llm/
│   ├── retrieval/
│   ├── storage/
│   ├── tui/
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
- **Rich** keeps the CLI answer loop readable.
- **Textual** provides the TUI shell without inventing a custom terminal framework.
- **OpenAI** powers answer generation and embedding/extraction integration points.
- **Postgres + pgvector** and **Neo4j** are represented as storage adapters, but tests use in-memory stores so the repository works without external services.

## Model configuration

The app supports **two text-generation roles** without turning model choice into a maze of knobs:

- `OPENAI_MODEL_DEFAULT`: main synthesis model for the user-facing answer.
- `OPENAI_MODEL_FAST`: optional cheaper/faster model reserved for lightweight preprocessing tasks.
- `OPENAI_EMBEDDING_MODEL`: dedicated embedding model.

Compatibility behavior:

- `OPENAI_MODEL` is still accepted temporarily as a deprecated alias.
- `OPENAI_MODEL_DEFAULT` wins when both are set.
- `OPENAI_MODEL_FAST` falls back to the resolved default model when omitted.

## Shared execution surface

Both the CLI and the TUI call the same service entrypoint:

- `run_query_with_trace(...)`
- `run_query_with_trace_sync(...)`

That shared layer returns a structured result with the answer, reasoning summary, selected strategy, config snapshot, vector hits, graph hits, citations, and trace lines. This keeps the TUI thin instead of becoming a second app.

## TUI v1

Launch it from the main entrypoint with:

```bash
uv run python -m rag_agent.main tui
```

The green-path TUI is a single-screen Textual app with:

- query editor
- config/override inputs
- answer panel
- inspection tabs for citations, vector hits, graph hits, and trace output
- in-memory session history

## Quick start

1. Create `.env` from `.env.example`.
2. Install dependencies with `uv sync`.
3. Run the CLI:

```bash
uv run python -m rag_agent.main ask "What changed in the design?"
```

4. Run the TUI:

```bash
uv run python -m rag_agent.main tui
```

5. Run tests:

```bash
uv run pytest
```

## Offline-friendly note

The execution environment used for this task did not allow fetching packages from PyPI, so the project includes graceful fallbacks that keep the code importable and testable even before `uv sync` succeeds. Once dependencies are available, the OpenAI/PydanticAI/Rich/Textual integrations are used automatically.
