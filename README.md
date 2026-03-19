# rag-agent

A CLI-first retrieval-augmented generation project built with **OpenAI**, **PydanticAI**,
**Rich**, and **uv**. The initial version intentionally keeps the architecture simple:
one agent, three retrieval tools, strict structured output, and an inspectable terminal loop.

## Why this shape

- **One agent, not a council of experts.** Routing is explicit and grounded in tool
  descriptions plus a lightweight heuristic pre-router.
- **CLI first.** Rich panels and tables make it easy to inspect routing, retrieved evidence,
  and citations before spending time on a web UI.
- **Hybrid retrieval.** Vector search handles semantic recall, graph search handles
  relationships, and hybrid search merges both when the query needs raw evidence plus links.
- **uv-native workflow.** The project is managed with `pyproject.toml`, `uv.lock`, `uv sync`,
  and `uv run`.

## Project layout

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

## Getting started

1. Copy the environment template.
   ```bash
   cp .env.example .env
   ```
2. Install dependencies and create the virtual environment.
   ```bash
   uv sync --extra dev
   ```
3. Run the CLI.
   ```bash
   uv run rag-agent ask "What does the corpus say about the ingestion pipeline?"
   ```

## Environment variables

The app reads configuration from `.env` with `pydantic-settings`.

- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_EMBEDDING_MODEL`
- `POSTGRES_DSN`
- `NEO4J_URI`
- `NEO4J_USER`
- `NEO4J_PASSWORD`
- `TOP_K_VECTOR`
- `TOP_K_GRAPH`
- `TOP_K_HYBRID`

## Retrieval flow

1. **Ingestion** loads documents, chunks them, embeds them, extracts entities/relations, then
   persists vector and graph representations.
2. **Retrieval tools** expose `vector_search`, `graph_search`, and `hybrid_search`.
3. **The agent** chooses a strategy, calls the tools, and returns an `AgentAnswer` schema with
   citations.
4. **The CLI** renders the selected strategy, reasoning summary, answer, and citations with Rich.

## Development commands

```bash
uv run pytest
uv run ruff check .
uv run mypy src
```
