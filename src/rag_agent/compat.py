from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field, fields, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generic, TypeVar, get_args, get_origin

try:
    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import Field as PydanticField
except ImportError:  # pragma: no cover - exercised only in dependency-light environments
    PydanticBaseModel = None
    PydanticField = None

try:
    from pydantic_ai import Agent as PydanticAIAgent
    from pydantic_ai import RunContext as PydanticAIRunContext
    from pydantic_ai.models.openai import OpenAIResponsesModel as PydanticOpenAIResponsesModel
except ImportError:  # pragma: no cover
    PydanticAIAgent = None
    PydanticAIRunContext = None
    PydanticOpenAIResponsesModel = None

try:
    import typer as typer_module
except ImportError:  # pragma: no cover
    typer_module = None

try:
    from rich.console import Console as RichConsole
    from rich.json import JSON as RichJSON
    from rich.panel import Panel as RichPanel
    from rich.table import Table as RichTable
except ImportError:  # pragma: no cover
    RichConsole = None
    RichJSON = None
    RichPanel = None
    RichTable = None

TDeps = TypeVar("TDeps")


if PydanticBaseModel is not None:
    BaseModel = PydanticBaseModel
    Field = PydanticField
else:
    def Field(*, default: Any = None, default_factory: Any | None = None) -> Any:
        if default_factory is not None:
            return field(default_factory=default_factory)
        return field(default=default)


    class BaseModel:
        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)
            dataclass(cls)

        def model_dump(self) -> dict[str, Any]:
            return asdict(self)

        def model_copy(self, *, update: dict[str, Any] | None = None) -> Any:
            return replace(self, **(update or {}))


if PydanticAIRunContext is not None:
    RunContext = PydanticAIRunContext
else:
    @dataclass
    class RunContext(Generic[TDeps]):
        deps: TDeps


if PydanticOpenAIResponsesModel is not None:
    OpenAIResponsesModel = PydanticOpenAIResponsesModel
else:
    class OpenAIResponsesModel:
        def __init__(self, model_name: str) -> None:
            self.model_name = model_name


if PydanticAIAgent is not None:
    Agent = PydanticAIAgent
else:
    class Agent(Generic[TDeps]):
        def __init__(self, **kwargs: Any) -> None:
            self.kwargs = kwargs

        async def run(self, prompt: str, *, deps: TDeps) -> SimpleNamespace:
            return SimpleNamespace(prompt=prompt, deps=deps, output=None)


if typer_module is not None:
    Typer = typer_module.Typer
else:
    class Typer:
        def __init__(self, **_: Any) -> None:
            self._commands: dict[str, Any] = {}

        def command(self, *_: Any, **__: Any):
            def decorator(func: Any) -> Any:
                self._commands[func.__name__] = func
                return func

            return decorator

        def __call__(self) -> None:
            import sys

            if len(sys.argv) < 2:
                raise RuntimeError("No command provided. Install Typer for the full CLI experience.")
            command = self._commands[sys.argv[1]]
            args = sys.argv[2:]
            kwargs: dict[str, Any] = {}
            if '--debug' in args:
                kwargs['debug'] = True
                args = [arg for arg in args if arg != '--debug']
            if args:
                kwargs['question'] = ' '.join(args)
            command(**kwargs)


if RichConsole is not None:
    Console = RichConsole
else:
    class Console:
        def print(self, value: Any) -> None:
            print(value)


if RichJSON is not None:
    JSON = RichJSON
else:
    class JSON:
        @classmethod
        def from_data(cls, data: Any) -> str:
            return str(data)


if RichPanel is not None:
    Panel = RichPanel
else:
    class Panel:
        def __init__(self, renderable: Any, title: str = "") -> None:
            self.renderable = renderable
            self.title = title

        def __str__(self) -> str:
            prefix = f"[{self.title}]\n" if self.title else ""
            return f"{prefix}{self.renderable}"


if RichTable is not None:
    Table = RichTable
else:
    class Table:
        def __init__(self, title: str = "") -> None:
            self.title = title
            self.columns: list[str] = []
            self.rows: list[tuple[Any, ...]] = []

        def add_column(self, label: str) -> None:
            self.columns.append(label)

        def add_row(self, *values: Any) -> None:
            self.rows.append(values)

        def __str__(self) -> str:
            lines = [self.title] if self.title else []
            if self.columns:
                lines.append(" | ".join(self.columns))
            lines.extend(" | ".join(str(value) for value in row) for row in self.rows)
            return "\n".join(lines)


class SettingsBase:
    env_file = ".env"

    @classmethod
    def from_env(cls):
        env_values: dict[str, str] = {}
        env_path = Path(cls.env_file)
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped or stripped.startswith("#") or "=" not in stripped:
                    continue
                key, value = stripped.split("=", 1)
                env_values[key] = value

        kwargs: dict[str, Any] = {}
        for item in fields(cls):
            env_name = item.name.upper()
            raw: Any = os.getenv(env_name, env_values.get(env_name, item.default))
            annotation = item.type
            origin = get_origin(annotation)
            if annotation is int:
                raw = int(raw)
            elif origin is not None and int in get_args(annotation):
                raw = None if raw is None else int(raw)
            kwargs[item.name] = raw
        return cls(**kwargs)


def parse_frontmatter(text: str) -> tuple[dict[str, Any], str]:
    if text.startswith("---\n"):
        _, rest = text.split("---\n", 1)
        header, _, body = rest.partition("\n---\n")
        metadata: dict[str, Any] = {}
        for line in header.splitlines():
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            metadata[key.strip()] = value.strip()
        return metadata, body.strip()
    return {}, text.strip()


class PdfReader:
    def __init__(self, path: str) -> None:
        text = Path(path).read_text(encoding="utf-8")
        self.pages = [SimpleNamespace(extract_text=lambda text=text: text)]
