"""Compatibility helpers for optional third-party dependencies."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from os import getenv
from typing import Any, Callable, TypeVar

T = TypeVar("T")

try:  # pragma: no cover - exercised when dependencies are installed.
    from pydantic import BaseModel as PydanticBaseModel
    from pydantic import Field as PydanticField
except Exception:  # pragma: no cover - deterministic fallback used in tests.
    PydanticBaseModel = None
    PydanticField = None

try:  # pragma: no cover - exercised when dependencies are installed.
    from pydantic_settings import BaseSettings as PydanticBaseSettings
    from pydantic_settings import SettingsConfigDict
except Exception:  # pragma: no cover
    PydanticBaseSettings = None
    SettingsConfigDict = dict


def Field(*, default: Any = MISSING, default_factory: Callable[[], Any] | None = None) -> Any:
    if PydanticField is not None:
        if default_factory is not None:
            return PydanticField(default_factory=default_factory)
        if default is MISSING:
            return PydanticField(...)
        return PydanticField(default)
    kwargs: dict[str, Any] = {}
    if default is not MISSING:
        kwargs["default"] = default
    if default_factory is not None:
        kwargs["default_factory"] = default_factory
    return field(**kwargs)


if PydanticBaseModel is not None:  # pragma: no cover
    class BaseModel(PydanticBaseModel):
        """Thin alias so application code can stay uniform."""

        pass
else:
    class BaseModel:
        """Very small dataclass-like fallback for offline testing."""

        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)
            dataclass(cls)

        def model_dump(self) -> dict[str, Any]:
            if is_dataclass(self):
                return asdict(self)
            return dict(self.__dict__)


if PydanticBaseSettings is not None:  # pragma: no cover
    BaseSettings = PydanticBaseSettings
else:
    class BaseSettings:
        model_config: dict[str, Any] = {}

        def __init__(self, **overrides: Any) -> None:
            for info in fields(self):
                env_name = info.name.upper()
                if info.name in overrides:
                    value = overrides[info.name]
                else:
                    value = getenv(env_name, MISSING)
                    if value is MISSING:
                        if info.default is not MISSING:
                            value = info.default
                        elif info.default_factory is not MISSING:  # type: ignore[comparison-overlap]
                            value = info.default_factory()  # type: ignore[misc]
                        else:
                            raise ValueError(f"Missing setting: {info.name}")
                setattr(self, info.name, value)

        def __init_subclass__(cls, **kwargs: Any) -> None:
            super().__init_subclass__(**kwargs)
            dataclass(cls)
