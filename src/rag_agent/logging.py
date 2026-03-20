from __future__ import annotations

import logging


def configure_logging() -> None:
    try:  # pragma: no cover - only active when Rich is installed.
        from rich.logging import RichHandler

        logging.basicConfig(
            level=logging.INFO,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)],
        )
    except Exception:
        logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
