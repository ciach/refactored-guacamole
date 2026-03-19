import logging

try:
    from rich.logging import RichHandler
except ImportError:  # pragma: no cover
    RichHandler = None


def configure_logging(level: int = logging.INFO) -> None:
    handlers = [RichHandler(rich_tracebacks=True)] if RichHandler is not None else None
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=handlers,
    )
