"""Small terminal styling helpers; files and pipes remain plain text."""

from __future__ import annotations

import copy
import logging
import os
import sys


STYLES = {"dim": "2", "cyan": "36", "green": "32", "yellow": "33", "red": "31"}

# These libraries emit per-request, transfer, cache, or model-loading chatter.
# Keep their diagnostics at WARNING even when NeuralSignal is in DEBUG mode.
QUIET_LOGGERS = (
    "httpx", "httpcore", "urllib3", "requests", "aiohttp",
    "boto3", "botocore", "s3transfer", "fsspec", "s3fs",
    "huggingface_hub", "datasets", "transformers", "filelock",
)


class ProviderNoiseFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno >= logging.WARNING or not any(
            record.name == name or record.name.startswith(name + ".")
            for name in QUIET_LOGGERS
        )


def color(text: str, style: str, stream=None) -> str:
    stream = sys.stdout if stream is None else stream
    if "NO_COLOR" in os.environ or os.environ.get("TERM") == "dumb" or not stream.isatty():
        return text
    return f"\033[{STYLES[style]}m{text}\033[0m"


class ConsoleFormatter(logging.Formatter):
    def __init__(self, stream):
        super().__init__("%(asctime)s %(levelname)-8s %(name)s: %(message)s", datefmt="%H:%M:%S")
        self.stream = stream

    def formatTime(self, record, datefmt=None):
        return color(super().formatTime(record, datefmt), "dim", self.stream)

    def format(self, record):
        # Avoid changing the record seen by file handlers or other consumers.
        styled = copy.copy(record)
        style = "red" if record.levelno >= logging.ERROR else "yellow" if record.levelno >= logging.WARNING else "cyan" if record.levelno >= logging.INFO else "dim"
        styled.levelname = color(f"{record.levelname:<8}", style, self.stream)
        styled.name = color(record.name, "dim", self.stream)
        return super().format(styled)


def configure_logging() -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(ConsoleFormatter(handler.stream))
    # A handler filter also catches children with explicitly enabled INFO/DEBUG.
    handler.addFilter(ProviderNoiseFilter())
    for name in QUIET_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    level = os.environ.get("NEURALSIGNAL_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO), handlers=[handler], force=True)
