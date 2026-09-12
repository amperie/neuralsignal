"""Small terminal styling helpers; files and pipes remain plain text."""

from __future__ import annotations

import copy
import logging
import os
import sys


STYLES = {"dim": "2", "cyan": "36", "green": "32", "yellow": "33", "red": "31"}


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
    level = os.environ.get("NEURALSIGNAL_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(level=getattr(logging, level, logging.INFO), handlers=[handler], force=True)
