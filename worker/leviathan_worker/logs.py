"""Structured logging: one JSON object (or one key=value line) per record.

Pass fields with ``extra``: ``log.info("request finished", extra={"request_id": rid})``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from typing import Any, TextIO

# Attributes every LogRecord has; anything else on a record came from ``extra``.
_RECORD_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", None, None))) | {"message", "asctime"}


def _fields(record: logging.LogRecord) -> dict[str, Any]:
    return {
        k: v for k, v in vars(record).items() if k not in _RECORD_ATTRS and not k.startswith("_") and v is not None
    }


def _timestamp(record: logging.LogRecord) -> str:
    return datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(timespec="milliseconds")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": _timestamp(record),
            "level": record.levelname.lower(),
            "logger": record.name,
            "msg": record.getMessage(),
        }
        payload.update(_fields(record))
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, default=str)


class TextFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        extra = " ".join(f"{k}={v}" for k, v in _fields(record).items())
        line = f"{_timestamp(record)} {record.levelname:<7} {record.name}: {record.getMessage()}"
        if extra:
            line = f"{line} {extra}"
        if record.exc_info:
            line = f"{line}\n{self.formatException(record.exc_info)}"
        return line


#: Libraries whose debug output is noise for worker operators.
_QUIET_LIBRARIES = ("asyncio", "grpc")


def configure(level: str = "info", fmt: str = "json", stream: TextIO | None = None) -> None:
    """Route the root logger to ``stream`` (stderr by default) with the chosen format."""
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(JsonFormatter() if fmt == "json" else TextFormatter())
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
    for name in _QUIET_LIBRARIES:
        logging.getLogger(name).setLevel(max(root.level, logging.INFO))
