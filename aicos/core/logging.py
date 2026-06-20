"""
Structured logging for AI-COS.

Every log line in production is a JSON object. In development (AICOS_LOG_JSON=false),
lines are human-readable coloured text. Both modes automatically include the
current X-Request-ID so any log aggregator can correlate a full request trace.

Usage:
    from aicos.core.logging import get_logger
    log = get_logger(__name__)
    log.info("Cache hit", extra={"hit_type": "semantic", "latency_ms": 1.5})
"""

from __future__ import annotations

import logging
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

import orjson

_request_id: ContextVar[str] = ContextVar("request_id", default="")
_session_id: ContextVar[str] = ContextVar("session_id", default="")

_BUILTIN_ATTRS = frozenset({
    "args", "asctime", "created", "exc_info", "exc_text", "filename",
    "funcName", "levelname", "levelno", "lineno", "message", "module",
    "msecs", "msg", "name", "pathname", "process", "processName",
    "relativeCreated", "stack_info", "thread", "threadName", "taskName",
})


def get_request_id() -> str:
    return _request_id.get()


def set_request_id(value: str) -> None:
    _request_id.set(value)


def set_session_id(value: str) -> None:
    _session_id.set(value)


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        data: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.message,
        }
        if rid := _request_id.get(""):
            data["request_id"] = rid
        if sid := _session_id.get(""):
            data["session_id"] = sid
        if record.exc_info:
            data["exc"] = self.formatException(record.exc_info)
        for key, val in record.__dict__.items():
            if key not in _BUILTIN_ATTRS and not key.startswith("_"):
                try:
                    orjson.dumps(val)
                    data[key] = val
                except (TypeError, ValueError):
                    data[key] = str(val)
        return orjson.dumps(data).decode()


class _TextFormatter(logging.Formatter):
    _COLORS = {
        "DEBUG": "\033[36m",
        "INFO": "\033[32m",
        "WARNING": "\033[33m",
        "ERROR": "\033[31m",
        "CRITICAL": "\033[35m",
    }
    _RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        color = self._COLORS.get(record.levelname, "")
        rid = _request_id.get("")
        rid_part = f" [{rid[:8]}]" if rid else ""
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        return (
            f"{color}{ts} {record.levelname:8}{self._RESET}"
            f"{rid_part} {record.name}: {record.getMessage()}"
        )


def configure_logging(level: str = "INFO", json_format: bool = False) -> None:
    """Configure the aicos logger hierarchy. Call once at application startup."""
    root = logging.getLogger("aicos")
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_JsonFormatter() if json_format else _TextFormatter())
    root.addHandler(handler)


def get_logger(name: str) -> logging.Logger:
    """Return a child logger under the aicos namespace."""
    qualified = name if name.startswith("aicos") else f"aicos.{name}"
    return logging.getLogger(qualified)
