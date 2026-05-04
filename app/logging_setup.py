"""Логирование. Простой stdlib без structlog — гарантированно идёт в stdout Railway.

Раньше использовался structlog с PrintLoggerFactory, но он перехватывал стандартный
поток и наши `logger.info("startup_done", extra={...})` не попадали в Deploy Logs.
Теперь просто stdlib + кастомный Formatter, который дописывает `extra` ключи в
конец строки `key=value`.
"""

from __future__ import annotations

import logging
import sys


_RESERVED = frozenset(
    {
        "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
        "module", "exc_info", "exc_text", "stack_info", "lineno", "funcName",
        "created", "msecs", "relativeCreated", "thread", "threadName",
        "processName", "process", "message", "asctime", "taskName",
    }
)


class _ExtraFormatter(logging.Formatter):
    """Формат: `LEVEL [logger] message key=value key=value`. Простой и парсится глазами."""

    def format(self, record: logging.LogRecord) -> str:
        base = f"{record.levelname:<5} [{record.name}] {record.getMessage()}"
        extras = []
        for key, value in record.__dict__.items():
            if key in _RESERVED or key.startswith("_"):
                continue
            extras.append(f"{key}={value}")
        if extras:
            base += " " + " ".join(extras)
        if record.exc_info:
            base += "\n" + self.formatException(record.exc_info)
        return base


def configure_logging(level: str = "INFO") -> None:
    """Настроить root logger так, чтобы наши `app.*` логи шли в stdout Railway.

    `force=True` важно — uvicorn по дороге уже мог поставить свой StreamHandler,
    нам нужно его перебить, иначе наши логи не покажутся в Deploy Logs.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(_ExtraFormatter())
    logging.basicConfig(level=log_level, handlers=[handler], force=True)

    # uvicorn и MCP SDK по умолчанию INFO; обеспечиваем что они тоже выводятся
    # через наш handler без duplication.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access", "mcp"):
        lg = logging.getLogger(name)
        lg.setLevel(log_level)
        lg.propagate = True
        lg.handlers = []  # уберём собственные handlers, чтобы не дублировались
