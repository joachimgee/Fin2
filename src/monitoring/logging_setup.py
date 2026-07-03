"""JSON structured logging for the whole system.

Convention everywhere (root CLAUDE.md): log.info("event_name", extra={...}) —
snake_case event name as the message, ALL variable data in extra={}.
Never f-strings as messages: they are unqueryable blobs in log aggregators.

One JSON object per line on stdout: ts, level, logger, event, plus every
extra key flattened at top level. Extra keys that collide with the payload's
own keys get an "x_" prefix instead of clobbering them.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

_HANDLER_NAME = "finbot-json"
_PAYLOAD_KEYS = {"ts", "level", "logger", "event"}
# Everything a bare LogRecord carries — plumbing, not user data.
_RESERVED_ATTRS = frozenset(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {
    "message",
    "asctime",
    "taskName",
}


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for key, value in record.__dict__.items():
            if key in _RESERVED_ATTRS:
                continue
            payload[f"x_{key}" if key in _PAYLOAD_KEYS else key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(level: str = "INFO") -> None:
    """Install the JSON stdout handler on the root logger. Called once per
    process by the entrypoint; safe to call again (no duplicate handlers)."""
    root = logging.getLogger()
    root.setLevel(level.upper())
    if any(h.get_name() == _HANDLER_NAME for h in root.handlers):
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.set_name(_HANDLER_NAME)
    handler.setFormatter(_JsonFormatter())
    root.addHandler(handler)
