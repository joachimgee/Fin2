"""JSON structured logging for the whole system.

Convention everywhere (root CLAUDE.md): log.info("event_name", extra={...}) —
snake_case event name as the message, ALL variable data in extra={}.
Never f-strings as messages: they are unqueryable blobs in log aggregators.
"""

from __future__ import annotations


def setup_logging(level: str = "INFO") -> None:
    """TODO(Phase 7): JSON formatter (timestamp, level, logger, event, extra
    fields flattened), stdout handler. Called once from each entrypoint."""
    raise NotImplementedError("Phase 7 — logging setup")
