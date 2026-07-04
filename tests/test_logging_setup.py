"""Phase 7 — JSON logging tests (docs/plan/PHASE_7_MONITORING.md)."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator

import pytest
from src.monitoring.logging_setup import _HANDLER_NAME, setup_logging


@pytest.fixture(autouse=True)
def _clean_root() -> Iterator[None]:
    root = logging.getLogger()
    previous_level = root.level
    yield
    root.handlers = [h for h in root.handlers if h.get_name() != _HANDLER_NAME]
    root.setLevel(previous_level)


def _last_line(capsys: pytest.CaptureFixture[str]) -> dict:
    lines = [line for line in capsys.readouterr().out.splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_each_line_is_valid_json(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO")
    logging.getLogger("finbot.test").info("order_filled", extra={"symbol": "AAPL"})
    payload = _last_line(capsys)
    assert payload["event"] == "order_filled"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "finbot.test"
    assert "ts" in payload


def test_extra_fields_at_top_level(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO")
    logging.getLogger("x").info("evt", extra={"symbol": "AAPL", "qty": 5})
    payload = _last_line(capsys)
    assert payload["symbol"] == "AAPL"
    assert payload["qty"] == 5


def test_payload_key_collision_prefixed(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO")
    logging.getLogger("x").info("real_event", extra={"event": "clash"})
    payload = _last_line(capsys)
    assert payload["event"] == "real_event"  # the message wins the key
    assert payload["x_event"] == "clash"  # the extra survives, prefixed


def test_double_setup_no_duplicate_handlers(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO")
    setup_logging("INFO")
    handlers = [h for h in logging.getLogger().handlers if h.get_name() == _HANDLER_NAME]
    assert len(handlers) == 1
    logging.getLogger("x").info("once")
    assert len([line for line in capsys.readouterr().out.splitlines() if "once" in line]) == 1


def test_level_from_config(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("WARNING")
    logging.getLogger("x").info("suppressed_event")
    logging.getLogger("x").warning("visible_event")
    out = capsys.readouterr().out
    assert "suppressed_event" not in out
    assert "visible_event" in out


def test_httpx_logger_silenced_to_warning() -> None:
    """httpx logs full URLs (apiKey included) at INFO — must never reach logs."""
    setup_logging("INFO")
    assert logging.getLogger("httpx").level == logging.WARNING


def test_non_serializable_extra_stringified(capsys: pytest.CaptureFixture[str]) -> None:
    setup_logging("INFO")
    logging.getLogger("x").info("evt", extra={"when": object()})
    assert isinstance(_last_line(capsys)["when"], str)  # default=str, never a crash
