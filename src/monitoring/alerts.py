"""Telegram alerts: circuit breaker trips, order rejections, daily P&L report.

Fire-and-forget: an alert failure is logged but NEVER propagates into the
trading path — monitoring must not be able to take down execution.
"""

from __future__ import annotations


async def send_telegram_alert(message: str) -> None:
    """TODO(Phase 7): httpx POST to Telegram Bot API; token/chat_id via
    require_env; try/except with log.error fallback, never raise."""
    raise NotImplementedError("Phase 7 — Telegram alerts")
