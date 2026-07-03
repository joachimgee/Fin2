"""Telegram alerts: circuit breaker trips, broker rejections, daily report.

Fire-and-forget: an alert failure is logged but NEVER propagates into the
trading path — monitoring must not be able to take down execution.

Credentials are parameters, resolved via require_env at the composition
root; this module holds no secrets.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable

import httpx

log = logging.getLogger(__name__)

_TELEGRAM_API = "https://api.telegram.org"


async def send_telegram_alert(
    message: str,
    bot_token: str,
    chat_id: str,
    timeout_s: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> None:
    try:
        async with httpx.AsyncClient(
            base_url=_TELEGRAM_API, timeout=timeout_s, transport=transport
        ) as client:
            response = await client.post(
                f"/bot{bot_token}/sendMessage", json={"chat_id": chat_id, "text": message}
            )
            response.raise_for_status()
    except httpx.HTTPError as exc:  # transport, timeout, and HTTP-status errors
        log.error("telegram_alert_failed", extra={"error": str(exc)})


def make_alert_dispatcher(
    bot_token: str,
    chat_id: str,
    timeout_s: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> Callable[[str], None]:
    """Sync fire-and-forget callable — the shape CircuitBreaker.on_trip and
    the StreamManager alert hook expect. Schedules on the running loop when
    there is one; runs to completion otherwise."""

    def dispatch(message: str) -> None:
        coro = send_telegram_alert(message, bot_token, chat_id, timeout_s, transport)
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(coro)
            return
        task = loop.create_task(coro)
        task.add_done_callback(lambda t: t.exception())  # retrieve, never raise

    return dispatch
