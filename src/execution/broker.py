"""AlpacaBrokerClient — the live/paper implementation of AbstractBrokerClient.

paper=True is the constructor default and is overridden EXCLUSIVELY from YAML
(execution.live_mode: true) at the composition root. Never from an env name,
branch name, or CLI flag — one accidental paper=False submits real orders.

execute_intent() is THE order pipeline: it is the only public path from an
OrderIntent to the broker, and it physically cannot skip validate().
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from alpaca.common.exceptions import APIError
from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

from src.risk import OrderIntent, RiskManager
from src.shared.exceptions import OrderRejectedError
from src.shared.interfaces import AbstractBrokerClient

log = logging.getLogger(__name__)


def _field(obj: Any, name: str) -> Any:
    """SDK calls return model objects or raw dicts depending on client config —
    normalize field access so plain dicts are what leaves this module."""
    return obj[name] if isinstance(obj, dict) else getattr(obj, name)


def build_order_request(intent: OrderIntent, adjusted_qty: float) -> Any:
    """Build the SDK request from the VALIDATED quantity. Taking adjusted_qty
    as an explicit parameter makes forgetting result.adjusted_qty impossible."""
    side = OrderSide.BUY if intent.side == "buy" else OrderSide.SELL
    if intent.order_type == "limit":
        return LimitOrderRequest(
            symbol=intent.symbol,
            qty=adjusted_qty,
            side=side,
            time_in_force=TimeInForce.DAY,
            limit_price=intent.limit_price,
        )
    return MarketOrderRequest(
        symbol=intent.symbol, qty=adjusted_qty, side=side, time_in_force=TimeInForce.DAY
    )


async def execute_intent(
    broker: AbstractBrokerClient, risk_manager: RiskManager, intent: OrderIntent
) -> str | None:
    """The order pipeline in this exact order (src/execution/CLAUDE.md).

    Returns the order_id, or None when risk rejected (already logged by risk).
    """
    result = risk_manager.validate(intent)  # 1 — mandatory, no exceptions
    if not result.approved:
        return None
    request = build_order_request(intent, result.adjusted_qty)  # 2 — never intent.qty
    order_id = await broker.submit_order(request)  # 3
    log.info(
        "order_submitted",  # 4
        extra={
            "order_id": order_id,
            "symbol": intent.symbol,
            "side": intent.side,
            "qty": result.adjusted_qty,
            "strategy_id": intent.strategy_id,
        },
    )
    return order_id  # 5


class AlpacaBrokerClient(AbstractBrokerClient):
    """Sole owner of the alpaca-py TradingClient. SDK types never leave this
    module — positions/account cross the boundary as plain dicts.

    TradingClient is synchronous; calls run in a worker thread so the event
    loop (stream callbacks) never blocks.
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        self._client = TradingClient(api_key, secret_key, paper=paper)

    async def submit_order(self, request: Any) -> str:
        try:
            order = await asyncio.to_thread(self._client.submit_order, order_data=request)
        except APIError as exc:
            log.error(
                "order_submit_failed",
                extra={"symbol": getattr(request, "symbol", None), "error": str(exc)},
            )
            raise OrderRejectedError(f"broker rejected order: {exc}") from exc
        return str(_field(order, "id"))

    async def cancel_order(self, order_id: str) -> None:
        await asyncio.to_thread(self._client.cancel_order_by_id, order_id)

    async def get_positions(self) -> list[dict[str, Any]]:
        positions = await asyncio.to_thread(self._client.get_all_positions)
        return [
            {
                "symbol": _field(p, "symbol"),
                "qty": float(_field(p, "qty")),
                "avg_entry_price": float(_field(p, "avg_entry_price")),
                "market_value": float(_field(p, "market_value")),
            }
            for p in positions
        ]

    async def get_account(self) -> dict[str, Any]:
        account = await asyncio.to_thread(self._client.get_account)
        return {
            "equity": float(_field(account, "equity")),
            "cash": float(_field(account, "cash")),
        }
