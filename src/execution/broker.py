"""AlpacaBrokerClient — the live/paper implementation of AbstractBrokerClient.

paper=True is the constructor default and is overridden EXCLUSIVELY from YAML
(execution.live_mode: true). Never from an env name, branch name, or CLI flag —
one accidental paper=False submits real orders.

Order pipeline, in this exact order (src/execution/CLAUDE.md):
  1. result = risk_manager.validate(intent)   # mandatory, reject -> no order
  2. use result.adjusted_qty                  # NEVER intent.qty
  3. client.submit_order(request)
  4. log.info("order_submitted", extra={order_id, symbol, side, qty, strategy_id})
  5. return order_id
"""

from __future__ import annotations

from typing import Any

from src.shared.interfaces import AbstractBrokerClient


class AlpacaBrokerClient(AbstractBrokerClient):
    """TODO(Phase 4): implement with alpaca-py TradingClient.
    - __init__(api_key, secret_key, paper: bool = True)
    - submit_order: build MarketOrderRequest/LimitOrderRequest from the
      validated intent, wrap the call in try/except APIError with logged
      fallback (never bare except).
    - get_positions/get_account: normalize SDK objects to plain dicts —
      alpaca-py types never leave this module.
    """

    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        raise NotImplementedError("Phase 4 — Alpaca broker client")

    async def submit_order(self, request: Any) -> str:
        raise NotImplementedError("Phase 4")

    async def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError("Phase 4")

    async def get_positions(self) -> list[dict[str, Any]]:
        raise NotImplementedError("Phase 4")

    async def get_account(self) -> dict[str, Any]:
        raise NotImplementedError("Phase 4")
