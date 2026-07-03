"""Phase 4 — Alpaca broker tests, mocked SDK, zero network (PHASE_4_EXECUTION.md)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
import src.execution.broker as broker_mod
from alpaca.common.exceptions import APIError
from src.execution.broker import AlpacaBrokerClient, build_order_request, execute_intent
from src.shared.exceptions import OrderRejectedError
from src.strategies.base import OrderIntent


class _BoomAPIError(APIError):
    def __init__(self) -> None:  # bypass APIError's response-object plumbing
        Exception.__init__(self, "boom")


class FakeTradingClient:
    last_init: ClassVar[dict[str, Any]] = {}

    def __init__(self, api_key: str, secret_key: str, paper: bool = True) -> None:
        FakeTradingClient.last_init = {"paper": paper}
        self.submitted: list[Any] = []
        self.fail = False

    def submit_order(self, order_data: Any) -> Any:
        if self.fail:
            raise _BoomAPIError()
        self.submitted.append(order_data)
        return SimpleNamespace(id="oid-1")

    def get_all_positions(self) -> list[Any]:
        return [
            SimpleNamespace(symbol="AAPL", qty="10", avg_entry_price="150.0", market_value="1600")
        ]

    def get_account(self) -> Any:
        return SimpleNamespace(equity="50000", cash="20000")


@pytest.fixture(autouse=True)
def _patch_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(broker_mod, "TradingClient", FakeTradingClient)


def _intent(qty: float = 50.0) -> OrderIntent:
    return OrderIntent(
        symbol="AAPL",
        side="buy",
        qty=qty,
        signal_strength=1.0,
        strategy_id="test",
        reference_price=100.0,
    )


def test_paper_defaults_true() -> None:
    AlpacaBrokerClient("k", "s")
    assert FakeTradingClient.last_init["paper"] is True


def test_paper_flag_forwarded_to_sdk() -> None:
    AlpacaBrokerClient("k", "s", paper=False)  # composition passes not live_mode
    assert FakeTradingClient.last_init["paper"] is False


async def test_submit_success_returns_id() -> None:
    client = AlpacaBrokerClient("k", "s")
    order_id = await client.submit_order(build_order_request(_intent(), 20.0))
    assert order_id == "oid-1"


async def test_api_error_wrapped_in_order_rejected() -> None:
    client = AlpacaBrokerClient("k", "s")
    client._client.fail = True
    with pytest.raises(OrderRejectedError, match="boom"):
        await client.submit_order(build_order_request(_intent(), 20.0))


async def test_positions_and_account_are_plain_dicts() -> None:
    client = AlpacaBrokerClient("k", "s")
    positions = await client.get_positions()
    assert positions == [
        {"symbol": "AAPL", "qty": 10.0, "avg_entry_price": 150.0, "market_value": 1600.0}
    ]
    assert await client.get_account() == {"equity": 50000.0, "cash": 20000.0}


def test_build_request_uses_adjusted_qty_param() -> None:
    request = build_order_request(_intent(qty=50.0), adjusted_qty=20.0)
    assert float(request.qty) == 20.0  # never intent.qty


class _StubRisk:
    def __init__(self, approved: bool, adjusted_qty: float) -> None:
        self._result = SimpleNamespace(approved=approved, adjusted_qty=adjusted_qty, reason="stub")
        self.calls: list[OrderIntent] = []

    def validate(self, intent: OrderIntent) -> Any:
        self.calls.append(intent)
        return self._result


class _StubBroker:
    def __init__(self) -> None:
        self.submitted: list[Any] = []

    async def submit_order(self, request: Any) -> str:
        self.submitted.append(request)
        return "oid-9"


async def test_execute_intent_never_submits_rejected() -> None:
    risk, broker = _StubRisk(approved=False, adjusted_qty=0.0), _StubBroker()
    assert await execute_intent(broker, risk, _intent()) is None  # type: ignore[arg-type]
    assert risk.calls  # validate WAS consulted
    assert broker.submitted == []  # and nothing reached the broker


async def test_execute_intent_submits_adjusted_qty() -> None:
    risk, broker = _StubRisk(approved=True, adjusted_qty=20.0), _StubBroker()
    order_id = await execute_intent(broker, risk, _intent(qty=50.0))  # type: ignore[arg-type]
    assert order_id == "oid-9"
    assert float(broker.submitted[0].qty) == 20.0
