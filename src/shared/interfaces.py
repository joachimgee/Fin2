"""Cross-module abstract interfaces (Dependency Inversion boundary).

These live in shared/ — NOT in execution/ or signals/ — because the dependency
graph forbids strategies/ and backtest/ from importing execution/, yet both need
to program against the broker abstraction. Same reasoning for SignalGenerator:
strategies consume signal generators by injection without importing signals/.

See ARCHITECTURE.md ADR-001 and ADR-002.
"""

from __future__ import annotations

import abc
from typing import Any, Protocol, runtime_checkable


class AbstractBrokerClient(abc.ABC):
    """Broker abstraction every order-touching component depends on.

    Implementations:
      - execution.broker.AlpacaBrokerClient  (live/paper — sole alpaca-py user)
      - backtest engine's simulated broker   (fills from historical bars)

    All methods are async — all I/O in this codebase is async (root CLAUDE.md).
    """

    @abc.abstractmethod
    async def submit_order(self, request: Any) -> str:
        """Submit an order, return the broker order_id.

        Callers MUST have passed the intent through RiskManager.validate()
        first and MUST use the risk-adjusted quantity. This is enforced at the
        call site in execution/, not here (shared/ knows nothing about risk/).
        """
        raise NotImplementedError

    @abc.abstractmethod
    async def cancel_order(self, order_id: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    async def get_positions(self) -> list[dict[str, Any]]:
        """Current open positions, normalized to plain dicts (no SDK types leak out)."""
        raise NotImplementedError

    @abc.abstractmethod
    async def get_account(self) -> dict[str, Any]:
        """Account snapshot; must include an "equity" key (str or float)."""
        raise NotImplementedError


@runtime_checkable
class SignalGenerator(Protocol):
    """Structural type for signal generators (src/signals/).

    Strategies receive instances via constructor injection and only rely on
    this Protocol — they never import from src/signals/ directly.

    Contract (src/signals/CLAUDE.md):
      - generate() returns a float clamped to [-1.0, 1.0]
      - deterministic, < 5 ms, no network calls, no DB reads
    """

    def generate(self, features: Any) -> float: ...
