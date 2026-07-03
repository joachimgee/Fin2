"""Base class for all signal generators.

Satisfies the shared.interfaces.SignalGenerator Protocol structurally —
strategies depend on the Protocol, never on this module.
"""

from __future__ import annotations

import abc
from typing import Any


class AbstractSignalGenerator(abc.ABC):
    """Invariants (src/signals/CLAUDE.md):
    1. generate() always returns float in [-1.0, 1.0] — always via _clamp().
    2. Every feature at time T uses only data from times <= T-1 (.shift(1)).
    3. generate() is deterministic and < 5 ms. No network calls, no DB reads.
    """

    @abc.abstractmethod
    def generate(self, features: Any) -> float:
        """Return signal strength in [-1.0, 1.0]. Implementations MUST end with
        `return self._clamp(raw)` — never return a raw model output."""
        raise NotImplementedError

    @staticmethod
    def _clamp(value: float) -> float:
        """Clamp to the output contract [-1.0, 1.0]."""
        return max(-1.0, min(1.0, value))
