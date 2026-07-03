"""Phase 3 — Kelly sizing tests (docs/plan/PHASE_3_RISK.md)."""

from __future__ import annotations

import pytest
from src.risk.kelly import kelly_fraction
from src.shared.exceptions import InvalidKellyFractionError


def test_above_half_kelly_raises() -> None:
    with pytest.raises(InvalidKellyFractionError, match=r"0\.6"):
        kelly_fraction(0.6, 2.0, 1.0, fraction=0.6)


def test_no_edge_returns_zero() -> None:
    # win_rate .3, payoff 1: full kelly = .3 - .7/1 = -.4 -> do not trade
    assert kelly_fraction(0.3, 1.0, 1.0) == 0.0


def test_zero_avg_loss_returns_zero() -> None:
    assert kelly_fraction(0.6, 2.0, 0.0) == 0.0
    assert kelly_fraction(0.6, 0.0, 1.0) == 0.0


def test_quarter_kelly_known_value() -> None:
    # win_rate .6, payoff 2: full = .6 - .4/2 = .4; quarter-Kelly -> .1
    assert kelly_fraction(0.6, 2.0, 1.0, fraction=0.25) == pytest.approx(0.1)
