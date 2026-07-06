"""Bootstrap confidence interval for the Sharpe ratio.

Answers the decision question the point WFO Sharpe cannot: "is the measured
OOS Sharpe reliably below the gate, or is the gate inside the noise band?"
by resampling the strategy's REALIZED daily returns and recomputing the
Sharpe on each resample.

Port provenance: Fin v1 backtest/monte_carlo.py used an IID bootstrap
(np.random.choice). Per the port protocol the method is RE-DERIVED and
CORRECTED — IID resampling destroys volatility clustering and serial
dependence, which understates the sampling variance of the Sharpe (falsely
tight intervals). This uses the STATIONARY BOOTSTRAP (Politis & Romano 1994):
geometric-length blocks wrapped circularly, which preserve serial dependence.
IID is the special case mean_block=1. The Sharpe is Fin2's single definition
(metrics.sharpe_ratio, ddof=1, rf=0) — never a second formula.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.backtest.metrics import sharpe_ratio

_PERCENTILES = (5, 25, 50, 75, 95)


@dataclass(frozen=True)
class SharpeInterval:
    """Bootstrap sampling distribution of a strategy's annualized Sharpe."""

    point: float  # Sharpe of the observed return series
    percentiles: dict[int, float]  # sampling-distribution percentiles
    p_at_least: float  # fraction of resamples with Sharpe >= threshold
    threshold: float
    n_resamples: int
    mean_block: float
    n_returns: int


def _stationary_resample_index(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano index sequence of length n: concatenate blocks that
    start at a uniform index and run for Geometric(p) steps (circularly),
    until length n is reached. mean_block = 1/p; p=1 -> all blocks length 1
    (pure IID bootstrap)."""
    pieces: list[np.ndarray] = []
    total = 0
    while total < n:
        start = int(rng.integers(n))
        length = int(rng.geometric(p))  # support {1, 2, ...}, mean 1/p
        pieces.append((start + np.arange(length)) % n)
        total += length
    return np.concatenate(pieces)[:n]


def bootstrap_sharpe(
    returns: pd.Series,
    periods_per_year: int,
    threshold: float,
    n_resamples: int = 5000,
    mean_block: float = 10.0,
    seed: int = 0,
) -> SharpeInterval:
    """Stationary-bootstrap sampling distribution of the annualized Sharpe.

    returns: realized per-period (daily) strategy returns. mean_block: expected
    bootstrap block length in periods (>=1; carries the serial dependence —
    ~2-3 weeks for daily bars). Degenerate input (<2 returns or zero variance)
    yields an all-zero interval rather than raising.
    """
    if mean_block < 1.0:
        raise ValueError(f"mean_block must be >= 1, got {mean_block}")
    clean = returns.dropna().to_numpy(dtype="float64")
    point = sharpe_ratio(pd.Series(clean), periods_per_year)
    n = len(clean)
    if n < 2 or clean.std(ddof=1) == 0.0:
        return SharpeInterval(
            point=point,
            percentiles=dict.fromkeys(_PERCENTILES, 0.0),
            p_at_least=0.0,
            threshold=threshold,
            n_resamples=n_resamples,
            mean_block=mean_block,
            n_returns=n,
        )

    rng = np.random.default_rng(seed)
    p = 1.0 / mean_block
    sharpes = np.empty(n_resamples, dtype="float64")
    for i in range(n_resamples):
        idx = _stationary_resample_index(n, p, rng)
        sharpes[i] = sharpe_ratio(pd.Series(clean[idx]), periods_per_year)

    return SharpeInterval(
        point=point,
        percentiles={q: float(np.percentile(sharpes, q)) for q in _PERCENTILES},
        p_at_least=float(np.mean(sharpes >= threshold)),
        threshold=threshold,
        n_resamples=n_resamples,
        mean_block=mean_block,
        n_returns=n,
    )
