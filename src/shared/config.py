"""Configuration loading.

Rules (root CLAUDE.md <coding_rules>):
  - All numeric parameters come from YAML (config/*.yaml).
  - All credentials come from environment variables via require_env().
  - Nothing is ever hardcoded in source files.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

from src.shared.exceptions import ConfigError

# Mirrors config/base.yaml exactly. A key absent from this schema is a typo
# by definition — load_config() rejects it instead of silently ignoring it
# (a silently dropped risk cap would trade without that cap).
_SCHEMA: dict[str, set[str]] = {
    "execution": {"live_mode", "broker"},
    "data": {"data_source", "db_path", "timeframe"},
    "risk": {
        "kelly_fraction",
        "max_risk_per_trade_pct",
        "max_position_pct",
        "max_total_exposure_pct",
        "max_sector_exposure_pct",
        "max_correlation",
        "circuit_breakers",
    },
    "signals": {"features", "training"},
    "wfo": {
        "min_windows",
        "min_wfe",
        "min_oos_trades",
        "min_oos_sharpe",
        "max_oos_drawdown_pct",
    },
    "stream": {"reconnect_backoff_initial_s", "reconnect_backoff_cap_s"},
    "llm": {"model", "max_tokens", "cache_ttl_s"},
    "monitoring": {"log_level", "daily_report_time_et"},
}
_CIRCUIT_BREAKER_KEYS: set[str] = {"daily_loss_pct", "max_drawdown_pct", "consecutive_losses"}
_FEATURE_KEYS: set[str] = {
    "ret_windows",
    "vol_short_window",
    "vol_long_window",
    "parkinson_window",
    "gk_window",
    "sma_window",
    "ema_window",
    "ma_fast_window",
    "ma_slow_window",
    "macd_fast",
    "macd_slow",
    "macd_signal",
    "zscore_window",
    "bb_window",
    "bb_std",
    "cci_window",
    "rsi_window",
    "stoch_window",
    "stoch_smooth",
    "roc_window",
    "mfi_window",
    "volume_window",
}
_TRAINING_KEYS: set[str] = {
    "label_horizon_bars",
    "train_frac",
    "valid_frac",
    "n_estimators",
    "learning_rate",
    "num_leaves",
    "hmm_states",
}

# Hard ceilings (src/risk/CLAUDE.md <warn_about>). Values still come from the
# YAML — these only bound what the YAML may ask for.
_MAX_KELLY_FRACTION = 0.50
_MAX_RISK_PER_TRADE_PCT = 0.05


def require_env(name: str) -> str:
    """Return the env var value or raise ConfigError. Never returns a default.

    Fail-fast at startup: a missing credential must stop the process before
    any component initializes, not surface as a broker 401 mid-session.
    """
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"required environment variable not set: {name}")
    return value


def _check_nested(path: Path, where: str, mapping: Any, allowed: set[str]) -> None:
    if not isinstance(mapping, dict):
        raise ConfigError(f"{path}: {where} must be a mapping")
    for key in mapping:
        if key not in allowed:
            raise ConfigError(f"{path}: unknown key {where}.{key}")


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a YAML config file into a plain dict.

    Rejects unknown sections/keys, enforces the risk hard caps, and applies
    the single safety default: execution.live_mode absent -> False (paper).
    """
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{path}: top level must be a mapping of sections")

    for section, content in raw.items():
        allowed = _SCHEMA.get(section)
        if allowed is None:
            raise ConfigError(f"{path}: unknown config section: {section!r}")
        if not isinstance(content, dict):
            raise ConfigError(f"{path}: section {section!r} must be a mapping")
        for key in content:
            if key not in allowed:
                raise ConfigError(f"{path}: unknown key {section}.{key}")

    risk = raw.get("risk", {})
    _check_nested(
        path, "risk.circuit_breakers", risk.get("circuit_breakers", {}), _CIRCUIT_BREAKER_KEYS
    )
    signals = raw.get("signals", {})
    _check_nested(path, "signals.features", signals.get("features", {}), _FEATURE_KEYS)
    _check_nested(path, "signals.training", signals.get("training", {}), _TRAINING_KEYS)

    kelly = risk.get("kelly_fraction")
    if kelly is not None and kelly > _MAX_KELLY_FRACTION:
        raise ConfigError(
            f"{path}: risk.kelly_fraction={kelly} exceeds the hard cap "
            f"{_MAX_KELLY_FRACTION} (half-Kelly)"
        )
    per_trade = risk.get("max_risk_per_trade_pct")
    if per_trade is not None and per_trade > _MAX_RISK_PER_TRADE_PCT:
        raise ConfigError(
            f"{path}: risk.max_risk_per_trade_pct={per_trade} exceeds the hard cap "
            f"{_MAX_RISK_PER_TRADE_PCT}"
        )

    execution = raw.setdefault("execution", {})
    live_mode = execution.setdefault("live_mode", False)
    if not isinstance(live_mode, bool):
        raise ConfigError(f"{path}: execution.live_mode must be a boolean, got {live_mode!r}")

    return raw
