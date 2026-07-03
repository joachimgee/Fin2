"""Phase 0 — config loading tests (docs/plan/PHASE_0_FOUNDATIONS.md)."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
import yaml
from src.shared.config import load_config, require_env
from src.shared.exceptions import ConfigError

REPO_ROOT = Path(__file__).resolve().parent.parent


def _variant(tmp_path: Path, mutate: Callable[[dict[str, Any]], None]) -> Path:
    """Copy config/base.yaml, apply one mutation, write to tmp, return path."""
    cfg = yaml.safe_load((REPO_ROOT / "config" / "base.yaml").read_text(encoding="utf-8"))
    mutate(cfg)
    out = tmp_path / "variant.yaml"
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out


def test_require_env_missing_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FINBOT_TEST_VAR", raising=False)
    with pytest.raises(ConfigError, match="FINBOT_TEST_VAR"):
        require_env("FINBOT_TEST_VAR")


def test_require_env_returns_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FINBOT_TEST_VAR", "abc")
    assert require_env("FINBOT_TEST_VAR") == "abc"


def test_load_base_yaml_valid() -> None:
    cfg = load_config(REPO_ROOT / "config" / "base.yaml")
    assert cfg["execution"]["live_mode"] is False
    assert cfg["risk"]["kelly_fraction"] <= 0.50
    assert cfg["data"]["data_source"] == "polygon"


def test_unknown_key_raises(tmp_path: Path) -> None:
    # typo'd risk cap must fail loudly, never be silently ignored
    path = _variant(tmp_path, lambda c: c["risk"].update({"max_risk_per_trad_pct": 0.02}))
    with pytest.raises(ConfigError, match="max_risk_per_trad_pct"):
        load_config(path)


def test_kelly_above_half_raises(tmp_path: Path) -> None:
    path = _variant(tmp_path, lambda c: c["risk"].update({"kelly_fraction": 0.6}))
    with pytest.raises(ConfigError, match="kelly_fraction"):
        load_config(path)


def test_risk_per_trade_above_5pct_raises(tmp_path: Path) -> None:
    path = _variant(tmp_path, lambda c: c["risk"].update({"max_risk_per_trade_pct": 0.06}))
    with pytest.raises(ConfigError, match="max_risk_per_trade_pct"):
        load_config(path)


def test_live_mode_defaults_false(tmp_path: Path) -> None:
    path = _variant(tmp_path, lambda c: c["execution"].pop("live_mode"))
    assert load_config(path)["execution"]["live_mode"] is False


def test_live_mode_non_bool_raises(tmp_path: Path) -> None:
    path = _variant(tmp_path, lambda c: c["execution"].update({"live_mode": "yes"}))
    with pytest.raises(ConfigError, match="live_mode"):
        load_config(path)
