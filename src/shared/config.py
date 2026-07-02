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

from src.shared.exceptions import ConfigError


def require_env(name: str) -> str:
    """Return the env var value or raise ConfigError. Never returns a default.

    Fail-fast at startup: a missing credential must stop the process before
    any component initializes, not surface as a broker 401 mid-session.
    """
    value = os.environ.get(name)
    if not value:
        raise ConfigError(f"required environment variable not set: {name}")
    return value


def load_config(path: Path) -> dict[str, Any]:
    """Load and validate a YAML config file into a plain dict.

    TODO(Phase 0): implement with yaml.safe_load + schema validation.
      - Reject unknown keys (typo in a risk cap must fail loudly, not be ignored).
      - Validate hard bounds here: kelly fraction <= 0.50,
        max_risk_per_trade_pct <= 0.05 (src/risk/CLAUDE.md <warn_about>).
      - execution.live_mode defaults to False; it is the ONLY source of
        paper=False anywhere in the codebase.
    """
    raise NotImplementedError("Phase 0 — config loader")
