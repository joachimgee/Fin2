"""Enforces the dependency graph from CLAUDE.md <dependency_graph>.

This test is the architecture: any import that violates the graph fails CI,
so the rules survive contact with future contributors (human or LLM).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / "src"

# module -> set of src/ modules it may import from (itself is always allowed)
ALLOWED: dict[str, set[str]] = {
    "shared": set(),
    "data": {"shared"},
    "strategies": {"data", "shared"},
    "signals": {"data", "shared"},
    "risk": {"strategies", "shared"},  # strategies/base only, for OrderIntent
    "execution": {"risk", "shared"},  # OrderIntent arrives via risk's re-export
    "backtest": {"strategies", "signals", "data", "risk", "shared"},
    "monitoring": {"shared"},
}

# only execution/ may import the broker SDK
FORBIDDEN_THIRD_PARTY: dict[str, set[str]] = {
    module: {"alpaca"} for module in ALLOWED if module != "execution"
}
# deprecated SDK banned everywhere (root CLAUDE.md <what_to_warn_about> #9)
BANNED_EVERYWHERE = {"alpaca_trade_api"}


def _imports_of(pyfile: Path) -> list[str]:
    tree = ast.parse(pyfile.read_text(encoding="utf-8"))
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append(node.module)
    return found


def _module_files() -> list[tuple[str, Path]]:
    files = []
    for module in ALLOWED:
        for pyfile in (SRC / module).rglob("*.py"):
            files.append((module, pyfile))
    return files


@pytest.mark.parametrize(("module", "pyfile"), _module_files(), ids=lambda p: str(p))
def test_dependency_graph(module: str, pyfile: Path) -> None:
    for imp in _imports_of(pyfile):
        root = imp.split(".")[0]
        assert root not in BANNED_EVERYWHERE, f"{pyfile}: imports deprecated SDK {imp!r}"
        if root in FORBIDDEN_THIRD_PARTY.get(module, set()):
            raise AssertionError(
                f"{pyfile}: {module}/ may not import {imp!r} — only execution/ touches alpaca-py"
            )
        if root == "src":
            target = imp.split(".")[1] if "." in imp else ""
            if target and target != module and target not in ALLOWED[module]:
                raise AssertionError(
                    f"{pyfile}: illegal import {imp!r} — "
                    f"{module}/ may only import from {sorted(ALLOWED[module])}"
                )


def test_risk_only_imports_strategies_base() -> None:
    """risk -> strategies is sanctioned ONLY for strategies.base (OrderIntent)."""
    for pyfile in (SRC / "risk").rglob("*.py"):
        for imp in _imports_of(pyfile):
            if imp.startswith("src.strategies"):
                assert imp == "src.strategies.base", (
                    f"{pyfile}: risk/ may import src.strategies.base only, got {imp!r}"
                )
