"""Daily P&L report (sent via Telegram at monitoring.daily_report_time_et).

The tracker arrives duck-typed (equity / total_exposure / position_values):
monitoring/ depends on shared only — it never imports risk/.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


def build_daily_report(tracker: Any, realized_pnls: Sequence[float]) -> str:
    positions = tracker.position_values()
    wins = sum(1 for p in realized_pnls if p > 0)
    lines = [
        "finbot daily report",
        f"equity: {tracker.equity:,.2f}",
        f"exposure: {tracker.total_exposure():,.2f}",
        f"open positions: {len(positions)}",
        f"trades closed: {len(realized_pnls)}",
        f"realized pnl: {sum(realized_pnls):+,.2f}",
    ]
    if realized_pnls:
        lines.append(f"win rate: {wins / len(realized_pnls):.0%}")
    lines.extend(f"  {symbol}: {value:,.2f}" for symbol, value in sorted(positions.items()))
    return "\n".join(lines)
