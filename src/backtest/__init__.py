"""Backtesting + walk-forward optimization.

Depends on: strategies + signals + data + risk + shared. NEVER execution —
the backtest implements its own simulated AbstractBrokerClient.
Data source: DuckDB (synced from Polygon.io), never yfinance.
"""
