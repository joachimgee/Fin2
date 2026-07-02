"""Project exception hierarchy. All custom exceptions derive from FinbotError."""

from __future__ import annotations


class FinbotError(Exception):
    """Base class for all project exceptions."""


class ConfigError(FinbotError):
    """Missing/invalid configuration or environment variable (fail-fast at startup)."""


class DataValidationError(FinbotError):
    """Bar failed validation (NaN, negative price, corrupt timestamp). Fail fast."""


class OrderRejectedError(FinbotError):
    """RiskManager.validate() rejected the order intent. Carries the reason."""


class InvalidKellyFractionError(FinbotError):
    """Kelly fraction > 0.50 requested — hard cap is half-Kelly (src/risk/CLAUDE.md)."""


class CircuitBreakerTrippedError(FinbotError):
    """An order was attempted while trading is halted."""
