"""Risk management — mandatory gatekeeper. No order exists without validate().

Depends on: strategies/base (OrderIntent) + shared.

Re-exports OrderIntent alongside ValidationResult so that execution/ can type
the validate() boundary while importing from risk/ only (its allowed edge).
"""

from src.risk.manager import RiskManager, ValidationResult
from src.strategies.base import OrderIntent

__all__ = ["OrderIntent", "RiskManager", "ValidationResult"]
