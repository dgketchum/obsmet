"""Base interface for QAQC rules (plan section 11).

All QC rules implement the QCRule protocol: given an observation value and
context, return a QCResult with state and reason code.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum


class QCTier(IntEnum):
    """QAQC tiers (plan section 11.1)."""

    SOURCE_NATIVE = 0  # Source-native flags only
    PHYSICAL = 1  # Physical and logical constraints
    TEMPORAL = 2  # Robust temporal checks
    SPATIAL = 3  # Spatial/background checks (optional)


@dataclass(frozen=True)
class QCResult:
    """Result of a single QC rule application."""

    state: str  # "pass", "fail", "suspect"
    reason: str  # machine-readable reason code
    tier: QCTier
    rule_name: str
    detail: str = ""  # optional human-readable detail


class QCRule(ABC):
    """Base class for all QAQC rules."""

    tier: QCTier
    name: str
    version: str = "1"

    @abstractmethod
    def check(self, value: float, **context) -> QCResult:
        """Apply this rule to a single observation value.

        Context kwargs vary by rule (e.g., variable name, timestamp,
        neighboring values, native flags).
        """
        ...
