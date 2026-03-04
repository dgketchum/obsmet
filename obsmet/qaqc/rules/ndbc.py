"""Tier 0: NDBC source-native QC rules (missing-value sentinel detection).

NDBC uses specific sentinel values to indicate missing data:
99, 999, 9999 (and their .0 float variants). These are field-width-dependent
missing codes rather than QC flags, so this rule simply detects whether
a value is a known sentinel.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from obsmet.qaqc.rules.base import QCResult, QCRule, QCTier

# Sentinel values used by NDBC (as floats for comparison)
DEFAULT_SENTINELS = {99.0, 999.0, 9999.0}


@dataclass
class NdbcSentinelRule(QCRule):
    """Tier 0: Detect NDBC missing-value sentinels.

    NDBC does not provide per-observation QC flags — instead, missing data
    is encoded as field-width-specific sentinel values (99, 999, 9999).
    This rule checks whether a value matches a known sentinel.
    """

    tier: QCTier = QCTier.SOURCE_NATIVE
    name: str = "ndbc_sentinel"
    version: str = "1"
    sentinels: set[float] = field(default_factory=lambda: DEFAULT_SENTINELS.copy())

    def check(self, value: float, **context) -> QCResult:
        """Check if value matches a known NDBC sentinel.

        Parameters
        ----------
        value : The observation value.
        **context : Optional 'variable' key for better diagnostics.
        """
        if value is None:
            return QCResult(
                state="fail",
                reason="value_none",
                tier=self.tier,
                rule_name=self.name,
                detail="Value is None",
            )

        variable = context.get("variable", "unknown")

        if value in self.sentinels:
            return QCResult(
                state="fail",
                reason="sentinel_detected",
                tier=self.tier,
                rule_name=self.name,
                detail=f"Value {value} is a known NDBC sentinel for {variable}",
            )

        return QCResult(
            state="pass",
            reason="not_sentinel",
            tier=self.tier,
            rule_name=self.name,
        )
