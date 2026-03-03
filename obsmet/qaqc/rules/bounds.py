"""Tier 1: Physical and logical bound checks (plan section 11.1)."""

from __future__ import annotations

from dataclasses import dataclass

from obsmet.qaqc.rules.base import QCResult, QCRule, QCTier

# Default physical bounds per canonical variable.
# These are intentionally generous operational limits.
PHYSICAL_BOUNDS: dict[str, tuple[float, float]] = {
    "tair": (-90.0, 60.0),
    "tmin": (-90.0, 60.0),
    "tmax": (-90.0, 60.0),
    "tmean": (-90.0, 60.0),
    "td": (-100.0, 50.0),
    "rh": (0.0, 100.0),
    "ea": (0.0, 10.0),
    "vpd": (0.0, 15.0),
    "wind": (0.0, 120.0),
    "u2": (0.0, 120.0),
    "u": (-120.0, 120.0),
    "v": (-120.0, 120.0),
    "wind_dir": (0.0, 360.0),
    "psfc": (30000.0, 110000.0),
    "slp": (85000.0, 110000.0),
    "rsds": (0.0, 50.0),
    "rsds_hourly": (0.0, 1400.0),
    "prcp": (0.0, 1000.0),
}


@dataclass
class PhysicalBoundsRule(QCRule):
    """Check that a value falls within physically plausible bounds."""

    tier: QCTier = QCTier.PHYSICAL
    name: str = "physical_bounds"
    version: str = "1"

    def check(self, value: float, **context) -> QCResult:
        variable = context.get("variable", "")
        bounds = PHYSICAL_BOUNDS.get(variable)
        if bounds is None:
            return QCResult(
                state="pass",
                reason="no_bounds_defined",
                tier=self.tier,
                rule_name=self.name,
                detail=f"No bounds for variable={variable!r}",
            )
        lo, hi = bounds
        if value < lo or value > hi:
            return QCResult(
                state="fail",
                reason="out_of_bounds",
                tier=self.tier,
                rule_name=self.name,
                detail=f"{variable}={value} outside [{lo}, {hi}]",
            )
        return QCResult(
            state="pass",
            reason="within_bounds",
            tier=self.tier,
            rule_name=self.name,
        )


@dataclass
class DewpointConsistencyRule(QCRule):
    """Check that dewpoint does not exceed air temperature."""

    tier: QCTier = QCTier.PHYSICAL
    name: str = "dewpoint_consistency"
    version: str = "1"

    def check(self, value: float, **context) -> QCResult:
        """value is dewpoint; context must include tair."""
        tair = context.get("tair")
        if tair is None:
            return QCResult(
                state="pass",
                reason="no_tair_context",
                tier=self.tier,
                rule_name=self.name,
                detail="No air temperature available for comparison",
            )
        tolerance = 0.5  # allow small measurement tolerance
        if value > tair + tolerance:
            return QCResult(
                state="fail",
                reason="td_exceeds_tair",
                tier=self.tier,
                rule_name=self.name,
                detail=f"td={value} > tair={tair} + {tolerance}",
            )
        return QCResult(
            state="pass",
            reason="td_consistent",
            tier=self.tier,
            rule_name=self.name,
        )
