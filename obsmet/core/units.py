"""Variable and unit standardization registry (plan section 9).

Each canonical variable has a defined name, standard unit, and optional
conversion functions keyed by (source, source_unit).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

# --------------------------------------------------------------------------- #
# Canonical variable definitions
# --------------------------------------------------------------------------- #

CANONICAL_UNITS: dict[str, str] = {
    # Temperature
    "tair": "degC",
    "tmin": "degC",
    "tmax": "degC",
    "tmean": "degC",
    # Humidity
    "rh": "percent",
    "ea": "kPa",
    "vpd": "kPa",
    "q": "kg kg-1",
    "td": "degC",
    # Wind
    "u": "m s-1",
    "v": "m s-1",
    "wind": "m s-1",
    "wind_dir": "deg",
    "u2": "m s-1",
    # Pressure
    "psfc": "Pa",
    "slp": "Pa",
    # Radiation
    "rsds_hourly": "W m-2",
    "rsds": "MJ m-2 day-1",
    # Precipitation
    "prcp": "mm",
}


# --------------------------------------------------------------------------- #
# Conversion helpers
# --------------------------------------------------------------------------- #


def fahrenheit_to_celsius(f: float) -> float:
    return (f - 32.0) * 5.0 / 9.0


def kelvin_to_celsius(k: float) -> float:
    return k - 273.15


def knots_to_ms(kt: float) -> float:
    return kt * 0.514444


def mph_to_ms(mph: float) -> float:
    return mph * 0.44704


def hpa_to_pa(hpa: float) -> float:
    return hpa * 100.0


def mb_to_pa(mb: float) -> float:
    return mb * 100.0


def inhg_to_pa(inhg: float) -> float:
    return inhg * 3386.389


def mm_to_mm(val: float) -> float:
    return val


def inches_to_mm(inches: float) -> float:
    return inches * 25.4


# --------------------------------------------------------------------------- #
# Conversion registry
# --------------------------------------------------------------------------- #


@dataclass
class ConversionRule:
    """A single unit conversion rule."""

    source_unit: str
    target_unit: str
    fn: Callable[[float], float]
    version: str = "1"


@dataclass
class UnitRegistry:
    """Registry of conversion rules keyed by (variable, source_unit)."""

    _rules: dict[tuple[str, str], ConversionRule] = field(default_factory=dict)

    def register(self, variable: str, rule: ConversionRule) -> None:
        self._rules[(variable, rule.source_unit)] = rule

    def convert(self, variable: str, value: float, source_unit: str) -> float:
        canonical = CANONICAL_UNITS.get(variable)
        if canonical is None:
            raise KeyError(f"Unknown canonical variable: {variable}")
        if source_unit == canonical:
            return value
        key = (variable, source_unit)
        rule = self._rules.get(key)
        if rule is None:
            raise KeyError(
                f"No conversion registered for variable={variable!r} "
                f"from {source_unit!r} to {canonical!r}"
            )
        return rule.fn(value)

    def get_version(self, variable: str, source_unit: str) -> str | None:
        rule = self._rules.get((variable, source_unit))
        return rule.version if rule else None


def build_default_registry() -> UnitRegistry:
    """Build the default unit conversion registry."""
    reg = UnitRegistry()

    # Temperature conversions
    for var in ("tair", "tmin", "tmax", "tmean", "td"):
        reg.register(var, ConversionRule("degF", "degC", fahrenheit_to_celsius))
        reg.register(var, ConversionRule("K", "degC", kelvin_to_celsius))

    # Wind conversions
    for var in ("u", "v", "wind", "u2"):
        reg.register(var, ConversionRule("kt", "m s-1", knots_to_ms))
        reg.register(var, ConversionRule("mph", "m s-1", mph_to_ms))

    # Pressure conversions
    for var in ("psfc", "slp"):
        reg.register(var, ConversionRule("hPa", "Pa", hpa_to_pa))
        reg.register(var, ConversionRule("mb", "Pa", mb_to_pa))
        reg.register(var, ConversionRule("inHg", "Pa", inhg_to_pa))

    # Precipitation conversions
    reg.register("prcp", ConversionRule("in", "mm", inches_to_mm))

    return reg
