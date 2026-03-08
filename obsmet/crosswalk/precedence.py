"""Per-variable source precedence configuration.

Defines which source to prefer when multiple sources report the same
station-time-variable. Config-driven with sensible defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class PrecedenceConfig:
    """Per-variable source priority ordering (highest first)."""

    hourly: dict[str, list[str]] = field(default_factory=dict)
    daily: dict[str, list[str]] = field(default_factory=dict)


DEFAULT_PRECEDENCE = PrecedenceConfig(
    hourly={
        "tair": ["ghcnh", "madis", "gdas", "ndbc"],
        "td": ["ghcnh", "madis", "gdas", "ndbc"],
        "rh": ["ghcnh", "madis"],
        "wind": ["ghcnh", "madis", "gdas", "ndbc"],
        "wind_dir": ["ghcnh", "madis", "gdas", "ndbc"],
        "wind_gust": ["ghcnh", "ndbc"],
        "slp": ["ghcnh", "gdas", "ndbc"],
        "psfc": ["ghcnh", "gdas"],
        "prcp": ["ghcnh", "madis"],
        "tw": ["ghcnh"],
    },
    daily={
        "tmax": ["ghcnd", "snotel", "raws_wrcc"],
        "tmin": ["ghcnd", "snotel", "raws_wrcc"],
        "tmean": ["ghcnd", "snotel", "raws_wrcc"],
        "prcp": ["ghcnd", "snotel", "raws_wrcc"],
        "swe": ["snotel"],
        "snow": ["ghcnd"],
        "snow_depth": ["ghcnd"],
        "wind": ["ghcnd", "raws_wrcc", "snotel"],
        "rh": ["raws_wrcc", "snotel"],
    },
)


def load_precedence(path: Path | None = None) -> PrecedenceConfig:
    """Load precedence from a TOML file, falling back to defaults.

    TOML format::

        [hourly]
        tair = ["ghcnh", "madis", "gdas", "ndbc"]

        [daily]
        tmax = ["ghcnd", "snotel", "raws_wrcc"]
    """
    if path is None:
        return DEFAULT_PRECEDENCE

    import tomllib

    with open(path, "rb") as f:
        data = tomllib.load(f)

    return PrecedenceConfig(
        hourly=data.get("hourly", DEFAULT_PRECEDENCE.hourly),
        daily=data.get("daily", DEFAULT_PRECEDENCE.daily),
    )
