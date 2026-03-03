"""Correction-factor interface (plan section 15).

Compute monthly correction factors between curated observations and
gridded baselines (ERA5-Land, NLDAS, RTMA, etc.).
"""

from __future__ import annotations

import pandas as pd


def compute_monthly_factors(
    obs: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    variable: str,
    station_col: str = "station_key",
    date_col: str = "date",
    obs_val_col: str | None = None,
    baseline_val_col: str | None = None,
    method: str = "ratio",
) -> pd.DataFrame:
    """Compute per-station monthly correction factors.

    Parameters
    ----------
    obs : DataFrame with station, date, and observation value columns.
    baseline : DataFrame with station, date, and baseline value columns.
    variable : canonical variable name.
    method : "ratio" (obs/baseline) or "delta" (obs - baseline).

    Returns
    -------
    DataFrame with columns: station_key, month, factor, obs_count.
    """
    obs_col = obs_val_col or variable
    base_col = baseline_val_col or variable

    merged = obs.merge(baseline, on=[station_col, date_col], suffixes=("_obs", "_base"))

    merged["month"] = pd.to_datetime(merged[date_col]).dt.month

    groups = merged.groupby([station_col, "month"])

    if method == "ratio":
        factors = groups.apply(
            lambda g: pd.Series(
                {
                    "factor": g[f"{obs_col}_obs"].mean() / g[f"{base_col}_base"].mean()
                    if g[f"{base_col}_base"].mean() != 0
                    else float("nan"),
                    "obs_count": len(g),
                }
            ),
            include_groups=False,
        )
    elif method == "delta":
        factors = groups.apply(
            lambda g: pd.Series(
                {
                    "factor": g[f"{obs_col}_obs"].mean() - g[f"{base_col}_base"].mean(),
                    "obs_count": len(g),
                }
            ),
            include_groups=False,
        )
    else:
        raise ValueError(f"Unknown method {method!r}; use 'ratio' or 'delta'")

    return factors.reset_index()
