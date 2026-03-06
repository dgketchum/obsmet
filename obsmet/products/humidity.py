"""Compiled humidity assembly using agweather-qaqc decision tree.

Produces best-available vapor pressure (ea) and dewpoint (td) from whatever
humidity variables a station reports, following the ASCE priority chain:
    ea → td → RHmax/RHmin → RHavg → Tmin-Ko
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from agweatherqaqc.calc_functions import calc_compiled_ea, calc_humidity_variables


def compile_humidity(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``ea_compiled`` and ``td_compiled`` columns using the best available source.

    Parameters
    ----------
    df : pd.DataFrame
        Daily DataFrame that may contain any subset of: ea, td, rhmax,
        rhmin, rhavg, tmax, tmin, tmean.

    Returns
    -------
    pd.DataFrame
        Copy with ``ea_compiled`` and ``td_compiled`` appended.
    """
    n = len(df)
    df = df.copy()

    tmax = df["tmax"].values.astype(np.float64) if "tmax" in df.columns else np.full(n, np.nan)
    tmin = df["tmin"].values.astype(np.float64) if "tmin" in df.columns else np.full(n, np.nan)
    tavg = df["tmean"].values.astype(np.float64) if "tmean" in df.columns else (tmax + tmin) / 2.0

    ea = df["ea"].values.astype(np.float64) if "ea" in df.columns else np.full(n, np.nan)
    tdew = df["td"].values.astype(np.float64) if "td" in df.columns else np.full(n, np.nan)
    rhmax = df["rhmax"].values.astype(np.float64) if "rhmax" in df.columns else np.full(n, np.nan)
    rhmin = df["rhmin"].values.astype(np.float64) if "rhmin" in df.columns else np.full(n, np.nan)
    rhavg = df["rh"].values.astype(np.float64) if "rh" in df.columns else np.full(n, np.nan)

    # Column sentinel: -1 means variable not provided
    ea_col = 0 if "ea" in df.columns and not np.all(np.isnan(ea)) else -1
    tdew_col = 0 if "td" in df.columns and not np.all(np.isnan(tdew)) else -1
    rhmax_col = 0 if "rhmax" in df.columns and not np.all(np.isnan(rhmax)) else -1
    rhmin_col = 0 if "rhmin" in df.columns and not np.all(np.isnan(rhmin)) else -1
    rhavg_col = 0 if "rh" in df.columns and not np.all(np.isnan(rhavg)) else -1

    # Tmin-based dewpoint estimate (ASCE appendix: Td ≈ Tmin)
    tdew_ko = tmin.copy()

    # First pass: get ea and td from best single source
    any_humidity = ea_col != -1 or tdew_col != -1 or rhmax_col != -1 or rhavg_col != -1
    if any_humidity:
        calc_ea, calc_td = calc_humidity_variables(
            tmax,
            tmin,
            tavg,
            ea,
            ea_col,
            tdew,
            tdew_col,
            rhmax,
            rhmax_col,
            rhmin,
            rhmin_col,
            rhavg,
            rhavg_col,
        )
    else:
        # No humidity columns — derive from Tmin estimate
        calc_ea = np.array(0.6108 * np.exp((17.27 * tdew_ko) / (tdew_ko + 237.3)))
        calc_td = tdew_ko.copy()
        tdew_col = 0  # mark as available for compiled_ea

    # Second pass: compile ea gap-filling across all sources
    compiled_ea = calc_compiled_ea(
        tmax,
        tmin,
        tavg,
        calc_ea,
        calc_td,
        tdew_col,
        rhmax,
        rhmax_col,
        rhmin,
        rhmin_col,
        rhavg,
        rhavg_col,
        tdew_ko,
    )

    # Derive Td from compiled Ea using inverse Magnus formula
    with np.errstate(divide="ignore", invalid="ignore"):
        compiled_td = (116.91 + 237.3 * np.log(compiled_ea)) / (16.78 - np.log(compiled_ea))

    df["ea_compiled"] = compiled_ea
    df["td_compiled"] = compiled_td
    return df
