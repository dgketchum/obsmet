"""Clear-sky solar radiation (Rso) for station locations.

Two sources:
  1. RSUN raster — terrain-corrected Rso from GRASS r.sun + r.horizon.
  2. Flat-earth ASCE — simple Rso from refet (needs only lat + elevation).
"""

from __future__ import annotations

import math

import numpy as np
from refet.calcs import _ra_daily, _rso_simple


def extract_station_rsun(
    lon: float,
    lat: float,
    rsun_path: str,
) -> np.ndarray:
    """Return 365-element array of clear-sky Rso (MJ/m²/day) from RSUN raster.

    The raster stores Wh/m²/day per band; we convert to MJ/m²/day so the
    values match the rsds units in the daily aggregation.

    Parameters
    ----------
    lon, lat : float
        Station coordinates in WGS-84 decimal degrees.
    rsun_path : str
        Path to the 365-band RSUN GeoTIFF (e.g. rsun_pnw_1km.tif).

    Returns
    -------
    np.ndarray
        Shape (365,) clear-sky Rso in MJ/m²/day.
    """
    import rasterio
    from pyproj import Transformer

    with rasterio.open(rsun_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        row, col = src.index(x, y)

        rso = np.empty(src.count, dtype=np.float64)
        for band_idx in range(1, src.count + 1):
            window = rasterio.windows.Window(col, row, 1, 1)
            val = src.read(band_idx, window=window)
            rso[band_idx - 1] = float(val[0, 0])

    # Convert Wh/m²/day → MJ/m²/day (1 Wh = 0.0036 MJ)
    rso = rso * 0.0036
    rso[rso < 0] = 0.0
    return rso


def compute_rso_asce(lat_deg: float, elev_m: float) -> np.ndarray:
    """Clear-sky Rso via ASCE flat-earth formula (refet).

    Uses the simple formulation: Rso = (0.75 + 2e-5 * elev) * Ra.
    Suitable for QC bounding at any location without raster data.

    Parameters
    ----------
    lat_deg : float
        Latitude in decimal degrees (positive north).
    elev_m : float
        Elevation in meters above sea level.

    Returns
    -------
    np.ndarray
        Shape (365,) clear-sky Rso in MJ/m²/day (same units as rsds in
        the daily aggregation).
    """
    lat_rad = lat_deg * (math.pi / 180.0)
    doy = np.arange(1, 366)
    ra = _ra_daily(lat_rad, doy, method="asce")
    rso = _rso_simple(ra, elev_m)
    return rso
