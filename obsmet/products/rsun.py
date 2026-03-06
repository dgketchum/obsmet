"""RSUN clear-sky solar radiation extraction for station locations.

Reads pre-computed terrain-corrected Rso from a 365-band GeoTIFF (one band
per DOY) produced by GRASS r.sun + r.horizon (see dads-mvp).
"""

from __future__ import annotations

import numpy as np
import rasterio
from pyproj import Transformer


def extract_station_rsun(
    lon: float,
    lat: float,
    rsun_path: str,
) -> np.ndarray:
    """Return 365-element array of clear-sky GHI (W/m²) for a station.

    The raster stores Wh/m²/day per band; we convert to average W/m² by
    dividing by 24 so the values are comparable to agweather-qaqc's Rso
    convention (W/m²).

    Parameters
    ----------
    lon, lat : float
        Station coordinates in WGS-84 decimal degrees.
    rsun_path : str
        Path to the 365-band RSUN GeoTIFF (e.g. rsun_pnw_1km.tif).

    Returns
    -------
    np.ndarray
        Shape (365,) clear-sky GHI in W/m².
    """
    with rasterio.open(rsun_path) as src:
        transformer = Transformer.from_crs("EPSG:4326", src.crs, always_xy=True)
        x, y = transformer.transform(lon, lat)
        row, col = src.index(x, y)

        rso = np.empty(src.count, dtype=np.float64)
        for band_idx in range(1, src.count + 1):
            window = rasterio.windows.Window(col, row, 1, 1)
            val = src.read(band_idx, window=window)
            rso[band_idx - 1] = float(val[0, 0])

    # Convert Wh/m²/day → W/m² (average over 24 hours)
    rso = rso / 24.0
    rso[rso < 0] = 0.0
    return rso
