"""
regions.py  –  Area-weighted regional averaging for vertical profiles.

Public API
----------
area_mean(da, lat_s, lat_n, lon_w, lon_e)
    → DataArray  (cos-latitude weighted mean over the box)

build_regional_profiles(composites, regions_df)
    → xr.DataArray  with an extra 'column' dimension (3 states × N regions)
"""

from __future__ import annotations

import dask
import numpy as np
import pandas as pd
import xarray as xr


def area_mean(
    da: xr.DataArray,
    lat_s: float, lat_n: float,
    lon_w: float, lon_e: float,
) -> xr.DataArray:
    """Return the cos-latitude weighted area mean over a lat-lon box."""
    sub = da.sel(lat=slice(lat_s, lat_n), lon=slice(lon_w, lon_e))
    weights = np.cos(np.deg2rad(sub.lat))
    weights.name = 'weights'
    return sub.weighted(weights).mean(('lat', 'lon'), skipna=True)


def build_regional_profiles(
    composites: tuple[xr.DataArray, xr.DataArray, xr.DataArray],
    regions_df: pd.DataFrame,
) -> xr.DataArray:
    """Compute area-averaged vertical profiles for every region and ENSO state.

    Parameters
    ----------
    composites  : (climo, nino_anom, nina_anom) – each a DataArray(lev, lat, lon)
    regions_df  : DataFrame from config.get_regions()

    Returns
    -------
    DataArray with a 'column' dimension of length N_regions × 3.
    The ordering is:  [reg0_climo, reg0_nino, reg0_nina, reg1_climo, ...]
    """
    climo, nino, nina = composites

    # Build all lazy area-mean tasks first, then compute the entire set in one
    # scheduler submission (N_regions × 3 states).  This lets Dask parallelise
    # across regions and reuse any shared intermediate results.
    lazy_profiles: list[xr.DataArray] = []
    for reg in regions_df.index:
        row   = regions_df.loc[reg]
        lat_s = float(row['lat_s']); lat_n = float(row['lat_n'])
        lon_w = float(row['lon_w']); lon_e = float(row['lon_e'])
        lazy_profiles.extend([
            area_mean(climo, lat_s, lat_n, lon_w, lon_e),
            area_mean(nino,  lat_s, lat_n, lon_w, lon_e),
            area_mean(nina,  lat_s, lat_n, lon_w, lon_e),
        ])

    computed = dask.compute(*lazy_profiles)
    return xr.concat(list(computed), dim='column')
