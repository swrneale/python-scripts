"""
enso.py  –  ENSO index computation and composite derivation.

Public API
----------
compute_enso_index(sst_data, nino_region, thresh)
    → (inino_mons, inina_mons)  – integer indices into the monthly time-series

derive_composites(case, var_name, ds, sst_ds, sst_vname, cfg, var_meta,
                  inino_mons, inina_mons)
    → (climo, nino_anom, nina_anom)  – xr.DataArray on pressure levels

derive_composites_climo(case, var_name, ds, var_meta)
    → (climo, nino_anom, nina_anom)  – from pre-computed climo/nino/nina files
"""

from __future__ import annotations

import dask
import numpy as np
import xarray as xr

from config import AnalysisConfig, CaseConfig


# ── Nino region bounding boxes (S, N, W, E) ──────────────────────────────────
_NINO_REGIONS: dict[str, tuple[float, float, float, float]] = {
    'nino12': (-10.,  0., 270., 280.),
    'nino3':  ( -5.,  5., 210., 270.),
    'nino34': ( -5.,  5., 190., 240.),
    'nino4':  ( -5.,  5., 160., 210.),
    'nino5':  ( -5.,  5., 120., 140.),
    'nino6':  (  8., 16., 140., 160.),
}


# ── ENSO index ────────────────────────────────────────────────────────────────

def compute_enso_index(
    sst_data: xr.DataArray,
    nino_region: str = 'nino34',
    thresh: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute El Niño / La Niña month indices from a monthly SST time-series.

    Parameters
    ----------
    sst_data : DataArray with dims (time, lat, lon)
    nino_region : one of the keys in _NINO_REGIONS
    thresh : SSTA threshold in Kelvin (default 0.5 K)

    Returns
    -------
    inino_mons : integer indices of El Niño months (SSTA > +thresh AND DJF)
    inina_mons : integer indices of La Niña months (SSTA < -thresh AND DJF)
    """
    s, n, w, e = _NINO_REGIONS[nino_region]
    sst_ts = sst_data.sel(lat=slice(s, n), lon=slice(w, e)).mean(('lat', 'lon'))
    sst_ts = sst_ts.compute()

    month_names = sst_ts.time.dt.strftime('%b').values
    if month_names[0] != 'Jan':
        print(f'  Warning: time-series starts with {month_names[0]}, not Jan.')

    # Remove annual cycle month-by-month
    for mon in np.unique(month_names):
        mask = month_names == mon
        sst_ts.values[mask] -= float(sst_ts.values[mask].mean())

    # El Niño / La Niña compositing months: DJF only
    djf_mask = np.isin(month_names, ['Dec', 'Jan', 'Feb'])
    inino = np.where((sst_ts.values >  thresh) & djf_mask)[0]
    inina = np.where((sst_ts.values < -thresh) & djf_mask)[0]

    print(f'  {nino_region}: {len(inino)} El Niño months, {len(inina)} La Niña months')
    return inino, inina


# ── Composite derivation from time-series ────────────────────────────────────

def derive_composites(
    case: CaseConfig,
    var_name: str,
    ds: xr.Dataset,
    vname_in_ds: str,
    cfg: AnalysisConfig,
    var_meta,           # pandas DataFrame row for var_name
    inino_mons: np.ndarray,
    inina_mons: np.ndarray,
    ps_ds: xr.Dataset | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray,
           tuple | None]:
    """Derive seasonal climatology and El Niño / La Niña anomalies.

    Parameters
    ----------
    ds          : Dataset containing vname_in_ds (and optionally PS)
    vname_in_ds : name of the variable inside ds
    ps_ds       : Dataset containing 'PS' (can be same as ds)
    cfg         : AnalysisConfig
    var_meta    : row from get_var_meta() for var_name
    inino_mons  : El Niño month indices in the full monthly time-series
    inina_mons  : La Niña month indices

    Returns
    -------
    climo    : seasonal mean (DataArray)
    nino_anom: El Niño minus climo (DataArray)
    nina_anom: La Niña minus climo (DataArray)
    ps_tuple : (ps_climo, ps_nino, ps_nina) or None if ps_ds not supplied
    """
    vscale = float(var_meta['vscale'])

    month_names = ds.time.dt.strftime('%b').values
    seas_mask   = np.isin(month_names, list(cfg.seas_mons))
    iseas       = np.where(seas_mask)[0]

    # Intersect nino/nina with the target season
    inino_seas = np.intersect1d(inino_mons, iseas)
    inina_seas = np.intersect1d(inina_mons, iseas)

    # Positions of nino/nina within the seasonal subset
    _, inino_in_seas, _ = np.intersect1d(iseas, inino_seas, return_indices=True)
    _, inina_in_seas, _ = np.intersect1d(iseas, inina_seas, return_indices=True)

    # Use .isel() (not raw [] indexing) so Dask can build a clean task graph
    # and avoid unoptimised fancy-index gathers on chunked arrays.
    var_seas_raw = vscale * ds[vname_in_ds].isel(time=iseas)

    climo     = var_seas_raw.mean('time')
    nino_mean = var_seas_raw.isel(time=inino_in_seas).mean('time')
    nina_mean = var_seas_raw.isel(time=inina_in_seas).mean('time')

    nino_anom = nino_mean - climo
    nina_anom = nina_mean - climo

    # Surface pressure composites (needed for hybrid-sigma → pressure interpolation)
    ps_tuple = None
    if ps_ds is not None and 'PS' in ps_ds:
        ps_seas  = ps_ds['PS'].isel(time=iseas)
        ps_climo = ps_seas.mean('time')
        ps_nino  = ps_seas.isel(time=inino_in_seas).mean('time')
        ps_nina  = ps_seas.isel(time=inina_in_seas).mean('time')
        # Compute all PS composites in one scheduler submission
        ps_climo, ps_nino, ps_nina = dask.compute(ps_climo, ps_nino, ps_nina)
        ps_tuple = (ps_climo, ps_nino, ps_nina)

    # Derived variable: divergence = -∂ω/∂p
    if var_name == 'DIV':
        climo     = -climo.differentiate('lev')
        nino_anom = -nino_anom.differentiate('lev')
        nina_anom = -nina_anom.differentiate('lev')

    # Submit climo/nino_anom/nina_anom as a single graph so the scheduler can
    # fuse operations and read each source file only once across all three.
    climo, nino_anom, nina_anom = dask.compute(climo, nino_anom, nina_anom)

    return climo, nino_anom, nina_anom, ps_tuple


# ── Composite derivation from pre-computed climo files ───────────────────────

def derive_composites_climo(
    var_name: str,
    ds: xr.Dataset,
    vname_in_ds: str,
    var_meta_df,          # full DataFrame from get_var_meta()
    p_levs: np.ndarray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Extract composites from a climo/nino/nina stacked dataset (time length 3).

    The dataset is expected to have time indices 0=climo, 1=nino, 2=nina,
    as produced by load_climo().

    Parameters
    ----------
    var_meta_df : the full var_meta DataFrame (not a single row), so that when
                  var_name=='DIV' we can look up the ovscale for the raw OMEGA data.
    """
    # The raw variable in the file may differ from the requested variable.
    # e.g. DIV is derived from OMEGA, so the file contains 'omega' and we
    # must use OMEGA's ovscale (= -1.0, which also handles the sign of divergence).
    _file_vname_to_meta_key = {
        'omega': 'OMEGA', 'hgt': 'Z3', 'ta': 'T',
        'hus': 'Q', 'ua': 'U', 'va': 'V',
    }
    raw_meta_key = _file_vname_to_meta_key.get(vname_in_ds, var_name)
    ovscale = float(var_meta_df.loc[raw_meta_key, 'ovscale'])

    raw_climo = ovscale * ds[vname_in_ds].isel(time=0).drop_vars('time')
    raw_nino  = ovscale * ds[vname_in_ds].isel(time=1).drop_vars('time')
    raw_nina  = ovscale * ds[vname_in_ds].isel(time=2).drop_vars('time')

    # Trim to requested pressure range
    pmax, pmin = float(max(p_levs)), float(min(p_levs))
    raw_climo = raw_climo.sel(lev=slice(pmax, pmin))
    raw_nino  = raw_nino.sel( lev=slice(pmax, pmin))
    raw_nina  = raw_nina.sel( lev=slice(pmax, pmin))

    # Divergence = -∂ω/∂p.  The negative sign is already in OMEGA's ovscale (-1.0),
    # so a plain differentiate gives the correct sign.
    if var_name == 'DIV':
        raw_climo = raw_climo.differentiate('lev')
        raw_nino  = raw_nino.differentiate( 'lev')
        raw_nina  = raw_nina.differentiate( 'lev')

    return raw_climo, raw_nino, raw_nina
