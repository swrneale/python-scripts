"""
forecast_diurnal_utils.py
Diurnal cycle analysis for CESM2 S2S hindcast ensemble forecasts.

Builds lead-day diurnal cycle composites from weekly-initialized 45-day
ensemble forecasts, then computes harmonics and panel plots with one map
per lead day.

Data format:
  cesm2cam6v2.YYYY-MM-DD.NN.cam.h3.YYYY-MM-DD-00000.nc
  Variables: PRECT (m/s, total precip), PRECC+PRECL also valid
  Time: 181 × 6-hourly (or variable dt_hours) starting at 00Z on init date
  Grid: 192 lat (Gaussian) × 288 lon (1.25°, 0–358.75°)
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import os
import sys

# ── Import parent diurnal cycle utilities ─────────────────────────────────────
_PARENT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if _PARENT not in sys.path:
    sys.path.insert(0, _PARENT)
import diurnal_cycle_utils as dcutils

# Re-export shared functions so callers only need to import this module
get_season_months   = dcutils.get_season_months
compute_harmonics   = dcutils.compute_harmonics
compute_raw_diurnal = dcutils.compute_raw_diurnal
phase_utc_to_lst    = dcutils.phase_utc_to_lst
SEASON_MONTHS       = dcutils.SEASON_MONTHS
_LEAP_YEARS         = dcutils._LEAP_YEARS

# Default data directories
DATA_DIR      = '/glade/campaign/cesm/development/cross-wg/S2S/CESM2/S2SHINDCASTS/6hourly'
DATA_DIR_ERA5 = '/lustre/desc1/espat/s2s/cesm/cesm2_era5/cesm2_era5/3hourly'
IMERG_DIR     = '/glade/derecho/scratch/rneale/IMERG/3hrly/1deg'

# Unit conversions
_MS_TO_MMDAY   = 1000.0 * 86400.0   # m/s  → mm/day
_MMHR_TO_MMDAY = 24.0                # mm/hr → mm/day

# ── Per-variable settings ─────────────────────────────────────────────────────
# Each entry: scale    — multiply raw CAM value to reach 'units'
#             units    — display unit string
#             long_name— descriptive label
#             min_amp / max_amp — Evans amplitude colour wheel range
#             levels   — contour levels for mean map (None = auto)
#             cmap     — matplotlib colormap for mean map
VAR_SETTINGS = {
    'PRECT': {
        'long_name': 'Total Precipitation',
        'units':     'mm/day',
        'scale':     _MS_TO_MMDAY,
        'min_amp':   0.0,   'max_amp': 4.0,
        'levels':    [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 10],
        'cmap':      None,          # None → use dcutils.CMAP_PRCP
    },
    'TMQ': {
        'long_name': 'Total Precipitable Water',
        'units':     'kg/m²',
        'scale':     1.0,
        'min_amp':   0.5,   'max_amp': 5.0,
        'levels':    None,          # auto
        'cmap':      'Blues',
    },
    'U10': {
        'long_name': '10-m Wind Speed',
        'units':     'm/s',
        'scale':     1.0,
        'min_amp':   0.1,   'max_amp': 1.5,
        'levels':    None,
        'cmap':      'YlOrRd',
    },
    'PSL': {
        'long_name': 'Sea Level Pressure',
        'units':     'hPa',
        'scale':     0.01,          # Pa → hPa
        'min_amp':   0.1,   'max_amp': 1.0,
        'levels':    None,
        'cmap':      'RdBu_r',
    },
    'TS': {
        'long_name': 'Surface Temperature',
        'units':     'K',
        'scale':     1.0,
        'min_amp':   0.5,   'max_amp': 5.0,
        'levels':    None,
        'cmap':      'RdBu_r',
    },
    'UBOT': {
        'long_name': 'Lowest Model Level Zonal Wind',
        'units':     'm/s',
        'scale':     1.0,
        'min_amp':   0.1,   'max_amp': 2.0,
        'levels':    None,
        'cmap':      'YlOrRd',
    },
    'QBOT': {
        'long_name': 'Lowest Model Level Water Vapour',
        'units':     'g/kg',
        'scale':     1000.0,        # kg/kg → g/kg
        'min_amp':   0.05,  'max_amp': 0.5,
        'levels':    None,
        'cmap':      'Blues',
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_start_dates(data_dir=DATA_DIR, season=None, year_start=None, year_end=None):
    """
    Return sorted list of start date strings 'YYYY-MM-DD' available in data_dir.
    Optionally filter by season and/or year range.

    Parameters
    ----------
    data_dir   : str
    season     : str  e.g. 'JJA', 'DJF', 'ANN' — filters start dates by month
    year_start : int  inclusive lower year bound
    year_end   : int  inclusive upper year bound

    Returns
    -------
    dates : list of str  'YYYY-MM-DD'
    """
    # Use ensemble member 00 as the reference file to find available dates
    files = glob.glob(os.path.join(data_dir, 'cesm2cam6v2.????-??-??.00.cam.h3.*.nc'))
    dates = sorted({os.path.basename(f).split('.')[1] for f in files})

    if year_start is not None:
        dates = [d for d in dates if int(d[:4]) >= year_start]
    if year_end is not None:
        dates = [d for d in dates if int(d[:4]) <= year_end]
    if season is not None:
        months, _ = get_season_months(season)
        dates = [d for d in dates if int(d[5:7]) in months]

    return dates


def get_start_dates_cesm2era5(data_dir=DATA_DIR_ERA5, season=None,
                               year_start=None, year_end=None):
    """
    Return sorted list of start date strings 'YYYY-MM-DD' available in the
    cesm2_era5 3-hourly dataset (member 00 used as reference).

    Parameters
    ----------
    data_dir   : str   path to the 3-hourly ERA5-init directory
    season     : str   e.g. 'JJA', 'DJF', 'ANN'
    year_start : int   inclusive lower year bound
    year_end   : int   inclusive upper year bound

    Returns
    -------
    dates : list of str 'YYYY-MM-DD'
    """
    files = glob.glob(os.path.join(
        data_dir, 'cesm2_era5_????-??-??.00.cam.h4.*.nc'))
    # basename: 'cesm2_era5_YYYY-MM-DD.00.cam.h4.YYYY-MM-DD-00000.nc'
    dates = sorted({
        os.path.basename(f).split('.')[0].split('_era5_')[1]
        for f in files
    })

    if year_start is not None:
        dates = [d for d in dates if int(d[:4]) >= year_start]
    if year_end is not None:
        dates = [d for d in dates if int(d[:4]) <= year_end]
    if season is not None:
        months, _ = get_season_months(season)
        dates = [d for d in dates if int(d[5:7]) in months]

    return dates


def _roll_lon(data, lon):
    """
    Roll longitude coordinate from 0–360 to -180–180.
    data : (..., nlon) array
    lon  : (nlon,) array in 0–360
    Returns (data_rolled, lon_rolled).
    """
    if lon.max() <= 180.:
        return data, lon
    idx = int(np.searchsorted(lon, 180.0))
    lon_r  = np.concatenate([lon[idx:] - 360., lon[:idx]])
    data_r = np.concatenate([data[..., idx:], data[..., :idx]], axis=-1)
    return data_r, lon_r


# ─────────────────────────────────────────────────────────────────────────────
# Main data loader
# ─────────────────────────────────────────────────────────────────────────────

def load_forecast_diurnal(data_dir=DATA_DIR, start_dates=None, ens_members=None,
                           nlead_days=6, dt_hours=6,
                           precip_vars=('PRECT',),
                           scale_mmday=True):
    """
    Load CESM2 S2S hindcast files and compute a lead-day diurnal cycle composite.

    Each 45-day forecast is divided into calendar days relative to the init date:
      lead day 1 = hours  0–(24-dt_hours)  after init (same calendar day)
      lead day 2 = hours 24–(48-dt_hours)  after init
      ...

    All (start_date × ensemble_member) samples are averaged for each
    (lead_day, time_of_day) slot to produce the composite.

    Parameters
    ----------
    data_dir     : str   hindcast file directory
    start_dates  : list of str  'YYYY-MM-DD'; if None uses all available dates
    ens_members  : list of int  0-based ensemble indices (0–20);
                   if None uses [0,1,2,3,4]
    nlead_days   : int   number of lead days to compute (max 45)
    dt_hours     : float data time step in hours (6 for 6-hourly files)
    precip_vars  : tuple of str  variable(s) to sum for total precip [m/s]
                   Use ('PRECT',) or ('PRECC', 'PRECL')
    scale_mmday  : bool  if True convert m/s → mm/day

    Returns
    -------
    dc_leads : xr.DataArray (lead_day, time_of_day, lat, lon)
               Composite diurnal cycle [mm/day or m/s].
               Longitude is rolled to -180–180.
    """
    if start_dates is None:
        start_dates = get_start_dates(data_dir)
    if ens_members is None:
        ens_members = list(range(5))

    n_tod  = int(24 // dt_hours)
    scale  = _MS_TO_MMDAY if scale_mmday else 1.0
    nlead_days = min(nlead_days, 45)

    dc_sum  = None   # (nlead_days, n_tod, nlat, nlon)
    cnt_sum = None
    lat = lon = None

    n_total   = len(start_dates) * len(ens_members)
    n_done    = 0
    n_skipped = 0
    n_no_var  = 0

    for start_date in start_dates:
        for ens in ens_members:
            fname = (f'cesm2cam6v2.{start_date}.{ens:02d}.cam.h3.'
                     f'{start_date}-00000.nc')
            fpath = os.path.join(data_dir, fname)
            if not os.path.isfile(fpath):
                n_skipped += 1
                n_done += 1
                continue

            ds = xr.open_dataset(fpath, decode_times=False)

            # Sum requested precip variables
            pr = None
            for v in precip_vars:
                if v in ds:
                    arr = ds[v].values.astype(np.float32)  # (ntime, nlat, nlon)
                    pr  = arr if pr is None else pr + arr
            if pr is None:
                n_no_var += 1
                ds.close()
                n_done += 1
                continue
            pr[pr < 0] = np.nan
            pr *= scale

            if lat is None:
                lat = ds['lat'].values.copy()
                lon = ds['lon'].values.copy()
                nlat, nlon = lat.size, lon.size
                dc_sum  = np.zeros((nlead_days, n_tod, nlat, nlon), np.float64)
                cnt_sum = np.zeros((nlead_days, n_tod, nlat, nlon), np.int32)
            ds.close()

            # Accumulate each lead day
            for d in range(nlead_days):
                i0 = d * n_tod
                i1 = i0 + n_tod
                if i1 > pr.shape[0]:
                    break
                chunk = pr[i0:i1]          # (n_tod, nlat, nlon)
                valid = np.isfinite(chunk)
                dc_sum[d]  += np.where(valid, chunk, 0.0)
                cnt_sum[d] += valid.astype(np.int32)

            n_done += 1
            if n_done % 100 == 0:
                n_loaded = n_done - n_skipped - n_no_var
                print(f'  {n_done}/{n_total}  loaded={n_loaded}  '
                      f'missing={n_skipped}  no_precip={n_no_var}')

    n_loaded = n_done - n_skipped - n_no_var
    print(f'  Done: {n_loaded}/{n_total} files loaded  '
          f'({n_skipped} missing, {n_no_var} no precip variable)')
    if n_no_var > 0:
        print(f'  Note: {n_no_var} files lacked {precip_vars}. '
              f'Only years 2022+ have PRECT in this dataset.')

    if dc_sum is None:
        raise RuntimeError('No forecast files found for the requested dates/members.')

    with np.errstate(invalid='ignore', divide='ignore'):
        dc = np.where(cnt_sum > 0,
                      dc_sum / cnt_sum,
                      np.nan).astype(np.float32)

    # Roll longitude to -180–180 for consistency with domain settings
    dc, lon = _roll_lon(dc, lon)

    tod       = np.arange(n_tod, dtype=float) * dt_hours
    lead_days = np.arange(1, nlead_days + 1)

    return xr.DataArray(
        dc,
        dims=['lead_day', 'time_of_day', 'lat', 'lon'],
        coords={'lead_day':    lead_days,
                'time_of_day': tod,
                'lat':         lat,
                'lon':         lon},
        attrs={'units':     'mm/day' if scale_mmday else 'm/s',
               'long_name': 'Forecast composite diurnal cycle',
               'dt_hours':  dt_hours},
    )


def load_forecast_diurnal_var(data_dir=DATA_DIR, start_dates=None, ens_members=None,
                               nlead_days=6, dt_hours=6,
                               var_name='TMQ', scale=None):
    """
    Load any CAM variable from the CESM2 S2S hindcast files and compute a
    lead-day diurnal cycle composite.

    This is a general version of load_forecast_diurnal for variables other
    than precipitation.  Unit scaling is taken from VAR_SETTINGS if not
    supplied explicitly.

    Parameters
    ----------
    data_dir    : str   hindcast file directory
    start_dates : list of str  'YYYY-MM-DD'
    ens_members : list of int  0-based ensemble indices
    nlead_days  : int   number of lead days
    dt_hours    : float data time step in hours
    var_name    : str   CAM variable name (e.g. 'TMQ', 'U10', 'PSL', 'PRECT')
                  For 'PRECT', falls back to PRECC+PRECL if PRECT is absent.
    scale       : float unit conversion factor; if None uses VAR_SETTINGS

    Returns
    -------
    dc_leads : xr.DataArray (lead_day, time_of_day, lat, lon)
    """
    if start_dates is None:
        start_dates = get_start_dates(data_dir)
    if ens_members is None:
        ens_members = list(range(5))

    vset  = VAR_SETTINGS.get(var_name, {})
    if scale is None:
        scale = vset.get('scale', 1.0)

    n_tod      = int(24 // dt_hours)
    nlead_days = min(nlead_days, 45)
    is_prect   = (var_name == 'PRECT')

    dc_sum  = None
    cnt_sum = None
    lat = lon = None

    n_total      = len(start_dates) * len(ens_members)
    n_done       = 0
    n_skipped    = 0
    n_no_var     = 0
    n_precc_precl = 0   # files where PRECC+PRECL was used instead of PRECT

    for start_date in start_dates:
        for ens in ens_members:
            fname = (f'cesm2cam6v2.{start_date}.{ens:02d}.cam.h3.'
                     f'{start_date}-00000.nc')
            fpath = os.path.join(data_dir, fname)
            if not os.path.isfile(fpath):
                n_skipped += 1
                n_done    += 1
                continue

            ds = xr.open_dataset(fpath, decode_times=False)

            # ── Variable extraction ──────────────────────────────────────────
            if is_prect:
                if 'PRECT' in ds:
                    raw = ds['PRECT'].values.astype(np.float32)
                elif 'PRECC' in ds and 'PRECL' in ds:
                    raw = (ds['PRECC'].values + ds['PRECL'].values
                           ).astype(np.float32)
                    n_precc_precl += 1
                else:
                    n_no_var += 1
                    ds.close()
                    n_done   += 1
                    continue
            else:
                if var_name not in ds:
                    n_no_var += 1
                    ds.close()
                    n_done   += 1
                    continue
                raw = ds[var_name].values.astype(np.float32)

            data = raw * scale

            if lat is None:
                lat  = ds['lat'].values.copy()
                lon  = ds['lon'].values.copy()
                nlat, nlon = lat.size, lon.size
                dc_sum  = np.zeros((nlead_days, n_tod, nlat, nlon), np.float64)
                cnt_sum = np.zeros((nlead_days, n_tod, nlat, nlon), np.int32)
            ds.close()

            for d in range(nlead_days):
                i0 = d * n_tod
                i1 = i0 + n_tod
                if i1 > data.shape[0]:
                    break
                chunk = data[i0:i1]
                valid = np.isfinite(chunk)
                dc_sum[d]  += np.where(valid, chunk, 0.0)
                cnt_sum[d] += valid.astype(np.int32)

            n_done += 1
            if n_done % 100 == 0:
                n_loaded = n_done - n_skipped - n_no_var
                print(f'  {n_done}/{n_total}  loaded={n_loaded}  '
                      f'missing={n_skipped}  no_var={n_no_var}')

    n_loaded = n_done - n_skipped - n_no_var
    print(f'  Done: {n_loaded}/{n_total} files loaded  '
          f'({n_skipped} missing, {n_no_var} without {var_name!r})')
    if is_prect and n_precc_precl > 0:
        print(f'  Note: {n_precc_precl} files used PRECC+PRECL '
              f'(PRECT not present)')

    if dc_sum is None:
        raise RuntimeError(
            f'No forecast files found for variable {var_name!r}.')

    with np.errstate(invalid='ignore', divide='ignore'):
        dc = np.where(cnt_sum > 0,
                      dc_sum / cnt_sum,
                      np.nan).astype(np.float32)

    dc, lon   = _roll_lon(dc, lon)
    tod       = np.arange(n_tod, dtype=float) * dt_hours
    lead_days = np.arange(1, nlead_days + 1)

    return xr.DataArray(
        dc,
        dims=['lead_day', 'time_of_day', 'lat', 'lon'],
        coords={'lead_day':    lead_days,
                'time_of_day': tod,
                'lat':         lat,
                'lon':         lon},
        attrs={'units':     vset.get('units', ''),
               'long_name': vset.get('long_name', var_name),
               'var_name':  var_name,
               'dt_hours':  dt_hours},
    )


def load_cesm2era5_diurnal(data_dir=DATA_DIR_ERA5, start_dates=None,
                       ens_members=None, nlead_days=10, dt_hours=3,
                       precip_vars=('PRECT',), scale_mmday=True):
    """
    Load CESM2/ERA5-init 3-hourly hindcast files and compute a lead-day
    diurnal cycle composite.

    File naming convention::

        cesm2_era5_YYYY-MM-DD.NN.cam.h4.YYYY-MM-DD-00000.nc

    Each 45-day forecast has 361 time steps at 3-hourly resolution
    (n_tod = 8 per calendar day).  Ensemble indices run 00–10 (11 members).

    Parameters
    ----------
    data_dir     : str   directory containing the 3-hourly ERA5-init files
    start_dates  : list of str 'YYYY-MM-DD'; if None uses all available dates
    ens_members  : list of int ensemble indices (0–10); if None uses [0..10]
    nlead_days   : int   number of lead days to composite (max 45)
    dt_hours     : float data time step in hours (default 3)
    precip_vars  : tuple of str variable name(s) to sum; only PRECT is in
                   these files so ('PRECT',) is the correct choice
    scale_mmday  : bool  if True convert m/s → mm/day

    Returns
    -------
    dc_leads : xr.DataArray (lead_day, time_of_day, lat, lon)
               Composite diurnal cycle [mm/day or m/s].
               Longitude is rolled to -180–180.
    """
    if start_dates is None:
        start_dates = get_start_dates_cesm2era5(data_dir)
    if ens_members is None:
        ens_members = list(range(11))

    n_tod      = int(24 // dt_hours)
    scale      = _MS_TO_MMDAY if scale_mmday else 1.0
    nlead_days = min(nlead_days, 45)

    dc_sum  = None
    cnt_sum = None
    lat = lon = None

    n_total   = len(start_dates) * len(ens_members)
    n_done    = 0
    n_skipped = 0
    n_no_var  = 0

    for start_date in start_dates:
        for ens in ens_members:
            fname = (f'cesm2_era5_{start_date}.{ens:02d}.cam.h4.'
                     f'{start_date}-00000.nc')
            fpath = os.path.join(data_dir, fname)
            if not os.path.isfile(fpath):
                n_skipped += 1
                n_done    += 1
                continue

            ds = xr.open_dataset(fpath, decode_times=False)

            pr = None
            for v in precip_vars:
                if v in ds:
                    arr = ds[v].values.astype(np.float32)
                    pr  = arr if pr is None else pr + arr
            if pr is None:
                n_no_var += 1
                ds.close()
                n_done   += 1
                continue
            pr[pr < 0] = np.nan
            pr *= scale

            if lat is None:
                lat = ds['lat'].values.copy()
                lon = ds['lon'].values.copy()
                nlat, nlon = lat.size, lon.size
                dc_sum  = np.zeros((nlead_days, n_tod, nlat, nlon), np.float64)
                cnt_sum = np.zeros((nlead_days, n_tod, nlat, nlon), np.int32)
            ds.close()

            for d in range(nlead_days):
                i0 = d * n_tod
                i1 = i0 + n_tod
                if i1 > pr.shape[0]:
                    break
                chunk = pr[i0:i1]
                valid = np.isfinite(chunk)
                dc_sum[d]  += np.where(valid, chunk, 0.0)
                cnt_sum[d] += valid.astype(np.int32)

            n_done += 1
            if n_done % 100 == 0:
                n_loaded = n_done - n_skipped - n_no_var
                print(f'  {n_done}/{n_total}  loaded={n_loaded}  '
                      f'missing={n_skipped}  no_precip={n_no_var}')

    n_loaded = n_done - n_skipped - n_no_var
    print(f'  Done: {n_loaded}/{n_total} files loaded  '
          f'({n_skipped} missing, {n_no_var} no precip variable)')

    if dc_sum is None:
        raise RuntimeError(
            'No CESM2-ERA5init forecast files found for the requested dates/members.')

    with np.errstate(invalid='ignore', divide='ignore'):
        dc = np.where(cnt_sum > 0,
                      dc_sum / cnt_sum,
                      np.nan).astype(np.float32)

    dc, lon   = _roll_lon(dc, lon)
    tod       = np.arange(n_tod, dtype=float) * dt_hours
    lead_days = np.arange(1, nlead_days + 1)

    return xr.DataArray(
        dc,
        dims=['lead_day', 'time_of_day', 'lat', 'lon'],
        coords={'lead_day':    lead_days,
                'time_of_day': tod,
                'lat':         lat,
                'lon':         lon},
        attrs={'units':     'mm/day' if scale_mmday else 'm/s',
               'long_name': 'ERA5-init composite diurnal cycle',
               'dt_hours':  dt_hours,
               'source':    'cesm2_era5 3-hourly hindcasts'},
    )


def load_cesm2era5_diurnal_var(data_dir=DATA_DIR_ERA5, start_dates=None,
                           ens_members=None, nlead_days=10, dt_hours=3,
                           var_name='TMQ', scale=None):
    """
    Load any CAM variable from the CESM2/ERA5-init 3-hourly files and compute
    a lead-day diurnal cycle composite.

    Available variables in these files: PRECT, PS, PSL, QBOT, TMQ, TS,
    UBOT, VBOT.  Unit scaling is taken from VAR_SETTINGS when not supplied.

    Parameters
    ----------
    data_dir    : str   path to the 3-hourly ERA5-init directory
    start_dates : list of str 'YYYY-MM-DD'
    ens_members : list of int ensemble indices (0–10)
    nlead_days  : int   number of lead days (max 45)
    dt_hours    : float data time step in hours (default 3)
    var_name    : str   CAM variable name
    scale       : float unit conversion; if None uses VAR_SETTINGS

    Returns
    -------
    dc_leads : xr.DataArray (lead_day, time_of_day, lat, lon)
    """
    if start_dates is None:
        start_dates = get_start_dates_cesm2era5(data_dir)
    if ens_members is None:
        ens_members = list(range(11))

    vset  = VAR_SETTINGS.get(var_name, {})
    if scale is None:
        scale = vset.get('scale', 1.0)

    n_tod      = int(24 // dt_hours)
    nlead_days = min(nlead_days, 45)
    is_prect   = (var_name == 'PRECT')

    dc_sum  = None
    cnt_sum = None
    lat = lon = None

    n_total   = len(start_dates) * len(ens_members)
    n_done    = 0
    n_skipped = 0
    n_no_var  = 0

    for start_date in start_dates:
        for ens in ens_members:
            fname = (f'cesm2_era5_{start_date}.{ens:02d}.cam.h4.'
                     f'{start_date}-00000.nc')
            fpath = os.path.join(data_dir, fname)
            if not os.path.isfile(fpath):
                n_skipped += 1
                n_done    += 1
                continue

            ds = xr.open_dataset(fpath, decode_times=False)

            if is_prect:
                if 'PRECT' in ds:
                    raw = ds['PRECT'].values.astype(np.float32)
                else:
                    n_no_var += 1
                    ds.close()
                    n_done   += 1
                    continue
            else:
                if var_name not in ds:
                    n_no_var += 1
                    ds.close()
                    n_done   += 1
                    continue
                raw = ds[var_name].values.astype(np.float32)

            data = raw * scale

            if lat is None:
                lat  = ds['lat'].values.copy()
                lon  = ds['lon'].values.copy()
                nlat, nlon = lat.size, lon.size
                dc_sum  = np.zeros((nlead_days, n_tod, nlat, nlon), np.float64)
                cnt_sum = np.zeros((nlead_days, n_tod, nlat, nlon), np.int32)
            ds.close()

            for d in range(nlead_days):
                i0 = d * n_tod
                i1 = i0 + n_tod
                if i1 > data.shape[0]:
                    break
                chunk = data[i0:i1]
                valid = np.isfinite(chunk)
                dc_sum[d]  += np.where(valid, chunk, 0.0)
                cnt_sum[d] += valid.astype(np.int32)

            n_done += 1
            if n_done % 100 == 0:
                n_loaded = n_done - n_skipped - n_no_var
                print(f'  {n_done}/{n_total}  loaded={n_loaded}  '
                      f'missing={n_skipped}  no_var={n_no_var}')

    n_loaded = n_done - n_skipped - n_no_var
    print(f'  Done: {n_loaded}/{n_total} files loaded  '
          f'({n_skipped} missing, {n_no_var} without {var_name!r})')

    if dc_sum is None:
        raise RuntimeError(
            f'No ERA5-init forecast files found for variable {var_name!r}.')

    with np.errstate(invalid='ignore', divide='ignore'):
        dc = np.where(cnt_sum > 0,
                      dc_sum / cnt_sum,
                      np.nan).astype(np.float32)

    dc, lon   = _roll_lon(dc, lon)
    tod       = np.arange(n_tod, dtype=float) * dt_hours
    lead_days = np.arange(1, nlead_days + 1)

    return xr.DataArray(
        dc,
        dims=['lead_day', 'time_of_day', 'lat', 'lon'],
        coords={'lead_day':    lead_days,
                'time_of_day': tod,
                'lat':         lat,
                'lon':         lon},
        attrs={'units':     vset.get('units', ''),
               'long_name': vset.get('long_name', var_name),
               'var_name':  var_name,
               'dt_hours':  dt_hours,
               'source':    'cesm2_era5 3-hourly hindcasts'},
    )


def load_imerg_diurnal(imerg_dir=IMERG_DIR, season=None,
                       year_start=None, year_end=None):
    """
    Load GPM IMERG 3-hourly 1-degree monthly files and compute the mean
    diurnal cycle for the requested season and year range.

    Files must be named  IMERG_3hr.YYYYMM.1deg.nc  with a 'precip' variable
    in mm/hr.  Year filtering matches the same convention used for the
    forecast start-dates: a file's month must be in the season AND its year
    must be in [year_start, year_end].

    Parameters
    ----------
    imerg_dir  : str  directory containing the monthly IMERG files
    season     : str  season code recognised by get_season_months
                 (e.g. 'DJF', 'JJA', 'ANN'); None = all months
    year_start : int  first year inclusive; None = no lower bound
    year_end   : int  last year inclusive;  None = no upper bound

    Returns
    -------
    dc_imerg : xr.DataArray (time_of_day, lat, lon)
               Mean precipitation at each UTC time-of-day [mm/day].
               Attributes: units, dt_hours=3, source.
    """
    # Build month whitelist
    months_ok = None
    if season is not None:
        months_ok, _ = get_season_months(season)

    # Collect matching files
    all_files = sorted(
        glob.glob(os.path.join(imerg_dir, 'IMERG_3hr.??????.1deg.nc')))
    files = []
    for f in all_files:
        ym = os.path.basename(f).split('.')[1]   # 'YYYYMM'
        yr, mo = int(ym[:4]), int(ym[4:])
        if year_start is not None and yr < year_start:
            continue
        if year_end   is not None and yr > year_end:
            continue
        if months_ok  is not None and mo not in months_ok:
            continue
        files.append(f)

    if not files:
        raise RuntimeError(
            f'No IMERG files found for season={season} '
            f'{year_start}–{year_end} in {imerg_dir}')

    print(f'  IMERG: loading {len(files)} monthly files '
          f'({season} {year_start}–{year_end}) ...')

    # 3-hourly time-of-day bins: 0, 3, 6, ..., 21
    dt_imerg  = 3.0
    n_tod     = int(24 // dt_imerg)
    tod_hours = np.arange(n_tod) * dt_imerg   # [0, 3, 6, ..., 21]

    dc_sum = None
    dc_cnt = None
    lat = lon = None

    for f in files:
        # drop 'datesec' — it has non-standard units ("seconds since start of day")
        # that xarray cannot decode; we derive the hour from 'yyyymmddhh' instead.
        ds  = xr.open_dataset(f, decode_times=False,
                              drop_variables=['datesec'])
        pr  = ds['precip'].values.astype(np.float32)  # (ntime, nlat, nlon)
        pr[pr < 0] = np.nan
        hour = ds['yyyymmddhh'].values % 100           # (ntime,) — last 2 digits = UTC hour

        if lat is None:
            lat = ds['lat'].values.copy()
            lon = ds['lon'].values.copy()
            dc_sum = np.zeros((n_tod, len(lat), len(lon)), np.float64)
            dc_cnt = np.zeros((n_tod, len(lat), len(lon)), np.int32)
        ds.close()

        for it, h in enumerate(tod_hours.astype(int)):
            idx = hour == h
            if not idx.any():
                continue
            chunk = pr[idx]                        # (n_at_h, nlat, nlon)
            valid = np.isfinite(chunk)
            dc_sum[it] += np.where(valid, chunk, 0.0).sum(axis=0)
            dc_cnt[it] += valid.astype(np.int32).sum(axis=0)

    print(f'  IMERG: done.')

    with np.errstate(invalid='ignore', divide='ignore'):
        dc = np.where(dc_cnt > 0,
                      dc_sum / dc_cnt,
                      np.nan).astype(np.float32)
    dc *= _MMHR_TO_MMDAY   # mm/hr → mm/day

    return xr.DataArray(
        dc,
        dims=['time_of_day', 'lat', 'lon'],
        coords={'time_of_day': tod_hours,
                'lat':         lat,
                'lon':         lon},
        attrs={'units':     'mm/day',
               'long_name': 'IMERG mean diurnal cycle',
               'dt_hours':  dt_imerg,
               'source':    'GPM IMERG V07B 3-hourly 1-degree'},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Experiment registry and unified loader
# ─────────────────────────────────────────────────────────────────────────────

# Maps experiment name → configuration used by load_experiment / get_start_dates_for.
# 'n_ens'  : total ensemble members available (use None to override via ens_members arg)
# 'label'  : human-readable description for plot titles
EXPERIMENTS = {
    'CESM2-CAM6v2': {
        'label':    'CESM2 CAM6v2 S2S hindcasts (6-hourly)',
        'data_dir': DATA_DIR,
        'dt_hours': 6,
        'n_ens':    21,
    },
    'CESM2-ERA5init': {
        'label':    'CESM2 ERA5-initialized hindcasts (3-hourly)',
        'data_dir': DATA_DIR_ERA5,
        'dt_hours': 3,
        'n_ens':    11,
    },
}


def get_start_dates_for(experiment, data_dir=None, season=None,
                         year_start=None, year_end=None):
    """
    Return available start dates for the given experiment name.

    Dispatches to the correct underlying get_start_dates function.
    experiment : str  key in EXPERIMENTS ('CESM2-CAM6v2' or 'CESM2-ERA5init')
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}. "
                         f"Choose: {list(EXPERIMENTS)}")
    cfg = EXPERIMENTS[experiment]
    if data_dir is None:
        data_dir = cfg['data_dir']
    if experiment == 'CESM2-CAM6v2':
        return get_start_dates(data_dir, season=season,
                               year_start=year_start, year_end=year_end)
    else:  # CESM2-ERA5init
        return get_start_dates_cesm2era5(data_dir, season=season,
                                          year_start=year_start, year_end=year_end)


def load_experiment(experiment, var_name='PRECT', data_dir=None,
                    start_dates=None, ens_members=None,
                    nlead_days=10, dt_hours=None, scale=None):
    """
    Load a diurnal cycle composite for the given experiment and variable.

    Dispatches to the correct underlying loader based on experiment name.
    All downstream analysis functions (compute_lead_harmonics, plotting)
    accept the returned DataArray without modification.

    Parameters
    ----------
    experiment  : str   key in EXPERIMENTS
    var_name    : str   CAM variable (e.g. 'PRECT', 'TMQ', 'PSL', 'TS', 'UBOT')
    data_dir    : str   override default data directory
    start_dates : list of str 'YYYY-MM-DD'; None = all available
    ens_members : list of int; None = all available for this experiment
    nlead_days  : int
    dt_hours    : float; None = use experiment default
    scale       : float; None = use VAR_SETTINGS default

    Returns
    -------
    dc_leads : xr.DataArray (lead_day, time_of_day, lat, lon)
    """
    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment {experiment!r}. "
                         f"Choose: {list(EXPERIMENTS)}")
    cfg = EXPERIMENTS[experiment]
    if data_dir is None:
        data_dir = cfg['data_dir']
    if dt_hours is None:
        dt_hours = cfg['dt_hours']
    if ens_members is None:
        ens_members = list(range(cfg['n_ens']))

    if experiment == 'CESM2-CAM6v2':
        return load_forecast_diurnal_var(
            data_dir=data_dir, start_dates=start_dates,
            ens_members=ens_members, nlead_days=nlead_days,
            dt_hours=dt_hours, var_name=var_name, scale=scale)
    else:  # CESM2-ERA5init
        return load_cesm2era5_diurnal_var(
            data_dir=data_dir, start_dates=start_dates,
            ens_members=ens_members, nlead_days=nlead_days,
            dt_hours=dt_hours, var_name=var_name, scale=scale)


def auto_plot_ranges(dc_leads, n_levels=12):
    """
    Derive data-driven contour levels and Evans amplitude bounds from the
    loaded diurnal composite.

    Uses percentiles of the actual data so the output is appropriate for
    any variable (precipitation, TMQ, U10, PSL, …).

    Parameters
    ----------
    dc_leads  : xr.DataArray (lead_day, time_of_day, lat, lon)
    n_levels  : int  approximate number of contour levels to produce

    Returns
    -------
    dict with keys
        'levels'  : list of float  contour levels for mean map
        'min_amp' : float  Evans colour wheel inner radius
        'max_amp' : float  Evans colour wheel outer radius
    """
    import matplotlib.ticker as mticker

    dc_vals = dc_leads.values   # (nlead, n_tod, nlat, nlon)

    # ── Mean-field levels ────────────────────────────────────────────────────
    mean_field = np.nanmean(dc_vals, axis=(0, 1))   # (nlat, nlon)
    flat       = mean_field[np.isfinite(mean_field)]
    vmin = float(np.nanpercentile(flat,  2))
    vmax = float(np.nanpercentile(flat, 98))

    locator = mticker.MaxNLocator(n_levels, symmetric=False)
    levels  = [float(v) for v in locator.tick_values(vmin, vmax)]

    # ── Amplitude range ──────────────────────────────────────────────────────
    # Proxy: (max - min) / 2 over time-of-day axis, then averaged over leads
    amp_proxy = (np.nanmax(dc_vals, axis=1) -
                 np.nanmin(dc_vals, axis=1)) / 2.0   # (nlead, nlat, nlon)
    amp_flat  = amp_proxy[np.isfinite(amp_proxy) & (amp_proxy > 0)]

    if len(amp_flat) > 100:
        min_amp = float(np.nanpercentile(amp_flat, 10))
        max_amp = float(np.nanpercentile(amp_flat, 95))
        # Round to 2 significant figures for cleaner wheel labels
        from math import log10, floor
        def _round2sf(x):
            if x <= 0:
                return 0.0
            mag = 10 ** floor(log10(abs(x)))
            return round(x / mag, 1) * mag
        min_amp = _round2sf(min_amp)
        max_amp = _round2sf(max_amp)
    else:
        min_amp, max_amp = 0.0, 1.0

    return {'levels': levels, 'min_amp': min_amp, 'max_amp': max_amp}


# ─────────────────────────────────────────────────────────────────────────────
# Harmonic analysis across lead days
# ─────────────────────────────────────────────────────────────────────────────

def compute_lead_harmonics(dc_leads, n_harm=2, dt_hours=6):
    """
    Compute harmonic decomposition for each lead day.

    Parameters
    ----------
    dc_leads : xr.DataArray (lead_day, time_of_day, lat, lon)
    n_harm   : int   number of harmonics
    dt_hours : float data time step

    Returns
    -------
    amplitude : ndarray (nlead, n_harm, nlat, nlon)
    phase_utc : ndarray (nlead, n_harm, nlat, nlon)
    var_exp   : ndarray (nlead, n_harm, nlat, nlon)
    mean_prcp : ndarray (nlead, nlat, nlon)
    """
    nlead = dc_leads.sizes['lead_day']
    amp_list, ph_list, ve_list, mp_list = [], [], [], []

    for d in range(nlead):
        dc_d = dc_leads.isel(lead_day=d)
        amp, ph, ve, mp = dcutils.compute_harmonics(dc_d, n_harm=n_harm,
                                                     dt_hours=dt_hours)
        amp_list.append(amp)
        ph_list.append(ph)
        ve_list.append(ve)
        mp_list.append(mp)

    return (np.stack(amp_list),   # (nlead, n_harm, nlat, nlon)
            np.stack(ph_list),
            np.stack(ve_list),
            np.stack(mp_list))    # (nlead, nlat, nlon)


# ─────────────────────────────────────────────────────────────────────────────
# Panel figure: 6 maps, one per lead day
# ─────────────────────────────────────────────────────────────────────────────

def plot_forecast_panel(amplitude, phase_lst, lat, lon,
                        lead_days,
                        min_amp=0.5, max_amp=6.0,
                        period_hours=24.0, hue_offset=0.5,
                        lat_range=(15., 60.), lon_range=(-140., -55.),
                        discrete_wheel=True, dt_hours=6,
                        show_states=False, n_cols=5,
                        title='', figsize=(24, 8)):
    """
    Panel figure: Evans phase/amplitude map for each lead day in a 2×n_cols
    grid with a shared color wheel legend on the right.

    Parameters
    ----------
    amplitude    : (nlead, nlat, nlon)
    phase_lst    : (nlead, nlat, nlon)
    lat, lon     : 1-D coordinate arrays
    lead_days    : 1-D array of lead day labels
    n_cols       : int  map columns per row (default 5 → 5×2 = 10 panels)
    """
    n_panels = min(len(lead_days), n_cols * 2)

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(
        2, n_cols + 1,
        width_ratios=[3] * n_cols + [0.8],
        left=0.03, right=0.97,
        bottom=0.06, top=0.90,
        wspace=0.06, hspace=0.22,
    )

    # Shared color wheel — right column, spans both rows
    ax_wheel = fig.add_subplot(gs[:, n_cols], projection='polar')

    for ip in range(n_panels):
        row, col = divmod(ip, n_cols)
        ax = fig.add_subplot(
            gs[row, col],
            projection=ccrs.PlateCarree(central_longitude=180),
        )
        ld = lead_days[ip]

        # Always pass the shared wheel axes — avoids fallback that adds a
        # new polar axes inside the map domain for panels 1–(n_panels-1).
        _wheel = ax_wheel

        dcutils.plot_evans_map(
            ax, fig,
            phase_lst    = phase_lst[ip],
            amplitude    = amplitude[ip],
            lat=lat, lon=lon,
            min_amp      = min_amp,
            max_amp      = max_amp,
            period_hours = period_hours,
            title        = f'Lead day {ld}',
            hue_offset   = hue_offset,
            lat_range    = lat_range,
            lon_range    = lon_range,
            ax_wheel     = _wheel,
            discrete_wheel = discrete_wheel,
            dt_hours     = dt_hours,
            show_states  = show_states,
        )

    if title:
        fig.suptitle(title, fontsize=12, y=0.96)

    return fig


def plot_forecast_mean_panel(mean_prcp, lat, lon,
                              lead_days,
                              lat_range=(15., 60.), lon_range=(-140., -55.),
                              show_states=False, n_cols=5, levels=None,
                              cmap=None, units='mm/day',
                              title='', figsize=(24, 8)):
    """
    Panel figure: mean value map for each lead day in a 2×n_cols grid.

    Parameters
    ----------
    mean_prcp : (nlead, nlat, nlon)  time-mean field
    lat, lon  : 1-D coordinate arrays
    lead_days : 1-D array of lead day labels
    lat_range, lon_range : map extent tuples
    levels    : contour levels; None = default precipitation levels
    cmap      : matplotlib colormap; None = default precipitation colormap
    units     : str  colorbar label
    title     : str
    figsize   : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    n_panels = min(len(lead_days), n_cols * 2)

    fig = plt.figure(figsize=figsize)
    gs  = gridspec.GridSpec(
        2, n_cols,
        left=0.03, right=0.97,
        bottom=0.08, top=0.90,
        wspace=0.06, hspace=0.22,
    )

    cf_last = None
    for ip in range(n_panels):
        row, col = divmod(ip, n_cols)
        ax = fig.add_subplot(
            gs[row, col],
            projection=ccrs.PlateCarree(central_longitude=180),
        )
        ld = lead_days[ip]
        cf_last = dcutils.plot_mean_precip(
            ax, mean_prcp[ip], lat, lon,
            title=f'Lead day {ld}',
            levels=levels,
            cmap=cmap if cmap is not None else dcutils.CMAP_PRCP,
            lat_range=lat_range, lon_range=lon_range,
            show_states=show_states,
        )

    # Shared colorbar below the panels
    cbar_ax = fig.add_axes([0.10, 0.02, 0.80, 0.02])
    cbar = fig.colorbar(cf_last, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(units, fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    if title:
        fig.suptitle(title, fontsize=12, y=0.96)

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Regional diurnal cycle line plots
# ─────────────────────────────────────────────────────────────────────────────

# Predefined regions: (lat_min, lat_max, lon_min, lon_max, mask_type)
# mask_type: 'land' = land points only, 'ocean' = ocean points only, None = all
REGIONS = {
    'Midwest':        (35., 45., -100.,  -90., None),
    'Southeast US':   (25., 35.,  -92.,  -78., 'land'),   # FL, GA, SC, MS, AL
    'SE Coast Ocean': (30., 35.,  -80.,  -70., 'ocean'),  # ocean off SE US coast
    'Great Plains':   (35., 48., -105.,  -95., None),
    'Gulf Coast':     (25., 32.,  -97.,  -80., None),
}

# ── Land/sea mask (cached per grid) ──────────────────────────────────────────
_land_mask_cache = {}   # key: (nlat, nlon)

def _get_land_mask(lat, lon):
    """
    Boolean array (nlat, nlon) — True where land, False where ocean.
    Uses regionmask Natural Earth 1:110m land polygons.
    Result is cached so the mask is only computed once per unique grid size.
    """
    key = (len(lat), len(lon))
    if key not in _land_mask_cache:
        import regionmask
        land = regionmask.defined_regions.natural_earth_v5_0_0.land_110
        mask_da = land.mask(lon, lat)   # DataArray: 0=land, NaN=ocean
        _land_mask_cache[key] = ~np.isnan(mask_da.values)  # True=land
    return _land_mask_cache[key]

# Distinct colors for up to 10 lead days (tab10 palette)
_LEAD_COLORS = [
    '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


def plot_region_boxes(regions=None,
                      lat_range=(20., 55.), lon_range=(-130., -60.),
                      show_states=True,
                      title='Averaging regions',
                      figsize=(8, 5)):
    """
    Map showing the lat/lon boxes used for regional averaging.

    Parameters
    ----------
    regions   : dict  {name: (lat_min, lat_max, lon_min, lon_max)}
                defaults to Midwest and Southeast US
    lat_range, lon_range : map extent
    show_states : bool
    title     : str
    figsize   : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    import matplotlib.patches as mpatches

    if regions is None:
        regions = {k: REGIONS[k] for k in ('Midwest', 'Southeast US')}

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
    )
    ax.set_extent([lon_range[0], lon_range[1], lat_range[0], lat_range[1]],
                  crs=ccrs.PlateCarree())
    ax.coastlines(linewidth=0.8, color='black', zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.4, edgecolor='grey', zorder=3)
    ax.add_feature(cfeature.LAND,   facecolor='#f5f5f0', zorder=0)
    ax.add_feature(cfeature.OCEAN,  facecolor='#d0e8f5', zorder=0)
    if show_states:
        ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='black', zorder=3)

    gl = ax.gridlines(linewidth=0.3, color='grey', linestyle='--',
                      draw_labels=True, zorder=2)
    gl.top_labels = gl.right_labels = False

    colors = _LEAD_COLORS[:len(regions)]
    handles = []
    for (reg_name, reg_def), color in zip(regions.items(), colors):
        lat_min, lat_max, lon_min, lon_max = reg_def[:4]
        mask_type = reg_def[4] if len(reg_def) > 4 else None
        mask_label = f'\n({mask_type} only)' if mask_type else ''

        rect = mpatches.Rectangle(
            xy=(lon_min, lat_min),
            width=lon_max - lon_min,
            height=lat_max - lat_min,
            linewidth=2.0, edgecolor=color,
            facecolor=color, alpha=0.25,
            transform=ccrs.PlateCarree(), zorder=4,
        )
        ax.add_patch(rect)
        # Label inside box
        ax.text(
            0.5 * (lon_min + lon_max), 0.5 * (lat_min + lat_max),
            f'{reg_name}{mask_label}', ha='center', va='center',
            fontsize=9, fontweight='bold', color=color,
            transform=ccrs.PlateCarree(), zorder=5,
        )
        handles.append(mpatches.Patch(facecolor=color, alpha=0.5,
                                       edgecolor=color, label=f'{reg_name}{mask_label}'))

    ax.legend(handles=handles, loc='lower left', fontsize=8)
    ax.set_title(title, fontsize=11)
    fig.tight_layout()
    return fig


def _region_mean(dc_leads, lat, lon, lat_min, lat_max, lon_min, lon_max,
                 mask_type=None):
    """
    Area-weighted (cos-lat) mean of dc_leads over a lat/lon box.

    Parameters
    ----------
    mask_type : str or None
        'land'  — include only land grid points (via regionmask)
        'ocean' — include only ocean grid points
        None    — include all grid points

    Returns (nlead, n_tod) float32.
    """
    data = dc_leads.values if isinstance(dc_leads, xr.DataArray) else dc_leads

    lat_mask = (lat >= lat_min) & (lat <= lat_max)
    lon_mask = (lon >= lon_min) & (lon <= lon_max)

    weights = np.cos(np.deg2rad(lat[lat_mask]))[:, np.newaxis]  # (nlat_r, 1)

    sub = data[:, :, lat_mask, :][:, :, :, lon_mask]   # (nlead, n_tod, nlat_r, nlon_r)

    # Apply land/ocean mask if requested
    if mask_type is not None:
        land_mask = _get_land_mask(lat, lon)              # (nlat, nlon) bool
        geo_sub   = land_mask[lat_mask, :][:, lon_mask]  # (nlat_r, nlon_r)
        if mask_type == 'ocean':
            geo_sub = ~geo_sub
        # Mask out unwanted points by setting to NaN
        sub = np.where(geo_sub[np.newaxis, np.newaxis, :, :], sub, np.nan)

    wsum  = np.nansum(sub  * weights[np.newaxis, np.newaxis], axis=2)
    wcnt  = np.nansum(np.isfinite(sub).astype(float) * weights[np.newaxis, np.newaxis], axis=2)
    wmean = np.where(wcnt > 0, wsum / wcnt, np.nan)    # (nlead, n_tod, nlon_r)
    return np.nanmean(wmean, axis=2).astype(np.float32)                       # (nlead, n_tod)


def plot_regional_diurnal_lines(dc_leads, lat, lon,
                                 regions=None,
                                 dt_hours=6,
                                 ylabel=None,
                                 title_prefix='',
                                 figsize=(10, 4),
                                 dir_fig=None,
                                 fname_prefix='forecast_regional',
                                 ylim=None):
    """
    For each region, plot one figure with one line per lead day showing the
    raw area-mean diurnal cycle (x = local solar time, y = mm/day).

    Parameters
    ----------
    dc_leads     : xr.DataArray (lead_day, time_of_day, lat, lon)
    lat, lon     : 1-D coordinate arrays (lon in -180–180)
    regions      : dict  {name: (lat_min, lat_max, lon_min, lon_max)}
                   defaults to Central US and Southeast US
    dt_hours     : float  data time step
    title_prefix : str   prepended to each plot title
    figsize      : tuple
    dir_fig      : str or None  save directory; None = don't save
    fname_prefix : str   filename prefix

    Returns
    -------
    figs : list of matplotlib Figure
    """
    if regions is None:
        regions = {k: REGIONS[k] for k in ('Central US', 'Southeast US')}

    lead_days = dc_leads['lead_day'].values
    tod_utc   = dc_leads['time_of_day'].values   # UTC hours

    colors = (_LEAD_COLORS * ((len(lead_days) // len(_LEAD_COLORS)) + 1))[:len(lead_days)]

    figs = []
    for reg_name, reg_def in regions.items():
        lat_min, lat_max, lon_min, lon_max = reg_def[:4]
        mask_type = reg_def[4] if len(reg_def) > 4 else None

        # UTC → LST using region longitude midpoint
        lon_mid    = 0.5 * (lon_min + lon_max)
        utc_to_lst = lon_mid / 15.0
        tod_lst    = (tod_utc + utc_to_lst) % 24.0
        sort_idx   = np.argsort(tod_lst)
        tod_plot   = tod_lst[sort_idx]

        reg_mean = _region_mean(dc_leads, lat, lon,
                                lat_min, lat_max, lon_min, lon_max,
                                mask_type=mask_type)  # (nlead, n_tod)

        fig, ax = plt.subplots(figsize=figsize)

        # Replicate data shifted ±24 h so the line connects cyclically through
        # midnight (0/24 h boundary), then clip with xlim — same idea as cyclic
        # longitude.  The three copies give a continuous line entering/exiting
        # both edges of the 0–24 h window.
        tod_cyc = np.concatenate([tod_plot - 24, tod_plot, tod_plot + 24])

        for il, ld in enumerate(lead_days):
            vals     = reg_mean[il][sort_idx]
            vals_cyc = np.tile(vals, 3)
            ax.plot(tod_cyc, vals_cyc, color=colors[il],
                    linewidth=1.8, marker='o', markersize=4,
                    label=f'Lead day {ld}')

        ax.set_xlim(0, 24)
        ax.set_xticks(range(0, 25, 3))
        _ylabel = ylabel if ylabel is not None else 'Precipitation (mm/day)'
        ax.set_xlabel('Local Solar Time (h)', fontsize=10)
        ax.set_ylabel(_ylabel, fontsize=10)
        # Tight y-limits from data range; can be overridden via ylim kwarg
        if ylim is not None:
            ax.set_ylim(ylim)
        else:
            _dmin, _dmax = float(reg_mean.min()), float(reg_mean.max())
            _pad = max(0.05 * (_dmax - _dmin), 0.05)
            ax.set_ylim(_dmin - _pad, _dmax + _pad)
        mask_label = f'  [{mask_type} only]' if mask_type else ''
        ax.set_title(
            f'{title_prefix}  {reg_name}{mask_label}\n'
            f'({lat_min}°–{lat_max}°N, {lon_min}°–{lon_max}°E)',
            fontsize=10)
        ax.legend(fontsize=8, ncol=2, loc='upper left')
        ax.grid(linewidth=0.4, linestyle='--', color='grey', alpha=0.6)
        ax.axhline(0, color='black', linewidth=0.5)
        fig.tight_layout()

        if dir_fig is not None:
            os.makedirs(dir_fig, exist_ok=True)
            safe  = reg_name.lower().replace(' ', '_')
            mask_tag = f'_{mask_type}' if mask_type else ''
            fpath = os.path.join(dir_fig,
                                 f'{fname_prefix}_{safe}{mask_tag}.png')
            fig.savefig(fpath, dpi=150, bbox_inches='tight')
            print(f'  Saved: {fpath}')

        figs.append(fig)

    print(f'Regional line plots: {len(figs)} figures saved to {dir_fig}')
    return figs
