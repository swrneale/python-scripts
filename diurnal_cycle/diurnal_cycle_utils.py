"""
diurnal_cycle_utils.py
Diurnal cycle analysis utilities — Python equivalent of NCL's evans_plot.ncl / calc_utils.ncl

Key capabilities:
  - Load TRMM 3-hourly monthly files and compute weighted seasonal composite diurnal cycles
  - Load ERA5 or CESM 3-hourly data
  - FFT harmonic analysis (amplitude, phase, variance explained)
  - Phase conversion from UTC to local solar time
  - Evans-style phase/amplitude maps (phase=hue, amplitude=saturation on a color wheel)
  - Color wheel legend in polar inset
  - Mean precipitation maps
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import glob
import os

# ---------------------------------------------------------------------------
# Season definitions
# ---------------------------------------------------------------------------

SEASON_MONTHS = {
    'ANN':  ([1,2,3,4,5,6,7,8,9,10,11,12], [31,28,31,30,31,30,31,31,30,31,30,31]),
    'DJF':  ([12,1,2],   [31,31,28]),
    'MAM':  ([3,4,5],    [31,30,31]),
    'JJA':  ([6,7,8],    [30,31,31]),
    'SON':  ([9,10,11],  [30,31,30]),
    'JFM':  ([1,2,3],    [31,28,31]),
    'AMJ':  ([4,5,6],    [30,31,30]),
    'JAS':  ([7,8,9],    [31,31,30]),
    'OND':  ([10,11,12], [31,30,31]),
    'NDJ':  ([11,12,1],  [30,31,31]),
    'MJ':   ([5,6],      [31,30]),
    'JA':   ([7,8],      [31,31]),
    'JJAS': ([6,7,8,9],  [30,31,31,30]),
    'MJJA': ([5,6,7,8],  [31,30,31,31]),
}

_LEAP_YEARS = {y for y in range(1900, 2200)
               if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0))}

# Precipitation colormap (approximating NCL's WhiteBlueGreenYellowRed)
_PRCP_COLORS = [
    'white',  '#d4f4ff', '#a8dff5', '#7bcaec', '#4db5e3',
    '#27a0da', '#1e8c31', '#82c91e', '#f5d400', '#f59900',
    '#f55800', '#cc0000', '#7a0000',
]
CMAP_PRCP = LinearSegmentedColormap.from_list('prcp', _PRCP_COLORS)

# Default contour levels for mean precipitation (mm/day)
PRCP_LEVELS = [0, 0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 5, 6, 7.5, 10]


def get_season_months(season):
    """Return (month_list, day_weights) for a season string."""
    if season in SEASON_MONTHS:
        return list(SEASON_MONTHS[season][0]), list(SEASON_MONTHS[season][1])
    try:
        m = int(season)
        day_map = {1:31,2:28,3:31,4:30,5:31,6:30,7:31,8:31,9:30,10:31,11:30,12:31}
        return [m], [day_map[m]]
    except ValueError:
        raise ValueError(f"Unknown season '{season}'. Choices: {list(SEASON_MONTHS)}")


# ---------------------------------------------------------------------------
# TRMM data loading
# ---------------------------------------------------------------------------

def load_trmm_composite(data_dir, year_start, year_end, season='DJF',
                        dt_hours=3, scale_mmday=True):
    """
    Load TRMM 3-hourly monthly files and build a weighted composite diurnal
    cycle for the requested season and year range.

    Files expected: <data_dir>/3B42.YYYYMM.3hr_V7.nc
    Variable 'precip' in mm/hr; yyyymmddhh gives integer timestamps.

    Parameters
    ----------
    data_dir : str
    year_start, year_end : int  inclusive
    season : str  e.g. 'DJF', 'JJA'
    dt_hours : float  data time step (3 for TRMM 3B42)
    scale_mmday : bool  if True, convert mm/hr → mm/day (* 24)

    Returns
    -------
    dc : xr.DataArray  (time_of_day, lat, lon)  mm/day composite
    """
    months, mwt_base = get_season_months(season)
    n_tod = int(24 // dt_hours)   # 8 for 3-hourly
    scale = 24.0 if scale_mmday else 1.0

    dc_sum  = None
    wgt_sum = None
    lat = lon = None

    for year in range(year_start, year_end + 1):
        for im, month in enumerate(months):
            fpath = os.path.join(data_dir,
                                 f'3B42.{year:04d}{month:02d}.3hr_V7.nc')
            if not os.path.isfile(fpath):
                print(f'  [skip] {os.path.basename(fpath)} not found')
                continue

            print(f'  {os.path.basename(fpath)} ...', end=' ')

            ds = xr.open_dataset(fpath, decode_times=False)
            precip      = ds['precip'].values.astype(np.float32)  # (ntime, nlat, nlon)
            yyyymmddhh  = ds['yyyymmddhh'].values                  # integer timestamps

            if lat is None:
                lat = ds['lat'].values.copy()
                lon = ds['lon'].values.copy()
            ds.close()

            # Fill value → NaN
            precip = np.where(precip < -9000., np.nan, precip)
            # Scale to mm/day
            precip *= scale

            # UTC hour from integer timestamp (last 2 digits)
            hours_utc = yyyymmddhh % 100

            # Composite by time-of-day
            dc_mon = np.full((n_tod, precip.shape[1], precip.shape[2]),
                              np.nan, dtype=np.float32)
            for itod in range(n_tod):
                target_hr = itod * dt_hours
                mask = (hours_utc == target_hr)
                if mask.sum() > 0:
                    dc_mon[itod] = np.nanmean(precip[mask], axis=0)

            # Day weight (leap-year February = 29 days)
            wgt = mwt_base[im]
            if month == 2 and year in _LEAP_YEARS:
                wgt = 29
            print(f'wgt={wgt} days')

            if dc_sum is None:
                dc_sum  = dc_mon * wgt
                wgt_sum = np.full_like(dc_mon, float(wgt))
            else:
                dc_sum  += dc_mon * wgt
                wgt_sum += wgt

    if dc_sum is None:
        raise RuntimeError('No TRMM files found for the requested period.')

    with np.errstate(invalid='ignore', divide='ignore'):
        dc = np.where(wgt_sum > 0, dc_sum / wgt_sum, np.nan)

    tod = np.arange(n_tod, dtype=float) * dt_hours  # UTC hours: 0,3,6,...,21
    return xr.DataArray(dc, dims=['time_of_day', 'lat', 'lon'],
                        coords={'time_of_day': tod, 'lat': lat, 'lon': lon},
                        attrs={'units': 'mm/day',
                               'long_name': f'Composite diurnal cycle ({season})',
                               'season': season})


# ---------------------------------------------------------------------------
# IMERG data loading
# ---------------------------------------------------------------------------

def load_imerg_composite(data_dir, year_start, year_end, season='DJF',
                         dt_hours=3, scale_mmday=True):
    """
    Load GPM IMERG V07 3-hourly monthly files and build a weighted composite
    diurnal cycle for the requested season and year range.

    Files expected: <data_dir>/IMERG_3hr.YYYYMM.0p25deg.nc  (0.25° directory)
                 or <data_dir>/IMERG_3hr.YYYYMM.1deg.nc     (1° directory)
    Variable 'precip' in mm/hr; yyyymmddhh gives integer timestamps.
    Point data_dir at the appropriate resolution subdirectory:
        .../IMERG/3hrly/0.25deg/  or  .../IMERG/3hrly/1deg/

    Parameters
    ----------
    data_dir : str
    year_start, year_end : int  inclusive
    season : str  e.g. 'DJF', 'JJA'
    dt_hours : float  data time step (3 for IMERG 3-hourly product)
    scale_mmday : bool  if True, convert mm/hr → mm/day (* 24)

    Returns
    -------
    dc : xr.DataArray  (time_of_day, lat, lon)  mm/day composite
    """
    # Detect filename suffix from the files present in data_dir
    _samples = (glob.glob(os.path.join(data_dir, 'IMERG_3hr.*.0p25deg.nc')) +
                glob.glob(os.path.join(data_dir, 'IMERG_3hr.*.1deg.nc')))
    if not _samples:
        raise FileNotFoundError(f'No IMERG_3hr.*.nc files found in {data_dir}')
    _suffix = '0p25deg' if '0p25deg' in os.path.basename(_samples[0]) else '1deg'

    months, mwt_base = get_season_months(season)
    n_tod = int(24 // dt_hours)
    scale = 24.0 if scale_mmday else 1.0

    dc_sum  = None
    wgt_sum = None
    lat = lon = None

    for year in range(year_start, year_end + 1):
        for im, month in enumerate(months):
            fpath = os.path.join(data_dir,
                                 f'IMERG_3hr.{year:04d}{month:02d}.{_suffix}.nc')
            if not os.path.isfile(fpath):
                print(f'  [skip] {os.path.basename(fpath)} not found')
                continue

            print(f'  {os.path.basename(fpath)} ...', end=' ')

            ds = xr.open_dataset(fpath, decode_times=False)
            precip     = ds['precip'].values.astype(np.float32)  # (ntime, nlat, nlon)
            yyyymmddhh = ds['yyyymmddhh'].values

            if lat is None:
                lat = ds['lat'].values.copy()
                lon = ds['lon'].values.copy()
            ds.close()

            # Fill value → NaN
            precip = np.where(precip < -9000., np.nan, precip)
            precip *= scale

            # UTC hour from integer timestamp (last 2 digits)
            hours_utc = yyyymmddhh % 100

            # Composite by time-of-day
            dc_mon = np.full((n_tod, precip.shape[1], precip.shape[2]),
                              np.nan, dtype=np.float32)
            for itod in range(n_tod):
                target_hr = itod * dt_hours
                mask = (hours_utc == target_hr)
                if mask.sum() > 0:
                    dc_mon[itod] = np.nanmean(precip[mask], axis=0)

            # Day weight (leap-year February = 29 days)
            wgt = mwt_base[im]
            if month == 2 and year in _LEAP_YEARS:
                wgt = 29
            print(f'wgt={wgt} days')

            if dc_sum is None:
                dc_sum  = dc_mon * wgt
                wgt_sum = np.full_like(dc_mon, float(wgt))
            else:
                dc_sum  += dc_mon * wgt
                wgt_sum += wgt

    if dc_sum is None:
        raise RuntimeError('No IMERG files found for the requested period.')

    with np.errstate(invalid='ignore', divide='ignore'):
        dc = np.where(wgt_sum > 0, dc_sum / wgt_sum, np.nan)

    tod = np.arange(n_tod, dtype=float) * dt_hours
    return xr.DataArray(dc, dims=['time_of_day', 'lat', 'lon'],
                        coords={'time_of_day': tod, 'lat': lat, 'lon': lon},
                        attrs={'units': 'mm/day',
                               'long_name': f'IMERG composite diurnal cycle ({season})',
                               'season': season,
                               'source': 'GPM IMERG V07B'})


# ---------------------------------------------------------------------------
# CAM / CESM diurnal cycle loader
# ---------------------------------------------------------------------------

def load_cam_composite(data_dir, case_name, hist_type='h3a',
                       year_start=None, year_end=None,
                       season='DJF', dt_hours=3,
                       precip_vars=('PRECC', 'PRECL')):
    """
    Load CAM/CESM sub-daily history files and compute a seasonal diurnal composite.

    Expects files named:
        {case_name}.cam.{hist_type}.{YYYY}-{MM}-{DD}-{SSSSS}.nc

    Parameters
    ----------
    data_dir : str   directory containing the CAM history files
    case_name : str  case name (e.g. 'b.e30_alpha08b.B1850C_LTso.ne30_t232_wgx3.316')
    hist_type : str  history tape ID — e.g. 'h3a', 'h1', 'h1a', 'h2', 'h2a', ...
    year_start, year_end : int  inclusive year range; None = all years
    season : str  season string (e.g. 'DJF', 'JJA', 'ANN')
    dt_hours : float  time step in hours (3 for h3a; set to match hist_type)
    precip_vars : tuple of str  variable names to sum for total precipitation [m/s]
        Default ('PRECC','PRECL') sums convective + large-scale precip.
        Use ('PRECT',) if a combined variable is present.

    Returns
    -------
    dc : xr.DataArray (time_of_day, lat, lon)  mm/day composite
    """
    n_tod = int(24 // dt_hours)
    tod   = np.arange(n_tod, dtype=float) * dt_hours

    # Collect files
    pattern   = os.path.join(data_dir, f'{case_name}.cam.{hist_type}.*.nc')
    all_files = sorted(glob.glob(pattern))
    if not all_files:
        raise FileNotFoundError(f'No files matching:\n  {pattern}')
    print(f'  CAM {hist_type}: {len(all_files)} files found')

    # Filter by year using the date field in the filename
    # Format: ...{hist_type}.YYYY-MM-DD-SSSSS.nc
    def _file_year(f):
        stem = os.path.basename(f).removesuffix('.nc')
        date_str = stem.split(f'.{hist_type}.')[-1]   # 'YYYY-MM-DD-SSSSS'
        return int(date_str.split('-')[0])

    if year_start is not None or year_end is not None:
        all_files = [f for f in all_files
                     if (year_start is None or _file_year(f) >= year_start)
                     and (year_end   is None or _file_year(f) <= year_end)]
        if not all_files:
            raise ValueError(f'No {hist_type} files in year range {year_start}–{year_end}')
        print(f'  Year range {year_start}–{year_end}: {len(all_files)} files')

    # Open lazily — use_cftime handles pre-1678 model years (e.g. year 0001)
    ds = xr.open_mfdataset(
        all_files,
        combine   = 'nested',
        concat_dim= 'time',
        chunks    = {'time': 40},    # one CAM h3a file = 40 timesteps
        decode_times = True,
        use_cftime   = True,
        parallel  = True,
    )

    lat_vals = ds['lat'].values
    lon_vals = ds['lon'].values

    # Total precipitation: sum requested variables and convert m/s → mm/day
    scale = 86400.0 * 1000.0
    prec  = sum(ds[v] for v in precip_vars).astype(np.float32) * scale

    # Seasonal mask
    months, _ = get_season_months(season)
    seas_mask  = ds['time.month'].isin(months)
    n_seas     = int(seas_mask.sum().values)
    if n_seas == 0:
        print(f'  WARNING: no {season} months found; using all steps.')
        seas_mask = xr.ones_like(ds['time.month'], dtype=bool)
    print(f'  Season {season}: {n_seas} steps selected')

    prec_seas = prec.isel(time=seas_mask)

    # Diurnal composite
    print('  Computing diurnal composite via dask groupby...')
    dc_grouped = prec_seas.groupby('time.hour').mean('time')
    dc_vals    = dc_grouped.compute().values
    ds.close()

    # Ensure output is sorted 0, dt, 2*dt, ...
    hours_out = dc_grouped['hour'].values
    sort_idx  = np.argsort(hours_out)
    dc_vals   = dc_vals[sort_idx]

    return xr.DataArray(dc_vals, dims=['time_of_day', 'lat', 'lon'],
                        coords={'time_of_day': tod,
                                'lat': lat_vals, 'lon': lon_vals},
                        attrs={'units': 'mm/day', 'season': season,
                               'case': case_name, 'hist_type': hist_type})


# ---------------------------------------------------------------------------
# ERA5 diurnal cycle loader
# ---------------------------------------------------------------------------

def load_era5_composite(fpath, season='DJF', scale_mmday=True, dt_hours=3,
                        year_start=None, year_end=None):
    """
    Load ERA5 3-hourly precipitation file(s) and compute a seasonal diurnal composite.

    fpath can be:
      - A directory of precip_YYYY_era5_3hr_clean.nc files (dask-optimised path:
        data stays chunked and lazy; groupby.mean triggers distributed compute)
      - A single file path (numpy path; used for legacy _3hrave.nc files with
        corrupted timestamps)

    year_start / year_end filter the directory file list (default = all).

    Returns xr.DataArray (time_of_day, lat, lon) in mm/day.
    """
    import glob, os

    n_tod = int(24 // dt_hours)   # 8 for 3-hourly
    tod   = np.arange(n_tod, dtype=float) * dt_hours

    # =================================================================
    # PATH A: directory of per-year clean files  →  dask-native
    # =================================================================
    if os.path.isdir(fpath):
        pattern   = os.path.join(fpath, 'precip_*_era5_3hr_clean.nc')
        all_files = sorted(glob.glob(pattern))
        if not all_files:
            raise FileNotFoundError(f'No precip_*_era5_3hr_clean.nc in {fpath}')

        def _year(f): return int(os.path.basename(f).split('_')[1])
        files = [f for f in all_files
                 if (year_start is None or _year(f) >= year_start)
                 and (year_end   is None or _year(f) <= year_end)]
        if not files:
            raise ValueError(f'No files in year range {year_start}–{year_end}')
        print(f'  ERA5 directory: {len(files)} files  '
              f'({_year(files[0])}–{_year(files[-1])})')

        # Detect units from history of first file (small metadata read)
        with xr.open_dataset(files[0], decode_times=False) as ds0:
            history = ds0.attrs.get('history', '')
        if scale_mmday:
            if 'divc,3600' in history and 'mulc,1000' in history:
                scale = 86400.0
                print(f'  CDO history: mm/s units → scale=×{scale:.0f}')
            elif 'divc,3600' in history:
                scale = 86400.0 * 1000.0
            else:
                scale = (1000.0 / dt_hours) * 24.0
        else:
            scale = 1.0

        # Open all files lazily with dask chunks (~1 month of 3-hourly per chunk)
        ds = xr.open_mfdataset(
            files,
            combine      = 'nested',
            concat_dim   = 'valid_time',
            chunks       = {'valid_time': 240},   # ~30 days; ~62 MB/chunk
            decode_times = True,
            parallel     = True,
        )

        # Rename spatial dims if needed
        rename = {k: v for k, v in [('latitude','lat'),('longitude','lon')]
                  if k in ds.dims}
        if rename:
            ds = ds.rename(rename)

        lat_vals = ds['lat'].values
        lon_vals = ds['lon'].values

        # Scale lazily — no data loaded yet
        tp = (ds['tp'] * scale).astype(np.float32)

        # Seasonal filter: boolean index along time (still lazy)
        months, _ = get_season_months(season)
        seas_mask  = ds['valid_time.month'].isin(months)
        n_seas     = int(seas_mask.sum().values)
        if n_seas == 0:
            print(f'  WARNING: no {season} months found; using all times.')
            seas_mask = xr.ones_like(ds['valid_time.month'], dtype=bool)
        print(f'  Season {season}: {n_seas} steps selected')

        tp_seas = tp.isel(valid_time=seas_mask)

        # Group by UTC hour → mean over all matching days  (distributed compute)
        print(f'  Computing diurnal composite via dask groupby...')
        dc_grouped = tp_seas.groupby('valid_time.hour').mean('valid_time')
        dc_vals    = dc_grouped.compute().values   # (n_tod, nlat, nlon)
        ds.close()

        # groupby reorders hours; ensure output is sorted 0,3,6,...,21
        hours_out = dc_grouped['hour'].values
        sort_idx  = np.argsort(hours_out)
        dc_vals   = dc_vals[sort_idx]

        return xr.DataArray(dc_vals, dims=['time_of_day', 'lat', 'lon'],
                            coords={'time_of_day': tod,
                                    'lat': lat_vals, 'lon': lon_vals},
                            attrs={'units': 'mm/day', 'season': season})

    # =================================================================
    # PATH B: single legacy file  →  numpy (file is small, ~750 MB)
    # =================================================================
    import pandas as pd

    ds = xr.open_dataset(fpath, decode_times=False)

    rename = {k: v for k, v in [('latitude','lat'),('longitude','lon')]
              if k in ds.dims}
    if rename:
        ds = ds.rename(rename)

    tname      = 'valid_time' if 'valid_time' in ds.coords else 'time'
    time_raw   = ds[tname].values.astype(float)
    time_units = ds[tname].attrs.get('units', 'seconds since 1970-01-01')
    origin     = time_units.split('since')[-1].strip()
    if 'seconds' in time_units:
        times_pd = pd.to_datetime(time_raw, unit='s', origin=origin)
    elif 'hours' in time_units:
        times_pd = pd.to_datetime(time_raw * 3600, unit='s', origin=origin)
    else:
        times_pd = pd.to_datetime(time_raw, unit='s', origin='unix')

    n_steps = len(times_pd)

    # Detect corrupted time axis
    unique_hours    = np.unique(times_pd.hour.values)
    hours_corrupted = len(unique_hours) <= n_tod // 4
    if hours_corrupted:
        print(f'  NOTE: valid_time hours corrupted '
              f'({unique_hours}). Using positional index.')

    # Unit scale from CDO history
    history = ds.attrs.get('history', '')
    if scale_mmday:
        if 'divc,3600' in history and 'mulc,1000' in history:
            scale = 86400.0
            print(f'  CDO history: mm/s units → scale=×{scale:.0f}')
        elif 'divc,3600' in history:
            scale = 86400.0 * 1000.0
        else:
            scale = (1000.0 / dt_hours) * 24.0
    else:
        scale = 1.0

    tp_vals  = ds['tp'].values.astype(np.float32) * scale
    lat_vals = ds['lat'].values
    lon_vals = ds['lon'].values
    ds.close()

    months, _  = get_season_months(season)
    month_vals = times_pd.month.values
    seas_mask  = np.isin(month_vals, months)
    if seas_mask.sum() == 0:
        print(f'  WARNING: no {season} months found; using all steps.')
        seas_mask = np.ones(n_steps, dtype=bool)
    seas_idx = np.where(seas_mask)[0]
    print(f'  Season {season}: {len(seas_idx)} steps selected out of {n_steps}')

    dc = np.full((n_tod, tp_vals.shape[1], tp_vals.shape[2]), np.nan,
                 dtype=np.float32)
    if hours_corrupted:
        for itod in range(n_tod):
            bin_idx = seas_idx[seas_idx % n_tod == itod]
            if len(bin_idx):
                dc[itod] = np.nanmean(tp_vals[bin_idx], axis=0)
    else:
        hours_utc = times_pd.hour.values
        for itod in range(n_tod):
            target_h = (itod * dt_hours) % 24
            idx = np.where(seas_mask & (hours_utc == target_h))[0]
            if len(idx):
                dc[itod] = np.nanmean(tp_vals[idx], axis=0)

    return xr.DataArray(dc, dims=['time_of_day', 'lat', 'lon'],
                        coords={'time_of_day': tod,
                                'lat': lat_vals, 'lon': lon_vals},
                        attrs={'units': 'mm/day', 'season': season})


# ---------------------------------------------------------------------------
# Harmonic analysis
# ---------------------------------------------------------------------------

def compute_harmonics(dc, n_harm=2, dt_hours=3):
    """
    FFT harmonic analysis of the composite diurnal cycle.

    Parameters
    ----------
    dc : xr.DataArray or ndarray  (n_tod, nlat, nlon)  in mm/day
    n_harm : int  Number of harmonics (1=diurnal, 2=+semi-diurnal)
    dt_hours : float  Time step in hours

    Returns
    -------
    amplitude    : ndarray (n_harm, nlat, nlon)  half peak-to-trough [mm/day]
    phase_utc    : ndarray (n_harm, nlat, nlon)  UTC time of max [0, T/k) [hrs]
    var_explained: ndarray (n_harm, nlat, nlon)  fraction of variance [0-1]
    mean_prcp    : ndarray (nlat, nlon)           time mean [mm/day]
    """
    data = dc.values if isinstance(dc, xr.DataArray) else np.asarray(dc)
    n_tod = data.shape[0]
    T = n_tod * dt_hours   # = 24 hours

    # FFT along time-of-day axis (axis 0)
    Xk = np.fft.rfft(data, axis=0)   # shape (n_tod//2+1, nlat, nlon)

    # Total variance from all non-zero harmonics
    power_all = (2.0 / n_tod**2) * np.sum(np.abs(Xk[1:])**2, axis=0)

    amplitude    = np.zeros((n_harm,) + data.shape[1:], dtype=np.float32)
    phase_utc    = np.zeros((n_harm,) + data.shape[1:], dtype=np.float32)
    var_explained = np.zeros((n_harm,) + data.shape[1:], dtype=np.float32)

    for k in range(1, n_harm + 1):
        ih = k - 1
        Xh = Xk[k]   # complex (nlat, nlon)

        # Amplitude: half peak-to-peak
        amplitude[ih] = (2.0 / n_tod) * np.abs(Xh)

        # Phase: UTC time of maximum
        # x(t) ~ (2/N)|X[k]| cos(2π k t/T + angle(X[k]))
        # max at t = -angle(X[k]) * T / (2π k)
        period_k = T / k   # hours (24 for k=1, 12 for k=2)
        phase_utc[ih] = (-np.angle(Xh) * T / (2 * np.pi * k)) % period_k

        # Variance explained relative to total diurnal variance
        harm_power = (2.0 / n_tod**2) * np.abs(Xh)**2
        with np.errstate(invalid='ignore', divide='ignore'):
            var_explained[ih] = np.where(power_all > 0,
                                          harm_power / power_all, 0.)

    mean_prcp = np.nanmean(data, axis=0)
    return amplitude, phase_utc, var_explained, mean_prcp


def compute_raw_diurnal(dc, dt_hours=3):
    """
    Amplitude and phase of the raw composite diurnal cycle (no harmonic fitting).

    Amplitude = (max − min) / 2  over the time-of-day axis — consistent with
                the half-peak-to-trough convention used for FFT harmonics.
    Phase_utc = UTC hour of the diurnal maximum.

    Parameters
    ----------
    dc : xr.DataArray or ndarray  (n_tod, lat, lon)  in mm/day
    dt_hours : float  time step (hours)

    Returns
    -------
    amplitude : ndarray (nlat, nlon)  [mm/day]
    phase_utc : ndarray (nlat, nlon)  UTC hour of max [0, 24)
    """
    data = dc.values if isinstance(dc, xr.DataArray) else np.asarray(dc)
    if isinstance(dc, xr.DataArray) and 'time_of_day' in dc.coords:
        tod = dc['time_of_day'].values.astype(float)
    else:
        tod = np.arange(data.shape[0], dtype=float) * dt_hours

    imax      = np.argmax(data, axis=0)          # (nlat, nlon)  index of max
    amplitude = (np.nanmax(data, axis=0) - np.nanmin(data, axis=0)) / 2.0
    phase_utc = tod[imax]                        # UTC hour of maximum

    return amplitude.astype(np.float32), phase_utc.astype(np.float32)


def phase_utc_to_lst(phase_utc, lon, period_hours):
    """
    Convert UTC phase to local solar time (LST).

    Parameters
    ----------
    phase_utc : ndarray (nlat, nlon)
    lon : array (nlon,)  degrees (-180 to 180 or 0 to 360)
    period_hours : float  24 or 12

    Returns
    -------
    phase_lst : ndarray (nlat, nlon)  in [0, period_hours)
    """
    # Hours ahead of UTC at each longitude
    lon_offset = lon * (24.0 / 360.0)          # shape (nlon,)
    phase_lst = (phase_utc + lon_offset[np.newaxis, :]) % period_hours
    return phase_lst


# ---------------------------------------------------------------------------
# Evans color helpers
# ---------------------------------------------------------------------------

def _phase_to_rgba(phase_lst, amplitude, min_amp, max_amp, period_hours,
                   hue_offset=0.5):
    """
    Map (phase_lst, amplitude) → RGBA using HSV color space.

    Convention (hue_offset=0.5, i.e. NCL's epHueOffset=180°):
      0h LST → cyan,   6h → blue/violet,  12h → red,  18h → yellow-green
    Amplitude below min_amp → white (sat=0).
    NaN → transparent.
    """
    hue = (phase_lst / period_hours + hue_offset) % 1.0
    sat = np.clip((amplitude - min_amp) / (max_amp - min_amp), 0.0, 1.0)
    val = np.ones_like(hue)

    hsv  = np.stack([hue, sat, val], axis=-1)
    rgb  = mcolors.hsv_to_rgb(hsv)
    alpha = np.where(np.isnan(amplitude) | np.isnan(phase_lst), 0.0, 1.0)
    return np.concatenate([rgb, alpha[..., np.newaxis]], axis=-1)  # (..., 4)


# ---------------------------------------------------------------------------
# Color wheel
# ---------------------------------------------------------------------------

def draw_color_wheel(ax_polar, period_hours, min_amp, max_amp,
                     hue_offset=0.5, n_seg=360, n_rad=60,
                     label_hours=None, tick_hours=None, fontsize=10,
                     discrete=False, dt_hours=3, n_amp_disc=10, deepen_val=0.82):
    """
    Fill a pre-created polar axes with the Evans-style color wheel.

    Angular direction = phase (hours LST, clockwise from top = 0h).
    Radial direction  = amplitude (inner = min_amp, outer = max_amp).

    Parameters
    ----------
    ax_polar : matplotlib polar axes  (must be created with projection='polar')
    period_hours : float  24 (diurnal) or 12 (semi-diurnal)
    min_amp, max_amp : float  amplitude thresholds [mm/day]
    hue_offset : float  HSV hue offset; 0.5 → 0h=cyan (NCL epHueOffset=180)
    n_seg, n_rad : int  angular and radial resolution (ignored when discrete=True)
    label_hours : list  hours to label (text); default every 2h (H1) or 1h (H2)
    tick_hours : list  hours to draw a tick stub; default every integer hour
    fontsize : int
    discrete : bool  if True use stepped appearance — 0.5h phase bins and
               n_amp_disc radial bands with deepening value at large amplitudes
    dt_hours : float  (unused; kept for API compatibility)
    n_amp_disc : int  number of discrete amplitude bands when discrete=True
    deepen_val : float  HSV value at the outermost amplitude band (discrete mode);
               1.0 = full brightness (no deepening), lower = darker/deeper hue
    """
    if discrete:
        n_seg = int(period_hours)   # one bin per hour: 24 for H1, 12 for H2
        n_rad = n_amp_disc          # default 10 radial bands
    ax_polar.set_theta_direction(-1)        # clockwise
    ax_polar.set_theta_zero_location('N')   # 0h at top

    r_outer = 0.96  # colored disk fills 96 % of the axes radius

    # Edge arrays.
    # In discrete mode shift by half a bin so each colour segment is centred on
    # its integer hour (edges at −0.5 h, +0.5 h, +1.5 h, …) rather than
    # starting at the hour (0 h, 1 h, 2 h, …).
    theta_e = np.linspace(0, 2 * np.pi, n_seg + 1)
    if discrete:
        theta_e = theta_e - np.pi / n_seg   # half-bin shift
    r_e     = np.linspace(0, r_outer, n_rad + 1)

    # Cell centers
    theta_c = 0.5 * (theta_e[:-1] + theta_e[1:])
    r_c     = 0.5 * (r_e[:-1]     + r_e[1:])

    # Phase and amplitude from coordinates
    phase_c = theta_c / (2 * np.pi) * period_hours   # (n_seg,) hours
    amp_c   = (r_c / r_outer) * (max_amp - min_amp) + min_amp  # (n_rad,) mm/day

    # HSV arrays
    hue = (phase_c / period_hours + hue_offset) % 1.0
    sat = np.clip((amp_c - min_amp) / (max_amp - min_amp), 0, 1)

    H = np.tile(hue[np.newaxis, :], (n_rad, 1))
    S = np.tile(sat[:, np.newaxis], (1, n_seg))

    # In discrete mode darken outer (large-amplitude) bands so hue looks deeper.
    # V ramps linearly from 1.0 at the innermost band to deepen_val at the outermost.
    if discrete and deepen_val < 1.0:
        val_profile = np.linspace(1.0, deepen_val, n_rad)   # (n_rad,) inner→outer
        V = np.tile(val_profile[:, np.newaxis], (1, n_seg))
    else:
        V = np.ones((n_rad, n_seg))

    RGB  = mcolors.hsv_to_rgb(np.stack([H, S, V], axis=-1))
    RGBA = np.concatenate([RGB, np.ones((n_rad, n_seg, 1))], axis=-1)

    # Draw via pcolormesh + set_facecolor
    pcm = ax_polar.pcolormesh(theta_e, r_e,
                               np.zeros((n_rad, n_seg)), shading='flat')
    pcm.set_array(None)
    pcm.set_facecolor(RGBA.reshape(-1, 4))

    # Tick stubs at every integer hour (short radial lines just outside wheel)
    if tick_hours is None:
        tick_hours = list(range(int(period_hours)))
    tick_len = 0.07
    for h in tick_hours:
        angle = (h / period_hours) * 2 * np.pi
        ax_polar.plot([angle, angle], [r_outer, r_outer + tick_len],
                      color='black', linewidth=0.8, zorder=3)

    # Hour labels around rim
    if label_hours is None:
        step = 2 if period_hours >= 20 else 1
        label_hours = list(range(0, int(period_hours), step))
    for h in label_hours:
        angle = (h / period_hours) * 2 * np.pi
        lbl = f'{int(h)}h' if h == int(h) else f'{h:.1f}h'
        ax_polar.text(angle, r_outer + 0.22, lbl,
                      ha='center', va='center',
                      fontsize=fontsize * 1.2, fontweight='bold')

    # Radial arrow along the 0h spoke (straight up), like NCL colour wheel
    ax_polar.annotate('',
        xy        = (0., r_outer),   # arrowhead at outer rim
        xytext    = (0., 0.),        # tail at centre
        arrowprops= dict(arrowstyle='->', color='black', lw=2.0, mutation_scale=12),
        zorder    = 4,
    )

    # Amplitude labels: min near origin, max slightly inside rim to clear 0h label.
    # Negative theta = counter-clockwise = left of the arrow; ha='right' aligns
    # both labels flush to the same horizontal position.
    lbl_theta = -0.15   # ~9° counter-clockwise (left of the 0h spoke)
    ax_polar.text(lbl_theta, 0.06, f'{min_amp:.1f}',
                  ha='right', va='center',
                  fontsize=(fontsize - 1) * 1.3, color='black', fontweight='bold')
    ax_polar.text(lbl_theta, r_outer * 0.83, f'{max_amp:.1f}',
                  ha='right', va='center',
                  fontsize=(fontsize - 1) * 1.3, color='black', fontweight='bold')

    # Unit label above the wheel — bold
    ax_polar.text(0., r_outer + 0.52, 'mm/d', ha='center', va='center',
                  fontsize=(fontsize - 1) * 1.3, color='dimgrey', fontweight='bold',
                  transform=ax_polar.transData)

    # Clean up default decorations
    ax_polar.set_xticks([])
    ax_polar.set_yticks([])
    ax_polar.spines['polar'].set_visible(False)
    ax_polar.set_ylim(0, 1.5)   # room for labels and ticks outside r_outer=0.8


# ---------------------------------------------------------------------------
# Map plots
# ---------------------------------------------------------------------------

def plot_mean_precip(ax, mean_prcp, lat, lon,
                     title='', levels=None, cmap='terrain_r',
                     lat_range=(-40., 40.), lon_range=(0., 360.),
                     show_states=False):
    """
    Filled-contour map of mean precipitation.

    Parameters
    ----------
    ax : cartopy GeoAxes
    mean_prcp : ndarray (nlat, nlon)  [mm/day]
    lat, lon : 1-D arrays
    title : str
    levels : list  contour levels in mm/day
    lat_range, lon_range : (min, max) tuples; None = use data extent

    Returns
    -------
    cf : QuadContourSet  (for colorbar)
    """
    if levels is None:
        levels = PRCP_LEVELS

    # Map extent
    lat0, lat1 = (lat.min(), lat.max()) if lat_range is None else lat_range
    lon0, lon1 = (lon.min(), lon.max()) if lon_range is None else lon_range
    ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())

    ax.coastlines(linewidth=0.7, color='black', zorder=3)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='grey', zorder=3)
    if show_states:
        ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='black', zorder=3)
    gl = ax.gridlines(linewidth=0.3, color='grey', linestyle='--',
                      draw_labels=True, zorder=2)
    gl.top_labels = gl.right_labels = False

    print(cmap)
    # Make contourf cyclic so the seam at lon=0/360° closes cleanly
    from cartopy.util import add_cyclic_point
    mean_cyc, lon_cyc = add_cyclic_point(mean_prcp, coord=lon)
    LON, LAT = np.meshgrid(lon_cyc, lat)
    cf = ax.contourf(LON, LAT, mean_cyc, levels=levels, cmap=cmap,
                     extend='max', transform=ccrs.PlateCarree(), zorder=1)
    ax.set_title(title, fontsize=10, pad=4)
    return cf


def plot_evans_map(ax, fig, phase_lst, amplitude, lat, lon,
                   min_amp, max_amp, period_hours, title='',
                   hue_offset=0.5, lat_range=(-40., 40.), lon_range=(0., 360.),
                   ax_wheel=None, wheel_fontsize=8,
                   discrete_wheel=False, dt_hours=3, n_amp_disc=10, deepen_val=0.82,
                   show_states=False):
    """
    Evans-style map: phase → hue, amplitude → color saturation.

    The color wheel is drawn into ax_wheel (a pre-created polar Axes that
    lives outside the map — see the notebook cell which uses GridSpec to
    reserve that space).  If ax_wheel is None a fallback position is used.

    Parameters
    ----------
    ax : cartopy GeoAxes  (the map)
    fig : matplotlib Figure
    phase_lst : ndarray (nlat, nlon)  local solar time of max [hours]
    amplitude : ndarray (nlat, nlon)  [mm/day]
    lat, lon : 1-D arrays
    min_amp, max_amp : float  amplitude thresholds
    period_hours : float  24 (diurnal) or 12 (semi-diurnal)
    title : str
    hue_offset : float  0.5 → 0h=cyan (NCL default)
    lat_range, lon_range : (min, max) tuples
    ax_wheel : polar Axes  pre-created wheel axes (from GridSpec); if None
               a small axes is added inside the figure right margin
    wheel_fontsize : int  font size for hour labels on wheel

    Returns
    -------
    ax_wheel : polar axes of the color wheel
    """
    # Compute RGBA for each grid cell
    rgba = _phase_to_rgba(phase_lst, amplitude, min_amp, max_amp,
                          period_hours, hue_offset)   # (nlat, nlon, 4)

    # Map extent
    lat0, lat1 = (lat.min(), lat.max()) if lat_range is None else lat_range
    lon0, lon1 = (lon.min(), lon.max()) if lon_range is None else lon_range

    dlat = abs(lat[1] - lat[0])
    dlon = abs(lon[1] - lon[0])

    # Subset lat/rgba to the display lat range BEFORE building img_ext.
    # imshow maps array rows linearly to the extent, so passing the full
    # global array with a smaller extent squashes all rows into that range,
    # displacing features toward the poles. Subsetting first gives a 1-to-1
    # row ↔ latitude correspondence.
    lat_mask = (lat >= lat0 - dlat/2) & (lat <= lat1 + dlat/2)
    lat_sub  = lat[lat_mask]
    rgba_sub = rgba[lat_mask, :, :]

    # Make imshow cyclic: append the lon=0 column at lon=360 so the image
    # fills the full 0–360° extent without a gap at the dateline.
    rgba_cyc = np.concatenate([rgba_sub, rgba_sub[:, :1, :]], axis=1)
    lon_cyc  = np.append(lon, lon[-1] + dlon)   # e.g. [..., 359, 360]

    ax.set_extent([lon0, lon1, lat0, lat1], crs=ccrs.PlateCarree())

    # Pixel edge extents (half a cell beyond the outermost centres)
    img_ext = [lon_cyc[0]  - dlon/2, lon_cyc[-1] + dlon/2,
               lat_sub.min() - dlat/2, lat_sub.max() + dlat/2]

    # origin: 'lower' if lat increases (S→N); 'upper' if lat decreases (N→S, ERA5)
    origin = 'lower' if lat_sub[-1] > lat_sub[0] else 'upper'

    # Do NOT pass aspect='auto' — that overrides cartopy's equal-aspect
    # enforcement and stretches the map in the lon direction.
    ax.imshow(rgba_cyc, origin=origin, interpolation='nearest',
              extent=img_ext, transform=ccrs.PlateCarree(), zorder=1)

    ax.coastlines(linewidth=0.7, color='black', zorder=2)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor='grey', zorder=2)
    if show_states:
        ax.add_feature(cfeature.STATES, linewidth=0.4, edgecolor='black', zorder=2)
    gl = ax.gridlines(linewidth=0.3, color='grey', linestyle='--',
                      draw_labels=True, zorder=2)
    gl.top_labels = gl.right_labels = False
    ax.set_title(title, fontsize=10, pad=4)

    # --- Color wheel ---
    if ax_wheel is None:
        # Fallback: add a small polar axes in the figure's right margin.
        # This is less reliable than the GridSpec approach in the notebook.
        fig_w, fig_h = fig.get_size_inches()
        wheel_in = min(fig_h * 0.45, 1.8)
        ww = wheel_in / fig_w
        wh = wheel_in / fig_h
        ax_wheel = fig.add_axes([0.87, 0.50 - wh/2, ww, wh],
                                 projection='polar', facecolor='white')

    ax_wheel.patch.set_facecolor('white')
    ax_wheel.patch.set_alpha(0.90)

    # Tick every integer hour; label every 2h (H1/24h) or every 1h (H2/12h)
    tick_hours  = list(range(int(period_hours)))
    label_step  = 2 if period_hours >= 20 else 1
    label_hours = list(range(0, int(period_hours), label_step))
    draw_color_wheel(ax_wheel, period_hours, min_amp, max_amp,
                     hue_offset=hue_offset,
                     label_hours=label_hours, tick_hours=tick_hours,
                     fontsize=wheel_fontsize,
                     discrete=discrete_wheel, dt_hours=dt_hours,
                     n_amp_disc=n_amp_disc, deepen_val=deepen_val)

    return ax_wheel


# ─────────────────────────────────────────────────────────────────────────────
# Regional diurnal-cycle analysis and line plots
# ─────────────────────────────────────────────────────────────────────────────

def get_land_mask(lat, lon, scale='110m'):
    """
    Build a boolean land mask on the (lat, lon) grid using cartopy Natural Earth.

    Parameters
    ----------
    lat, lon : 1-D arrays  (any longitude convention)
    scale    : str  Natural Earth resolution — '110m' (fast), '50m', or '10m'

    Returns
    -------
    mask : ndarray (nlat, nlon)  True = land, False = ocean/lake
    """
    from shapely.ops import unary_union
    import shapely

    print(f'  Building land mask (Natural Earth {scale}) …', end=' ', flush=True)
    land_geoms = list(
        cfeature.NaturalEarthFeature('physical', 'land', scale).geometries())
    land_poly = unary_union(land_geoms)

    # Normalise lon to ±180 for shapely (which uses standard geographic coords)
    lon_std = np.where(lon > 180., lon - 360., lon)
    lons2d, lats2d = np.meshgrid(lon_std, lat)

    # Three fallback paths covering shapely ≥ 2.0, 1.x vectorized, and the loop
    try:                                        # shapely ≥ 2.0
        mask = shapely.contains_xy(
            land_poly, lons2d.ravel(), lats2d.ravel()
        ).reshape(lons2d.shape)
    except AttributeError:
        try:                                    # shapely 1.x
            from shapely.vectorized import contains as _sv_contains
            mask = _sv_contains(land_poly, lons2d, lats2d)
        except ImportError:                     # fallback loop
            from shapely.prepared import prep
            from shapely.geometry import Point
            prep_land = prep(land_poly)
            mask = np.array(
                [prep_land.contains(Point(lo, la))
                 for lo, la in zip(lons2d.ravel(), lats2d.ravel())],
                dtype=bool).reshape(lons2d.shape)

    n_land = int(mask.sum())
    print(f'{n_land} land / {mask.size} total ({n_land / mask.size:.1%})')
    return mask


def _lon_mask(lon, lon_b0, lon_b1):
    """
    Boolean mask for lon in [lon_b0, lon_b1], handling wrap-around and
    mixed 0-360 / ±180 conventions.

    Input lon_b0/lon_b1 may be in 0-360 convention regardless of what the
    data array uses; this function normalises them automatically.
    """
    if lon.max() > 180.5:          # data is 0–360
        lon_b0 = lon_b0 % 360.
        lon_b1 = lon_b1 % 360.
    else:                           # data is −180–180
        if lon_b0 > 180.: lon_b0 -= 360.
        if lon_b1 > 180.: lon_b1 -= 360.
        if lon_b0 < -180.: lon_b0 += 360.
        if lon_b1 < -180.: lon_b1 += 360.

    if lon_b0 <= lon_b1:
        return (lon >= lon_b0) & (lon <= lon_b1)
    else:                           # straddles 0°/180° meridian
        return (lon >= lon_b0) | (lon <= lon_b1)


def region_mean_dc(dc, lat, lon, lat_box, lon_box, point_mask=None):
    """
    Cosine-latitude-weighted spatial mean of the composite diurnal cycle
    over a lat/lon bounding box.

    Parameters
    ----------
    dc         : xr.DataArray or ndarray  (time_of_day, nlat, nlon)
    lat, lon   : 1-D arrays  (any longitude convention)
    lat_box    : (lat_min, lat_max)
    lon_box    : (lon_min, lon_max)  — may span the dateline; store in 0-360
    point_mask : ndarray (nlat, nlon) bool or None
        If provided, only grid points where point_mask is True are included.
        Pass land_mask for land-only, ~land_mask for ocean-only.

    Returns
    -------
    dc_mean : ndarray  (n_tod,)
    """
    data = dc.values if isinstance(dc, xr.DataArray) else np.asarray(dc, dtype=float)

    lat_mask = (lat >= lat_box[0]) & (lat <= lat_box[1])
    lmask    = _lon_mask(lon, lon_box[0], lon_box[1])

    if not lat_mask.any() or not lmask.any():
        raise ValueError(
            f'No grid points in box lat={lat_box}, lon={lon_box}. '
            f'Data lat=[{lat.min():.1f},{lat.max():.1f}], '
            f'lon=[{lon.min():.1f},{lon.max():.1f}]')

    subset = data[:, lat_mask, :][:, :, lmask]          # (n_tod, nlat_sub, nlon_sub)
    wgt    = np.cos(np.deg2rad(lat[lat_mask]))           # (nlat_sub,)
    wgt2d  = wgt[:, np.newaxis] * np.ones(lmask.sum())  # (nlat_sub, nlon_sub)

    if point_mask is not None:
        pmask_sub = point_mask[lat_mask, :][:, lmask]   # (nlat_sub, nlon_sub)
        wgt2d     = wgt2d * pmask_sub.astype(float)
        if wgt2d.sum() == 0:
            raise ValueError(
                f'No valid points in box lat={lat_box}, lon={lon_box} '
                f'after applying point mask.')

    dc_mean = (np.nansum(subset * wgt2d[np.newaxis], axis=(1, 2))
               / np.nansum(wgt2d))
    return dc_mean.astype(np.float32)


def compute_harmonics_1d(dc_1d, dt_hours=3, n_harm=2):
    """
    FFT harmonic analysis on a 1-D composite diurnal cycle.

    Returns
    -------
    harmonics : list of dicts  [{amp, phase_utc, period_h}, ...]  length n_harm
    mean_val  : float  time mean
    """
    data  = np.asarray(dc_1d, dtype=float)
    n_tod = len(data)
    T     = n_tod * dt_hours   # 24 h

    Xk  = np.fft.rfft(data)
    out = []
    for k in range(1, n_harm + 1):
        period_k  = T / k
        amp       = (2.0 / n_tod) * abs(Xk[k])
        phase_utc = (-np.angle(Xk[k]) * T / (2 * np.pi * k)) % period_k
        out.append({'amp': float(amp), 'phase_utc': float(phase_utc),
                    'period_h': float(period_k)})
    return out, float(data.mean())


def reconstruct_harmonic_fit(tod, harmonics, n_harm=None, mean=0.):
    """
    Reconstruct harmonic fit(s) from a list of {amp, phase_utc, period_h} dicts.

    Returns
    -------
    fit_total : ndarray (len(tod),)   mean + all harmonics summed
    fit_each  : list of ndarray       mean + each harmonic individually
    """
    tod = np.asarray(tod, dtype=float)
    if n_harm is None:
        n_harm = len(harmonics)
    fit_total = np.full(len(tod), mean)
    fit_each  = []
    for h in harmonics[:n_harm]:
        comp = h['amp'] * np.cos(2 * np.pi * (tod - h['phase_utc']) / h['period_h'])
        fit_total = fit_total + comp
        fit_each.append(mean + comp)
    return fit_total, fit_each


def _add_region_inset(ax, reg, map_extent=None, show_states=False):
    """
    Add a small cartopy map inset to *ax* (a regular matplotlib Axes) that
    shows where the region bounding box sits geographically.
    Regions that straddle the dateline use ccrs.PlateCarree(central_longitude=180)
    so that 180° is centred rather than at the edges.
    Falls back silently if the inset axes cannot be created.
    """
    try:
        from mpl_toolkits.axes_grid1.inset_locator import inset_axes
        import cartopy.mpl.geoaxes as cgax

        lat_b0, lat_b1 = reg['lat_box']
        lon_b0, lon_b1 = reg['lon_box']   # 0-360 convention

        # Convert to ±180 standard to check for dateline crossing
        b0 = lon_b0 if lon_b0 <= 180. else lon_b0 - 360.
        b1 = lon_b1 if lon_b1 <= 180. else lon_b1 - 360.
        straddles = b0 > b1

        if straddles:
            # Centre the inset on 180° so the box is seamless.
            # In PlateCarree(180) coords: x = (lon % 360) - 180
            clon  = 180.
            x0_p  = (b0 % 360.) - 180.   # e.g. 160°E → -20
            x1_p  = (b1 % 360.) - 180.   # e.g. -160°E(200°E) → +20
        else:
            clon  = 0.
            x0_p, x1_p = b0, b1

        proj_inset = ccrs.PlateCarree(central_longitude=clon)

        if map_extent is None:
            w   = max(x1_p - x0_p, 15.)
            h   = max(lat_b1 - lat_b0, 10.)
            ie  = [x0_p - w * 0.7, x1_p + w * 0.7,
                   lat_b0 - h * 0.6, lat_b1 + h * 0.6]
        else:
            ie = list(map_extent)

        ie[2] = max(ie[2], -90.)
        ie[3] = min(ie[3],  90.)

        # Create inset axes (try both cartopy kwarg spellings)
        try:
            axi = inset_axes(ax, width='30%', height='36%', loc='lower right',
                             axes_class=cgax.GeoAxes,
                             axes_kwargs={'map_projection': proj_inset})
        except TypeError:
            axi = inset_axes(ax, width='30%', height='36%', loc='lower right',
                             axes_class=cgax.GeoAxes,
                             axes_kwargs={'projection': proj_inset})

        axi.set_extent(ie, crs=proj_inset)
        axi.add_feature(cfeature.LAND,  facecolor='#ccc2a8', zorder=0)
        axi.add_feature(cfeature.OCEAN, facecolor='#c0dff0', zorder=0)
        axi.coastlines(linewidth=0.4, color='k', zorder=1)
        if show_states:
            axi.add_feature(cfeature.NaturalEarthFeature(
                'cultural', 'admin_1_states_provinces_lines', '10m'),
                edgecolor='#555555', facecolor='none', linewidth=0.4, zorder=2)
            axi.add_feature(cfeature.NaturalEarthFeature(
                'cultural', 'admin_0_countries', '10m'),
                edgecolor='#222222', facecolor='none', linewidth=0.65, zorder=3)

        # Single seamless rectangle in the inset projection's coordinate space
        axi.add_patch(mpatches.Rectangle(
            (x0_p, lat_b0), x1_p - x0_p, lat_b1 - lat_b0,
            linewidth=1.5, edgecolor='red', facecolor='red', alpha=0.25,
            transform=proj_inset, zorder=2))

        axi.set_xticks([])
        axi.set_yticks([])

    except Exception:
        pass   # text annotation already present; inset is best-effort


def annotate_regions_on_map(ax, regions,
                             edgecolor='black', linewidth=2.0, linestyle='-',
                             label=True, label_fontsize=7):
    """
    Overlay bounding-box rectangles on a cartopy GeoAxes for a list of regions.

    Regions that straddle the dateline (e.g. West Pacific 160°E–200°E) are drawn
    as a single seamless rectangle using ccrs.PlateCarree(central_longitude=180)
    as the patch transform, avoiding the visible split line at 180°.

    Parameters
    ----------
    ax        : cartopy GeoAxes
    regions   : list of dicts  [{name, lat_box, lon_box}, ...]
                lon_box may be in 0-360 convention.
    linestyle : str  matplotlib linestyle (e.g. '-', '--', ':', '-.')
    """
    for reg in regions:
        lat_b0, lat_b1 = reg['lat_box']
        lon_b0, lon_b1 = reg['lon_box']

        # Convert to ±180 standard
        b0 = lon_b0 if lon_b0 <= 180. else lon_b0 - 360.
        b1 = lon_b1 if lon_b1 <= 180. else lon_b1 - 360.

        if b0 <= b1:
            # Non-wrapping: standard PlateCarree transform
            ax.add_patch(mpatches.Rectangle(
                (b0, lat_b0), b1 - b0, lat_b1 - lat_b0,
                linewidth=linewidth, edgecolor=edgecolor, facecolor='none',
                linestyle=linestyle,
                transform=ccrs.PlateCarree(), zorder=5))
            mid_lon = (b0 + b1) / 2.
            mid_crs = ccrs.PlateCarree()
        else:
            # Straddles ±180°: use PlateCarree(180) so the box is one piece.
            # In that CRS: x = (lon % 360) - 180
            crs_180 = ccrs.PlateCarree(central_longitude=180.)
            x0_p = (b0 % 360.) - 180.   # e.g. 160°E → -20
            x1_p = (b1 % 360.) - 180.   # e.g. -160°E → +20
            ax.add_patch(mpatches.Rectangle(
                (x0_p, lat_b0), x1_p - x0_p, lat_b1 - lat_b0,
                linewidth=linewidth, edgecolor=edgecolor, facecolor='none',
                linestyle=linestyle,
                transform=crs_180, zorder=5))
            mid_lon = (x0_p + x1_p) / 2.
            mid_crs = crs_180

        if label:
            ax.text(mid_lon, lat_b1 + 0.5, reg['name'],
                    fontsize=label_fontsize, color=edgecolor,
                    ha='center', va='bottom',
                    transform=mid_crs, zorder=6,
                    fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.6, pad=1,
                              boxstyle='round,pad=0.2',
                              edgecolor=edgecolor, linewidth=linewidth,
                              linestyle=linestyle))


def plot_diurnal_region_group(dc, lat, lon, regions,
                               dt_hours=3, n_harm=2,
                               title='', ncols=3,
                               inset_map_extent=None,
                               land_mask=None,
                               show_states=False,
                               ylim=None):
    """
    Panel of regional-mean diurnal cycle line plots with harmonic overlays.

    For each region in *regions*, plots:
      • Raw composite DC (black solid)
      • H1 fit = mean + A1·cos(2π(t−φ1)/24)  (red dashed)
      • H2 fit = mean + A2·cos(4π(t−φ2)/12)  (blue dashed)
      • H1+H2 combined fit (orange, if n_harm ≥ 2)
    A small cartopy inset map in each panel shows the box location.

    Parameters
    ----------
    dc        : xr.DataArray  (time_of_day, lat, lon)
    lat, lon  : 1-D arrays
    regions   : list of dicts [{name, lat_box, lon_box,
                                mask_type (optional: 'land'|'ocean'|None),
                                land_only  (optional bool, legacy)}]
                mask_type='land'  → average land points only
                mask_type='ocean' → average ocean points only
                mask_type=None    → average all points
                land_only=True is treated as mask_type='land' for backward compat.
    dt_hours  : float
    n_harm    : int  (1 or 2)
    title     : str  figure suptitle
    ncols     : int  subplot columns
    inset_map_extent : [lon0,lon1,lat0,lat1] shared for all insets (or None)
    land_mask : ndarray (nlat, nlon) bool or None
                Full-grid land mask from get_land_mask().  Required when any
                region has mask_type='land' or 'ocean'; ignored otherwise.

    Returns
    -------
    fig : matplotlib Figure
    """
    tod = (dc['time_of_day'].values.astype(float)
           if isinstance(dc, xr.DataArray)
           else np.arange(dc.shape[0], dtype=float) * dt_hours)

    # Pre-pass: compute data range across all regions for shared y-limits
    if ylim is None:
        _all_vals = []
        for reg in regions:
            _mt = reg.get('mask_type', 'land' if reg.get('land_only', False) else None)
            _pm = (land_mask if _mt == 'land'
                   else (~land_mask if (_mt == 'ocean' and land_mask is not None) else None))
            try:
                _dc_reg = region_mean_dc(dc, lat, lon, reg['lat_box'], reg['lon_box'],
                                         point_mask=_pm)
                _all_vals.append(float(_dc_reg.min()))
                _all_vals.append(float(_dc_reg.max()))
            except (ValueError, Exception):
                pass
        if _all_vals:
            _pad = max(0.05 * (max(_all_vals) - min(_all_vals)), 0.05)
            ylim = (min(_all_vals) - _pad, max(_all_vals) + _pad)

    n_reg  = len(regions)
    nrows  = max(1, int(np.ceil(n_reg / ncols)))
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 3.8 * nrows),
                              sharex=True, squeeze=False)
    fig.suptitle(title, fontsize=11, fontweight='bold', y=1.005)

    harm_colors = ['tab:red', 'tab:blue']
    harm_labels = ['H1 (24 h)', 'H2 (12 h)']
    t_dense_lst = np.linspace(0., 24., 241)   # dense LST grid for smooth fits

    for idx, reg in enumerate(regions):
        row, col = divmod(idx, ncols)
        ax = axes[row, col]

        # Resolve mask_type: prefer explicit 'mask_type' field; fall back to land_only
        _mt = reg.get('mask_type', 'land' if reg.get('land_only', False) else None)
        if _mt == 'land':
            point_mask = land_mask   # None if not provided
        elif _mt == 'ocean':
            point_mask = ~land_mask if land_mask is not None else None
        else:
            point_mask = None

        try:
            dc_reg = region_mean_dc(dc, lat, lon, reg['lat_box'], reg['lon_box'],
                                     point_mask=point_mask)
        except ValueError as exc:
            ax.text(0.5, 0.5, f'No data\n{exc}',
                    ha='center', va='center', transform=ax.transAxes, fontsize=7,
                    wrap=True)
            ax.set_title(reg['name'], fontsize=9, fontweight='bold')
            continue

        harmonics, mean_val = compute_harmonics_1d(dc_reg, dt_hours=dt_hours,
                                                    n_harm=n_harm)

        # ── Convert to local solar time ──────────────────────────────────────
        # Centre longitude of region (0-360 convention)
        lon_c = ((reg['lon_box'][0] + reg['lon_box'][1]) / 2.) % 360.
        lst_offset = lon_c / 15.   # hours ahead of UTC

        # Shift raw DC time-of-day to LST and sort to 0–24 order
        tod_lst  = (tod + lst_offset) % 24.
        sort_idx = np.argsort(tod_lst)
        tod_plot = tod_lst[sort_idx]
        dc_plot  = dc_reg[sort_idx]

        # Shift harmonic phases from UTC to LST
        harmonics_lst = [
            {'amp':      h['amp'],
             'phase_utc': (h['phase_utc'] + lst_offset) % h['period_h'],
             'period_h':  h['period_h']}
            for h in harmonics
        ]
        fit_total, fit_each = reconstruct_harmonic_fit(
            t_dense_lst, harmonics_lst, n_harm=n_harm, mean=mean_val)

        # Raw DC
        ax.plot(tod_plot, dc_plot, color='k', lw=1.8, label='Raw DC', zorder=3)

        # Individual harmonics
        for ih in range(n_harm):
            ax.plot(t_dense_lst, fit_each[ih],
                    color=harm_colors[ih], lw=1.2, ls='--',
                    label=harm_labels[ih], zorder=2)

        # Combined fit
        if n_harm >= 2:
            ax.plot(t_dense_lst, fit_total,
                    color='darkorange', lw=1.5, ls='-', alpha=0.75,
                    label='H1+H2', zorder=2)

        # Mean reference line
        ax.axhline(mean_val, color='grey', lw=0.7, ls=':', zorder=1)

        # Axes formatting
        ax.set_xlim(0., 24.)
        ax.set_xticks([0, 6, 12, 18, 24])
        ax.set_xlabel('LST hour', fontsize=8)
        ax.set_ylabel('mm day⁻¹', fontsize=8)
        ax.tick_params(labelsize=8)
        if ylim is not None:
            ax.set_ylim(ylim)
        ax.set_title(reg['name'], fontsize=9, fontweight='bold')

        # Coordinate + LST-offset annotation
        lb, ob = reg['lat_box'], reg['lon_box']
        _mt_tag = reg.get('mask_type', 'land' if reg.get('land_only', False) else None)
        mask_tag = {'land': '  land only', 'ocean': '  ocean only'}.get(_mt_tag, '')
        ax.text(0.02, 0.97,
                f'{lb[0]:.0f}°–{lb[1]:.0f}°N, {ob[0]:.0f}°–{ob[1]:.0f}°E'
                f'  (LST≈UTC+{lst_offset:.0f}h){mask_tag}',
                transform=ax.transAxes, fontsize=7, va='top', color='dimgrey')

        if idx == 0:
            ax.legend(fontsize=7, loc='upper right', framealpha=0.85)

        # Inset map
        _add_region_inset(ax, reg, inset_map_extent, show_states=show_states)

    # Hide unused axes
    for idx in range(n_reg, nrows * ncols):
        row, col = divmod(idx, ncols)
        axes[row, col].set_visible(False)

    plt.tight_layout()
    return fig
