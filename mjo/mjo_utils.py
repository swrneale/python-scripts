"""
mjo_utils.py
Utilities for MJO lag correlation / lag regression analysis.

Mirrors the NCL script lag_corr_CESM3_315_316_lat_time.ncl.
Supports TRMM + ERAI observations and CAM/CESM model output.
"""

import numpy as np
import pandas as pd
import xarray as xr
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ─────────────────────────────────────────────────────────────────────────────
# Time utilities
# ─────────────────────────────────────────────────────────────────────────────

def decode_cftime(da_time):
    """
    Convert a cftime coordinate (CAM/CESM model output) to a pandas DatetimeIndex.

    Works for NoLeap, Gregorian, and proleptic_gregorian calendars.
    Uses year/month/day only (ignores sub-daily info).

    Parameters
    ----------
    da_time : xarray DataArray  (decoded with use_cftime=True)

    Returns
    -------
    DatetimeIndex
    """
    cf = da_time.values
    return pd.DatetimeIndex(
        [pd.Timestamp(t.year, t.month, t.day) for t in cf]
    )


def decode_julian_day(da_time):
    """
    Convert an ERAI Julian-day float coordinate to a pandas DatetimeIndex.

    ERAI stores time as Julian Day numbers (JD). JD 2440587.5 = 1970-01-01 UTC.

    Parameters
    ----------
    da_time : xarray DataArray  (the raw 'time' coordinate, decoded_times=False)

    Returns
    -------
    DatetimeIndex
    """
    jd = da_time.values.astype(float)
    days_since_1970 = jd - 2440587.5
    return pd.to_datetime(days_since_1970, unit='D', origin='1970-01-01').normalize()


# ─────────────────────────────────────────────────────────────────────────────
# CAM / CESM data loader
# ─────────────────────────────────────────────────────────────────────────────

def load_cam_daily(data_dir, case_name,
                   year_start=None, year_end=None,
                   lat_s=-90., lat_n=90., lon_w=0., lon_e=360.,
                   precip_vars=('PRECC', 'PRECL')):
    """
    Load CAM/CESM daily mean data for precipitation and U850.

    Expects files of the form:
        {data_dir}/{case_name}_dmeans_ts_PRECC.nc
        {data_dir}/{case_name}_dmeans_ts_PRECL.nc
        {data_dir}/{case_name}_dmeans_ts_U850.nc

    Parameters
    ----------
    data_dir    : str, directory containing the _dmeans_ts_*.nc files
    case_name   : str, CESM case name (used to build filenames)
    year_start  : int or None, first model year to include
    year_end    : int or None, last model year to include
    lat_s/n     : float, latitude bounds to subset
    lon_w/e     : float, longitude bounds to subset
    precip_vars : tuple of str, variables to sum for total precip
                  (default ('PRECC','PRECL'); use ('PRECT',) if available)

    Returns
    -------
    prect : ndarray (time, lat, lon)  mm/day
    u850  : ndarray (time, lat, lon)  m/s
    lat   : 1-D ndarray
    lon   : 1-D ndarray
    times : DatetimeIndex
    """
    import os

    def _open(var):
        fpath = os.path.join(data_dir, f'{case_name}_dmeans_ts_{var}.nc')
        return xr.open_dataset(fpath, use_cftime=True)

    # ── Precipitation ──────────────────────────────────────────────────────
    ds0 = _open(precip_vars[0])
    times_cf = decode_cftime(ds0['time'])

    lat = ds0['lat'].values.astype(float)
    lon = ds0['lon'].values.astype(float)

    # Ensure lat is S→N
    if lat[0] > lat[-1]:
        lat = lat[::-1]
        flip_lat = True
    else:
        flip_lat = False

    lat_mask = (lat >= lat_s) & (lat <= lat_n)
    lon_mask = (lon >= lon_w) & (lon <= lon_e)

    def _load_var(ds, varname):
        arr = ds[varname].values.astype(float)   # (time, lat, lon)
        if flip_lat:
            arr = arr[:, ::-1, :]
        return arr[:, lat_mask, :][:, :, lon_mask]

    prect = _load_var(ds0, precip_vars[0])
    for v in precip_vars[1:]:
        prect += _load_var(_open(v), v)
    prect *= 86400.0 * 1000.0   # m/s → mm/day

    # ── U850 ───────────────────────────────────────────────────────────────
    ds_u = _open('U850')
    u850 = _load_var(ds_u, 'U850')

    # ── Year subset ────────────────────────────────────────────────────────
    ymask = np.ones(len(times_cf), dtype=bool)
    if year_start is not None:
        ymask &= times_cf.year >= year_start
    if year_end is not None:
        ymask &= times_cf.year <= year_end

    return (prect[ymask], u850[ymask],
            lat[lat_mask], lon[lon_mask],
            times_cf[ymask])


# ─────────────────────────────────────────────────────────────────────────────
# Lanczos bandpass filter
# ─────────────────────────────────────────────────────────────────────────────

def lanczos_weights(n_weights, fca, fcb):
    """
    Compute Lanczos bandpass filter weights.

    Parameters
    ----------
    n_weights : int   total number of weights (odd, e.g. 201)
    fca       : float low-frequency  cutoff (cycles/day, e.g. 1/100)
    fcb       : float high-frequency cutoff (cycles/day, e.g. 1/20)

    Returns
    -------
    w : ndarray, shape (n_weights,)
    """
    M = n_weights // 2
    n = np.arange(-M, M + 1, dtype=float)

    # Lanczos sigma window  sigma[n] = sinc(n/M)
    sigma = np.sinc(n / M)

    # Ideal bandpass = difference of two ideal lowpass filters
    # numpy.sinc(x) = sin(pi*x)/(pi*x)
    w = 2.0 * fcb * np.sinc(2.0 * fcb * n) * sigma \
      - 2.0 * fca * np.sinc(2.0 * fca * n) * sigma

    return w


def apply_lanczos_bpf(data, n_weights=201, fca=1.0 / 100.0, fcb=1.0 / 20.0):
    """
    Apply Lanczos bandpass filter along the first (time) dimension.

    Parameters
    ----------
    data      : ndarray, shape (time,) or (time, ...) — any trailing dims
    n_weights : int
    fca, fcb  : float, cutoff frequencies (cycles/day)

    Returns
    -------
    filtered : same shape as data; NaN inserted at the M-point edges
    """
    w = lanczos_weights(n_weights, fca, fcb)
    M = n_weights // 2

    orig_shape = data.shape
    if data.ndim == 1:
        data2d = data[:, np.newaxis]
    else:
        nt = data.shape[0]
        data2d = data.reshape(nt, -1)

    out = np.full_like(data2d, np.nan, dtype=float)
    for j in range(data2d.shape[1]):
        col = data2d[:, j].astype(float)
        if np.all(np.isnan(col)):
            continue
        # Replace NaN with 0 for convolution, then restore NaN edges
        col_nofill = np.where(np.isnan(col), 0.0, col)
        out[:, j] = np.convolve(col_nofill, w, mode='same')

    # Edge values are unreliable
    out[:M, :] = np.nan
    out[-M:, :] = np.nan

    return out.reshape(orig_shape)


# ─────────────────────────────────────────────────────────────────────────────
# Season utilities
# ─────────────────────────────────────────────────────────────────────────────

SEASON_MONTHS = {
    'winter': (12, 1, 2),
    'DJF':    (12, 1, 2),
    'summer': (6, 7, 8),
    'JJA':    (6, 7, 8),
    'MAM':    (3, 4, 5),
    'SON':    (9, 10, 11),
    'ANN':    tuple(range(1, 13)),
}


def season_mask(times, season='winter'):
    """Return boolean array selecting days that fall in the given season."""
    months = SEASON_MONTHS[season]
    return np.isin(times.month.values, months)


# ─────────────────────────────────────────────────────────────────────────────
# Base index
# ─────────────────────────────────────────────────────────────────────────────

def compute_base_index(data, lat, lon,
                       lat_s, lat_n, lon_w, lon_e,
                       n_weights=201, fca=1.0 / 100.0, fcb=1.0 / 20.0):
    """
    Compute the MJO precipitation base index over the specified region.

    Steps
    -----
    1. Subset to the base region
    2. Cosine-latitude weighted area average → 1-D time series
    3. Linear detrend
    4. Lanczos 20–100-day bandpass filter

    Parameters
    ----------
    data             : ndarray (time, lat, lon)
    lat, lon         : 1-D arrays (degrees)
    lat_s … lon_e   : float, region bounds
    n_weights        : int, Lanczos filter length (default 201)
    fca, fcb         : float, cutoff frequencies (cycles/day)

    Returns
    -------
    idx : 1-D ndarray (time,)
    """
    lat_mask = (lat >= lat_s) & (lat <= lat_n)
    lon_mask = (lon >= lon_w) & (lon <= lon_e)

    sub = data[:, lat_mask, :][:, :, lon_mask]
    lat_sub = lat[lat_mask]

    # Cosine-latitude weights
    wgt = np.cos(np.deg2rad(lat_sub))
    wgt_norm = wgt / wgt.sum()

    # Weighted average over lat; then plain mean over lon
    idx = (sub * wgt_norm[np.newaxis, :, np.newaxis]).sum(axis=1).mean(axis=1)

    idx = signal.detrend(idx.astype(float), type='linear')
    idx = apply_lanczos_bpf(idx, n_weights=n_weights, fca=fca, fcb=fcb)

    return idx


# ─────────────────────────────────────────────────────────────────────────────
# Lat/lon-averaged time series
# ─────────────────────────────────────────────────────────────────────────────

def compute_time_lon_series(data, lat, lat_s, lat_n,
                             n_weights=201, fca=1.0 / 100.0, fcb=1.0 / 20.0):
    """
    Latitude-average over [lat_s, lat_n], detrend, bandpass filter.

    Returns
    -------
    series : ndarray (time, lon)
    """
    lat_mask = (lat >= lat_s) & (lat <= lat_n)
    avg = data[:, lat_mask, :].mean(axis=1).astype(float)   # (time, lon)
    avg = signal.detrend(avg, axis=0, type='linear')
    return apply_lanczos_bpf(avg, n_weights=n_weights, fca=fca, fcb=fcb)


def compute_time_lat_series(data, lon, lon_w, lon_e,
                             n_weights=201, fca=1.0 / 100.0, fcb=1.0 / 20.0):
    """
    Longitude-average over [lon_w, lon_e], detrend, bandpass filter.

    Returns
    -------
    series : ndarray (time, lat)
    """
    lon_mask = (lon >= lon_w) & (lon <= lon_e)
    avg = data[:, :, lon_mask].mean(axis=2).astype(float)   # (time, lat)
    avg = signal.detrend(avg, axis=0, type='linear')
    return apply_lanczos_bpf(avg, n_weights=n_weights, fca=fca, fcb=fcb)


# ─────────────────────────────────────────────────────────────────────────────
# Lag correlation / regression
# ─────────────────────────────────────────────────────────────────────────────

def _align_lag(base, field, lag):
    """
    Align 1-D base index and (time, space) field at the given lag (days).

    Convention (matches NCL mjo_xcor_lag):
      lag > 0  →  base leads field  (base at t, field at t+lag)
      lag < 0  →  field leads base
      lag = 0  →  simultaneous
    """
    n = len(base)
    if lag > 0:
        return base[:n - lag], field[lag:]
    elif lag < 0:
        return base[-lag:], field[:n + lag]
    else:
        return base, field


def _tmask_at_lag(smask, lag):
    """Shift the season mask to match the base-index slice at this lag."""
    n = len(smask)
    if lag > 0:
        return smask[:n - lag]
    elif lag < 0:
        return smask[-lag:]
    else:
        return smask


def mjo_lag_corr(base_idx, field, times, mxlag=25, season='winter'):
    """
    Lag correlations between the base index and a 2-D (time, space) field,
    restricted to days in the chosen season.

    Parameters
    ----------
    base_idx : 1-D ndarray (time,)
    field    : ndarray (time, n_space)  — space = lon or lat
    times    : DatetimeIndex, length time
    mxlag    : int, maximum lag in days
    season   : str, 'winter'/'DJF' or 'summer'/'JJA' etc.

    Returns
    -------
    r    : ndarray (2*mxlag+1, n_space)
    lags : 1-D ndarray  (integer day lags from -mxlag to +mxlag)
    """
    smask = season_mask(times, season)
    lags = np.arange(-mxlag, mxlag + 1)
    n_space = field.shape[1]
    r = np.full((len(lags), n_space), np.nan)

    for i, lag in enumerate(lags):
        b, f = _align_lag(base_idx, field, lag)
        tm = _tmask_at_lag(smask, lag)

        b_s = b[tm].astype(float)
        f_s = f[tm].astype(float)

        valid = ~np.isnan(b_s)
        b_s, f_s = b_s[valid], f_s[valid]
        if len(b_s) < 20:
            continue

        b_std = b_s.std()
        if b_std == 0:
            continue
        b_z = (b_s - b_s.mean()) / b_std

        f_std = f_s.std(axis=0)
        f_std[f_std == 0] = np.nan
        f_z = (f_s - f_s.mean(axis=0)) / f_std

        with np.errstate(all='ignore'):
            r[i] = np.nanmean(b_z[:, np.newaxis] * f_z, axis=0)

    return r, lags


def mjo_lag_regr(base_idx, field, times, mxlag=25, season='winter'):
    """
    Lag regressions of a 2-D (time, space) field onto the base index.

    The base index is normalised by its own standard deviation so that
    the returned coefficients have units of (field units) per (1 std of base).

    Parameters
    ----------
    base_idx, field, times, mxlag, season : same as mjo_lag_corr

    Returns
    -------
    beta : ndarray (2*mxlag+1, n_space)
    lags : 1-D ndarray
    """
    smask = season_mask(times, season)
    lags = np.arange(-mxlag, mxlag + 1)
    n_space = field.shape[1]
    beta = np.full((len(lags), n_space), np.nan)

    for i, lag in enumerate(lags):
        b, f = _align_lag(base_idx, field, lag)
        tm = _tmask_at_lag(smask, lag)

        b_s = b[tm].astype(float)
        f_s = f[tm].astype(float)

        valid = ~np.isnan(b_s)
        b_s, f_s = b_s[valid], f_s[valid]
        if len(b_s) < 20:
            continue

        b_std = b_s.std()
        if b_std == 0:
            continue
        b_n = (b_s - b_s.mean()) / b_std          # normalised base index

        f_dm = f_s - f_s.mean(axis=0)
        # beta = <b_n * f_dm> / <b_n^2>  ≈  <b_n * f_dm>  (b_n is unit-variance)
        with np.errstate(all='ignore'):
            beta[i] = np.nanmean(b_n[:, np.newaxis] * f_dm, axis=0)

    return beta, lags


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_lag_hovmoller(ax, data, lags, x_coords,
                       x_label='', title='',
                       clevs=None, cmap='RdBu_r',
                       x_lim=None,
                       ref_lines=None,
                       shading_box=None,
                       ylabel=True):
    """
    Plot a lag-lon or lag-lat Hovmöller diagram.

    Parameters
    ----------
    ax          : matplotlib Axes
    data        : 2-D ndarray (n_lags, n_x)
    lags        : 1-D array, lag values in days (y-axis)
    x_coords    : 1-D array, spatial coordinate in degrees (x-axis)
    clevs       : contour levels; auto-scaled if None
    cmap        : colormap string
    x_lim       : (xmin, xmax) tuple or None
    ref_lines   : list of x-values for vertical green dashed reference lines
    shading_box : (x1, x2) tuple for semi-transparent green shading
    ylabel      : bool, whether to label y-axis

    Returns
    -------
    cf : QuadContourSet (filled contours, for colorbar)
    """
    if clevs is None:
        vmax = np.nanpercentile(np.abs(data), 97)
        step = vmax / 10.0
        clevs = np.arange(-vmax, vmax + step / 2.0, step)
        clevs = clevs[clevs != 0]

    cf = ax.contourf(x_coords, lags, data,
                     levels=clevs, cmap=cmap, extend='both')
    pos_levs = clevs[clevs > 0]
    neg_levs = clevs[clevs < 0]
    if len(pos_levs):
        ax.contour(x_coords, lags, data, levels=pos_levs,
                   colors='k', linewidths=0.6, linestyles='solid')
    if len(neg_levs):
        ax.contour(x_coords, lags, data, levels=neg_levs,
                   colors='k', linewidths=0.6, linestyles='dashed')

    ax.axhline(0, color='k', linewidth=1.2)

    if ref_lines is not None:
        for xl in ref_lines:
            ax.axvline(xl, color='green', linewidth=1.5, linestyle='--', alpha=0.8)

    if shading_box is not None:
        x1, x2 = shading_box
        ax.axvspan(x1, x2, alpha=0.15, color='green')

    if x_lim is not None:
        ax.set_xlim(x_lim)
    ax.set_ylim(lags[0], lags[-1])

    ax.set_xlabel(x_label, fontsize=10)
    if ylabel:
        ax.set_ylabel('Lag (days)', fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.tick_params(labelsize=9)

    return cf


def make_lag_panel(r_prect, r_u850, lags,
                   lon_coords, lat_coords,
                   season_label, case_label,
                   clevs_p=None, clevs_u=None,
                   lon_lim=None, lat_lim=None,
                   ref_lon=None, ref_lat=None,
                   mc_box=(115., 145.),
                   analysis_label='Correlation',
                   cbar_label_p='Correlation', cbar_label_u='Correlation',
                   lon_coords_u=None, lat_coords_u=None):
    """
    Make a 2×2 panel:
      Row 0: lag-lon (EW)  for PRECT and U850
      Row 1: lag-lat (NS)  for PRECT and U850

    Parameters
    ----------
    r_prect   : dict with keys 'lon' and 'lat', each (n_lags, n_x)
    r_u850    : dict with keys 'lon' and 'lat', each (n_lags, n_x)
    lags      : 1-D ndarray
    lon_coords    : 1-D array of longitudes for PRECT EW plots
    lat_coords    : 1-D array of latitudes  for PRECT NS plots
    lon_coords_u  : 1-D array of longitudes for U850 EW plots (defaults to lon_coords)
    lat_coords_u  : 1-D array of latitudes  for U850 NS plots (defaults to lat_coords)
    season_label, case_label : str
    clevs_p, clevs_u : contour levels for precip and U850 (None → auto)
    lon_lim, lat_lim : (min, max) tuples or None
    ref_lon   : x reference lines for EW plots (list of float)
    ref_lat   : x reference lines for NS plots (list of float)
    mc_box    : (x1, x2) for Maritime Continent shading on EW plots
    analysis_label : 'Correlation' or 'Regression'

    Returns
    -------
    fig, axes
    """
    if lon_coords_u is None:
        lon_coords_u = lon_coords
    if lat_coords_u is None:
        lat_coords_u = lat_coords

    fig, axes = plt.subplots(2, 2, figsize=(12, 9),
                             sharex='row', sharey=True)
    fig.suptitle(f'{case_label}  —  {season_label}  ({analysis_label})',
                 fontsize=13, fontweight='bold', y=0.98)

    plot_cfg = [
        # (ax,        data,              lags, x_coords,      x_label,      clevs,    x_lim,   ref,      shade,   ylabel)
        (axes[0, 0], r_prect['lon'],     lags, lon_coords,    'Longitude',  clevs_p,  lon_lim, ref_lon,  mc_box,  True),
        (axes[0, 1], r_u850['lon'],      lags, lon_coords_u,  'Longitude',  clevs_u,  lon_lim, ref_lon,  mc_box,  False),
        (axes[1, 0], r_prect['lat'],     lags, lat_coords,    'Latitude',   clevs_p,  lat_lim, ref_lat,  None,    True),
        (axes[1, 1], r_u850['lat'],      lags, lat_coords_u,  'Latitude',   clevs_u,  lat_lim, ref_lat,  None,    False),
    ]

    titles = [
        f'PRECT — lag-lon',
        f'U850 — lag-lon',
        f'PRECT — lag-lat',
        f'U850 — lag-lat',
    ]
    cbar_labels = [cbar_label_p, cbar_label_u, cbar_label_p, cbar_label_u]

    cfs = []
    for (ax, dat, lg, xc, xl, clv, xlm, rl, shd, yl), ttl, cbl in zip(
            plot_cfg, titles, cbar_labels):
        cf = plot_lag_hovmoller(ax, dat, lg, xc,
                                x_label=xl, title=ttl,
                                clevs=clv, x_lim=xlm,
                                ref_lines=rl, shading_box=shd,
                                ylabel=yl)
        cfs.append(cf)

    # Individual colourbars (right of each panel)
    for ax, cf, cbl in zip(axes.flat, cfs, cbar_labels):
        cb = fig.colorbar(cf, ax=ax, orientation='vertical',
                          shrink=0.9, pad=0.02, fraction=0.04)
        cb.set_label(cbl, fontsize=8)
        cb.ax.tick_params(labelsize=8)

    plt.subplots_adjust(hspace=0.25, wspace=0.35)
    return fig, axes
