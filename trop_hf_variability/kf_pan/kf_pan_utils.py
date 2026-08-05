"""
Wheeler-Kiladis wavenumber-frequency spectral analysis utilities.
Python port of NCL diagnostics_cam_kf_pan_col.ncl.

References:
  Wheeler, M. and G.N. Kiladis (1999), J. Atmos. Sci., 56, 374-399.
  Hayashi, Y. (1971), J. Meteor. Soc. Japan, 49, 125-128.
"""

import numpy as np
import scipy.signal
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.ticker as mticker
from pathlib import Path
from glob import glob
import warnings

# ---------------------------------------------------------------------------
# Signal-processing utilities
# ---------------------------------------------------------------------------

def cosine_taper_window(n, p=0.1):
    """1D cosine bell taper window of length n; fraction p tapered at each end."""
    w = np.ones(n)
    n_taper = int(n * p)
    if n_taper > 0:
        t = np.arange(n_taper)
        bell = 0.5 * (1.0 - np.cos(np.pi * t / n_taper))
        w[:n_taper]  = bell
        w[n-n_taper:] = bell[::-1]
    return w


def smooth121_1d(x, n_passes=1):
    """Apply 1-2-1 smoother (in-place) n_passes times to 1D array x."""
    for _ in range(n_passes):
        x[1:-1] = 0.25 * (x[:-2] + 2.0 * x[1:-1] + x[2:])
    return x


# ---------------------------------------------------------------------------
# Symmetric / antisymmetric decomposition
# ---------------------------------------------------------------------------

def decompose_sym_asym(x):
    """Decompose x (time, lat, lon) about the equator.

    The NCL convention:
      SH (lower lat indices)  → symmetric  = (x(lat) + x(-lat)) / 2
      NH (upper lat indices)  → antisymmetric = (x(lat) - x(-lat)) / 2

    Assumes lats are ordered S→N so that index nl pairs with nlat-1-nl.
    Returns xsa with the same shape as x.
    """
    xsa = x.copy()
    nlat = x.shape[1]
    N2 = nlat // 2
    for nl in range(N2):
        xsa[:, nl, :]        = 0.5 * (x[:, nlat-1-nl, :] + x[:, nl, :])  # SH → sym
        xsa[:, nlat-1-nl, :] = 0.5 * (x[:, nlat-1-nl, :] - x[:, nl, :])  # NH → asym
    return xsa


# ---------------------------------------------------------------------------
# Annual cycle removal
# ---------------------------------------------------------------------------

def remove_annual_cycle(x, spd, f_crit):
    """Remove low-frequency signals (< f_crit cpd) via FFT zeroing.

    Parameters
    ----------
    x      : (time, lat, lon) float array
    spd    : samples per day
    f_crit : cutoff frequency (cpd); all periods longer than 1/f_crit are removed
             Typically 1/nDayWin.

    Returns modified x (same array, modified in-place and returned).
    """
    ntim = x.shape[0]
    n_day_tot = ntim / spd
    cf = np.fft.rfft(x, axis=0)          # (ntim//2+1, nlat, mlon) complex
    # freq[k] in cpd = k / n_day_tot
    k_crit = int(n_day_tot * f_crit)     # zero out k = 0 .. k_crit
    cf[:k_crit + 1] = 0.0
    x[:] = np.fft.irfft(cf, n=ntim, axis=0)
    return x


# ---------------------------------------------------------------------------
# Hayashi wave decomposition
# ---------------------------------------------------------------------------

def resolve_waves_hayashi(q, n_day_win, spd):
    """Rearrange 2D FFT result into Hayashi signed-wave/signed-freq power spectrum.

    Parameters
    ----------
    q        : complex (mlon, N) – after spatial then temporal FFT and normalization
    n_day_win: temporal window length (days)
    spd      : samples per day

    Returns
    -------
    pee      : float (mlon+1, N+1) power, wave axis -mlon/2..+mlon/2,
                                           freq axis -spd/2..+spd/2 cpd
    wave_arr : 1D array of wavenumbers (length mlon+1)
    freq_arr : 1D array of frequencies in cpd (length N+1)
    """
    mlon, N = q.shape
    power = q.real**2 + q.imag**2   # (mlon, N)

    # fftshift re-orders both axes so DC is at centre
    # wave axis after shift: [±mlon/2, -(mlon/2-1), ..., -1, 0, 1, ..., mlon/2-1]
    # freq axis after shift: [±N/2,    -(N/2-1),    ..., -1, 0, 1, ..., N/2-1    ]
    ps = np.fft.fftshift(power)      # (mlon, N)

    # Build (mlon+1, N+1): duplicate Nyquist endpoints so axis spans -Nyq .. +Nyq
    pee = np.zeros((mlon + 1, N + 1))
    pee[:mlon, :N] = ps
    pee[mlon, :N]  = ps[0, :]        # +wavenumber Nyquist = same as -wavenumber Nyquist
    pee[:mlon, N]  = ps[:, 0]        # +freq Nyquist = same as -freq Nyquist
    pee[mlon,  N]  = ps[0, 0]

    wave_arr = np.arange(-mlon // 2, mlon // 2 + 1, dtype=float)
    freq_arr = np.linspace(-spd / 2.0, spd / 2.0, N + 1)

    return pee, wave_arr, freq_arr


# ---------------------------------------------------------------------------
# Main WK spectral computation
# ---------------------------------------------------------------------------

def compute_wk_spectrum(x_np, lat_arr, spd=1, n_day_win=96, n_day_skip=-65,
                        vscale=1.0, lat_bound=15.0):
    """Compute Wheeler-Kiladis wavenumber-frequency power spectrum.

    Parameters
    ----------
    x_np     : (time, lat, lon) float32/64, no missing values, pre-trimmed to lat_bound
    lat_arr  : 1D latitude array (S→N, e.g. -15..+15)
    spd      : samples per day (1=daily, 4=6-hourly, 8=3-hourly)
    n_day_win: window length in days (default 96)
    n_day_skip: days between window starts; negative = overlap (default -65)
    vscale   : scale factor applied to x before analysis
    lat_bound: symmetric equatorial lat bound (for information only; subset externally)

    Returns
    -------
    dict with keys:
      psumanti_nl : (wave, freq) antisymmetric power (linear)
      psumsym_nl  : (wave, freq) symmetric power (linear)
      psumb_nl    : (wave, freq) background power after heavy smoothing (linear)
      wave        : 1D wavenumber array
      freq        : 1D frequency array (cpd)
    """
    x = x_np.astype(np.float64) * vscale
    ntim, nlat, mlon = x.shape
    n_day_tot = ntim // spd

    if n_day_tot < n_day_win:
        raise ValueError(f"n_day_tot={n_day_tot} < n_day_win={n_day_win}")

    # Temporal sampling
    n_samp_win  = n_day_win  * spd
    n_samp_skip = n_day_skip * spd
    n_window    = (ntim - n_samp_win) // (n_samp_win + n_samp_skip) + 1
    N           = n_samp_win         # convenience alias

    # ----- Pre-processing -----
    # Linear detrend along time
    x = scipy.signal.detrend(x, axis=0)

    # Remove annual cycle (if enough data)
    f_crit = 1.0 / n_day_win
    if n_day_tot >= 365:
        remove_annual_cycle(x, spd, f_crit)

    # Sym / Asym decomposition (in time, lat, lon)
    xAS = decompose_sym_asym(x)   # (time, lat, lon)

    # Taper window
    taper_win = cosine_taper_window(N, p=0.1)   # (N,)

    # ----- Pre-compute detrended + tapered windows (all lats at once) -----
    windows = []   # list of (nlat, mlon, N) float arrays
    nt_strt = 0
    for _nw in range(n_window):
        nt_end = nt_strt + N
        seg = xAS[nt_strt:nt_end, :, :].copy()      # (N, nlat, mlon)
        # Detrend along time axis (axis 0) for each lat, lon
        seg = scipy.signal.detrend(seg, axis=0)
        # Taper along time
        seg *= taper_win[:, np.newaxis, np.newaxis]
        windows.append(seg)                          # (N, nlat, mlon)
        nt_strt += N + n_samp_skip

    # ----- Accumulate power spectrum (lat, wave, freq) -----
    # Placeholder: first resolve to get axis sizes
    _q0 = np.fft.fft(windows[0][:, 0, :], axis=-1) / mlon  # (N, mlon)
    _q0 = np.fft.fft(_q0, axis=0) / N                       # (N, mlon)
    _q0 = _q0.T                                              # (mlon, N)
    _, wave_arr, freq_arr = resolve_waves_hayashi(_q0, n_day_win, spd)

    peeAS = np.zeros((nlat, mlon + 1, N + 1))

    for nl in range(nlat):
        for nw, seg in enumerate(windows):
            work = seg[:, nl, :]   # (N, mlon) – time first
            # Spatial FFT (over lon, axis 1), normalize
            q = np.fft.fft(work, axis=1) / mlon    # (N, mlon) complex
            # Temporal FFT (over time, axis 0), normalize
            q = np.fft.fft(q, axis=0) / N          # (N, mlon) complex
            q = q.T                                  # (mlon, N)
            pee, _, _ = resolve_waves_hayashi(q, n_day_win, spd)
            peeAS[nl] += pee / n_window

    # ----- Sum latitudes for antisym / sym / background -----
    N2 = nlat // 2
    if nlat % 2 == 0:
        # NH (antisymmetric): indices N2..nlat-1
        # SH (symmetric):     indices 0..N2-1
        psumanti_nl = 2.0 * peeAS[N2:nlat].sum(axis=0)
        psumsym_nl  = 2.0 * peeAS[0:N2].sum(axis=0)
    else:
        psumanti_nl = 2.0 * peeAS[N2+1:nlat].sum(axis=0)
        psumsym_nl  = 2.0 * peeAS[0:N2+1].sum(axis=0)

    # Background = sum over all latitudes
    psumb_nl = peeAS.sum(axis=0)

    # Set DC (freq=0) to NaN
    i_dc = N // 2  # index of zero frequency in freq_arr
    psumanti_nl[:, i_dc] = np.nan
    psumsym_nl[:, i_dc]  = np.nan
    psumb_nl[:, i_dc]    = np.nan

    # ----- Smooth raw spectra (psumanti, psumsym) – freq dimension only -----
    # One pass of 1-2-1 over positive freq range, wavenumbers -27..27
    i_w_lo = np.searchsorted(wave_arr, -27)
    i_w_hi = np.searchsorted(wave_arr,  27) + 1
    i_f_lo = i_dc + 1          # first positive freq (after DC)
    i_f_hi = N                  # up to (but not including) +Nyquist endpoint

    for iw in range(i_w_lo, i_w_hi):
        sl = psumanti_nl[iw, i_f_lo:i_f_hi]
        sl_c = sl.copy()
        sl_c[1:-1] = 0.25 * (sl[:-2] + 2.0 * sl[1:-1] + sl[2:])
        psumanti_nl[iw, i_f_lo:i_f_hi] = sl_c

        sl = psumsym_nl[iw, i_f_lo:i_f_hi]
        sl_c = sl.copy()
        sl_c[1:-1] = 0.25 * (sl[:-2] + 2.0 * sl[1:-1] + sl[2:])
        psumsym_nl[iw, i_f_lo:i_f_hi] = sl_c

    # ----- Background spectrum: heavy smoothing -----
    # Pass 1: frequency-dependent wavenumber smoothing
    for it in range(i_dc + 1, N + 1):
        f_val = freq_arr[it]
        if f_val < 0.1:
            n_passes = 5
        elif f_val < 0.2:
            n_passes = 10
        elif f_val < 0.3:
            n_passes = 20
        else:
            n_passes = 40
        sl = psumb_nl[i_w_lo:i_w_hi, it].copy()
        smooth121_1d(sl, n_passes=n_passes)
        psumb_nl[i_w_lo:i_w_hi, it] = sl

    # Pass 2: frequency smoothing for each wavenumber up to 0.8 cpd
    i_f_08 = int(np.searchsorted(freq_arr, 0.8))
    i_f_08 = min(i_f_08, N)
    for iw in range(i_w_lo, i_w_hi):
        sl = psumb_nl[iw, i_f_lo:i_f_08 + 1].copy()
        smooth121_1d(sl, n_passes=10)
        psumb_nl[iw, i_f_lo:i_f_08 + 1] = sl

    # Re-zero DC after smoothing
    psumb_nl[:, i_dc] = np.nan

    return dict(
        psumanti_nl=psumanti_nl,
        psumsym_nl=psumsym_nl,
        psumb_nl=psumb_nl,
        wave=wave_arr,
        freq=freq_arr,
    )


# ---------------------------------------------------------------------------
# Equatorial wave dispersion curves  (port of genDispersionCurves.ncl)
# ---------------------------------------------------------------------------

def gen_dispersion_curves(ahe=(50., 25., 12.), n_wave_type=6, n_planetary_wave=50):
    """Compute equatorial wave dispersion curves.

    Wave types (1-indexed as in NCL):
      1 = MRG (Mixed Rossby-Gravity)      antisymmetric
      2 = n=0 IG (inertio-gravity)        antisymmetric
      3 = n=2 IG                          antisymmetric
      4 = n=1 ER (equatorial Rossby)      symmetric
      5 = Kelvin                          symmetric
      6 = n=1 IG                          symmetric

    Parameters
    ----------
    ahe            : equivalent depths (m), e.g. (50, 25, 12)
    n_wave_type    : number of wave types (default 6)
    n_planetary_wave: number of planetary wavenumber points (default 50)

    Returns
    -------
    Apzwn : (n_wave_type, n_equiv_depth, n_planetary_wave) wavenumber arrays
    Afreq : (n_wave_type, n_equiv_depth, n_planetary_wave) frequency arrays (cpd)
    """
    pi    = np.pi
    re    = 6.37122e6     # m, Earth radius
    g     = 9.80665       # m/s^2
    omega = 7.292e-5      # 1/s, Earth angular velocity
    rlat  = 0.0           # equator
    ll    = 2.0 * pi * re * np.cos(np.abs(rlat))
    Beta  = 2.0 * omega * np.cos(np.abs(rlat)) / re
    fillval = np.nan

    n_equiv_depth = len(ahe)
    Apzwn = np.full((n_wave_type, n_equiv_depth, n_planetary_wave), fillval)
    Afreq = np.full((n_wave_type, n_equiv_depth, n_planetary_wave), fillval)

    for ww in range(1, n_wave_type + 1):
        for ed, he in enumerate(ahe):
            T = 1.0 / np.sqrt(Beta) * (g * he) ** 0.25
            L = (g * he) ** 0.25 / np.sqrt(Beta)

            for wn in range(1, n_planetary_wave + 1):
                # planetary wavenumber s ranges from +20 to -20 over the wn loop
                s = -20.0 * (wn - 1) * 2.0 / (n_planetary_wave - 1) + 20.0
                k = 2.0 * pi * s / ll

                deif = fillval

                if ww == 1:   # MRG
                    if k < 0.0:
                        del_ = np.sqrt(1.0 + 4.0 * Beta / (k**2 * np.sqrt(g * he)))
                        deif = k * np.sqrt(g * he) * (0.5 - 0.5 * del_)
                    elif k == 0.0:
                        deif = (g * he * Beta) ** 0.25
                    # k > 0: fillval

                elif ww == 2:  # n=0 IG
                    if k < 0.0:
                        pass  # fillval
                    elif k == 0.0:
                        deif = (g * he * Beta) ** 0.25
                    else:
                        del_ = np.sqrt(1.0 + 4.0 * Beta / (k**2 * np.sqrt(g * he)))
                        deif = k * np.sqrt(g * he) * (0.5 + 0.5 * del_)

                elif ww == 3:  # n=2 IG
                    n = 2.0
                    del_ = Beta * np.sqrt(g * he)
                    deif = np.sqrt((2.0 * n + 1.0) * del_ + g * he * k**2)
                    for _ in range(5):
                        deif = np.sqrt((2.0 * n + 1.0) * del_ + g * he * k**2
                                       + g * he * Beta * k / deif)

                elif ww == 4:  # n=1 ER
                    n = 1.0
                    if k < 0.0:
                        del_ = Beta / np.sqrt(g * he) * (2.0 * n + 1.0)
                        deif = -Beta * k / (k**2 + del_)
                    # k >= 0: fillval

                elif ww == 5:  # Kelvin
                    deif = k * np.sqrt(g * he)

                elif ww == 6:  # n=1 IG
                    n = 1.0
                    del_ = Beta * np.sqrt(g * he)
                    deif = np.sqrt((2.0 * n + 1.0) * del_ + g * he * k**2)
                    for _ in range(5):
                        deif = np.sqrt((2.0 * n + 1.0) * del_ + g * he * k**2
                                       + g * he * Beta * k / deif)

                Apzwn[ww-1, ed, wn-1] = s
                if np.isfinite(deif):
                    P = 2.0 * pi / (deif * 24.0 * 3600.0)   # period in seconds
                    Afreq[ww-1, ed, wn-1] = 1.0 / P          # cpd

    return Apzwn, Afreq


# ---------------------------------------------------------------------------
# Data loaders
# ---------------------------------------------------------------------------

def _lat_slice(ds, lat_bound, lat_name='lat'):
    """Return dataset subset to ±lat_bound, detecting coordinate name."""
    for cname in [lat_name, 'latitude', 'nav_lat']:
        if cname in ds.coords or cname in ds.dims:
            return ds.sel({cname: slice(-lat_bound, lat_bound)})
    raise KeyError(f"Cannot find latitude coordinate in dataset: {list(ds.coords)}")


def _ensure_s2n(da, lat_name='lat'):
    """Ensure latitude dimension is sorted South→North."""
    for cname in [lat_name, 'latitude']:
        if cname in da.dims:
            if da[cname].values[0] > da[cname].values[-1]:
                da = da.sortby(cname)
            return da
    return da


def load_imerg_data(yr0, yr1, lat_bound=15, resolution='1deg',
                    data_dir='/glade/derecho/scratch/rneale/IMERG/3hrly'):
    """Load IMERG 3-hourly precipitation data.

    Parameters
    ----------
    yr0, yr1    : start/end year (inclusive)
    lat_bound   : symmetric equatorial lat bound
    resolution  : '1deg' or '0.25deg'
    data_dir    : parent directory containing <resolution>/ subfolder

    Returns
    -------
    x_np  : (time, lat, lon) float32 array in mm/day
    lat   : 1D latitude array (S→N)
    lon   : 1D longitude array
    """
    res_suffix = resolution  # '1deg' or '0.25deg' → '1deg' or '0p25deg'
    if resolution == '0.25deg':
        res_suffix = '0p25deg'
    fdir = Path(data_dir) / resolution
    files = sorted(fdir.glob(f'IMERG_3hr.??????.{res_suffix}.nc'))
    files = [f for f in files if yr0 <= int(str(f.name)[11:15]) <= yr1]
    if not files:
        raise FileNotFoundError(f"No IMERG files found in {fdir} for {yr0}-{yr1}")

    ds = xr.open_mfdataset(files, combine='by_coords')
    da = ds['precip']   # mm/hr
    da = _ensure_s2n(da, lat_name='lat')
    da = _lat_slice(da, lat_bound, lat_name='lat')
    da = da * 24.0      # mm/hr → mm/day
    x_np = da.values.astype(np.float32)
    lat  = da['lat'].values
    lon  = da['lon'].values
    ds.close()
    return x_np, lat, lon


def load_era5_prec_data(yr0, yr1, lat_bound=15,
                         data_dir='/glade/derecho/scratch/rneale/ERA5/download'
                                  '/dcycle_3hrave/precip/clean_3hr'):
    """Load ERA5 3-hourly total precipitation.

    Returns x_np in mm/day, lat (S→N), lon.
    File variable: 'tp' (m per 3-hr accumulation).
    """
    fdir = Path(data_dir)
    files = sorted(fdir.glob('precip_????_era5_3hr_clean.nc'))
    files = [f for f in files
             if yr0 <= int(str(f.name).split('_')[1]) <= yr1]
    if not files:
        raise FileNotFoundError(f"No ERA5 precip files in {fdir} for {yr0}-{yr1}")

    ds = xr.open_mfdataset(files, combine='by_coords')
    # Detect time dimension name
    time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
    da = ds['tp'].rename({time_dim: 'time'})
    da = _ensure_s2n(da, lat_name='latitude')
    da = _lat_slice(da, lat_bound, lat_name='latitude')
    da = da * 1000.0 * 8.0   # m/3hr → mm/3hr * 8 → mm/day
    x_np = da.values.astype(np.float32)
    lat  = da['latitude'].values
    lon  = da['longitude'].values
    ds.close()
    return x_np, lat, lon


def load_era5_wind_data(var, yr0, yr1, lat_bound=15, level_hPa=None,
                         data_dir='/glade/derecho/scratch/rneale/ERA5/download'
                                  '/dcycle_3hrave'):
    """Load ERA5 3-hourly wind data (u or v).

    Parameters
    ----------
    var       : ERA5 short name, e.g. 'u', 'v', 'w'
    level_hPa : pressure level in hPa (None for surface fields)
    """
    fdir = Path(data_dir) / var / 'clean_3hr'
    files = sorted(fdir.glob(f'{var}_????_era5_3hr_clean.nc'))
    files = [f for f in files
             if yr0 <= int(str(f.name).split('_')[1]) <= yr1]
    if not files:
        raise FileNotFoundError(f"No ERA5 {var} files in {fdir} for {yr0}-{yr1}")

    ds = xr.open_mfdataset(files, combine='by_coords')
    time_dim = 'valid_time' if 'valid_time' in ds.dims else 'time'
    da = ds[var].rename({time_dim: 'time'})
    if level_hPa is not None and 'pressure_level' in da.dims:
        da = da.sel(pressure_level=level_hPa, method='nearest')
    da = _ensure_s2n(da, lat_name='latitude')
    da = _lat_slice(da, lat_bound, lat_name='latitude')
    x_np = da.values.astype(np.float32)
    lat  = da['latitude'].values
    lon  = da['longitude'].values
    ds.close()
    return x_np, lat, lon


def load_trmm_gpcp_data(source, var='PRECT', yr0=None, yr1=None, lat_bound=15,
                         data_dir='/glade/work/rneale/data'):
    """Load TRMM or GPCP daily mean data.

    Expects file: {data_dir}/{source}/{source}_dmeans_ts_{var}.nc
    """
    fpath = Path(data_dir) / source / f'{source}_dmeans_ts_{var}.nc'
    if not fpath.exists():
        raise FileNotFoundError(f"File not found: {fpath}")
    try:
        ds = xr.open_dataset(fpath)
    except ValueError as e:
        if 'conflicting sizes' in str(e):
            ds = xr.open_dataset(fpath, decode_coords=False)
        else:
            raise
    # Identify the precipitation variable
    vname = var if var in ds else list(ds.data_vars)[0]
    da = ds[vname]
    if yr0 is not None:
        da = da.sel(time=slice(str(yr0), str(yr1)))
    da = _ensure_s2n(da)
    da = _lat_slice(da, lat_bound)
    x_np = da.values.astype(np.float32)
    lat  = da['lat'].values if 'lat' in da.dims else da['latitude'].values
    lon  = da['lon'].values if 'lon' in da.dims else da['longitude'].values
    ds.close()
    return x_np, lat, lon


_RNEALE_ARCHIVE = '/glade/derecho/scratch/rneale/archive'
_HANNAY_ARCHIVE = '/glade/derecho/scratch/hannay/archive'


def _make_dmeans_from_h2a(case, var, out_dir):
    """Build a dmeans tseries file from h2a history files in hannay's archive.

    h2a files already contain daily means (cell_methods='time: mean').
    Concatenates all h2a files, extracts *var*, and writes to *out_dir*.

    Returns
    -------
    Path to the newly written file.
    """
    hist_dir = Path(_HANNAY_ARCHIVE) / case / 'atm' / 'hist'
    files = sorted(hist_dir.glob(f'{case}.cam.h2a.*.nc'))
    if not files:
        raise FileNotFoundError(
            f"No h2a files for case={case} in {hist_dir}"
        )

    print(f"  Building dmeans tseries from {len(files)} h2a files in hannay archive...")
    ds = xr.open_mfdataset(
        files,
        combine='by_coords',
        data_vars='minimal',
        coords='minimal',
        compat='override',
    )
    if var not in ds:
        raise KeyError(
            f"Variable '{var}' not found in h2a files. "
            f"Available: {list(ds.data_vars)}"
        )

    da = ds[[var]]   # keep as Dataset so to_netcdf preserves coords cleanly

    out_path = Path(out_dir) / f'{case}_dmeans_ts_{var}.nc'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"  Writing {out_path} ...")
    da.to_netcdf(out_path)
    ds.close()
    print(f"  Done.")
    return out_path


def load_model_data(case, var='PRECT', yr0=None, yr1=None, lat_bound=15,
                    vscale=86400. * 1000.,
                    data_dir=_RNEALE_ARCHIVE):
    """Load CESM/CAM daily time-series data.

    Looks first in {data_dir}/{case}/tseries/ for pre-built dmeans files.
    If none are found, builds one from h2a files in hannay's archive and
    saves it to data_dir for future use.

    vscale: unit conversion (default: m/s → mm/day for PRECT).
    """
    fdir = Path(data_dir) / case / 'tseries'
    candidates = sorted(fdir.glob(f'{case}_dmeans_ts_{var}*.nc'))

    if not candidates:
        # Fall back: build from h2a files in hannay's archive
        out_file = _make_dmeans_from_h2a(case, var, fdir)
        candidates = [out_file]

    ds = xr.open_mfdataset(candidates, combine='by_coords')
    da = ds[var]
    if yr0 is not None:
        da = da.sel(time=slice(str(yr0), str(yr1)))
    da = _ensure_s2n(da)
    da = _lat_slice(da, lat_bound)
    da = da * vscale
    x_np = da.values.astype(np.float32)
    lat  = da['lat'].values if 'lat' in da.dims else da['latitude'].values
    lon  = da['lon'].values if 'lon' in da.dims else da['longitude'].values
    ds.close()
    return x_np, lat, lon


def load_data(source, var='PRECT', resolution='1deg', freq='daily',
              yr0=None, yr1=None, lat_bound=15, level_hPa=None, data_dir=None):
    """Unified data loader dispatching to the appropriate loader.

    Parameters
    ----------
    source     : 'TRMM', 'GPCP', 'ERA5', 'IMERG', or a CESM case string
    var        : variable name; for ERA5 use short name ('tp', 'u', 'v')
    resolution : '1deg' or '0.25deg' (IMERG only)
    freq       : 'daily' or '3hourly' (affects spd)
    yr0, yr1   : year range
    lat_bound  : equatorial belt half-width
    level_hPa  : pressure level for 4D wind data
    data_dir   : override default data directory

    Returns
    -------
    x_np : (time, lat, lon) float32
    lat  : 1D latitude array
    lon  : 1D longitude array
    spd  : samples per day (1 for daily, 8 for 3-hourly)
    """
    spd = 1 if freq == 'daily' else 8

    if source == 'IMERG':
        kw = dict(data_dir=data_dir) if data_dir else {}
        x, lat, lon = load_imerg_data(yr0, yr1, lat_bound=lat_bound,
                                       resolution=resolution, **kw)
        spd = 8  # IMERG is always 3-hourly
    elif source == 'ERA5' and var in ('tp', 'precip', 'PRECT'):
        kw = dict(data_dir=data_dir) if data_dir else {}
        x, lat, lon = load_era5_prec_data(yr0, yr1, lat_bound=lat_bound, **kw)
        spd = 8
    elif source == 'ERA5':
        kw = dict(data_dir=data_dir) if data_dir else {}
        x, lat, lon = load_era5_wind_data(var, yr0, yr1, lat_bound=lat_bound,
                                           level_hPa=level_hPa, **kw)
        spd = 8
    elif source in ('TRMM', 'GPCP'):
        kw = dict(data_dir=data_dir) if data_dir else {}
        x, lat, lon = load_trmm_gpcp_data(source, var=var, yr0=yr0, yr1=yr1,
                                            lat_bound=lat_bound, **kw)
    else:
        # Assume it is a CESM/model case string
        kw = dict(data_dir=data_dir) if data_dir else {}
        x, lat, lon = load_model_data(source, var=var, yr0=yr0, yr1=yr1,
                                       lat_bound=lat_bound, **kw)

    return x, lat, lon, spd


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def add_hor_vert_lines(ax, min_wav=-15, max_wav=15):
    """Add horizontal dashed period lines (3, 6, 30 days) and vertical wavenumber-0 line."""
    wvs = np.array([min_wav, max_wav], dtype=float)
    ax.axhline(1.0 / 3.0,  color='k', lw=0.7, ls='--', alpha=0.6)
    ax.axhline(1.0 / 6.0,  color='k', lw=0.7, ls='--', alpha=0.6)
    ax.axhline(1.0 / 30.0, color='k', lw=0.7, ls='--', alpha=0.6)
    ax.axvline(0.0,         color='k', lw=0.7, ls='--', alpha=0.6)
    # Labels
    ax.text(min_wav + 0.3, 1.0 / 3.0 + 0.005,  '3 days',  fontsize=7, va='bottom')
    ax.text(min_wav + 0.3, 1.0 / 6.0 + 0.005,  '6 days',  fontsize=7, va='bottom')
    ax.text(min_wav + 0.3, 1.0 / 30.0 + 0.003, '30 days', fontsize=7, va='bottom')


_ASYM_WAVE_TYPES = [0, 1, 2]   # MRG, n=0 IG, n=2 IG
_SYM_WAVE_TYPES  = [3, 4, 5]   # n=1 ER, Kelvin, n=1 IG
_DISP_COLORS = {
    0: 'DarkGreen', 1: 'blue',      2: 'red',        # antisymmetric
    3: 'DarkGreen', 4: 'blue',      5: 'red',         # symmetric
}
_WAVE_LABELS = {
    3: 'n=1 ER',   4: 'Kelvin',  5: 'n=1 IG',
    0: 'MRG',      1: 'n=0 IG', 2: 'n=2 IG',
}


def add_dispersion_curves(ax, Apzwn, Afreq, plot_type='sym',
                           min_wav=-15, max_wav=15, max_freq=0.8,
                           add_labels=True):
    """Overlay equatorial wave dispersion curves on a WK panel axes.

    Parameters
    ----------
    plot_type : 'sym' or 'asym' – which wave families to plot
    """
    wtypes = _SYM_WAVE_TYPES if plot_type == 'sym' else _ASYM_WAVE_TYPES
    for wt in wtypes:
        col = _DISP_COLORS[wt]
        for ed in range(Apzwn.shape[1]):
            s   = Apzwn[wt, ed, :]
            frq = Afreq[wt, ed, :]
            mask = np.isfinite(s) & np.isfinite(frq) & \
                   (s >= min_wav) & (s <= max_wav) & (frq <= max_freq)
            if mask.any():
                ax.plot(s[mask], frq[mask], color=col, lw=1.5, zorder=5)

    if add_labels:
        # Equivalent depth labels at h=50, 25, 12 for Kelvin (wt=4) or MJO box
        for wt in wtypes:
            if wt == 4:  # Kelvin
                for ed, h_lbl in enumerate(('h=50', 'h=25', 'h=12')):
                    s   = Apzwn[wt, ed, :]
                    frq = Afreq[wt, ed, :]
                    # find label position near wavenumber 8
                    idx = np.argmin(np.abs(s - 8))
                    if np.isfinite(frq[idx]) and frq[idx] < max_freq:
                        ax.text(s[idx], frq[idx], h_lbl, fontsize=6,
                                color=_DISP_COLORS[wt], va='bottom')
            if wt in (3, 5):  # n=1 ER, n=1 IG
                s   = Apzwn[wt, 0, :]
                frq = Afreq[wt, 0, :]
                mid = len(s) // 2
                if np.isfinite(frq[mid]) and frq[mid] < max_freq:
                    ax.text(s[mid], frq[mid], _WAVE_LABELS[wt], fontsize=6,
                            color=_DISP_COLORS[wt], va='bottom')


def _wk_axes_setup(ax, min_wav, max_wav, max_freq, title=''):
    """Set common axis formatting for a WK panel."""
    ax.set_xlim(min_wav, max_wav)
    ax.set_ylim(0.0, max_freq)
    ax.set_xlabel('Zonal Wave Number', fontsize=8)
    ax.set_ylabel('Frequency (cpd)', fontsize=8)
    ax.set_title(title, fontsize=9)
    ax.tick_params(labelsize=7)
    freq_ticks = np.linspace(0.0, max_freq, 9)
    ax.set_yticks(freq_ticks)
    ax.set_yticklabels([f'{t:.2f}' for t in freq_ticks], fontsize=7)
    ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))


def plot_wk_panel(results, case_labels, var_name='PRECT', lat_bound=15,
                  fig_1_levels=None, fig_2_levels=None,
                  fig_3a_levels=None, fig_3b_levels=None,
                  anom_plot=False, anom_levels=None,
                  Apzwn=None, Afreq=None, add_disp_lines=True,
                  add_mjo_box=True,
                  min_wav=-15, max_wav=15, max_freq=0.8,
                  cmap='RdBu_r', cmap_ratio='RdBu_r',
                  save_dir='.', fig_prefix='kf_pan'):
    """Create 2×2 (or n-panel) WK spectral plots.

    For each of 5 plot types (Asym log10, Sym log10, BG log10,
    Asym/BG ratio, Sym/BG ratio) a figure is produced with
    one panel per case.

    Parameters
    ----------
    results     : list of dicts, each from compute_wk_spectrum()
    case_labels : list of str, one per case
    var_name    : variable label for titles
    lat_bound   : equatorial belt (for title)
    fig_*_levels: explicit contour levels (list/array); if None, auto
    anom_plot   : if True, plot ratio to results[0] rather than full field
    Apzwn, Afreq: dispersion curve arrays from gen_dispersion_curves()
    save_dir    : directory for saved figures
    fig_prefix  : filename prefix

    Returns
    -------
    figs : list of 5 matplotlib Figure objects
    """
    ncases = len(results)
    ny = int(np.ceil(ncases / 2))
    nx = min(ncases, 2)

    # Default contour levels (log10 power for PRECT)
    if fig_1_levels is None:
        fig_1_levels = np.linspace(-1.0, 0.4, 15)
    if fig_2_levels is None:
        fig_2_levels = np.linspace(-1.0, 0.4, 15)
    if fig_3a_levels is None:
        fig_3a_levels = np.array([0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.15,
                                   1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5, 1.6])
    if fig_3b_levels is None:
        fig_3b_levels = fig_3a_levels.copy()
    if anom_levels is None:
        anom_levels = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8,
                                 0.7, 0.9, 1.1, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0])

    # Prepare wave/freq grids (from first result)
    wave = results[0]['wave']
    freq = results[0]['freq']
    # Subset for plotting
    iw0 = np.searchsorted(wave, min_wav)
    iw1 = np.searchsorted(wave, max_wav) + 1
    if0 = np.searchsorted(freq, 0.0)          # DC index
    if1 = np.searchsorted(freq, max_freq) + 1
    wave_p = wave[iw0:iw1]
    freq_p = freq[if0:if1]

    plot_types = [
        ('Antisymmetric (log10)',    'asym',  fig_1_levels,  cmap),
        ('Symmetric (log10)',        'sym',   fig_2_levels,  cmap),
        ('Background (log10)',       'bg',    fig_2_levels,  cmap),
        ('Antisymmetric / Background', 'asym_r', fig_3a_levels, cmap_ratio),
        ('Symmetric / Background',   'sym_r', fig_3b_levels, cmap_ratio),
    ]

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    figs = []

    for pt_label, pt_key, levels, cm in plot_types:
        fig, axes = plt.subplots(ny, nx, figsize=(6 * nx, 5 * ny))
        axes = np.array(axes).ravel()
        fig.suptitle(
            f"CESM Equatorial Wave Spectrum : {var_name} (±{lat_bound}°) — {pt_label}",
            fontsize=11
        )

        for ic, (res, clabel) in enumerate(zip(results, case_labels)):
            ax = axes[ic]
            w  = res['wave']
            f  = res['freq']

            # Select data for this plot type
            anti_nl = res['psumanti_nl']
            sym_nl  = res['psumsym_nl']
            bg_nl   = res['psumb_nl']

            if anom_plot and ic > 0:
                ref = results[0]
                if pt_key == 'asym':
                    data = anti_nl / ref['psumanti_nl']
                elif pt_key == 'sym':
                    data = sym_nl / ref['psumsym_nl']
                elif pt_key == 'bg':
                    data = bg_nl / ref['psumb_nl']
                elif pt_key == 'asym_r':
                    data = (anti_nl / bg_nl) / (ref['psumanti_nl'] / ref['psumb_nl'])
                else:  # sym_r
                    data = (sym_nl / bg_nl) / (ref['psumsym_nl'] / ref['psumb_nl'])
                lev = anom_levels
            else:
                if pt_key == 'asym':
                    data = np.log10(np.where(anti_nl > 0, anti_nl, np.nan))
                elif pt_key == 'sym':
                    data = np.log10(np.where(sym_nl > 0, sym_nl, np.nan))
                elif pt_key == 'bg':
                    data = np.log10(np.where(bg_nl > 0, bg_nl, np.nan))
                elif pt_key == 'asym_r':
                    data = anti_nl / bg_nl
                else:
                    data = sym_nl / bg_nl
                lev = levels

            # Subset to plot range (wave × freq, positive freq only)
            iw0_ = np.searchsorted(w, min_wav)
            iw1_ = np.searchsorted(w, max_wav) + 1
            if0_ = np.searchsorted(f, 0.0)
            if1_ = np.searchsorted(f, max_freq) + 1
            plot_data = data[iw0_:iw1_, if0_:if1_].T   # (freq, wave) for contourf

            W, F = np.meshgrid(w[iw0_:iw1_], f[if0_:if1_])
            cf = ax.contourf(W, F, plot_data, levels=lev, cmap=cm, extend='both')
            plt.colorbar(cf, ax=ax, shrink=0.8, pad=0.02)

            add_hor_vert_lines(ax, min_wav=min_wav, max_wav=max_wav)

            if add_disp_lines and Apzwn is not None and Afreq is not None:
                disp_type = 'sym' if pt_key in ('sym', 'bg', 'sym_r') else 'asym'
                add_dispersion_curves(ax, Apzwn, Afreq, plot_type=disp_type,
                                      min_wav=min_wav, max_wav=max_wav,
                                      max_freq=max_freq)

            if add_mjo_box and pt_key in ('asym_r', 'sym_r', 'sym', 'asym'):
                # MJO box: wavenumber 0-5, freq 0.015-0.05
                from matplotlib.patches import Rectangle
                rect = Rectangle((0, 0.015), 5, 0.05 - 0.015,
                                  fill=False, edgecolor='orange', lw=1.5)
                ax.add_patch(rect)

            _wk_axes_setup(ax, min_wav, max_wav, max_freq,
                           title=f'{clabel}')

        # Hide unused axes
        for iex in range(ncases, len(axes)):
            axes[iex].set_visible(False)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fname = save_path / f'{fig_prefix}_{var_name}_{pt_key}.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        print(f"Saved: {fname}")
        figs.append(fig)

    return figs
