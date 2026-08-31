"""
Regional wave-mode variance from wavenumber-frequency filtering.

Given daily-anomaly data (time, lat, lon), for each named mode
(MJO, Kelvin, ER1, MRG, MRG, ...):
  1. Split the record into overlapping time windows.
  2. Detrend + cosine-taper each window.
  3. 2D FFT over (time, lon) at each latitude.
  4. Multiply by a wavenumber-frequency mask that keeps only the
     (|k|, |ω|, propagation-direction) box for the mode.
  5. Inverse 2D FFT → reconstructed lat/lon/time contribution from that mode.
  6. If the mode is nominally symmetric (Kelvin, MJO) or antisymmetric (MRG),
     apply the equatorial (a)symmetric projection.
  7. Variance across time → 2D map per mode.

Ported from /glade/u/home/rneale/ncl/CESM3_CAM7/wave_var/wave_var_cesm3_tiedke.ncl.
Uses the loaders from kf_pan_utils so both diagnostics can share cases/observations.
"""

from pathlib import Path
import numpy as np
import scipy.signal

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature


# ---------------------------------------------------------------------------
# Mode definitions (extending the standard NCL set)
# ---------------------------------------------------------------------------
#
#   k_lo, k_hi   |k| range (integer zonal wavenumbers)
#   p_lo, p_hi   period range in days (positive; p_lo < p_hi)
#   direction    'east' | 'west' | 'both'
#   symtype      'sym' | 'asym' | 'all' — equatorial (a)symmetric projection
#                 applied to the reconstructed field before variance
#
# Values mirror the standard NCL wave_var setup.

MODES = {
    "AllWave": dict(k_lo=0, k_hi=15, p_lo=1.25, p_hi=128.0,
                     direction="both", symtype="all"),
    "MJO":     dict(k_lo=1, k_hi=5,  p_lo=20.0, p_hi=100.0,
                     direction="east", symtype="sym"),
    "Kelvin":  dict(k_lo=1, k_hi=14, p_lo=2.5,  p_hi=30.0,
                     direction="east", symtype="sym"),
    "ER1":     dict(k_lo=1, k_hi=10, p_lo=6.25, p_hi=48.0,
                     direction="west", symtype="sym"),
    "MRG":     dict(k_lo=0, k_hi=10, p_lo=2.5,  p_hi=10.0,
                     direction="west", symtype="asym"),
}


# ---------------------------------------------------------------------------
# Signal-processing helpers
# ---------------------------------------------------------------------------

def cosine_taper(n, p=0.1):
    """1D cosine-bell taper, fraction p at each end.  p in (0, 0.5]."""
    w = np.ones(n)
    n_taper = int(n * p)
    if n_taper > 0:
        t = np.arange(n_taper)
        bell = 0.5 * (1.0 - np.cos(np.pi * t / n_taper))
        w[:n_taper] = bell
        w[n - n_taper:] = bell[::-1]
    return w


def remove_annual_cycle_doy(x, spd=1, dpy=365):
    """Subtract the day-of-year climatology from daily data.

    x : (time, lat, lon) float array.  If time is not a whole number of years,
        the trailing partial year still gets the same climatology subtracted.
    spd, dpy : samples-per-day, days-per-year (noleap default).
    """
    n = x.shape[0]
    period = int(dpy * spd)
    doy = np.arange(n) % period
    clim = np.zeros((period,) + x.shape[1:], dtype=x.dtype)
    counts = np.zeros(period, dtype=np.int64)
    for d in range(period):
        idx = np.where(doy == d)[0]
        if idx.size:
            clim[d] = x[idx].mean(axis=0)
            counts[d] = idx.size
    return x - clim[doy]


# ---------------------------------------------------------------------------
# Wavenumber-frequency mask
# ---------------------------------------------------------------------------

def build_wnfreq_mask(N, nlon, spd, k_lo, k_hi, p_lo, p_hi, direction):
    """Build a boolean (N, nlon) mask over raw np.fft.fft2 indices that keeps
    only the FFT cells belonging to one wave mode.

    Sign convention (empirical, matches numpy's fft2 on cos(k·λ − ω·t)):
      - Eastward physical wave (k>0, ω>0) occupies cells where either
          k_signed > 0 and ω_signed < 0, or k_signed < 0 and ω_signed > 0.
      - Westward wave (physical k<0, ω>0) occupies cells where either
          k_signed > 0 and ω_signed > 0, or k_signed < 0 and ω_signed < 0.
    """
    om_min = 1.0 / p_hi          # cpd — lower bound (longest period)
    om_max = 1.0 / p_lo          # cpd — upper bound (shortest period)

    ti = np.arange(N)[:, None]
    wi = np.arange(nlon)[None, :]
    # Signed physical wavenumber and frequency (cpd) from FFT indices
    om_signed = np.where(ti <= N // 2, ti * spd / N, (ti - N) * spd / N)
    k_signed  = np.where(wi <= nlon // 2, wi, wi - nlon)

    om_mag = np.abs(om_signed)
    k_mag  = np.abs(k_signed)

    in_om = (om_mag >= om_min) & (om_mag <= om_max)
    in_k  = (k_mag  >= k_lo)   & (k_mag  <= k_hi)
    mask = in_om & in_k

    if direction == "east":
        east = ((k_signed > 0) & (om_signed < 0)) | \
               ((k_signed < 0) & (om_signed > 0))
        mask &= east
    elif direction == "west":
        west = ((k_signed > 0) & (om_signed > 0)) | \
               ((k_signed < 0) & (om_signed < 0))
        mask &= west
    # 'both' — no direction filter beyond magnitude constraints
    return mask


# ---------------------------------------------------------------------------
# Sym/asym projection about the equator
# ---------------------------------------------------------------------------

def _project_sym(x):
    """Symmetric part about equator: (x(lat) + x(-lat))/2.  lat assumed S→N."""
    return 0.5 * (x + x[:, ::-1, :])


def _project_asym(x):
    """Antisymmetric part about equator: (x(lat) - x(-lat))/2."""
    return 0.5 * (x - x[:, ::-1, :])


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------

def compute_wave_mode_variance(x_np, lat, lon, spd=1,
                               n_day_win=365, n_day_olap=10, n_day_taper=20,
                               modes=None, remove_ac=True,
                               ldetrend=True, vscale=1.0, verbose=True):
    """Compute wave-mode variance maps for a lat/lon field.

    Memory-conscious: the full (time, lat, lon) input is kept as float32.
    Per-window segments are upcast to float64 for detrend + FFT.  ``vscale``
    is applied inside the window loop (rather than the caller pre-scaling the
    whole array) so we don't need a second full-record copy.

    Parameters
    ----------
    x_np       : (time, lat, lon) float array (daily means).  Retained as
                 float32 internally to keep the memory footprint down.
    lat, lon   : 1D coordinate arrays.  lat assumed S→N.
    spd        : samples per day (1 for daily).
    n_day_win  : window length (days).  365 = one year.
    n_day_olap : overlap between successive windows (days).
    n_day_taper: taper length at each window end (days).
    modes      : dict of {name: mode_dict}.  Defaults to MODES.
    remove_ac  : if True, subtract day-of-year climatology first.
    ldetrend   : linear-detrend each window before tapering.
    vscale     : scalar unit conversion (e.g. 86400*1000 for CAM PRECT m/s→mm/day)

    Returns
    -------
    dict {mode_name: variance (lat, lon)}  in units² of (x_np * vscale).
    """
    modes = modes if modes is not None else MODES
    if x_np.dtype != np.float32:
        x = np.asarray(x_np, dtype=np.float32)
    else:
        x = x_np                                       # avoid a copy
    ntim, nlat, nlon = x.shape

    nan_mask_all_time = np.isnan(x).all(axis=0)        # (nlat, nlon)
    if np.isnan(x).any():
        n_nan = int(np.isnan(x).sum())
        if verbose:
            print(f"  - filling {n_nan} NaN cells with column time-means")
        col_mean = np.nanmean(x, axis=0)
        col_mean = np.where(np.isnan(col_mean), 0.0, col_mean).astype(np.float32)
        # in-place fill to avoid another full-record temporary
        idx = np.isnan(x)
        # broadcast column mean into x wherever NaN
        col_mean_b = np.broadcast_to(col_mean[None, :, :], x.shape)
        x = np.where(idx, col_mean_b, x)               # one temp, then dropped

    if remove_ac and ntim >= 365 * spd:
        if verbose:
            print("  - removing day-of-year climatology")
        # daily-mean climatology + in-place subtract to keep peak memory low
        period = int(365 * spd)
        doy = np.arange(ntim, dtype=np.int64) % period
        clim = np.zeros((period, nlat, nlon), dtype=np.float32)
        counts = np.zeros(period, dtype=np.int64)
        for d in range(period):
            sel = doy == d
            if sel.any():
                clim[d] = x[sel].mean(axis=0)
                counts[d] = int(sel.sum())
        # in-place: x -= clim[doy]
        for i in range(ntim):
            x[i] -= clim[doy[i]]
        del clim

    N = int(n_day_win * spd)
    step = int((n_day_win - n_day_olap) * spd)
    step = max(step, 1)
    n_win = 1 + max(0, (ntim - N) // step) if ntim >= N else 0
    if n_win == 0:
        raise ValueError(
            f"Not enough data: ntim={ntim}, need at least "
            f"{n_day_win*spd} samples for one window."
        )
    if verbose:
        print(f"  - windowing: N={N}, step={step}, n_win={n_win}")

    masks = {name: build_wnfreq_mask(N, nlon, spd,
                                     cfg["k_lo"], cfg["k_hi"],
                                     cfg["p_lo"], cfg["p_hi"],
                                     cfg["direction"])
             for name, cfg in modes.items()}
    if verbose:
        for name, m in masks.items():
            print(f"      mask[{name}]: {int(m.sum())} FFT cells "
                  f"({100.*m.mean():.2f}% of grid)")

    p_taper = n_day_taper * spd / N
    taper = cosine_taper(N, p=p_taper).astype(np.float64)

    var_sum = {name: np.zeros((nlat, nlon), dtype=np.float64) for name in modes}

    for w in range(n_win):
        s0 = w * step
        s1 = s0 + N
        if s1 > ntim:
            break
        if verbose:
            print(f"  - window {w+1}/{n_win}  samples [{s0}:{s1}]")

        # upcast just this segment to float64; apply unit scale here
        seg = np.asarray(x[s0:s1], dtype=np.float64) * vscale
        if ldetrend:
            seg = scipy.signal.detrend(seg, axis=0)
        seg *= taper[:, None, None]

        C = np.fft.fft2(seg, axes=(0, 2))                # complex128
        del seg                                          # free before mode loop

        for name, cfg in modes.items():
            m = masks[name][:, None, :]
            recon = np.fft.ifft2(C * m, axes=(0, 2)).real
            symtype = cfg.get("symtype", "all")
            if symtype == "sym":
                recon = _project_sym(recon)
            elif symtype == "asym":
                recon = _project_asym(recon)
            var_sum[name] += recon.var(axis=0)
            del recon

        del C

    out = {name: v / n_win for name, v in var_sum.items()}
    for name in out:
        out[name][nan_mask_all_time] = np.nan
    return out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_DEFAULT_LEVELS = {
    # broad-band precipitation-like variance (mm/day)² tuned to typical values
    "AllWave": np.array([1, 2, 4, 8, 12, 16, 20, 30, 40, 60, 80, 100, 150]),
    "MJO":     np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 6, 8, 10]),
    "Kelvin":  np.array([0.2, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 15, 20, 30]),
    "ER1":     np.array([0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 6, 8, 10]),
    "MRG":     np.array([0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 1.5, 2, 3, 4, 6, 8]),
}


def plot_wave_variance(vars_by_case, case_labels, mode_names=None,
                       lat=None, lon=None,
                       var_name="PRECT", var_units="mm² day⁻²",
                       lat_range=(-45, 45), lon_center=180,
                       levels_per_mode=None, cmap="gnuplot_r",
                       save_dir=".", fig_prefix="wave_var"):
    """Panel-plot per-mode variance maps.

    Layout: rows = modes, cols = cases.
    Plot domain: full 360° longitude centred at lon_center, latitudes lat_range.

    Parameters
    ----------
    vars_by_case : list of dicts {mode_name: (nlat, nlon) variance array},
                   one dict per case (in the order of case_labels).
    case_labels  : list of str
    mode_names   : subset/order of modes to plot; defaults to MODES.keys()
    lat, lon     : 1D coords (assumed common across cases; if they differ each
                   var dict may provide its own lat/lon via keys 'lat','lon').
    levels_per_mode : optional {mode: level_array} to override defaults.
    """
    modes = list(mode_names) if mode_names else list(MODES.keys())
    ncases = len(case_labels)
    nmodes = len(modes)

    levels_per_mode = levels_per_mode or {}
    proj = ccrs.PlateCarree(central_longitude=lon_center)
    data_crs = ccrs.PlateCarree()

    fig, axes = plt.subplots(nmodes, ncases,
                             figsize=(4.5 * ncases, 2.6 * nmodes),
                             subplot_kw={"projection": proj},
                             squeeze=False)

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    cmap_obj = plt.get_cmap(cmap)

    for im, mode in enumerate(modes):
        lev = levels_per_mode.get(mode, _DEFAULT_LEVELS.get(mode))
        cf_last = None
        for ic, (var_dict, clabel) in enumerate(zip(vars_by_case, case_labels)):
            ax = axes[im, ic]
            v = var_dict[mode]
            _lat = var_dict.get("lat", lat)
            _lon = var_dict.get("lon", lon)
            if lev is None:
                vmin = float(np.nanpercentile(v, 5))
                vmax = float(np.nanpercentile(v, 99))
                lev = np.linspace(vmin, vmax, 13)
            # BoundaryNorm gives each interval an equal slice of the colormap
            # (independent of the numeric spacing between contour levels).
            # Combined with spacing='uniform' on the colorbar below, this makes
            # the colorbar visually evenly stepped even for geometric level
            # sequences like [0.1, 0.2, 0.4, 0.8, ...].
            norm = mcolors.BoundaryNorm(lev, ncolors=cmap_obj.N, extend="both")
            cf = ax.contourf(_lon, _lat, v,
                             levels=lev, cmap=cmap, norm=norm, extend="both",
                             transform=data_crs)
            cf_last = cf
            ax.set_extent([lon_center - 180, lon_center + 180,
                           lat_range[0], lat_range[1]], crs=data_crs)
            ax.coastlines(linewidth=0.6, color="black")
            ax.add_feature(cfeature.LAND, facecolor="none", edgecolor="none")
            if im == 0:
                ax.set_title(clabel, fontsize=11, fontweight="bold")
            if ic == 0:
                ax.text(-0.08, 0.5, mode, transform=ax.transAxes,
                        rotation=90, va="center", ha="right",
                        fontsize=11, fontweight="bold")

        # One colorbar per row on the right — uniform per-interval spacing.
        if cf_last is not None:
            cbar_ax = fig.add_axes([0.93,
                                    0.05 + (nmodes - 1 - im) * (0.88 / nmodes) + 0.02,
                                    0.012,
                                    0.88 / nmodes - 0.04])
            cb = fig.colorbar(cf_last, cax=cbar_ax,
                              spacing="uniform", ticks=lev)
            cb.ax.tick_params(labelsize=7)
            cb.set_label(var_units, fontsize=8)

    fig.suptitle(f"Wave-mode variance of {var_name}  (±{abs(lat_range[0])}°)",
                 fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.06, right=0.91, top=0.94, bottom=0.05,
                        wspace=0.10, hspace=0.15)

    fname = save_path / f"{fig_prefix}_{var_name}.png"
    fig.savefig(fname, dpi=140, bbox_inches="tight")
    print(f"Saved: {fname}")
    return fig, axes


# ---------------------------------------------------------------------------
# Convenience: subset a loaded array to +/- lat_bound
# ---------------------------------------------------------------------------

def subset_lat(x, lat, lat_bound):
    """Return x, lat trimmed to |lat| <= lat_bound (assumes S→N ordering)."""
    m = np.abs(lat) <= lat_bound
    return x[:, m, :], lat[m]
