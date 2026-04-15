"""
skewt_utils.py
==============
Skew-T log-P diagram utilities for RAOB sounding data.

Data notes (from raob_soundings_*.cdf):
  tpMan / tpSigT  : temperature     [K]
  tdMan / tdSigT  : dewpoint depression T−Td  [K]  →  Td = T − DPD
  prMan           : pressure at mandatory levels  [hPa]  (index 0 = surface)
  prSigT          : pressure at significant-T levels [Pa] → divide by 100
  wsMan / wsSigW  : wind speed  [m/s]
  wdMan / wdSigW  : wind direction [degrees true]
  numSigT         : number of valid significant-T levels per sounding
  numSigW         : number of valid significant-wind levels per sounding

Requires: MetPy (available in npl conda env on NCAR systems)
"""

import io
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.animation as manimation
import os
import glob

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from metpy.plots import SkewT, Hodograph
import metpy.calc as mpcalc
from metpy.units import units as munits


# ── Color cycle for multiple soundings ────────────────────────────────────────
_SOUNDING_COLORS = [
    '#1f77b4', '#d62728', '#2ca02c', '#ff7f0e', '#9467bd',
    '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
]


# ─────────────────────────────────────────────────────────────────────────────
# Time helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_sounding_times(ds):
    """
    Decode synTime from the RAOB dataset to a pandas DatetimeIndex.

    Handles fill values (very large integers) by masking them out before
    conversion; masked entries become NaT.

    Parameters
    ----------
    ds : xr.Dataset  opened with decode_times=False

    Returns
    -------
    pandas.DatetimeIndex
    """
    times_raw = ds['synTime'].values.astype(float)
    # Fill values are typically INT_MAX or similar huge numbers.
    # Cap at a plausible upper bound (~year 2100 = 4.1e9 seconds since epoch).
    max_valid = 4_200_000_000.0
    times_raw = np.where((times_raw > 0) & (times_raw < max_valid),
                         times_raw, np.nan)
    return pd.to_datetime(times_raw, unit='s', origin='unix', errors='coerce')


def find_nearest_soundings(times, target_dts, max_hours=3):
    """
    For each target datetime, find the index of the nearest available sounding.

    Parameters
    ----------
    times      : pandas.DatetimeIndex  all available sounding times
    target_dts : list of str or datetime-like  requested times
                 Strings parsed by pd.Timestamp (e.g. '2005-07-04 00:00')
    max_hours  : float  maximum allowed offset in hours; unmatched → None

    Returns
    -------
    indices : list of int or None  one per target_dt
    matched : list of pd.Timestamp  actual sounding times
    """
    # Build a boolean mask of valid (non-NaT) entries
    valid_mask = ~pd.isnull(times)
    valid_idx  = np.where(valid_mask)[0]
    valid_times = times[valid_mask]

    indices, matched = [], []
    for t in target_dts:
        tgt = pd.Timestamp(t)
        deltas = np.abs((valid_times - tgt).total_seconds()) / 3600.0
        best   = int(np.argmin(deltas))
        if deltas[best] <= max_hours:
            indices.append(int(valid_idx[best]))
            matched.append(valid_times[best])
        else:
            print(f'  WARNING: no sounding within {max_hours}h of {t} '
                  f'(nearest is {deltas[best]:.1f}h away)')
            indices.append(None)
            matched.append(None)
    return indices, matched


# ─────────────────────────────────────────────────────────────────────────────
# Sounding extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_sounding(ds, idx, lev_type='significant'):
    """
    Extract one sounding from the RAOB dataset.

    For the T / Td profiles, significant levels are used when
    lev_type='significant' (more vertical detail); mandatory levels are
    used for wind barbs in all cases.

    Parameters
    ----------
    ds       : xr.Dataset  RAOB file opened with decode_times=False
    idx      : int         sounding index (row in the dataset)
    lev_type : str         'significant' | 'mandatory'
                           Controls which levels are used for T/Td

    Returns
    -------
    dict with keys:
        p_hpa   : ndarray  pressure [hPa], monotonically decreasing, NaN-free
        T_degC  : ndarray  temperature [°C]
        Td_degC : ndarray  dewpoint temperature [°C]
        p_wind  : ndarray  pressure at wind levels [hPa]
        u_kts   : ndarray  u-component [knots]
        v_kts   : ndarray  v-component [knots]
        sfc_p   : float    surface pressure [hPa]
        sfc_T   : float    surface temperature [°C]
        sfc_Td  : float    surface dewpoint [°C]
    """
    def _K_to_C(x):
        return x - 273.15

    # ── Significant levels for T/Td profile ──────────────────────────────────
    if lev_type == 'significant' and 'prSigT' in ds:
        n_sig = int(ds['numSigT'].values[idx])
        p_raw = ds['prSigT'].values[idx, :n_sig].astype(float)
        T_raw = ds['tpSigT'].values[idx, :n_sig].astype(float)
        d_raw = ds['tdSigT'].values[idx, :n_sig].astype(float)

        # prSigT is typically in Pa — convert if values look like Pa
        if np.nanmax(p_raw[p_raw > 0]) > 2000:
            p_raw = p_raw / 100.0   # Pa → hPa

        T_C  = _K_to_C(T_raw)
        Td_C = _K_to_C(T_raw - d_raw)   # DPD = T − Td  →  Td = T − DPD

        # Valid mask: p and T physically plausible; DPD non-negative
        ok = (np.isfinite(p_raw) & (p_raw > 0) &
              np.isfinite(T_raw)  & (T_raw > 170.) & (T_raw < 340.) &
              np.isfinite(d_raw)  & (d_raw >= 0.)  & (d_raw < 100.))
        p_prof  = p_raw[ok]
        T_prof  = T_C[ok]
        Td_C_ok = Td_C[ok]
        Td_prof = np.where(np.isfinite(Td_C_ok), Td_C_ok, np.nan)

    else:
        # Fall back to mandatory levels (skip surface index 0 here;
        # surface is handled separately below)
        p_raw = ds['prMan'].values[idx, 1:].astype(float)   # hPa
        T_raw = ds['tpMan'].values[idx, 1:].astype(float)
        d_raw = ds['tdMan'].values[idx, 1:].astype(float)

        T_C  = _K_to_C(T_raw)
        Td_C = _K_to_C(T_raw - d_raw)

        ok = (np.isfinite(p_raw) & (p_raw > 0) &
              np.isfinite(T_raw)  & (T_raw > 170.) & (T_raw < 340.) &
              np.isfinite(d_raw)  & (d_raw >= 0.)  & (d_raw < 100.))
        p_prof  = p_raw[ok]
        T_prof  = T_C[ok]
        Td_C_ok = Td_C[ok]
        Td_prof = np.where(np.isfinite(Td_C_ok), Td_C_ok, np.nan)

    # ── Surface observation (mandatory index 0) ───────────────────────────────
    sfc_p_hpa = float(ds['prMan'].values[idx, 0])
    sfc_T_K   = float(ds['tpMan'].values[idx, 0])
    sfc_d_K   = float(ds['tdMan'].values[idx, 0])
    sfc_T_C  = (_K_to_C(sfc_T_K)
                if np.isfinite(sfc_T_K) and 170. < sfc_T_K < 340.
                else np.nan)
    sfc_Td_C = (_K_to_C(sfc_T_K - sfc_d_K)
                if (np.isfinite(sfc_T_K) and np.isfinite(sfc_d_K)
                    and 170. < sfc_T_K < 340.
                    and 0. <= sfc_d_K < 100.)
                else np.nan)

    # Prepend surface to profile if not already the lowest level
    if np.isfinite(sfc_p_hpa) and (len(p_prof) == 0 or sfc_p_hpa > p_prof[0]):
        p_prof  = np.concatenate([[sfc_p_hpa],  p_prof])
        T_prof  = np.concatenate([[sfc_T_C],    T_prof])
        Td_prof = np.concatenate([[sfc_Td_C],   Td_prof])

    # Sort by decreasing pressure (surface first)
    order  = np.argsort(p_prof)[::-1]
    p_prof  = p_prof[order]
    T_prof  = T_prof[order]
    Td_prof = Td_prof[order]

    # ── Wind: use mandatory levels ────────────────────────────────────────────
    p_w  = ds['prMan'].values[idx, :].astype(float)   # hPa (includes sfc)
    ws_w = ds['wsMan'].values[idx, :].astype(float)   # m/s
    wd_w = ds['wdMan'].values[idx, :].astype(float)   # degrees

    ok_w = np.isfinite(p_w) & np.isfinite(ws_w) & np.isfinite(wd_w) & (p_w > 0)
    p_wind = p_w[ok_w]
    ws_ms  = ws_w[ok_w]
    wd_deg = wd_w[ok_w]

    u_ms, v_ms = mpcalc.wind_components(
        ws_ms * munits('m/s'), wd_deg * munits.degrees)
    u_kts = u_ms.to('knots').magnitude
    v_kts = v_ms.to('knots').magnitude

    return {
        'p_hpa':   p_prof,
        'T_degC':  T_prof,
        'Td_degC': Td_prof,
        'p_wind':  p_wind,
        'u_kts':   u_kts,
        'v_kts':   v_kts,
        'sfc_p':   sfc_p_hpa,
        'sfc_T':   sfc_T_C,
        'sfc_Td':  sfc_Td_C,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Skew-T plot
# ─────────────────────────────────────────────────────────────────────────────

def plot_skewt(soundings, labels, station_info,
               show_parcel=True,
               show_hodograph=True,
               lev_type='significant',
               rotation=45,
               p_top=100.,
               title=None,
               dir_fig=None,
               fname=None,
               figsize=(11, 12)):
    """
    Plot one or more soundings on a Skew-T log-P diagram.

    Parameters
    ----------
    soundings    : list of dict  output from extract_sounding()
    labels       : list of str   datetime label for each sounding
    station_info : dict  keys: 'station', 'lat', 'lon', 'elev'
    show_parcel  : bool  lift surface parcel; shade CAPE/CIN; annotate indices
    show_hodograph : bool  add hodograph inset (only plotted for first sounding)
    lev_type     : str   'significant' | 'mandatory' (for title annotation)
    rotation     : float Skew-T rotation angle (degrees); 45 is standard
    p_top        : float upper pressure limit [hPa]
    title        : str   figure title; None = auto-generated
    dir_fig      : str   directory to save PNG; None = don't save
    fname        : str   filename (without path); None = auto-generated
    figsize      : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    n_snd  = len(soundings)
    colors = (_SOUNDING_COLORS * ((n_snd // len(_SOUNDING_COLORS)) + 1))[:n_snd]

    # ── Figure / GridSpec ────────────────────────────────────────────────────
    fig = plt.figure(figsize=figsize)

    if show_hodograph:
        # Skew-T takes most of the figure; hodograph sits top-right
        gs  = gridspec.GridSpec(2, 2, figure=fig,
                                height_ratios=[1, 3],
                                width_ratios=[3, 1],
                                hspace=0.04, wspace=0.04)
        ax_skew = fig.add_subplot(gs[:, 0])
        ax_hodo = fig.add_subplot(gs[0, 1])
        ax_legend = fig.add_subplot(gs[1, 1])
        ax_legend.axis('off')
    else:
        ax_skew  = fig.add_subplot(111)
        ax_hodo  = None
        ax_legend = None

    skew = SkewT(fig, rotation=rotation, subplot=ax_skew)
    skew.ax.set_ylim(1050, p_top)
    skew.ax.set_xlim(-100, 40)

    # ── Plot each sounding ───────────────────────────────────────────────────
    legend_handles = []
    cape_cin_labels = []

    for i, (snd, lbl, col) in enumerate(zip(soundings, labels, colors)):
        p   = snd['p_hpa']   * munits.hPa
        T   = snd['T_degC']  * munits.degC
        Td  = snd['Td_degC'] * munits.degC

        # Remove NaN rows in T before passing to MetPy
        ok  = np.isfinite(snd['T_degC'])
        p_  = p[ok];  T_ = T[ok]
        ok2 = np.isfinite(snd['Td_degC'])

        alpha = 1.0 if n_snd == 1 else max(0.6, 1.0 - 0.1 * i)
        lw    = 2.0 if n_snd == 1 else 1.8

        h_T,  = skew.plot(p_, T_,  color=col, linewidth=lw, alpha=alpha,
                          label=lbl)
        if ok2.any():
            p_d = p[ok2]; Td_ = Td[ok2]
            skew.plot(p_d, Td_, color=col, linewidth=lw, alpha=alpha,
                      linestyle='--')
        legend_handles.append(h_T)

        # Wind barbs
        if len(snd['p_wind']) > 0:
            skew.plot_barbs(
                snd['p_wind'] * munits.hPa,
                snd['u_kts']  * munits.knots,
                snd['v_kts']  * munits.knots,
                color=col, alpha=alpha,
                x_clip_radius=0.12,
                barbcolor=col,
            )

        # Parcel profile — only for the first (or single) sounding
        if show_parcel and i == 0:
            try:
                # Use surface parcel
                parcel_p = p_
                parcel_T = mpcalc.parcel_profile(parcel_p, T_[0],
                                                  Td[ok][0]).to('degC')
                skew.plot(parcel_p, parcel_T, color='black',
                          linewidth=1.5, linestyle='--',
                          label='Parcel (surface)')
                skew.shade_cape(parcel_p, T_, parcel_T,
                                alpha=0.25, color='red')
                skew.shade_cin(parcel_p, T_, parcel_T,
                               alpha=0.25, color='blue')

                cape, cin = mpcalc.cape_cin(parcel_p, T_, Td[ok2[:len(ok)][ok]],
                                            parcel_T)
                lcl_p, lcl_T = mpcalc.lcl(parcel_p[0], T_[0],
                                            Td[ok2[:len(ok)][ok]][0])
                skew.ax.axhline(lcl_p.magnitude, color='grey',
                                linestyle=':', linewidth=1, alpha=0.7)
                cape_cin_labels.append(
                    f'CAPE={cape.magnitude:.0f} J/kg  '
                    f'CIN={cin.magnitude:.0f} J/kg  '
                    f'LCL={lcl_p.magnitude:.0f} hPa')
            except Exception as e:
                cape_cin_labels.append(f'Parcel calc failed: {e}')

        # Hodograph — first sounding only
        if show_hodograph and ax_hodo is not None and i == 0:
            hodo = Hodograph(ax_hodo, component_range=40.)
            hodo.add_grid(increment=10)
            p_w  = snd['p_wind']
            u_ms, v_ms = mpcalc.wind_components(
                (snd['u_kts'] * munits.knots).to('m/s'),
                np.zeros(len(snd['u_kts'])) * munits.degrees)
            # Re-extract u,v in m/s properly
            ws = np.sqrt(snd['u_kts']**2 + snd['v_kts']**2) * munits.knots
            # Draw coloured by pressure
            hodo.plot_colormapped(
                snd['u_kts'] * munits.knots,
                snd['v_kts'] * munits.knots,
                p_w * munits.hPa)
            ax_hodo.set_title('Hodograph', fontsize=8)

    # ── Reference lines ──────────────────────────────────────────────────────
    skew.plot_dry_adiabats(alpha=0.25, colors='brown', linewidths=0.8)
    skew.plot_moist_adiabats(alpha=0.25, colors='green', linewidths=0.8)
    skew.plot_mixing_lines(alpha=0.25, colors='blue', linewidths=0.8)
    skew.ax.axvline(0, color='grey', linewidth=0.8, linestyle='-', alpha=0.5)

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_handles.append(
        plt.Line2D([0], [0], color='k', linewidth=1.5, linestyle='--',
                   label='Parcel'))
    all_labels = labels + ['Parcel']
    if ax_legend is not None:
        ax_legend.legend(handles=legend_handles, labels=all_labels,
                         loc='upper left', fontsize=8,
                         title='─ T   -- Td', title_fontsize=7)
    else:
        skew.ax.legend(handles=legend_handles, labels=all_labels,
                       loc='upper left', fontsize=8,
                       title='─ T   -- Td')

    # ── Title & annotations ──────────────────────────────────────────────────
    sname = station_info.get('station', '')
    lat   = station_info.get('lat', np.nan)
    lon   = station_info.get('lon', np.nan)
    elev  = station_info.get('elev', np.nan)

    if title is None:
        title = (f'{sname}  ({lat:.2f}°N, {lon:.2f}°E,  {elev:.0f} m)\n'
                 f'{lev_type} levels')
    skew.ax.set_title(title, fontsize=11, loc='left')
    skew.ax.set_xlabel('Temperature (°C)', fontsize=10)
    skew.ax.set_ylabel('Pressure (hPa)',   fontsize=10)

    if cape_cin_labels:
        skew.ax.text(0.01, 0.01, '\n'.join(cape_cin_labels),
                     transform=skew.ax.transAxes,
                     fontsize=8, va='bottom', ha='left',
                     bbox=dict(boxstyle='round,pad=0.3',
                               facecolor='white', alpha=0.8))

    fig.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    if dir_fig is not None:
        os.makedirs(dir_fig, exist_ok=True)
        if fname is None:
            safe_times = '_'.join(
                l.replace(' ', 'T').replace(':', '').replace('-', '')
                for l in labels[:3])
            if len(labels) > 3:
                safe_times += f'_plus{len(labels)-3}more'
            fname = f'skewt_{sname}_{safe_times}.png'
        fpath = os.path.join(dir_fig, fname)
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f'Saved: {fpath}')

    return fig


# ─────────────────────────────────────────────────────────────────────────────
# Skew-T animation
# ─────────────────────────────────────────────────────────────────────────────

def animate_skewt(ds, start_idx, n_frames=10,
                  station_info=None,
                  lev_type='significant',
                  show_parcel=True,
                  show_hodograph=False,
                  p_top=100.,
                  interval=1500,
                  figsize=(9, 10),
                  dir_fig=None,
                  fname='skewt_animation.gif'):
    """
    Animate successive soundings as a looping Skew-T sequence.

    Each frame shows a single sounding.  The animation plays in the Jupyter
    notebook window via ``HTML(anim.to_jshtml())`` — no file is required,
    though an optional GIF can also be saved.

    Parameters
    ----------
    ds          : xr.Dataset  RAOB file opened with decode_times=False
    start_idx   : int         index of the first sounding to animate
    n_frames    : int         number of successive soundings to include
    station_info: dict        keys: station, lat, lon, elev
    lev_type    : str         'significant' | 'mandatory'
    show_parcel : bool
    show_hodograph : bool     (disabled by default to speed up rendering)
    p_top       : float       upper pressure limit [hPa]
    interval    : int         ms between frames
    figsize     : tuple
    dir_fig     : str         directory to save GIF; None = don't save
    fname       : str         GIF filename

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        Call ``HTML(anim.to_jshtml())`` in a notebook cell to display.
    """
    if station_info is None:
        station_info = {}

    times = get_sounding_times(ds)
    n_total = len(times)

    # Clamp to available soundings
    end_idx = min(start_idx + n_frames, n_total)
    indices = list(range(start_idx, end_idx))

    print(f'Pre-rendering {len(indices)} frames …')

    # ── Pre-render each frame as an RGBA image array ─────────────────────────
    frame_images = []
    frame_titles = []

    for idx in indices:
        try:
            snd = extract_sounding(ds, idx, lev_type=lev_type)
        except Exception as e:
            print(f'  skip idx={idx}: {e}')
            continue

        t = times[idx]
        lbl = str(t)[:16] if not pd.isnull(t) else f'idx={idx}'
        frame_titles.append(lbl)

        fig = plot_skewt(
            soundings      = [snd],
            labels         = [lbl],
            station_info   = station_info,
            show_parcel    = show_parcel,
            show_hodograph = show_hodograph,
            lev_type       = lev_type,
            p_top          = p_top,
            figsize        = figsize,
        )

        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=90, bbox_inches='tight')
        buf.seek(0)
        img = plt.imread(buf)
        frame_images.append(img)
        plt.close(fig)

    if not frame_images:
        raise RuntimeError('No valid frames rendered.')

    print(f'  Done. Building animation ({len(frame_images)} frames) …')

    # ── Build animation using imshow ──────────────────────────────────────────
    fig_anim, ax_anim = plt.subplots(figsize=figsize)
    ax_anim.axis('off')
    fig_anim.subplots_adjust(0, 0, 1, 1)

    im = ax_anim.imshow(frame_images[0], animated=True)

    def _update(i):
        im.set_data(frame_images[i])
        return [im]

    anim = manimation.FuncAnimation(
        fig_anim, _update,
        frames=len(frame_images),
        interval=interval,
        blit=True,
        repeat=True,
    )

    # ── Optionally save GIF ───────────────────────────────────────────────────
    if dir_fig is not None:
        os.makedirs(dir_fig, exist_ok=True)
        fpath = os.path.join(dir_fig, fname)
        anim.save(fpath, writer='pillow', fps=1000 // interval, dpi=90)
        print(f'Saved: {fpath}')

    plt.close(fig_anim)
    return anim


# ─────────────────────────────────────────────────────────────────────────────
# Station inventory and map
# ─────────────────────────────────────────────────────────────────────────────

def collect_station_info(rdir):
    """
    Scan all raob_soundings_*.cdf files in *rdir* and return a list of dicts
    with station metadata.

    Parameters
    ----------
    rdir : str  directory containing raob_soundings_*.cdf files

    Returns
    -------
    list of dict, each with keys:
        station, lat, lon, elev, n_soundings
    """
    files = sorted(glob.glob(os.path.join(rdir, 'raob_soundings_*.cdf')))
    stations = []
    for fpath in files:
        try:
            ds = xr.open_dataset(fpath, decode_times=False)
            name = os.path.basename(fpath).replace('raob_soundings_', '').replace('.cdf', '')
            lat  = float(ds['staLat'].values.flat[0])  if 'staLat'  in ds else np.nan
            lon  = float(ds['staLon'].values.flat[0])  if 'staLon'  in ds else np.nan
            elev = float(ds['staElev'].values.flat[0]) if 'staElev' in ds else np.nan
            # Count valid soundings (non-fill synTime entries)
            times = get_sounding_times(ds)
            n_valid = int((~pd.isnull(times)).sum())
            ds.close()
            stations.append({'station': name, 'lat': lat, 'lon': lon,
                             'elev': elev, 'n_soundings': n_valid})
        except Exception as e:
            print(f'  WARNING: could not read {fpath}: {e}')
    return stations


def plot_station_map(rdir=None,
                     station_catalog=None,
                     lat_range=(8., 38.),
                     lon_range=(-105., -60.),
                     highlight=None,
                     title='Available RAOB Stations',
                     dir_fig=None,
                     fname='raob_station_map.png',
                     figsize=(12, 8)):
    """
    Plot a map of radiosonde stations.

    Two modes:
    1. Pass *station_catalog* (dict from gulf_caribbean_stations.STATION_NAMES):
       plots all known stations from the catalog; local CDF files in *rdir* are
       used to mark which stations have data already downloaded (filled vs open).
    2. Pass only *rdir*: scans CDF files in that directory (legacy mode).

    Parameters
    ----------
    rdir            : str   directory with raob_soundings_*.cdf files (optional)
    station_catalog : dict  STATION_NAMES dict from gulf_caribbean_stations.py
    lat_range       : tuple (lat_min, lat_max)
    lon_range       : tuple (lon_min, lon_max)
    highlight       : list of str  station names (keys) to mark with orange star
    title           : str
    dir_fig         : str   save directory; None = don't save
    fname           : str   filename
    figsize         : tuple

    Returns
    -------
    fig : matplotlib Figure
    """
    if station_catalog is not None:
        # Build station list from catalog; mark locally available ones
        local_names = set()
        if rdir is not None:
            for fp in glob.glob(os.path.join(rdir, 'raob_soundings_*.cdf')):
                name = os.path.basename(fp).replace('raob_soundings_', '').replace('.cdf', '')
                local_names.add(name)
        stations = []
        for name, info in station_catalog.items():
            stations.append({
                'station':     name,
                'lat':         info['lat'],
                'lon':         info['lon'],
                'elev':        info.get('elev', 0),
                'n_soundings': info.get('active', 2000),   # use active year as proxy for size
                'local':       name in local_names,
                'country':     info.get('country', ''),
                'active':      info.get('active', 0),
            })
    elif rdir is not None:
        stations = collect_station_info(rdir)
        for s in stations:
            s['local'] = True
    else:
        print('Provide rdir or station_catalog.')
        return None

    if not stations:
        print('No stations found.')
        return None

    if highlight is None:
        highlight = []

    # ── Figure / map ─────────────────────────────────────────────────────────
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(1, 1, figsize=figsize,
                           subplot_kw={'projection': proj})
    ax.set_extent([lon_range[0], lon_range[1],
                   lat_range[0], lat_range[1]], crs=proj)

    ax.add_feature(cfeature.LAND,       facecolor='#f0f0e8', zorder=0)
    ax.add_feature(cfeature.OCEAN,      facecolor='#d0e8f8', zorder=0)
    ax.add_feature(cfeature.COASTLINE,  linewidth=0.8,       zorder=1)
    ax.add_feature(cfeature.BORDERS,    linewidth=0.6,       linestyle='--', zorder=1)
    ax.add_feature(cfeature.STATES,     linewidth=0.4,       linestyle=':',  zorder=1)

    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color='grey',
                      alpha=0.6, linestyle='--')
    gl.top_labels   = False
    gl.right_labels = False

    # ── Plot stations ────────────────────────────────────────────────────────
    dot_size = 60.0

    for s in stations:
        lon_s  = s['lon']
        lat_s  = s['lat']
        name   = s['station']
        active = s.get('active', 0)
        local  = s.get('local', False)

        # Skip if outside map extent
        if not (lon_range[0] <= lon_s <= lon_range[1] and
                lat_range[0] <= lat_s <= lat_range[1]):
            continue

        if name in highlight:
            ax.plot(lon_s, lat_s, marker='*', color='orange',
                    markersize=16, markeredgecolor='k',
                    markeredgewidth=0.8, transform=proj, zorder=5)
        else:
            # Active (≥2010): filled green; inactive: open grey
            is_active = active >= 2010
            color     = '#2ca02c' if is_active else '#aaaaaa'
            edge      = 'k'       if is_active else '#666666'
            fc        = color     if local     else 'none'   # open = not downloaded
            ax.scatter(lon_s, lat_s, s=dot_size,
                       facecolors=fc, edgecolors=edge, linewidths=1.0,
                       transform=proj, zorder=3)

        country = s.get('country', '')
        ax.text(lon_s + 0.3, lat_s + 0.2,
                name,
                fontsize=5.5, transform=proj, zorder=6,
                va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.1',
                          facecolor='white', alpha=0.55, linewidth=0))

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_handles = [
        plt.scatter([], [], s=dot_size, facecolors='#2ca02c', edgecolors='k',
                    linewidths=1.0, label='Active (≥2010), downloaded'),
        plt.scatter([], [], s=dot_size, facecolors='none', edgecolors='k',
                    linewidths=1.0, label='Active (≥2010), catalog only'),
        plt.scatter([], [], s=dot_size, facecolors='none', edgecolors='#666666',
                    linewidths=1.0, label='Inactive / historical'),
        plt.scatter([], [], s=dot_size*1.5, marker='*', facecolors='orange',
                    edgecolors='k', linewidths=0.8, label='Selected station'),
    ]
    ax.legend(handles=legend_handles, loc='lower right', fontsize=7,
              framealpha=0.85)

    ax.set_title(title, fontsize=12)

    fig.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────────
    if dir_fig is not None:
        os.makedirs(dir_fig, exist_ok=True)
        fpath = os.path.join(dir_fig, fname)
        fig.savefig(fpath, dpi=150, bbox_inches='tight')
        print(f'Saved: {fpath}')

    return fig
