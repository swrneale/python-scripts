"""
hindcast_blocking_figs.py
Visualisation routines for hindcast blocking frequency diagnostics.

  block_plot_1d_hindcast  – longitude line plot per lead day (mean ± spread)
  block_plot_2d_hindcast  – polar stereographic panel map per lead day
  polarCentral_set_latlim – helper for circular polar-stereo boundary (shared)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.cm as mplcm
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm


DIR_FIG = '/glade/u/home/rneale/python/python-figs/hindcast_blocking/'

# ── 2-D colormap shared with blocking_figs.py ────────────────────────────────
_BCONTOURS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30]
_BCOLORS   = [
    (1, 1, 1, 0), 'cyan', 'cornflowerblue', 'blue', 'green', 'darkgreen',
    'yellow', 'gold', 'orange', 'red', 'darkred', 'lightpink', 'hotpink', 'magenta',
]
_CMAP2D = LinearSegmentedColormap.from_list('blocking2d', _BCOLORS, N=256)
_NORM2D = BoundaryNorm(boundaries=_BCONTOURS, ncolors=256)


# ─────────────────────────────────────────────────────────────────────────────
# 1-D line plot – multiple lead days coloured by forecast day
# ─────────────────────────────────────────────────────────────────────────────

def block_plot_1d_hindcast(block_freq_dict: dict,
                            lead_days:      list[int] | None = None,
                            pshade:         str  = '1',
                            season:         str  = '',
                            years:          list[int] | None = None,
                            ylim:           tuple = (0, 35),
                            era5_da         = None,
                            era5_clim_da    = None,
                            era5_verify_da  = None,
                            fig_out:        bool = False,
                            fig_name:       str  = 'block_1d_hindcast') -> None:
    """
    Longitude line plot of 1D blocking frequency for multiple lead days.

    Each lead day is drawn as a distinct colour (qualitative palette) with a
    matching shaded spread band.  A legend identifies lines by lead day.

    Parameters
    ----------
    block_freq_dict : dict  {lead_day: DataArray(member, lon)}
    lead_days       : which lead days to plot (default: all keys)
    pshade          : '1' ±1 std | '2' ±2 std | 'mm' min/max
    season          : label string for title / filename
    years           : list of years used (shown in title)
    ylim            : y-axis limits  (%, blocking frequency)
    era5_da         : optional DataArray(lon) of ERA5 blocking frequency [0,1]
                      plotted as a black line with open-circle markers
    era5_clim_da    : optional DataArray(lon) of full ERA5 season climatology [0,1]
                      plotted as a mid-gray dotted line
    era5_verify_da  : optional DataArray(lon) of ERA5 verification blocking [0,1]
                      (season days from the hindcast year files); plotted as a
                      black dashed line labelled 'ERA5 verify'
    fig_out         : save PNG to DIR_FIG
    fig_name        : output filename stem
    """
    t0 = time.time()
    plt.rcParams.update({'font.size': 18})

    LON_OFFSET = 90.    # degrees to roll eastward
    DLON_TICK  = 30.

    all_days  = sorted(block_freq_dict.keys())
    lead_days = lead_days if lead_days is not None else all_days

    # Evenly span jet across however many lead days are being plotted
    cmap   = mplcm.get_cmap('jet')
    n_ld   = len(lead_days)
    colors = [cmap(i / max(n_ld - 1, 1)) for i in range(n_ld)]

    fig, ax = plt.subplots(figsize=(17, 8))

    for col, ld in zip(colors, lead_days):
        if ld not in block_freq_dict:
            continue
        da  = block_freq_dict[ld]                     # (member, lon)

        # Roll + smooth + scale to %
        n_roll = int(LON_OFFSET / float(da.lon[1] - da.lon[0]))
        da     = da.roll(lon=n_roll).rolling(lon=3, center=True).mean() * 100.

        da_mean = da.mean('member')

        if pshade == 'mm':
            da_lo = da.min('member')
            da_hi = da.max('member')
            shade_lbl = 'min/max'
        else:
            k      = int(pshade)
            std    = da.std('member')
            da_lo  = da_mean - k * std
            da_hi  = da_mean + k * std
            shade_lbl = f'±{k} std'

        ax.plot(da_mean.lon, da_mean, lw=2.5, color=col, label=f'Day {ld}')
        ax.fill_between(da_mean.lon.values, da_lo.values, da_hi.values,
                        alpha=0.20, color=col)

    # --- ERA5 verification line (single observed line for the hindcast years) --
    if era5_verify_da is not None:
        n_roll_v = int(LON_OFFSET / float(era5_verify_da.lon[1] - era5_verify_da.lon[0]))
        v_plot   = (era5_verify_da.roll(lon=n_roll_v)
                                  .rolling(lon=3, center=True).mean() * 100.)
        ax.plot(v_plot.lon, v_plot,
                color='black', lw=2.5, linestyle='--',
                marker='o', markevery=12, markersize=12, markerfacecolor='none',
                markeredgewidth=2.5,
                label='ERA5 verify', zorder=5)

    # --- ERA5 observed line ---------------------------------------------------
    if era5_da is not None:
        n_roll_era5 = int(LON_OFFSET / float(era5_da.lon[1] - era5_da.lon[0]))
        era5_plot   = (era5_da.roll(lon=n_roll_era5)
                               .rolling(lon=3, center=True).mean() * 100.)
        ax.plot(era5_plot.lon, era5_plot,
                color='black', lw=2.5, linestyle='-',
                marker='o', markevery=12, markersize=12, markerfacecolor='none',
                markeredgewidth=2.5,
                label='ERA5', zorder=5)

    # --- ERA5 full-season climatology -----------------------------------------
    if era5_clim_da is not None:
        n_roll_clim  = int(LON_OFFSET / float(era5_clim_da.lon[1] - era5_clim_da.lon[0]))
        clim_plot    = (era5_clim_da.roll(lon=n_roll_clim)
                                    .rolling(lon=3, center=True).mean() * 100.)
        ax.plot(clim_plot.lon, clim_plot,
                color='#808080', lw=2.5, linestyle=':',
                label='ERA5 clim', zorder=4)

    ax.legend(fontsize=14, framealpha=0.8, title='Lead day', title_fontsize=13)

    # Axis formatting
    ax.set_xlim([0, 360])
    ax.set_ylim(ylim)
    xticks = np.arange(0, 361, DLON_TICK)
    xlbls  = np.arange(-LON_OFFSET, 360 - LON_OFFSET + 1, DLON_TICK)
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [f'{int(abs(t))}°W' if t < 0 else f'{int(t)}°E' for t in xlbls],
        fontsize=14,
    )
    if years is not None:
        yr_str = (f'{years[0]}' if len(years) == 1
                  else f'{min(years)}–{max(years)}')
    else:
        yr_str = ''
    title_parts = ['1D Hindcast']
    if season:
        title_parts.append(season)
    if yr_str:
        title_parts.append(yr_str)
    title_parts.append(f'[{shade_lbl}]')

    ax.set_xlabel('Longitude', fontsize=16)
    ax.set_ylabel('(%)', fontsize=16)
    ax.set_title('  '.join(title_parts), fontsize=18)
    ax.grid(axis='y', alpha=0.4)

    if fig_out:
        Path(DIR_FIG).mkdir(parents=True, exist_ok=True)
        seas_tag = f'_{season}' if season else ''
        yr_tag   = (f'_{yr_str}' if yr_str else '').replace('–', '-')
        fpath    = Path(DIR_FIG) / f'{fig_name}{seas_tag}{yr_tag}.png'
        plt.savefig(fpath, dpi=100, bbox_inches='tight')
        print(f'  Saved: {fpath}')

    print(f'block_plot_1d_hindcast: done in {time.time()-t0:.1f}s')


# ─────────────────────────────────────────────────────────────────────────────
# 1-D gradient line plot – GHGS or GHGN vs longitude
# ─────────────────────────────────────────────────────────────────────────────

def block_plot_1d_gradient(block_grad_dict: dict,
                            diag_name:      str,
                            lead_days:      list[int] | None = None,
                            pshade:         str  = '1',
                            season:         str  = '',
                            years:          list[int] | None = None,
                            ylim:           tuple | None = None,
                            era5_clim_da    = None,
                            era5_verify_da  = None,
                            fig_out:        bool = False,
                            fig_name:       str | None = None) -> None:
    """
    Longitude line plot of GHGS or GHGN gradient strength for multiple lead days.

    Mirrors block_plot_1d_hindcast but works in m / degree-lat.

    Parameters
    ----------
    block_grad_dict : dict  {lead_day: DataArray(member, lon)}  in m/deg-lat
    diag_name       : 'GHGS' or 'GHGN' — used in titles and filename
    lead_days       : which lead days to plot (default: all keys)
    pshade          : '1' ±1 std | '2' ±2 std | 'mm' min/max
    season          : label string for title / filename
    years           : list of years used (shown in title)
    ylim            : y-axis limits in m/deg-lat; None = auto from data
    era5_clim_da    : optional DataArray(lon) of ERA5 full-season climatology
                      gradient (m/deg-lat); plotted as a mid-gray dotted line
    era5_verify_da  : optional DataArray(lon) of ERA5 verification gradient
                      (m/deg-lat) for the hindcast years; plotted as a black
                      dashed line with open-circle markers
    fig_out         : save PNG to DIR_FIG
    fig_name        : output filename stem (default: block_{diag_name}_hindcast)
    """
    t0 = time.time()
    plt.rcParams.update({'font.size': 18})

    LON_OFFSET = 90.
    DLON_TICK  = 30.

    all_days  = sorted(block_grad_dict.keys())
    lead_days = lead_days if lead_days is not None else all_days

    cmap   = mplcm.get_cmap('jet')
    n_ld   = len(lead_days)
    colors = [cmap(i / max(n_ld - 1, 1)) for i in range(n_ld)]

    fig, ax = plt.subplots(figsize=(17, 8))

    # Track data range for auto ylim
    all_lo, all_hi = [], []

    for col, ld in zip(colors, lead_days):
        if ld not in block_grad_dict:
            continue
        da = block_grad_dict[ld]                          # (member, lon)  m/deg

        n_roll = int(LON_OFFSET / float(da.lon[1] - da.lon[0]))
        da     = da.roll(lon=n_roll).rolling(lon=3, center=True).mean()

        da_mean = da.mean('member')

        if pshade == 'mm':
            da_lo     = da.min('member')
            da_hi     = da.max('member')
            shade_lbl = 'min/max'
        else:
            k         = int(pshade)
            std       = da.std('member')
            da_lo     = da_mean - k * std
            da_hi     = da_mean + k * std
            shade_lbl = f'±{k} std'

        ax.plot(da_mean.lon, da_mean, lw=2.5, color=col, label=f'Day {ld}')
        ax.fill_between(da_mean.lon.values, da_lo.values, da_hi.values,
                        alpha=0.20, color=col)

        all_lo.append(float(da_lo.min()))
        all_hi.append(float(da_hi.max()))

    # Add a zero reference line — useful for both GHGS and GHGN
    ax.axhline(0, color='black', lw=1.0, linestyle='--', alpha=0.5, zorder=2)

    # Mark the relevant blocking threshold
    if diag_name == 'GHGN':
        ax.axhline(-5, color='dimgray', lw=1.5, linestyle=':', alpha=0.7,
                   label='Block threshold (−5)', zorder=2)

    # --- ERA5 verification line -----------------------------------------------
    if era5_verify_da is not None:
        n_roll_v = int(LON_OFFSET / float(era5_verify_da.lon[1] - era5_verify_da.lon[0]))
        v_plot   = (era5_verify_da.roll(lon=n_roll_v)
                                  .rolling(lon=3, center=True).mean())
        ax.plot(v_plot.lon, v_plot,
                color='black', lw=2.5, linestyle='--',
                marker='o', markevery=12, markersize=12, markerfacecolor='none',
                markeredgewidth=2.5,
                label='ERA5 verify', zorder=5)
        all_lo.append(float(v_plot.min()))
        all_hi.append(float(v_plot.max()))

    # --- ERA5 full-season climatology -----------------------------------------
    if era5_clim_da is not None:
        n_roll_clim = int(LON_OFFSET / float(era5_clim_da.lon[1] - era5_clim_da.lon[0]))
        clim_plot   = (era5_clim_da.roll(lon=n_roll_clim)
                                   .rolling(lon=3, center=True).mean())
        ax.plot(clim_plot.lon, clim_plot,
                color='#808080', lw=2.5, linestyle=':',
                label='ERA5 clim', zorder=4)
        all_lo.append(float(clim_plot.min()))
        all_hi.append(float(clim_plot.max()))

    ax.legend(fontsize=14, framealpha=0.8, title='Lead day', title_fontsize=13)

    ax.set_xlim([0, 360])
    if ylim is not None:
        ax.set_ylim(ylim)
    elif all_lo and all_hi:
        pad = (max(all_hi) - min(all_lo)) * 0.1
        ax.set_ylim(min(all_lo) - pad, max(all_hi) + pad)

    xticks = np.arange(0, 361, DLON_TICK)
    xlbls  = np.arange(-LON_OFFSET, 360 - LON_OFFSET + 1, DLON_TICK)
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [f'{int(abs(t))}°W' if t < 0 else f'{int(t)}°E' for t in xlbls],
        fontsize=14,
    )

    if years is not None:
        yr_str = (f'{years[0]}' if len(years) == 1
                  else f'{min(years)}–{max(years)}')
    else:
        yr_str = ''

    title_parts = [f'{diag_name} Hindcast']
    if season:
        title_parts.append(season)
    if yr_str:
        title_parts.append(yr_str)
    title_parts.append(f'[{shade_lbl}]')

    ax.set_xlabel('Longitude', fontsize=16)
    ax.set_ylabel('m / deg-lat', fontsize=16)
    ax.set_title('  '.join(title_parts), fontsize=18)
    ax.grid(axis='y', alpha=0.4)

    if fig_out:
        Path(DIR_FIG).mkdir(parents=True, exist_ok=True)
        stem     = fig_name or f'block_{diag_name.lower()}_hindcast'
        seas_tag = f'_{season}' if season else ''
        yr_tag   = (f'_{yr_str}' if yr_str else '').replace('–', '-')
        fpath    = Path(DIR_FIG) / f'{stem}{seas_tag}{yr_tag}.png'
        plt.savefig(fpath, dpi=100, bbox_inches='tight')
        print(f'  Saved: {fpath}')

    print(f'block_plot_1d_gradient ({diag_name}): done in {time.time()-t0:.1f}s')


# ─────────────────────────────────────────────────────────────────────────────
# 2-D polar map – panel per lead day
# ─────────────────────────────────────────────────────────────────────────────

def block_plot_2d_hindcast(block_freq_dict: dict,
                            lead_days:      list[int] | None = None,
                            season:         str  = '',
                            years:          list[int] | None = None,
                            era5_da         = None,
                            era5_clim_da    = None,
                            era5_verify_da  = None,
                            max_cols:       int  = 5,
                            fig_out:        bool = True,
                            fig_name:       str  = 'block_2d_hindcast') -> None:
    """
    North Polar Stereographic panel maps of 2D blocking frequency,
    one panel per lead day (ensemble-mean blocking frequency).
    Optional ERA5 panels are appended at the end.

    Parameters
    ----------
    block_freq_dict : dict  {lead_day: DataArray(member, lat, lon)}
    lead_days       : which lead days to plot (default: all keys)
    season          : label string for title / filename
    era5_da         : optional DataArray(lat, lon) of ERA5 matched-date blocking [0,1]
    era5_clim_da    : optional DataArray(lat, lon) of full ERA5 season climatology [0,1]
                      appended as the final panel
    era5_verify_da  : optional DataArray(lat, lon) of ERA5 verification blocking [0,1]
                      (season days from the hindcast year files); appended as an
                      'ERA5 verify' panel before the clim panel
    max_cols        : maximum columns in the panel layout
    fig_out         : save PNG to DIR_FIG
    fig_name        : output filename stem
    """
    all_days  = sorted(block_freq_dict.keys())
    lead_days = lead_days if lead_days is not None else all_days
    n_hcast   = len(lead_days)
    n_panels  = (n_hcast
                 + (1 if era5_da        is not None else 0)
                 + (1 if era5_verify_da is not None else 0)
                 + (1 if era5_clim_da   is not None else 0))

    ncols = min(max_cols, n_panels)
    nrows = (n_panels + ncols - 1) // ncols

    proj = ccrs.NorthPolarStereo()
    fig, axes = plt.subplots(
        nrows, ncols,
        subplot_kw={'projection': proj},
        figsize=(7 * ncols, 7 * nrows),
        constrained_layout=True,
    )
    ax_flat   = np.array(axes).flat
    text_size = max(10, ncols * 90. * 3 / fig.get_size_inches()[0])

    def _draw_panel(ax, da_pct, title):
        """Plot one 2D blocking panel; return the contourf object."""
        da_cyc, lon_cyc = add_cyclic_point(da_pct.values, coord=da_pct.lon.values)
        cf = ax.contourf(
            lon_cyc, da_pct.lat, da_cyc,
            levels=_BCONTOURS, norm=_NORM2D,
            cmap=_CMAP2D, extend='max',
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines(linewidth=1.5, color='black', resolution='110m')
        ax.add_feature(cfeature.LAND, facecolor='silver')
        gl = ax.gridlines(color='C7', lw=0.8, ls=':', draw_labels=True,
                          rotate_labels=False, ylocs=[40, 60, 80])
        gl.xlabel_style = {'size': text_size * 0.5}
        polarCentral_set_latlim((40, 90), ax)
        ax.set_title(title, fontsize=text_size)
        return cf

    n_era5   = 1 if era5_da        is not None else 0
    n_verify = 1 if era5_verify_da is not None else 0
    n_clim   = 1 if era5_clim_da   is not None else 0

    cf_last = None
    for idx, ax in enumerate(ax_flat):
        if idx >= n_panels:
            fig.delaxes(ax)
            continue

        if idx < n_hcast:
            ld = lead_days[idx]
            if ld not in block_freq_dict:
                fig.delaxes(ax)
                continue
            da_pct  = block_freq_dict[ld].mean('member') * 100.
            cf_last = _draw_panel(ax, da_pct, f'Lead day {ld}')
        elif n_era5 and idx == n_hcast:
            da_pct  = era5_da * 100.
            cf_last = _draw_panel(ax, da_pct, 'ERA5')
        elif n_verify and idx == n_hcast + n_era5:
            da_pct  = era5_verify_da * 100.
            cf_last = _draw_panel(ax, da_pct, 'ERA5 verify')
        else:
            # ERA5 full-season climatology panel (always last)
            da_pct  = era5_clim_da * 100.
            cf_last = _draw_panel(ax, da_pct, 'ERA5 clim (1979–2005)')

    if cf_last is not None:
        cbar = fig.colorbar(
            cf_last, ax=list(np.array(axes).flat)[:n_panels],
            ticks=_BCONTOURS, location='right', shrink=0.4, pad=0.02,
        )
        cbar.set_label('%', fontsize=text_size * 0.7)

    yr_str   = (f'{years[0]}' if years and len(years) == 1
                else f'{min(years)}-{max(years)}' if years else '')
    seas_str = f'  {season}' if season else ''
    yr_suptitle = f'  {yr_str}' if yr_str else ''
    fig.suptitle(
        f'2D Blocking Frequency (%) – Hindcast ensemble mean{seas_str}{yr_suptitle}',
        fontsize=text_size * 0.9,
    )

    if fig_out:
        Path(DIR_FIG).mkdir(parents=True, exist_ok=True)
        seas_tag = f'_{season}' if season else ''
        yr_tag   = f'_{yr_str}' if yr_str else ''
        fpath    = Path(DIR_FIG) / f'{fig_name}{seas_tag}{yr_tag}.png'
        plt.savefig(fpath, dpi=100, bbox_inches='tight')
        print(f'  Saved: {fpath}')


# ─────────────────────────────────────────────────────────────────────────────
# Lead-day evolution plot – single longitude or region vs lead day
# ─────────────────────────────────────────────────────────────────────────────

def block_plot_evolution(block_freq_dict: dict,
                          lon_range:   tuple[float, float] = (0., 360.),
                          season:      str  = '',
                          years:       list[int] | None = None,
                          ylim:        tuple = (0, 25),
                          era5_clim_da = None,
                          fig_out:     bool = False,
                          fig_name:    str  = 'block_evolution_hindcast') -> None:
    """
    Plot longitudinally-averaged 1D blocking frequency vs lead day.

    Useful for seeing how blocking skill / climatology evolves with forecast
    length.  Shading shows ±1 std across ensemble members.

    Parameters
    ----------
    block_freq_dict : dict  {lead_day: DataArray(member, lon)}
    lon_range       : (lon_min, lon_max) for averaging – default: global
    era5_clim_da    : optional DataArray(lon) of full ERA5 season climatology [0,1];
                      drawn as a mid-gray dotted horizontal reference line
    """
    t0 = time.time()
    plt.rcParams.update({'font.size': 18})

    lead_days = sorted(block_freq_dict.keys())
    means     = []
    stds      = []

    for ld in lead_days:
        da   = block_freq_dict[ld]                      # (member, lon)
        da_r = da.sel(lon=slice(*lon_range)) * 100.
        m    = da_r.mean('lon').mean('member').item()
        s    = da_r.mean('lon').std('member').item()
        means.append(m)
        stds.append(s)

    means = np.array(means)
    stds  = np.array(stds)
    xd    = np.array(lead_days)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(xd, means, lw=3, color='royalblue', label='Ens mean')
    ax.fill_between(xd, means - stds, means + stds,
                    alpha=0.3, color='royalblue', label='±1 std')

    if era5_clim_da is not None:
        clim_val = (era5_clim_da.sel(lon=slice(*lon_range)).mean('lon').item() * 100.)
        ax.axhline(clim_val, color='#808080', lw=2.5, linestyle=':',
                   label='ERA5 clim', zorder=3)

    ax.set_xlabel('Lead day', fontsize=16)
    ax.set_ylabel('Blocking Frequency (%)', fontsize=16)
    lon_lbl = (f'{lon_range[0]:.0f}°–{lon_range[1]:.0f}°E'
               if lon_range != (0., 360.) else 'Global')
    ax.set_title(
        f'1D Blocking Frequency ({lon_lbl}) vs Lead Day'
        f'{("  " + season) if season else ""}',
        fontsize=18,
    )
    ax.set_ylim(ylim)
    ax.legend(fontsize=14)
    ax.grid(alpha=0.4)

    if fig_out:
        Path(DIR_FIG).mkdir(parents=True, exist_ok=True)
        yr_str   = (f'{years[0]}' if years and len(years) == 1
                    else f'{min(years)}-{max(years)}' if years else '')
        seas_tag = f'_{season}' if season else ''
        yr_tag   = f'_{yr_str}' if yr_str else ''
        fpath    = Path(DIR_FIG) / f'{fig_name}{seas_tag}{yr_tag}.png'
        plt.savefig(fpath, dpi=100, bbox_inches='tight')
        print(f'  Saved: {fpath}')

    print(f'block_plot_evolution: done in {time.time()-t0:.1f}s')


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def polarCentral_set_latlim(lat_lims, ax):
    """Set a circular boundary for a polar-stereographic axes."""
    import matplotlib.path as mpath
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    theta  = np.linspace(0, 2 * np.pi, 100)
    verts  = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * 0.5 + 0.5)
    ax.set_boundary(circle, transform=ax.transAxes)
