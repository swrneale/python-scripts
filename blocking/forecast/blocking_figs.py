"""
blocking_figs.py
Visualisation routines for blocking frequency diagnostics.

  block_plot_1d         – longitude line plot of 1D blocking frequency
  block_plot_2d         – polar stereographic map of 2D blocking frequency
  block_plot_1d_pdf     – (stub) regional PDFs
  jet_var_plot          – (stub) jet latitude diagnostics
  polarCentral_set_latlim – helper for circular polar-stereo boundary
"""

import sys
import time

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm


DIR_FIG = '/glade/u/home/rneale/python/python-figs/'

# ── Shared plot style constants ──────────────────────────────────────────────
_MODEL_COLORS = ['blue', 'red', 'green', 'purple']
_OBS_DASHES   = ['-', '--', ':']
_OBS_MARKERS  = ['o', 's', '+']

# 2D colormap: transparent white → cyan → blue → … → magenta
_BCONTOURS = [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 25, 30]
_BCOLORS   = [
    (1, 1, 1, 0), 'cyan', 'cornflowerblue', 'blue', 'green', 'darkgreen',
    'yellow', 'gold', 'orange', 'red', 'darkred', 'lightpink', 'hotpink', 'magenta',
]
_CMAP2D = LinearSegmentedColormap.from_list('blocking2d', _BCOLORS, N=256)
_NORM2D = BoundaryNorm(boundaries=_BCONTOURS, ncolors=256)


# ─────────────────────────────────────────────────────────────────────────────
# 1D line plot
# ─────────────────────────────────────────────────────────────────────────────

def block_plot_1d(block_meta, ens_block_1d, bseason,
                  pshade='mm', ens_plot='av', fig_out=False):
    """
    Longitude line plot of 1D blocking frequency.

    Parameters
    ----------
    block_meta   : pd.DataFrame  from ens_setup
    ens_block_1d : dict          {ens_name: DataArray}  from block_z500_freq
    bseason      : str           season label
    pshade       : 'mm' min/max | '1' ±1 std | '2' ±2 std
    ens_plot     : 'av' ensemble mean (default) — kept for API symmetry with block_plot_2d
    fig_out      : bool  save PNG to DIR_FIG
    """
    t0 = time.time()
    plt.rcParams.update({'font.size': 22})

    LON_OFFSET = 90.   # degrees to roll eastward
    DLON_TICK  = 30.

    ens_names  = list(block_meta.index)
    ens_ystarts = block_meta['Start Year'].values
    ens_yends   = block_meta['End Year'].values

    fig, ax = plt.subplots(figsize=(20, 10))
    imod = 0; iobs = 0

    for ens_name in ens_names:
        ens_type = block_meta.loc[ens_name]['Ensemble Type']
        n_runs   = len(block_meta.loc[ens_name]['Run Name'])

        if ens_type == 'model':
            col   = _MODEL_COLORS[imod % len(_MODEL_COLORS)]
            dash  = '-'; marker = None; msize = None
            imod += 1
        else:
            col   = 'black'
            dash  = _OBS_DASHES[iobs % len(_OBS_DASHES)]
            marker= _OBS_MARKERS[iobs % len(_OBS_MARKERS)]
            msize = 15; iobs += 1

        da = ens_block_1d[ens_name]

        # Longitude roll + 3-point smoothing + scale to %
        n_roll = int(LON_OFFSET / float(da.lon[1] - da.lon[0]))
        da = da.roll(lon=n_roll).rolling(lon=3, center=True).mean() * 100.

        da_mean = da.mean('name')

        if pshade == 'mm':
            shade_lbl = 'min/max range'
            da_lo = da.min('name')
            da_hi = da.max('name')
        else:
            k = int(pshade)
            shade_lbl = f'±{k} std'
            std = da.std('name')
            da_lo = da_mean - k * std
            da_hi = da_mean + k * std

        label = ens_name if n_runs == 1 else f'{ens_name} ({n_runs})'
        ax.plot(da_mean.lon, da_mean, lw=4, color=col, linestyle=dash,
                marker=marker, markersize=msize, markevery=10,
                mew=3, fillstyle='none', label=label)

        if n_runs > 1:
            ax.fill_between(da_mean.lon, da_lo, da_hi, alpha=0.35, color=col)

    # Axis formatting
    ax.set_xlim([0, 360])
    ax.set_ylim([0.01, 35])
    xticks = np.arange(0, 361, DLON_TICK)
    xlbls  = np.arange(-LON_OFFSET, 360 - LON_OFFSET + 1, DLON_TICK)
    ax.set_xticks(xticks)
    ax.set_xticklabels(
        [f'{int(abs(t))}°W' if t < 0 else f'{int(t)}°E' for t in xlbls]
    )
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Blocking Frequency (%)')

    if (ens_ystarts[0] == ens_ystarts[-1] and ens_yends[0] == ens_yends[-1]):
        yr_str = f' ({ens_ystarts[0]}–{ens_yends[0]})'
        fig_yr = f'_{ens_ystarts[0]}-{ens_yends[0]}'
    else:
        yr_str = ''; fig_yr = ''

    ax.set_title(f'{bseason}{yr_str}  {shade_lbl}')
    ax.legend()

    if fig_out:
        fname = (DIR_FIG + 'block_1d_freq_'
                 + '_'.join(ens_names) + fig_yr + f'_{bseason}.png')
        plt.savefig(fname, dpi=80, bbox_inches='tight')
        print(f'  Saved: {fname}')

    print(f'block_plot_1d: done in {time.time()-t0:.1f}s')


# ─────────────────────────────────────────────────────────────────────────────
# 2D polar map
# ─────────────────────────────────────────────────────────────────────────────

def block_plot_2d(block_meta, ens_block_2d, bseason,
                  ens_plot='av', fig_out=True):
    """
    North Polar Stereographic panel maps of 2D blocking frequency.

    Parameters
    ----------
    block_meta   : pd.DataFrame
    ens_block_2d : dict  {ens_name: DataArray}
    bseason      : str
    ens_plot     : 'av' ensemble mean | '0' first member only
    fig_out      : bool  save PNG
    """
    ens_names   = list(block_meta.index)
    ens_ystarts = block_meta['Start Year'].values
    ens_yends   = block_meta['End Year'].values
    nens  = len(ens_names)
    ncols = 3
    nrows = (nens + ncols - 1) // ncols

    proj = ccrs.NorthPolarStereo()
    fig, axes = plt.subplots(
        nrows, ncols,
        subplot_kw={'projection': proj},
        figsize=(8 * ncols, 8 * nrows),
        constrained_layout=True,
    )
    ax_flat = axes.flat
    text_size = ncols * 100. * 3 / fig.get_size_inches()[0]

    cf_last = None
    for iens, ax in enumerate(ax_flat):

        if iens >= nens:
            fig.delaxes(ax)
            continue

        ens_name = ens_names[iens]
        da = ens_block_2d[ens_name] * 100.

        # Ensemble averaging / member selection
        if ens_plot == 'av':
            da_plot = da.mean('name').squeeze()
        else:
            da_plot = da.isel(name=0).squeeze()

        lat = da_plot.lat
        lon = da_plot.lon

        # MERRA: lon may not be 0-360-sorted; regrid to regular spacing
        if ens_name == 'MERRA':
            lon_vals = lon.values % 360
            sort_idx = np.argsort(lon_vals)
            da_plot  = da_plot.isel(lon=sort_idx)
            dlon     = np.round(np.median(np.diff(lon_vals[sort_idx])), 6)
            lon_reg  = np.arange(0, 360, dlon)
            da_plot  = da_plot.assign_coords(lon=lon_vals[sort_idx]).interp(
                lon=lon_reg, kwargs={'fill_value': 'extrapolate'}
            )
            lon = da_plot.lon

        da_cyc, lon_cyc = add_cyclic_point(da_plot.values, coord=lon.values)
        lon_cyc_da = xr.DataArray(lon_cyc, dims='lon', coords={'lon': lon_cyc})
        da_cyc_da  = xr.DataArray(da_cyc,  dims=('lat', 'lon'),
                                   coords={'lat': lat, 'lon': lon_cyc_da})

        cf_last = ax.contourf(
            lon_cyc_da, lat, da_cyc_da,
            levels=_BCONTOURS, norm=_NORM2D,
            cmap=_CMAP2D, extend='max',
            transform=ccrs.PlateCarree(),
        )
        ax.coastlines(linewidth=2, color='black', resolution='110m')
        ax.add_feature(cfeature.LAND, facecolor='silver')
        gl = ax.gridlines(color='C7', lw=1, ls=':', draw_labels=True,
                          rotate_labels=False, ylocs=[40, 60, 80])
        gl.xlabel_style = {'size': text_size * 0.5}
        polarCentral_set_latlim((40, 90), ax)
        ax.set_title(ens_name, fontsize=text_size)

    if cf_last is not None:
        cbar = fig.colorbar(cf_last, ax=axes[:, -1],
                            ticks=_BCONTOURS, location='right', shrink=0.25)
        cbar.set_label('%', fontsize=text_size * 0.6)

    if (ens_ystarts[0] == ens_ystarts[-1] and ens_yends[0] == ens_yends[-1]):
        yr_str = f' ({ens_ystarts[0]}–{ens_yends[0]})'
        fig_yr = f'_{ens_ystarts[0]}-{ens_yends[0]}'
    else:
        yr_str = ''; fig_yr = ''

    fig.suptitle(f'Blocking Frequency (%) – {bseason}{yr_str}',
                 fontsize=text_size, ha='center', va='bottom')

    if fig_out:
        fname = (DIR_FIG + 'block_2d_freq_'
                 + '_'.join(ens_names) + fig_yr + f'_{bseason}.png')
        plt.savefig(fname, dpi=80, bbox_inches='tight')
        print(f'  Saved: {fname}')


# ─────────────────────────────────────────────────────────────────────────────
# Stubs
# ─────────────────────────────────────────────────────────────────────────────

def block_plot_1d_pdf(block_meta, ens_block_2d, bseason, fig_out=True):
    """(stub) Regional PDFs of blocking strength."""
    pass


def jet_var_plot():
    """(stub) Jet latitude diagnostics."""
    pass


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
