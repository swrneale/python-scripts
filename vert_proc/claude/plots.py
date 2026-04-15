"""
plots.py  –  All plotting routines for the vertical-processes analysis.

Public functions
----------------
plot_sst_index(run_case, sst_data, nino_region, inino, inina, cfg)
plot_div_pressure_level(case, var_name, composites, ps_composites, ds_ptr, cfg)
plot_scatter_2vars(case, var1, var2, comp1, comp2, ps_comp, regions_df, ds_ptr, cfg)
plot_region_boxes(regions_df, cfg)
plot_vertical_profiles(all_case_profiles, p_levs, var_name, regions_df,
                       var_meta, cases, years, pref_out, cfg)
cam_hybrid_to_pressure(da, ps, ds_ptr)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point
import cartopy.mpl.ticker as cticker
import numpy as np
import pandas as pd
import xarray as xr

from config import AnalysisConfig, CaseConfig, DIR_FIGURES


# ── Reference pressure levels (hPa) ─────────────────────────────────────────
_REF_LEVS = [
    1000, 975, 950, 925, 900, 850, 800, 750, 700,
    600,  500, 450, 400, 300, 250, 225, 200, 175,
    150,  125, 100,  50,
]

FIG_DPI = 150


# ── Legend / line-style helpers ───────────────────────────────────────────────
_COLORS = ['blue', 'orange', 'green', 'red', 'purple', 'cyan', 'brown', 'pink']
_MARKER_CYCLE = ['.', 's', '^', 'D', 'v', 'P', '*']
_ENSEMBLE_TYPES = {'lens1', 'lens2', 'lense2', 'c6_amip'}

_TYPE_STYLE: dict[str, dict] = {
    'reanal':       dict(marker='x', lw=3, ls='-'),
    'lens1':        dict(marker=None, color='red',   lw=1, ls='-'),
    'lens2':        dict(marker=None, color='blue',  lw=1, ls='-'),
    'lense2':       dict(marker=None, color='green', lw=1, ls='-'),
    'c6_amip':      dict(marker=None, color='blue',  lw=1, ls='-'),
    'cam6_revert':  dict(marker='.',  lw=1, ls='-'),
    'cesm3_dev':    dict(marker='.',  lw=1, ls='-'),
}


def _case_line_styles(cases: list[CaseConfig]):
    """Return per-case plotting attributes as parallel lists."""
    markers, colors, lwidths, lstyles = [], [], [], []
    col_idx = 0

    for case in cases:
        style = dict(_TYPE_STYLE[case.case_type])   # copy
        if 'color' not in style:
            style['color'] = _COLORS[col_idx % len(_COLORS)]
            col_idx += 1
        else:
            col_idx += 1
        markers.append(style.get('marker'))
        colors.append(style['color'])
        lwidths.append(style.get('lw', 1))
        lstyles.append(style.get('ls', '-'))

    return markers, colors, lwidths, lstyles


def make_legend(cases: list[CaseConfig]):
    """Build (elements, labels, markers, colors, lwidths, lstyles) for a legend."""
    from matplotlib.lines import Line2D

    markers, colors, lwidths, lstyles = _case_line_styles(cases)
    elements, labels = [], []
    seen_types: set[str] = set()

    for i, case in enumerate(cases):
        is_ensemble = case.case_type in _ENSEMBLE_TYPES
        if is_ensemble and case.case_type in seen_types:
            continue   # show only one entry per ensemble type
        seen_types.add(case.case_type)
        elements.append(Line2D([0], [0],
                               marker=markers[i],
                               color=colors[i],
                               lw=lwidths[i],
                               ls=lstyles[i]))
        labels.append(case.case_type if is_ensemble else case.name)

    return elements, labels, markers, colors, lwidths, lstyles


# ── Hybrid sigma → pressure interpolation ────────────────────────────────────

def cam_hybrid_to_pressure(
    da: xr.DataArray,
    ps: xr.DataArray,
    ds_ptr: xr.Dataset,
) -> xr.DataArray:
    """Interpolate from hybrid sigma-pressure coordinates to standard pressure levels.

    Parameters
    ----------
    da     : DataArray on CAM hybrid levels (dim 'lev')
    ps     : DataArray of surface pressure (Pa)
    ds_ptr : Dataset that contains 'hyam' and 'hybm'

    Returns
    -------
    DataArray on pressure levels (hPa), dim renamed to 'lev'.
    """
    from geocat.comp import interp_hybrid_to_pressure

    p0 = 100000.          # reference surface pressure, Pa
    new_levels = np.array(_REF_LEVS, dtype=float) * 100.  # hPa → Pa

    hyam = ds_ptr['hyam']
    hybm = ds_ptr['hybm']
    if hyam.ndim == 2:
        hyam = hyam[0]
    if hybm.ndim == 2:
        hybm = hybm[0]

    # interp_hybrid_to_pressure requires dask-backed arrays; rechunk if the
    # composites were already computed down to plain numpy by dask.compute().
    if da.chunks is None:
        da = da.chunk({'lev': -1})
    if ps.chunks is None:
        ps = ps.chunk()

    da_p = interp_hybrid_to_pressure(da, ps, hyam, hybm, p0=p0,
                                     new_levels=new_levels, method='log')
    da_p = da_p.rename({'plev': 'lev'})
    da_p = da_p.assign_coords(lev=da_p.lev * 0.01)   # Pa → hPa
    return da_p


# ── SST index time-series plot ────────────────────────────────────────────────

def plot_sst_index(
    run_case: str,
    sst_data: xr.DataArray,
    nino_region: str,
    inino: np.ndarray,
    inina: np.ndarray,
    cfg: AnalysisConfig,
) -> None:
    """Plot the Niño SST-anomaly time-series and mean nino/nina SST maps."""
    from enso import _NINO_REGIONS

    s, n, w, e = _NINO_REGIONS[nino_region]
    sst_ts = sst_data.sel(lat=slice(s, n), lon=slice(w, e)).mean(('lat', 'lon')).compute()

    # Remove annual cycle
    mnames = sst_ts.time.dt.strftime('%b').values
    for mon in np.unique(mnames):
        mask = mnames == mon
        sst_ts.values[mask] -= sst_ts.values[mask].mean()

    year_all = sst_ts.time.dt.strftime('%Y').values
    time_ax  = np.arange(len(year_all))
    it_ticks = np.arange(0, len(year_all), 12)
    thresh   = cfg.ssta_thresh

    cart_proj  = ccrs.PlateCarree()
    tcart_proj = ccrs.PlateCarree(central_longitude=180)

    fig = plt.figure(figsize=(22, 7))
    ax_ts  = fig.add_subplot(1, 2, 1)
    ax_nino = fig.add_subplot(2, 2, 2, projection=tcart_proj)
    ax_nina = fig.add_subplot(2, 2, 4, projection=tcart_proj)

    # Time-series
    ax_ts.plot(time_ax, sst_ts.values, color='black')
    ax_ts.fill_between(time_ax, sst_ts.values,  thresh, where=sst_ts.values >  thresh, color='red',  interpolate=True)
    ax_ts.fill_between(time_ax, sst_ts.values, -thresh, where=sst_ts.values < -thresh, color='blue', interpolate=True)
    ax_ts.hlines([0., thresh, -thresh], time_ax[0], time_ax[-1], colors='black',
                 linestyles=['solid', 'dashed', 'dashed'], lw=1)
    ax_ts.set_title(f'{nino_region} SSTA – {run_case}', fontsize=20)
    ax_ts.set_xlabel('Year', fontsize=15)
    ax_ts.set_ylabel('K', fontsize=15)
    ax_ts.set_xticks(it_ticks + 6)
    ax_ts.set_xticks(it_ticks, minor=True)
    ax_ts.set_xticklabels(year_all[it_ticks], rotation=90)

    # SST anomaly maps
    djf = sst_data[np.isin(mnames, ['Dec', 'Jan', 'Feb'])]
    djf_climo  = djf.mean('time')
    nino_anom  = sst_data[inino].mean('time') - djf_climo
    nina_anom  = sst_data[inina].mean('time') - djf_climo

    plevels = np.arange(-3., 3.25, 0.25)
    nino_cyc, lons_cyc = add_cyclic_point(nino_anom, coord=sst_data['lon'])
    nina_cyc, _        = add_cyclic_point(nina_anom, coord=sst_data['lon'])

    for ax, data, title in [(ax_nino, nino_cyc, 'El Niño'), (ax_nina, nina_cyc, 'La Niña')]:
        ax.contourf(lons_cyc, sst_data.lat, data, levels=plevels, cmap='bwr',
                    extend='both', transform=cart_proj)
        ax.contour(lons_cyc, sst_data.lat, data, levels=plevels,
                   colors='black', linewidths=0.5, transform=cart_proj)
        ax.set_extent((-10, 360, -45, 45), crs=tcart_proj)
        ax.set_title(f'{title} {nino_region} SST anomalies', fontsize=20)
        ax.add_feature(cfeature.LAND, zorder=100, color='black')
        ax.coastlines()
        x_tik = np.arange(0, 361, 30.); x_tik[-1] -= 1e-9
        ax.set_xticks(x_tik, crs=cart_proj)
        ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
        ax.set_yticks(np.arange(-45, 46, 15), crs=cart_proj)
        ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
        # Nino region box
        nbox = mpatches.Rectangle([w, s], e - w, n - s, facecolor='gray',
                                   alpha=0.7, edgecolor='black', transform=ccrs.PlateCarree())
        ax.add_patch(nbox)

    plt.tight_layout()
    out = DIR_FIGURES / f'{run_case}_{nino_region}_ssta.png'
    fig.savefig(out, dpi=FIG_DPI)
    plt.show()
    print(f'    Saved: {out}')


# ── Divergence / Omega max-min pressure-level maps ───────────────────────────

def plot_div_pressure_level(
    case: CaseConfig,
    var_name: str,
    composites: tuple,
    ps_composites: tuple | None,
    ds_ptr: xr.Dataset,
    cfg: AnalysisConfig,
) -> None:
    """3×2 panel map of the pressure level of max/min divergence."""
    clevsp = [1008, 992, 962, 938, 912, 875, 825, 775, 725, 650, 550,
              475, 425, 350, 275, 232.5, 212.5, 187, 162, 132.5, 112.5, 75, 25]
    clevsr = list(reversed(_REF_LEVS))

    ccols = ['lightgray','darkgray','gray','tan','khaki','yellow','gold',
             'darkorange','lightsalmon','red','greenyellow','green','darkgreen',
             'lightseagreen','cyan','deepskyblue','blue','navy','purple',
             'slateblue','violet','pink']
    cmap  = mcolors.ListedColormap(list(reversed(ccols)))

    cc_pc  = ccrs.PlateCarree(central_longitude=180)
    tcc_pc = ccrs.PlateCarree()

    fig, axl = plt.subplots(3, 2, subplot_kw={'projection': cc_pc}, figsize=(38, 20))
    fig.patch.set_facecolor('white')
    axl = axl.flatten()

    ens_labels = ['Climatology', 'El Niño', 'La Niña']

    for iens, da_in in enumerate(composites):
        if case.case_type != 'reanal' and ps_composites is not None:
            da_in = cam_hybrid_to_pressure(da_in, ps_composites[iens], ds_ptr)

        for imm, (mname, idxfn, threshold) in enumerate([
            ('Maximum', lambda d: d.idxmax('lev'),  1.5e-4),
            ('Minimum', lambda d: d.idxmin('lev'), -1.5e-4),
        ]):
            da_plot = idxfn(da_in)
            lev_val = da_in.max('lev') if mname == 'Maximum' else da_in.min('lev')
            da_plot = da_plot.where(lev_val > threshold if mname == 'Maximum'
                                    else lev_val < threshold)

            iax = 2 * iens + imm
            axl[iax].coastlines(color='black', linewidth=3)
            im = da_plot.plot.pcolormesh(
                ax=axl[iax], transform=tcc_pc, levels=clevsp,
                cmap=cmap, rasterized=True, add_colorbar=False, shading='auto',
            )
            axl[iax].set_title(f'{ens_labels[iens]} {mname}', fontsize=25)
            axl[iax].hlines(0., -180, 180, color='black', lw=1, linestyle='--')

    plt.subplots_adjust(bottom=0.25)
    fig.suptitle(f'{case.name} – Level of Maximum/Minimum {var_name}', fontsize=50)

    cbar_ax = fig.add_axes([0.5, 0.34, 0.01, 0.46])
    cbar_ax.set_title('Pressure (hPa)', fontsize=20)
    plt.colorbar(im, cax=cbar_ax, orientation='vertical', ticks=clevsr)
    cbar_ax.set_yticklabels(clevsr, fontsize=20)
    cbar_ax.invert_yaxis()

    out = DIR_FIGURES / f'{case.name}_{var_name}_minmax_level.png'
    plt.savefig(out, dpi=FIG_DPI)
    plt.show()
    print(f'    Saved: {out}')


# ── Two-variable KDE scatter plot ─────────────────────────────────────────────

def plot_scatter_2vars(
    case: CaseConfig,
    var1_name: str, var2_name: str,
    composites1: tuple, composites2: tuple,
    ps_composites: tuple | None,
    regions_df: pd.DataFrame,
    ds_ptr: xr.Dataset,
    var_meta: pd.DataFrame,
    cfg: AnalysisConfig,
) -> None:
    """KDE scatter plot of the column-maximum of two 2D fields, per region."""
    import seaborn as sb

    v1 = var_meta.loc[var1_name]
    v2 = var_meta.loc[var2_name]
    tav_names = ['Seasonal', 'El Niño', 'La Niña']

    for itav, (da1, da2) in enumerate(zip(composites1, composites2)):
        if case.case_type != 'reanal' and ps_composites is not None:
            ps = ps_composites[0]
            da1 = cam_hybrid_to_pressure(da1, ps, ds_ptr)
            da2 = cam_hybrid_to_pressure(da2, ps, ds_ptr)

        records = []
        for reg in regions_df.index:
            row  = regions_df.loc[reg]
            reg1 = da1.sel(lat=slice(row['lat_s'], row['lat_n']),
                           lon=slice(row['lon_w'], row['lon_e']))
            reg2 = da2.sel(lat=slice(row['lat_s'], row['lat_n']),
                           lon=slice(row['lon_w'], row['lon_e']))
            x = (float(v1['vscale']) * reg1.max('lev').values).ravel()
            y = (float(v2['vscale']) * reg2.max('lev').values).ravel()
            for xi, yi in zip(x, y):
                records.append({'xvar': xi, 'yvar': yi, 'Region': row['long_name']})

        df = pd.DataFrame(records)
        slevels = [0.02, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

        plt.figure(figsize=(15, 8))
        ax = sb.kdeplot(df, x='xvar', y='yvar', hue='Region',
                        levels=slevels, common_norm=True)
        ax.axhline(0., color='k', linestyle='--')
        ax.axvline(0., color='k', linestyle='--')
        plt.setp(ax.get_legend().get_texts(), fontsize=20)
        plt.setp(ax.get_legend().get_title(), fontsize=28)
        plt.xlabel(f'Maximum {v1["long_name"]} ({v1["vunits"]})', fontsize=20)
        plt.ylabel(f'Maximum {v2["long_name"]} ({v2["vunits"]})', fontsize=20)
        plt.suptitle(f'{case.name} – {tav_names[itav]}', fontsize=20)
        plt.ticklabel_format(axis='y', style='sci', scilimits=(1, 4))

        out = DIR_FIGURES / f'{case.name}_{tav_names[itav]}_scatter.png'
        plt.savefig(out, dpi=FIG_DPI)
        plt.show()
        print(f'    Saved: {out}')


# ── Region box map ────────────────────────────────────────────────────────────

def plot_region_boxes(regions_df: pd.DataFrame, cfg: AnalysisConfig) -> None:
    """Draw a map showing the analysis region boxes."""
    desired_proj = ccrs.PlateCarree(central_longitude=180.)
    facecolors   = ['b', 'darkorange', 'g', 'r', 'purple']

    fig = plt.figure(figsize=(10, 6))
    ax  = plt.subplot(projection=desired_proj)
    ax.set_extent([80, 280, -20, 50])
    ax.coastlines()
    ax.add_feature(cfeature.LAND, color='k')
    ax.gridlines()

    print('Regions:')
    for ireg, reg in enumerate(regions_df.index):
        row = regions_df.loc[reg]
        lat_s, lat_n = float(row['lat_s']), float(row['lat_n'])
        lon_w, lon_e = float(row['lon_w']), float(row['lon_e'])
        print(f'  {reg}: {lat_s}–{lat_n}°N, {lon_w}–{lon_e}°E')
        ax.add_patch(mpatches.Rectangle(
            [lon_w, lat_s], lon_e - lon_w, lat_n - lat_s,
            facecolor=facecolors[ireg % len(facecolors)],
            alpha=0.3, transform=ccrs.PlateCarree(),
        ))

    out = DIR_FIGURES / 'analysis_regions.png'
    plt.savefig(out, dpi=FIG_DPI, bbox_inches='tight')
    plt.show()
    print(f'    Saved: {out}')


# ── Multi-case vertical profile panels ───────────────────────────────────────

def plot_vertical_profiles(
    all_case_profiles: dict[str, xr.DataArray],
    p_levs: np.ndarray,
    var_name: str,
    regions_df: pd.DataFrame,
    var_meta: pd.DataFrame,
    cases: list[CaseConfig],
    years: tuple[int, int],
    pref_out: str,
    cfg: AnalysisConfig,
) -> None:
    """Plot vertical profiles for all cases, regions, and ENSO states.

    all_case_profiles : {case.name: DataArray(column, lev)}
        column ordering: [reg0_climo, reg0_nino, reg0_nina, reg1_climo, ...]
    """
    nino_labels = [
        f'Climatology ({years[0]}–{years[1]})',
        'El Niño Anomaly',
        'La Niña Anomaly',
    ]
    nino_colors = ['k', 'r', 'b']
    nnino  = 3
    nreg   = len(regions_df)
    ncases = len(cases)

    leg_elements, leg_labels, markers, colors, lwidths, lstyles = make_legend(cases)

    vm    = var_meta.loc[var_name]
    xmin, xmax   = float(vm['xmin']),  float(vm['xmax'])
    axmin, axmax = float(vm['axmin']), float(vm['axmax'])
    vunits = vm['vunits']
    var_text = vm['long_name']

    lloc = 'lower right' if var_name in ('ZMDQ', 'STEND_CLUBB') else 'lower left'

    fig, axn = plt.subplots(nreg, 3, figsize=(26, 9 * nreg))

    for icase, case in enumerate(cases):
        profiles = all_case_profiles[case.name]

        for ireg, reg in enumerate(regions_df.index):
            row = regions_df.loc[reg]
            reg_name = row['long_name']
            lat_s, lat_n = float(row['lat_s']), float(row['lat_n'])
            lon_w, lon_e = float(row['lon_w']), float(row['lon_e'])
            reg_str = f'{lon_w:.0f}–{lon_e:.0f}°E  {lat_s:.1f}–{lat_n:.0f}°N'

            for inino in range(nnino):
                icol = ireg * nnino + inino
                prof = profiles[icol]

                axn[ireg, inino].plot(
                    prof, prof.lev,
                    lw=lwidths[icase], markersize=9,
                    marker=markers[icase],
                    color=colors[icase],
                    linestyle=lstyles[icase],
                )

                if icase == 0:   # Decorate axes only once
                    axn[ireg, inino].set_title(nino_labels[inino], fontsize=20,
                                               color=nino_colors[inino])
                    axn[ireg, inino].set_ylabel('hPa', fontsize=16)
                    axn[ireg, inino].set_xlabel(vunits, fontsize=16)
                    axn[ireg, inino].set_yticks(p_levs)
                    axn[ireg, inino].set_yticklabels(p_levs.astype(int), fontsize=14)
                    axn[ireg, inino].invert_yaxis()
                    axn[ireg, inino].tick_params(axis='both', which='major', labelsize=14)
                    axn[ireg, inino].grid(linestyle='--')
                    axn[ireg, inino].vlines(0., p_levs.max(), p_levs.min(),
                                            linestyle='--', lw=1, color='black')

                    axn[ireg, 0].set_xlim([xmin,  xmax])
                    axn[ireg, 1].set_xlim([axmin, axmax])
                    axn[ireg, 2].set_xlim([axmin, axmax])

                    axn[ireg, 0].legend(leg_elements, leg_labels, fontsize=15, loc=lloc)
                    axn[ireg, 0].text(0., 1., reg_name, transform=axn[ireg, 0].transAxes,
                                      ha='left', va='top', fontsize=20)
                    axn[ireg, 0].text(0., 0.95, reg_str, transform=axn[ireg, 0].transAxes,
                                      ha='left', va='top', fontsize=16)

    fig.suptitle(f'ENSO Vertical Profiles – {var_text}', fontsize=24)
    plt.tight_layout()

    out = DIR_FIGURES / f'{pref_out}_nino_vprof_{var_name}_{years[0]}_to_{years[1]}.png'
    fig.savefig(out, dpi=80)
    plt.show()
    print(f'    Saved: {out}')
