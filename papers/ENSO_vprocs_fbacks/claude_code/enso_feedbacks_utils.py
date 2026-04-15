'''
    Utility Routines for ENSO Diagnostics (currently a redo of the ncl quick look panels)
'''

import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.path as mpath
import cartopy
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
from scipy.stats import gaussian_kde
from windspharm.xarray import VectorWind

import geocat.comp as gc
import glob as glob
import os as os
import re as _re


### May need to CHANGE for your own local directory to write out derived timeseries (not used for LENS, already in tseries format.)
dir_ncout = '/glade/work/rneale/python-netcdf/enso/'
dir_data  = '/glade/work/rneale/data/'

# Cache for CESM3 h0a datasets — keyed by (case, yr0, yr1, lread_in_all_hist)
# so open_mfdataset is only called once per case/year-range combination.
_cesm3_ds_cache: dict = {}


# ─────────────────────────────────────────────────────────────────────────────
# Simple helpers
# ─────────────────────────────────────────────────────────────────────────────

def nino_region(nino_name):
    """Return (west, east, north, south) bounds for a Nino region."""
    match nino_name:
        case 'nino3':  return 210., 270.,  5., -5.
        case 'nino34': return 190., 240.,  5., -5.
        case 'nino4':  return 160., 210.,  5., -5.
        case 'nino5':  return 120., 140.,  5., -5.
        case 'nino6':  return 140., 160., 16.,  8.
        case _: raise ValueError(f'Unknown nino region: {nino_name!r}')


def infer_ctype(cname):
    """Infer the model/obs type string from a case name."""
    if isinstance(cname, list):
        cname = cname[0]
    if 'b.e3' in cname: return 'cesm3'
    if 'b.e2' in cname: return 'cesm2'
    if 'b.e1' in cname: return 'cesm1'
    if 'f.e1' in cname: return 'cam5'
    if 'f.e2' in cname: return 'cam6'
    if 'GOGA' in cname: return 'cam6'
    if 'v3.'  in cname: return 'e3smv3'
    if 'v2.'  in cname: return 'e3smv2'
    return 'OBS'


def make_ensemble_names(base_case: str, n_members: int,
                        e3sm_step: int = 10,
                        cesm2_members_per_macro: int = 10,
                        cesm2_macro_step: int = 20) -> list:
    """Generate n_members case name strings from a representative base_case.

    Three naming conventions are auto-detected:

    CESM1 / generic trailing .NNN
        'b.e11.B20TRC5CNBDRD.f09_g16.001'  → .001, .002, ...

    CESM2 LENS BHISTcmip6  (LE2-MMMM.NNN pattern, irregular macro years)
        'b.e21.BHISTcmip6.f09_g17.LE2-1001.001'
        Block 1 (10): LE2-1001.001, LE2-1021.002, ..., LE2-1181.010
        Blocks 2-5 (10 each): LE2-1231.001-010, LE2-1251.001-010,
                               LE2-1281.001-010, LE2-1301.001-010

    CESM2 LENS other  (LE2-MMMM.NNN pattern, uniform macro step)
        'b.e21.BHISTsmbb.f09_g17.LE2-1011.001'
        Inner NNN cycles 001..cesm2_members_per_macro, then
        MMMM increments by cesm2_macro_step.

    E3SMv2  (trailing _NNNN number)
        'v2.FV1.historical_0101'  → increments by e3sm_step from parsed start.

    E3SMv3  (trailing .enNN suffix)
        'v3.LR.historical.en00'  → .en00, .en01, .en02, ...

    Parameters
    ----------
    base_case            : representative case name (used to infer convention)
    n_members            : total number of names to generate
    e3sm_step            : E3SMv2 member-number increment (default 10)
    cesm2_members_per_macro : members per macro-year group (default 10)
    cesm2_macro_step     : CESM2 macro-year increment (default 20)
    """
    ctype = infer_ctype(base_case)

    if ctype == 'cesm2' and 'LE2-' in base_case:
        m = _re.search(r'LE2-(\d+)\.(\d+)', base_case)
        if not m:
            raise ValueError(f'Cannot parse LE2-MMMM.NNN from {base_case!r}')
        prefix = base_case[:m.start()]

        if 'BHISTcmip6' in base_case:
            # Block 1: each of the first 10 members has its own macro year
            # (1001, 1021, ..., 1181) with matching NNN (001..010).
            # Blocks 2-5: 10 members each from macro years 1231, 1251, 1281, 1301.
            _b1 = [(1001 + i * 20, i + 1) for i in range(10)]
            _b2 = [(macro, mem)
                   for macro in [1231, 1251, 1281, 1301]
                   for mem in range(1, 11)]
            _all = _b1 + _b2
            if n_members > len(_all):
                raise ValueError(
                    f'BHISTcmip6 only has {len(_all)} known members; '
                    f'requested {n_members}')
            return [f'{prefix}LE2-{macro:04d}.{mem:03d}'
                    for macro, mem in _all[:n_members]]

        # Generic CESM2-LE: uniform macro step
        macro_start = int(m.group(1))
        names, count, macro = [], 0, macro_start
        while count < n_members:
            for mem in range(1, cesm2_members_per_macro + 1):
                if count >= n_members:
                    break
                names.append(f'{prefix}LE2-{macro:04d}.{mem:03d}')
                count += 1
            macro += cesm2_macro_step
        return names

    elif ctype == 'e3smv2':
        m = _re.search(r'_(\d+)$', base_case)
        if not m:
            raise ValueError(f'Cannot parse trailing _NNNN from {base_case!r}')
        start, width = int(m.group(1)), len(m.group(1))
        prefix = base_case[:m.start()]
        return [f'{prefix}_{start + i * e3sm_step:0{width}d}' for i in range(n_members)]

    elif ctype == 'e3smv3':
        # E3SMv3 suffix is .enNN  (e.g. 'v3.LR.historical.en00' → en00, en01, en02 ...)
        m = _re.search(r'\.en(\d+)$', base_case)
        if not m:
            raise ValueError(f'Cannot parse trailing .enNN from {base_case!r}')
        start, width = int(m.group(1)), len(m.group(1))
        prefix = base_case[:m.start()]
        return [f'{prefix}.en{start + i:0{width}d}' for i in range(n_members)]

    elif ctype in ('cam5', 'cam6'):
        # CAM5/CAM6 GOGA suffix is .ensNN  (e.g. '...toga.ens01' → ens01, ens02, ...)
        m = _re.search(r'\.ens(\d+)$', base_case)
        if not m:
            raise ValueError(f'Cannot parse trailing .ensNN from {base_case!r}')
        start, width = int(m.group(1)), len(m.group(1))
        prefix = base_case[:m.start()]
        return [f'{prefix}.ens{start + i:0{width}d}' for i in range(n_members)]

    else:  # CESM1 / generic trailing .NNN
        m = _re.search(r'\.(\d+)$', base_case)
        if not m:
            raise ValueError(f'Cannot parse trailing .NNN from {base_case!r}')
        prefix, width = base_case[:m.start()], len(m.group(1))
        return [f'{prefix}.{i:0{width}d}' for i in range(1, n_members + 1)]


def monthly_anom(da):
    """Remove monthly climatology from a DataArray."""
    clim = da.groupby('time.month').mean('time', keep_attrs=True)
    return da.groupby('time.month') - clim


# ─────────────────────────────────────────────────────────────────────────────
# Time trimming
# ─────────────────────────────────────────────────────────────────────────────

def _normalize_time(da):
    """Reset time coordinate to YYYY-MM-01 to ensure consistent alignment."""
    import pandas as pd
    new_time = pd.DatetimeIndex([
        pd.Timestamp(int(y), int(m), 1)
        for y, m in zip(da.time.dt.year.values, da.time.dt.month.values)
    ])
    return da.assign_coords(time=new_time)


def dataset_ts_trim(da_x, da_y, yr0, yr1, var_x, var_y, get_months):
    """Check time overlap and trim both arrays to a common period."""

    print('  - Requested variable year range = ', yr0, '-', yr1)

    print('  - Time details for x-axis - ' + var_x)
    fyr0 = da_x.time.min().dt.strftime('%Y.%b').item()
    fyr1 = da_x.time.max().dt.strftime('%Y.%b').item()
    print('    - Available year range = ', fyr0, '-', fyr1)

    print('  - Time details for y-axis - ' + var_y)
    fyr0 = da_y.time.min().dt.strftime('%Y.%b').item()
    fyr1 = da_y.time.max().dt.strftime('%Y.%b').item()

    da_x_min = da_x.time.min().values
    da_x_max = da_x.time.max().values
    da_y_min = da_y.time.min().values
    da_y_max = da_y.time.max().values

    print('    - Available year range = ', fyr0, '-', fyr1)

    if da_x_min != da_x_max and da_y_min != da_y_max:
        print('    - DataArray time range of axes do not match')

    # Subset to requested years and season
    tx_mask = ((da_x.time.dt.year  >= yr0) & (da_x.time.dt.year  <= yr1) &
               (da_x.time.dt.month.isin(get_months)))
    ty_mask = ((da_y.time.dt.year  >= yr0) & (da_y.time.dt.year  <= yr1) &
               (da_y.time.dt.month.isin(get_months)))

    # For piControl / simulation-year cases the requested calendar years won't match.
    # Fall back to season-only filtering so we don't end up with empty arrays.
    if tx_mask.sum() == 0:
        print(f'    *** WARNING: no {var_x} data in years {yr0}–{yr1}; '
              f'using all available years (simulation-year dataset?)')
        tx_mask = da_x.time.dt.month.isin(get_months)
    if ty_mask.sum() == 0:
        print(f'    *** WARNING: no {var_y} data in years {yr0}–{yr1}; '
              f'using all available years (simulation-year dataset?)')
        ty_mask = da_y.time.dt.month.isin(get_months)

    da_x = da_x.sel(time=tx_mask)
    da_y = da_y.sel(time=ty_mask)

    # Align to common (year, month) pairs — handles day-of-month mismatches
    x_ym = set(zip(da_x.time.dt.year.values.tolist(), da_x.time.dt.month.values.tolist()))
    y_ym = set(zip(da_y.time.dt.year.values.tolist(), da_y.time.dt.month.values.tolist()))
    common_ym = x_ym & y_ym

    xm = np.array([(int(y), int(m)) in common_ym
                   for y, m in zip(da_x.time.dt.year.values, da_x.time.dt.month.values)])
    ym = np.array([(int(y), int(m)) in common_ym
                   for y, m in zip(da_y.time.dt.year.values, da_y.time.dt.month.values)])

    da_x = da_x.sel(time=xm)
    da_y = da_y.sel(time=ym)

    # Normalise to YYYY-MM-01 so xr.align works correctly across all callers
    da_x = _normalize_time(da_x)
    da_y = _normalize_time(da_y)

    print(f'    - Common time range: {str(da_x.time.values[0])[:10]} – '
          f'{str(da_x.time.values[-1])[:10]}  ({da_x.sizes["time"]} steps)')

    return da_x, da_y


# ─────────────────────────────────────────────────────────────────────────────
# Binning
# ─────────────────────────────────────────────────────────────────────────────

def bin_mean_var_by_level(
    field_ts,        # (time, lev) or (lev, time) — already aligned with index_ts
    index_ts,        # (time,)
    bins,
    bin_centers,
    lev_dim="plev",
    time_dim="time",
):
    """
    Bin a 3-D field by a 1-D index and return mean, std and counts per bin.

    Precondition: field_ts and index_ts must have the same time length.
    Use xr.align(..., join='inner') at the call site if needed.

    Returns
    -------
    mean_da   : DataArray (lev, bin)
    std_da    : DataArray (lev, bin)
    count_arr : ndarray  (bin,)
    """
    x    = index_ts.values
    y    = field_ts.transpose(time_dim, lev_dim).values  # (time, lev)
    levs = field_ts[lev_dim].values

    bin_idx   = np.digitize(x, bins) - 1   # 0-based
    nbins     = len(bin_centers)
    nlevs     = len(levs)

    mean_arr  = np.full((nlevs, nbins), np.nan)
    std_arr   = np.full((nlevs, nbins), np.nan)
    count_arr = np.zeros(nbins, dtype=int)

    for b in range(nbins):
        mask = bin_idx == b
        count_arr[b] = int(mask.sum())
        if mask.any():
            vals           = y[mask, :]
            mean_arr[:, b] = np.nanmean(vals, axis=0)
            std_arr[:, b]  = np.nanstd(vals,  ddof=1, axis=0)

    coords  = {lev_dim: levs, "bin": bin_centers}
    mean_da = xr.DataArray(mean_arr, coords=coords, dims=[lev_dim, "bin"])
    std_da  = xr.DataArray(std_arr,  coords=coords, dims=[lev_dim, "bin"])

    return mean_da, std_da, count_arr


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly timeseries and PDFs
# ─────────────────────────────────────────────────────────────────────────────

def nino_anom_ts(da_axis, nino_reg, axis_vals, lev_dim='plev', kde_bw=0.5):
    """
    Compute area-weighted Nino anomaly timeseries and KDE PDFs.

    Parameters
    ----------
    da_axis   : DataArray  monthly data, 2-D (time, lat, lon) or
                           3-D (time, lev, lat, lon)
    nino_reg  : str        Nino region key ('nino34', etc.)
    axis_vals : 1-D array  x-axis values at which to evaluate PDFs
    lev_dim   : str        name of the vertical dimension (default 'plev')

    Returns
    -------
    nino_ts  : DataArray  anomaly time series  — (time,) or (lev, time)
    nino_pdf : DataArray  KDE PDF(s)           — (axis,) or (lev, axis)
    """
    nino_w, nino_e, nino_n, nino_s = nino_region(nino_reg)

    nino_axis  = da_axis.sel(lat=slice(nino_s, nino_n), lon=slice(nino_w, nino_e))
    weights    = np.cos(np.deg2rad(nino_axis.lat))
    nino_waxis = nino_axis.weighted(weights).mean(dim=["lat", "lon"])

    clim       = nino_waxis.groupby("time.month").mean("time")
    nino_anom  = nino_waxis.groupby("time.month") - clim

    # ── PDF helper ────────────────────────────────────────────────────────
    def _kde_da(x, lev_val=None):
        x = x[np.isfinite(x)]
        if np.all(x == 0):   # e.g. CLOUD in stratosphere — skip KDE
            pdf_vals = np.full_like(axis_vals, np.nan)
        else:
            pdf_vals = gaussian_kde(x, bw_method=kde_bw)(axis_vals)
        da = xr.DataArray(pdf_vals, coords={"axis": axis_vals}, dims=["axis"])
        if lev_val is not None:
            da = da.expand_dims({lev_dim: [lev_val]})
        return da

    # ── PDF construction ──────────────────────────────────────────────────
    if lev_dim in nino_anom.dims:
        # Compute the full array once to avoid repeated Dask materialisation
        nino_np  = nino_anom.compute()
        levs     = nino_np[lev_dim].values
        nino_arr = nino_np.transpose(lev_dim, "time").values  # (lev, time)
        nino_pdf = xr.concat(
            [_kde_da(nino_arr[il], lev_val=lev) for il, lev in enumerate(levs)],
            dim=lev_dim,
        )
    else:
        nino_pdf = _kde_da(nino_anom.values.flatten())

    return nino_anom, nino_pdf


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def plot_latlon(ax, fig, icase, var, pvar, vunits, case, cname, seas, pregion,
                nino_reg, mask_ocn, lanom_plot, last_plot, fscale,
                levels=None, mask_thresh=0.5,
                equator_line=False, transparent_zero=False):

    preg_area = get_pregion(pregion)

    lat_min = preg_area['lat_min']
    lat_max = preg_area['lat_max']
    lon_min = preg_area['lon_min']
    lon_max = preg_area['lon_max']

    pvar    = pvar.sortby('lat').sel(lat=slice(lat_min, lat_max),
                                    lon=slice(lon_min, lon_max))
    dpproj  = ccrs.PlateCarree()

    if pregion in ['NPolar', 'SPolar']:
        theta      = np.linspace(0, 2 * np.pi, 100)
        center     = np.array([0.5, 0.5])
        circle     = np.vstack([np.sin(theta), np.cos(theta)]).T
        ax.set_boundary(mpath.Path(circle * 0.5 + center), transform=ax.transAxes)
    else:
        if pregion != 'Global':
            ax.set_yticks(np.arange(lat_min, lat_max + 10.0, 10.0), crs=dpproj)
            ax.yaxis.set_major_formatter(LatitudeFormatter())
            ax.set_xticks(np.arange(lon_min, lon_max + 30.0, 30.0), crs=dpproj)
            ax.xaxis.set_major_formatter(LongitudeFormatter())

    if levels is None:
        levels = np.array([-50., -40., -30, -25, -20, -15, -10,
                            0, 10, 15, 20, 25, 30, 40, 50.]) * 1.

    # Build colormap — optionally make the two bands nearest zero transparent
    _base = plt.get_cmap('RdBu_r')
    _n    = len(levels) - 1                                   # number of filled intervals
    _cols = [_base(i / (_n - 1)) for i in range(_n)]
    if transparent_zero and 0.0 in levels:
        _zi = int(np.searchsorted(levels, 0.0))               # index of the 0 level
        _cols[_zi - 1] = (1., 1., 1., 0.)                    # interval just below 0
        _cols[_zi]     = (1., 1., 1., 0.)                    # interval just above 0
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmapo = ListedColormap(_cols)
    cmapo.set_bad('lightgray')
    cmapo.set_under(_base(0.))
    cmapo.set_over(_base(1.))
    _norm = BoundaryNorm(levels, _n)

    pplot = ax.contourf(pvar.lon, pvar.lat, pvar,
                        transform=dpproj, levels=levels, norm=_norm,
                        extend='both', cmap=cmapo)

    # Horizontal colour bar below the axes
    cbar_ax = ax.inset_axes([0.02, -0.15, 0.9, 0.05], transform=ax.transAxes)
    cbar = fig.colorbar(pplot, cax=cbar_ax, orientation='horizontal')
    cbar.set_label(vunits)

    levels_nozero = [l for l in levels if l != 0]
    ax.contour(pvar.lon, pvar.lat, pvar,
               transform=dpproj, levels=levels_nozero,
               colors='black', linewidths=0.4)

    ax.set_title(r"$\bf{" + case + "}$ - " + var + "  (" + nino_reg + ") - " + seas)
    ax.text(0.01, 0.99, cname, transform=ax.transAxes,
            ha="left", va="top", fontweight="bold", fontsize=10 * fscale)

    ax.coastlines(linewidth=1, color='black', resolution='110m')
    ax.gridlines()
    ax.add_feature(cartopy.feature.LAND, zorder=0, linewidth=1.2)
    ax.add_feature(cfeature.LAND, facecolor='silver')

    if pregion not in ['NPolar', 'SPolar', 'Global']:
        ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=dpproj)

    if pregion == "US":
        ax.add_feature(cfeature.STATES, linewidth=0.5, edgecolor="black")

    # Bold equator line
    if equator_line:
        ax.plot([lon_min, lon_max], [0., 0.],
                transform=dpproj, color='black', linewidth=2.0,
                linestyle='-', zorder=7)

    # Niño region box
    _nw, _ne, _nn, _ns = nino_region(nino_reg)
    ax.plot([_nw, _ne, _ne, _nw, _nw],
            [_ns, _ns, _nn, _nn, _ns],
            transform=dpproj, color='red', linewidth=1.5,
            linestyle='--', zorder=6)


# ─────────────────────────────────────────────────────────────────────────────
# Plot domain / axis ranges
# ─────────────────────────────────────────────────────────────────────────────

def fig_domains(vname):
    """Return (vmin, vmax, axis_vals) for a given variable name."""
    match vname:
        case 'TS':
            vmin, vmax = -4., 4.
        case 'PRECT':
            vmin, vmax = -4., 4.
        case 'TAUX':
            vmin, vmax = -0.08, 0.08
        case 'OMEGA500':
            vmin, vmax = -0.8, 0.8
        case _ if vname in ['DTCOND500', 'DTCOND300', 'DTCOND700']:
            vmin, vmax = -3., 3.
        case _:
            print('  - Variable not recognized ' + vname)
    return vmin, vmax, np.linspace(vmin, vmax, 100)


# ─────────────────────────────────────────────────────────────────────────────
# CAM pressure-level interpolation helpers
# ─────────────────────────────────────────────────────────────────────────────

def cam_2d_from_3d(var_2d, var_3dget, var_plev, files_suff=None, ds_h0=None):
    """Extract a single pressure-level slice from a 3-D CAM variable."""

    var_derived = ['DIV']

    if files_suff is not None:   # CESM1/2 tseries

        files_ps = files_suff.replace(var_2d, 'PS')
        ds_ps = xr.open_mfdataset(files_ps, parallel=True,
                                  combine="by_coords",
                                  data_vars="minimal", coords="minimal")

        if var_2d not in var_derived:
            files_var = files_suff.replace(var_2d, var_3dget)
            ds_var = xr.open_mfdataset(files_var, parallel=True,
                                       combine="by_coords",
                                       data_vars="minimal", coords="minimal")
            da_out = get_plev_cam(ds_var, ds_ps,
                                  np.array([float(var_plev) * 100.]), var_3dget)
        else:
            if var_2d == 'DIV':
                da_u = xr.open_mfdataset(files_suff.replace(var_2d, 'U'),
                                         parallel=True, combine="by_coords",
                                         data_vars="minimal", coords="minimal")
                da_v = xr.open_mfdataset(files_suff.replace(var_2d, 'V'),
                                         parallel=True, combine="by_coords",
                                         data_vars="minimal", coords="minimal")
                da_uplev = get_plev_cam(da_u, ds_ps,
                                        np.array([float(var_plev) * 100.]), 'U')
                da_vplev = get_plev_cam(da_v, ds_ps,
                                        np.array([float(var_plev) * 100.]), 'V')
                da_out = calc_div(da_uplev, da_vplev)

    if ds_h0 is not None:   # CESM3 h0a files

        if var_2d not in var_derived:
            da_out = get_plev_cam(ds_h0, ds_h0,
                                  np.array([float(var_plev) * 100.]), var_3dget)
        else:
            if var_2d == 'DIV':
                da_uplev = get_plev_cam(ds_h0, ds_h0,
                                        np.array([float(var_plev) * 100.]), 'U')
                da_vplev = get_plev_cam(ds_h0, ds_h0,
                                        np.array([float(var_plev) * 100.]), 'V')
                da_out = calc_div(da_uplev, da_vplev)

    return da_out


def get_plev_cam(ds_var, ds_ps, var_plev, var_2interp):
    """Interpolate a 3-D CAM field from hybrid-sigma to pressure levels."""
    print('-Interpolating ', var_2interp, ' to ', var_plev, ' mb')

    da_var = ds_var[var_2interp]
    hyam   = ds_var['hyam']
    hybm   = ds_var['hybm']
    p0     = ds_var['P0'] if 'P0' in ds_var else xr.DataArray(100000.)
    da_ps  = ds_ps['PS']

    if hyam.ndim == 2: hyam = hyam[0]
    if hybm.ndim == 2: hybm = hybm[0]

    da_var = da_var.chunk({"lev": -1})

    da_var = gc.interp_hybrid_to_pressure(
        da_var, da_ps, hyam, hybm, p0=p0,
        new_levels=var_plev, method='log')

    # Rescale to hPa
    da_var = da_var.assign_coords(plev=0.01 * da_var.plev)

    print('Done')
    return da_var


# ─────────────────────────────────────────────────────────────────────────────
# Plot region domains
# ─────────────────────────────────────────────────────────────────────────────

def get_pregion(pregion):
    """Return a dict of lat/lon bounds and scale factors for a named region."""
    reg_info = {
        'Global':  {'lat_min': -90,  'lat_max': 90,   'lon_min': 0.,   'lon_max': 360., 'plev_scale': 0.2, 'aplev_scale': 0.2},
        'LabSea':  {'lat_min':  35,  'lat_max': 70,   'lon_min': 280., 'lon_max': 340., 'plev_scale': 0.2, 'aplev_scale': 0.2},
        'IO':      {'lat_min': -10,  'lat_max': 35,   'lon_min': 50.,  'lon_max': 120., 'plev_scale': 1.,  'aplev_scale': 1.},
        'US':      {'lat_min':  25,  'lat_max': 55,   'lon_min': -120.,'lon_max': -70., 'plev_scale': 0.25,'aplev_scale': 0.25},
        'SAm':     {'lat_min': -40,  'lat_max': 15,   'lon_min': 250., 'lon_max': 330., 'plev_scale': 0.5, 'aplev_scale': 0.5},
        'Aus':     {'lat_min': -20,  'lat_max': 10,   'lon_min': 120., 'lon_max': 150., 'plev_scale': 0.5, 'aplev_scale': 0.5},
        'TP':      {'lat_min': -10,  'lat_max': 10,   'lon_min': 120., 'lon_max': 290., 'plev_scale': 0.5, 'aplev_scale': 0.5},
        'WP':      {'lat_min': -20,  'lat_max': 40,   'lon_min': 110., 'lon_max': 270., 'plev_scale': 0.5, 'aplev_scale': 0.5},
        'IndoPac': {'lat_min': -40,  'lat_max': 40,   'lon_min': 40.,  'lon_max': 200., 'plev_scale': 0.5, 'aplev_scale': 0.5},
        'Tropics': {'lat_min': -45,  'lat_max': 45,   'lon_min': 0.,   'lon_max': 360., 'plev_scale': 0.8, 'aplev_scale': 1.},
        'EPac':    {'lat_min': -30,  'lat_max': 30,   'lon_min': 200., 'lon_max': 280., 'plev_scale': 0.8, 'aplev_scale': 1.},
        'NPac':    {'lat_min': -10,  'lat_max': 80,   'lon_min': 120., 'lon_max': 310., 'plev_scale': 0.8, 'aplev_scale': 1.},
        'Boreal':  {'lat_min':  40,  'lat_max': 75,   'lon_min': 190., 'lon_max': 280., 'plev_scale': 0.8, 'aplev_scale': 1.},
        'NPolar':  {'lat_min':  60., 'lat_max': 90.,  'lon_min':-180., 'lon_max': 180., 'plev_scale': 0.8, 'aplev_scale': 1.},
        'TropPac': {'lat_min': -15., 'lat_max': 70.,  'lon_min':  90., 'lon_max': 300., 'plev_scale': 0.8, 'aplev_scale': 1.},
    }
    return reg_info[pregion]


# ─────────────────────────────────────────────────────────────────────────────
# Dynamics helpers
# ─────────────────────────────────────────────────────────────────────────────

def _prep_uv_for_windspharm(u, v):
    """
    Flip to north-to-south and interpolate to a windspharm-compatible lat grid.
    Must be called before constructing VectorWind.
    """
    if float(u.lat[0]) < float(u.lat[-1]):   # south-to-north → flip
        u = u.isel(lat=slice(None, None, -1))
        v = v.isel(lat=slice(None, None, -1))

    nlat = len(u.lat)
    if nlat % 2:
        std_lats = np.linspace(90., -90., nlat)
    else:
        delta    = 180. / nlat
        std_lats = np.linspace(90. - 0.5 * delta, -90. + 0.5 * delta, nlat)

    u = u.compute().interp(lat=std_lats, kwargs={"fill_value": "extrapolate"})
    v = v.compute().interp(lat=std_lats, kwargs={"fill_value": "extrapolate"})
    return u, v


def calc_div(da_u, da_v, scale=1.e6):
    """
    Finite-difference horizontal divergence.

    Parameters
    ----------
    da_u, da_v : DataArray  (time, lat, lon) or (lat, lon)
    scale      : float      output scale factor (default 1e6 → 1e-6 s⁻¹)
    """
    erad   = 6.371e6
    lat    = np.deg2rad(da_u.lat)
    coslat = np.cos(lat).where(np.abs(np.cos(lat)) > 1e-3)
    dlon   = np.deg2rad(float(da_u.lon.diff("lon").mean()))
    dlat   = np.deg2rad(float(da_u.lat.diff("lat").mean()))
    return (da_u.diff("lon") / dlon / (erad * coslat) +
            da_v.diff("lat") / dlat / erad) * scale


def calc_rws(u, v):
    """
    Rossby Wave Source at a single pressure level.

    RWS = -ζ_a · D - V_χ · ∇ζ_a

    Parameters
    ----------
    u, v : DataArray  (time, lat, lon)  full-globe regular/Gaussian grid

    Returns
    -------
    rws : DataArray, units s⁻²
    """
    u, v = _prep_uv_for_windspharm(u, v)
    w    = VectorWind(u, v)

    eta        = w.absolutevorticity()
    div        = w.divergence()
    uchi, vchi = w.irrotationalcomponent()
    etay, etax = w.gradient(eta)

    return -eta * div - (uchi * etax + vchi * etay)


def calc_div_wind(u, v):
    """
    Divergent (irrotational) wind components and divergence via spherical harmonics.

    Parameters
    ----------
    u, v : DataArray  (time, lat, lon)

    Returns
    -------
    u_chi : DataArray  divergent zonal wind      (m s⁻¹)
    v_chi : DataArray  divergent meridional wind  (m s⁻¹)
    div   : DataArray  divergence                 (s⁻¹)
    """
    u, v         = _prep_uv_for_windspharm(u, v)
    w            = VectorWind(u, v)
    u_chi, v_chi = w.irrotationalcomponent()
    div          = w.divergence()
    return u_chi, v_chi, div


# ─────────────────────────────────────────────────────────────────────────────
# ERA5 pressure levels
# ─────────────────────────────────────────────────────────────────────────────

def grab_era5_levs():
    """Return the 16 ERA5 pressure levels (Pa) used for interpolation."""
    return np.array([1000, 925, 850, 700, 600, 500,
                     400, 300, 250, 200, 150, 100,
                     70, 50, 30, 20]) * 100.


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loader  (DO NOT MODIFY — routing logic is case-specific)
# ─────────────────────────────────────────────────────────────────────────────

''' Read in Different Datasets For each Axis '''



def get_dataset(case,case_type,var_axis,yr0,yr1,lread_in_all_hist,lwrite_ts_file ,lread_ts_file):

    cvars =      ['TS','TAUX','PRECT',   'OMEGA500', 'DTCOND300',' DTCOND500','DTCOND700','OMEGA','RELHUM','Q',    'DIV',  'Z3', 'U', 'V','T','CLOUD','DTCOND','DCQ']
    cvar_scales = [1.,  1.,  86400.*1000.   ,36.      ,86400.,     86400.      ,86400.,     36.,     1.,  1000.,   1.e+6,   1.,   1.,  1., 1., 100., 86400.,86400.*1000]
    cvar_scale = cvar_scales[cvars.index(var_axis)]

    evars = ['sst','avg_iews','tp',   'w',      'd',      'd',      'd', 'w', 'r', 'q', 'd', 'z', 'u','v','t','cc','mmpdt','mmpdq']
    efvars = ['sst','taux','prect','omega500','div200','div300','div400', 'omega', 'rh', 'q','div','z','u','v','t','cloud','dtdt_param','dqdt_param']


    lvar_from3d = False
    var_3dget = var_axis

# Variable requiring 3D interp.

    if var_axis[-3:].isdigit():  # Test if the last 3 letters of the variable name are a digit (e.g., 200)
        lvar_from3d = True
        var_plev = var_axis[-3:]   # Extract pressure level (e.g., '200' from 'Z3200')
        var_3dget = var_axis[:-3]  # Extract the variable name (e.g., 'Z3' from 'Z3200')



    match case:


        case _ if case_type == 'OBS':


            # Selct obs. source.

            print('  - Grabbing ',case,' data for',var_axis)

            match case:

                case 'ERA5':



                    dir_era5 = '/glade/derecho/scratch/rneale/ERA5/mmean/1deg/'


# Some ERA5 variable mappings to CESM vars.

                    evar = evars[cvars.index(var_axis)]
                    efvar = efvars[cvars.index(var_axis)]

                    da_axis = xr.open_dataset(dir_era5+efvar+'/'+efvar+'_era5_monthly_1x1.nc')[evar]

                    if 'valid_time' in da_axis.dims:
                        da_axis = da_axis.rename({'valid_time': 'time'})

                    # Change pressure level name.
                    da_axis = da_axis.rename({"pressure_level": "plev"}) if "pressure_level" in da_axis.dims else da_axis

                    vscale = 1.


                    if var_axis in ['TS','T']:    vscale = 1.
                    if var_axis in ['Z3']:    vscale = 0.1
                    if var_axis in ['PRECT']: vscale = 1000.
                    if var_axis in ['TAUX','TAUY']:  vscale = -1
                    if var_axis in ['OMEGA500','OMEGA']:  vscale = 36.
                    if var_axis in ['Q']:  vscale = 1000.
                    if var_axis in ['DIV']:  vscale = 1.e+6
                    if var_axis in ['CLOUD']:  vscale = 100.
                    if var_axis in ['DTCOND','DTCOND300','DTCOND500','DTCOND700']:  vscale = 86400.  # K s-1 -> K day-1


                case 'MERRA2':

                    dir_merra2 = '/glade/derecho/scratch/rneale/MERRA2/mmean/1deg/'

                    # Reuse identical ERA5 variable mappings — output files
                    # were written with the same short names (sst, w, r, …).
                    evar  = evars[cvars.index(var_axis)]
                    efvar = efvars[cvars.index(var_axis)]

                    da_axis = xr.open_dataset(
                        dir_merra2 + efvar + '/' + efvar + '_merra2_monthly_1x1.nc'
                    )[evar]

                    if 'valid_time' in da_axis.dims:
                        da_axis = da_axis.rename({'valid_time': 'time'})

                    da_axis = da_axis.rename({"pressure_level": "plev"}) \
                        if "pressure_level" in da_axis.dims else da_axis

                    # Same scale factors as ERA5
                    vscale = 1.
                    if var_axis in ['TS', 'T']:   vscale = 1.
                    if var_axis in ['Z3']:         vscale = 1.      # MERRA2 z is geopotential height (m), not geopotential (m2/s2)
                    if var_axis in ['PRECT']:      vscale = 86400.  # kg/m²/s → mm/day
                    if var_axis in ['TAUX','TAUY']: vscale = -1.
                    if var_axis in ['OMEGA500','OMEGA']: vscale = 36.  # Pa/s → mb/hr
                    if var_axis in ['Q']:          vscale = 1000.   # kg/kg → g/kg
                    if var_axis in ['DIV']:        vscale = 1.e+6
                    if var_axis in ['CLOUD']:      vscale = 1.      # already % after processing
                    if var_axis in ['DTCOND','DTCOND300','DTCOND500','DTCOND700']: vscale = 86400.

                case 'TROPFLUX' if var_axis=='TAUX':

                    da_axis = xr.open_dataset(dir_data+'tropflux/taux_tropflux_1m_1979-2018.nc')['taux']
                    da_axis = da_axis.rename({'latitude': 'lat', 'longitude': 'lon'})
                    vscale = -1.

                case _:

                    print("  - No obs, product match for ",case_type)



        case _ if case_type in ['cesm1','cesm2','cam5','cam6']:

            lens_chunk = {
                                "time": 12,     # one chunk per file (monthly)
                                "lev": -1,      # keep full vertical column
                                "lat": 64,      # or 48 / 72 depending on grid
                                "lon": 64}

            vscale = cvar_scale



            if case_type == 'cesm1':
                dir_lens = '/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly/'
                fyrs_str = '.04*' if case == 'b.e11.B1850C5CN.f09_g16.005' else '*'

            if case_type == 'cesm2':
                dir_lens = '/glade/campaign/cgd/cesm/CESM2-LE/timeseries/atm/proc/tseries/month_1/'
                fyrs_str = '*'

            if case_type == 'cam5':
                dir_lens = '/glade/campaign/cesm/development/cvcwg/cvwg/f.e11.FAMIPC5CN.f09_f09.hist-rcp85.toga/atm/proc/tseries/monthly/'
                fyrs_str = '*'


            if case_type == 'cam6':
                dir_lens = '/glade/campaign/collections/gdex/data/d651010/global/CESM2.1_GOGA_ERSSTv5/atm/proc/tseries/month_1/'
                fyrs_str = '*'





            file_suff = var_axis+'/'+case+'.cam.h0.'+var_axis+fyrs_str+'.nc'
            files_hist = dir_lens+file_suff




            files_ls  = glob.glob(files_hist)




            print(f'  - Grabbing file(s) for {case_type.upper()}')

#            da_axis = xr.open_mfdataset(files_ls,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal")[var_axis]


            if case_type in ['cesm1','cam5','cam6'] and var_axis == 'PRECT': # CAM5/CAM6/CESM1: no PRECT — need PRECC+PRECL

                files_hist_pc = files_hist.replace('PRECT', 'PRECC')
                files_hist_pl = files_hist.replace('PRECT', 'PRECL')

                files_ls_pc  = glob.glob(files_hist_pc)
                files_ls_pl  = glob.glob(files_hist_pl)

                da_pc = xr.open_mfdataset(files_ls_pc,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal",chunks=lens_chunk)['PRECC']
                da_pl = xr.open_mfdataset(files_ls_pl,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal",chunks=lens_chunk)['PRECL']

                da_axis = da_pc+da_pl

            else:

                # If empty file then try my derived directory
                if not files_ls:
                    print('-Checking for local CESM copy, likely a derived variable if it exists')
                    files_hist = dir_ncout+file_suff
                    files_ls  = glob.glob(files_hist)



                # Now try a 3D derived data grab
                if not files_ls: # Either the variable just isn't there or we have to derive it from a 3D variable

                    if lvar_from3d: # Test if the last 3 letters of the variable name are a digit (e.g., 200) to determine that it is a single level

                        da_axis = cam_2d_from_3d(var_axis,var_3dget,var_plev,files_suff=dir_lens+var_3dget+'/'+case+'.cam.h0.'+var_3dget+fyrs_str+'.nc')

                    else:
                        raise FileNotFoundError(
                            f'No files found for {case!r} / {var_axis!r}\n'
                            f'  Searched: {files_hist!r}'
                        )

                else: # File(s) exist!


                    ds_axis = xr.open_mfdataset(files_ls,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal",compat='override',chunks=lens_chunk)


                    if "lev" in ds_axis[var_axis].dims:
                        file_psuff = 'PS/'+case+'.cam.h0.PS'+fyrs_str+'.nc'
                        files_phist = dir_lens+file_psuff
                        files_pls  = glob.glob(files_phist)
                        ds_ps = xr.open_mfdataset(files_pls,parallel=True, combine="by_coords",data_vars="minimal", coords="minimal",compat='override',chunks=lens_chunk)

                        da_axis = get_plev_cam(ds_axis,ds_ps,grab_era5_levs(),var_axis)
                    else:
                        da_axis = ds_axis[var_axis]




        case _ if case_type == 'cesm3':


            # Does a timeseries dataset exist for this case?
            # Check for this file name lwoer down


            ofile_case = dir_ncout+case+'_'+var_axis+'_mmeans_ts.nc'


            # add vscale here
            vscale = cvar_scale

            if os.path.exists(ofile_case) and lread_ts_file:

                print('  - Timeseries files exist for the case - using them')

                print('     ',ofile_case)

                da_axis = xr.open_dataset(ofile_case)[var_axis]


            else:

                # Pick the right directory (hannay/gmarques)

                dir_c3 = '/glade/derecho/scratch/hannay/archive/'

                if os.path.isdir(dir_c3+case):
                    print("   - Cecile's Run")
                # Your operation here, e.g. read/write files
                else:
                    print("   - Gustavo's Run")
                    dir_c3 = '/glade/derecho/scratch/gmarques/archive/'


    # Grab files and read in.

                # Trim down range of files to read in requested years, otherwise read in all.

                dir_hist = dir_c3+case+'/atm/hist/'

                # Monthly mean files only (h0a = monthly, skip h0i instantaneous etc.)
                file_list = sorted(glob.glob(dir_hist+case+'.cam.h0a.*.nc'))
                if not file_list:
                    file_list = sorted(glob.glob(dir_hist+case+'.cam.h0.*.nc'))
                print(f'   - Found {len(file_list)} monthly hist files in {dir_hist}')

                # Filter files by year range (YYYY-MM format in filename)
                def _cesm3_file_year(fpath):
                    """Return year from filename ...cam.h0a.YYYY-MM.nc, or None if unparseable."""
                    try:
                        date_str = os.path.basename(fpath).rsplit('.', 2)[-2]  # 'YYYY-MM'
                        return int(date_str[:4])
                    except (ValueError, IndexError):
                        return None

                if lread_in_all_hist:
                    filtered_files = file_list
                else:
                    filtered_files = [f for f in file_list
                                      if _cesm3_file_year(f) is not None
                                      and yr0 <= _cesm3_file_year(f) <= yr1]
                print(f'   - Using {len(filtered_files)} files for years {yr0}–{yr1}')


                # ne30 SE grid uses ncol, not lat/lon — chunk only on time
                lens_chunk_c3 = {"time": 12}

                _cache_key = (case, yr0, yr1, lread_in_all_hist)
                if _cache_key in _cesm3_ds_cache:
                    print('   - Using cached CESM3 dataset')
                    ds_hist = _cesm3_ds_cache[_cache_key]
                else:
                    ds_hist = xr.open_mfdataset(
                        filtered_files,
                        combine="by_coords",
                        data_vars="minimal",
                        coords="minimal",
                        compat='override',
                        parallel=True,
                        chunks=lens_chunk_c3
                    )
                    _cesm3_ds_cache[_cache_key] = ds_hist
                    print('   - Opened and cached CESM3 dataset')

                if "lev" in ds_hist[var_axis].dims:

                    ds_ps = ds_hist  # PS is in the same h0a files

                    if lvar_from3d:
                        da_axis = get_plev_cam(ds_hist,ds_ps,np.array([float(var_plev) * 100.]),var_3dget)
                    else:
                        da_axis = get_plev_cam(ds_hist,ds_ps,grab_era5_levs(),var_axis)
                else:
                    da_axis = ds_hist[var_axis]


                if lwrite_ts_file:

                    print("  - Write out files of 2D field "+var_axis)

                    print('    ',ofile_case)


                    da_axis.to_netcdf(ofile_case,mode="w")
                    da_axis.close()


                    print(' - Done')

        case _ if case_type == 'e3smv2':

            dir_e3sm = '/glade/campaign/cgd/ccr/E3SMv2/FV_regridded/' + case + '/atm/proc/tseries/month_1/'

            lens_chunk = {"time": 12, "lev": -1, "lat": 64, "lon": 64}
            vscale = cvar_scale

            print('  - Grabbing E3SM file(s) for', var_axis)

            if var_axis == 'PRECT':  # E3SM stores PRECC + PRECL separately
                files_pc = glob.glob(dir_e3sm + case + '.eam.h0.PRECC.*.nc')
                files_pl = glob.glob(dir_e3sm + case + '.eam.h0.PRECL.*.nc')
                da_pc = xr.open_mfdataset(files_pc, parallel=True, combine="by_coords",
                                           data_vars="minimal", coords="minimal",
                                           chunks=lens_chunk)['PRECC']
                da_pl = xr.open_mfdataset(files_pl, parallel=True, combine="by_coords",
                                           data_vars="minimal", coords="minimal",
                                           chunks=lens_chunk)['PRECL']
                da_axis = da_pc + da_pl

            else:
                var_get = var_3dget if lvar_from3d else var_axis
                files_hist = glob.glob(dir_e3sm + case + '.eam.h0.' + var_get + '.*.nc')

                if not files_hist:
                    print('  - E3SM: no files found for', var_get)
                else:
                    ds_axis = xr.open_mfdataset(files_hist, parallel=True, combine="by_coords",
                                                 data_vars="minimal", coords="minimal",
                                                 compat='override', chunks=lens_chunk)

                    if "lev" in ds_axis[var_get].dims:
                        files_ps = glob.glob(dir_e3sm + case + '.eam.h0.PS.*.nc')
                        ds_ps = xr.open_mfdataset(files_ps, parallel=True, combine="by_coords",
                                                   data_vars="minimal", coords="minimal",
                                                   compat='override', chunks=lens_chunk)
                        if lvar_from3d:
                            da_axis = get_plev_cam(ds_axis, ds_ps,
                                                   np.array([float(var_plev) * 100.]), var_get)
                        else:
                            da_axis = get_plev_cam(ds_axis, ds_ps, grab_era5_levs(), var_get)
                    else:
                        da_axis = ds_axis[var_get]


        case _ if case_type == 'e3smv3':

            dir_e3smv3 = f'/glade/campaign/cgd/ccr/E3SMv3-LE/{case}/atm/proc/tseries/month_1/'
            lens_chunk = {"time": 12, "lev": -1, "lat": 64, "lon": 64}
            vscale = cvar_scale

            print('  - Grabbing E3SMv3 file(s) for', var_axis)

            if var_axis == 'PRECT':  # E3SMv3 stores PRECC + PRECL separately
                files_pc = glob.glob(dir_e3smv3 + case + '.PRECC.*.nc')
                files_pl = glob.glob(dir_e3smv3 + case + '.PRECL.*.nc')
                da_pc = xr.open_mfdataset(files_pc, parallel=True, combine="by_coords",
                                           data_vars="minimal", coords="minimal",
                                           chunks=lens_chunk)['PRECC']
                da_pl = xr.open_mfdataset(files_pl, parallel=True, combine="by_coords",
                                           data_vars="minimal", coords="minimal",
                                           chunks=lens_chunk)['PRECL']
                da_axis = da_pc + da_pl

            else:
                var_get = var_3dget if lvar_from3d else var_axis
                files_hist = glob.glob(dir_e3smv3 + case + '.' + var_get + '.*.nc')

                if not files_hist:
                    print('  - E3SMv3: no files found for', var_get)
                else:
                    ds_axis = xr.open_mfdataset(files_hist, parallel=True, combine="by_coords",
                                                 data_vars="minimal", coords="minimal",
                                                 compat='override', chunks=lens_chunk)

                    if "lev" in ds_axis[var_get].dims:
                        files_ps = glob.glob(dir_e3smv3 + case + '.PS.*.nc')
                        ds_ps = xr.open_mfdataset(files_ps, parallel=True, combine="by_coords",
                                                   data_vars="minimal", coords="minimal",
                                                   compat='override', chunks=lens_chunk)
                        if lvar_from3d:
                            da_axis = get_plev_cam(ds_axis, ds_ps,
                                                   np.array([float(var_plev) * 100.]), var_get)
                        else:
                            da_axis = get_plev_cam(ds_axis, ds_ps, grab_era5_levs(), var_get)
                    else:
                        da_axis = ds_axis[var_get]

            # E3SMv3 tseries plev is in Pa; convert to hPa for consistency
            if 'plev' in da_axis.dims and float(da_axis.plev.values.max()) > 2000.:
                da_axis = da_axis.assign_coords(plev=da_axis.plev / 100.)
                da_axis['plev'].attrs['units'] = 'hPa'


        case _:

            print('  - No case_type match for '+case)

# Squeeze out single value dimensions (usually pressure)
    da_axis = da_axis.squeeze()


# Just scale the variable right at the end

    da_axis = vscale *  da_axis



    return da_axis


# ─────────────────────────────────────────────────────────────────────────────
# Binned line plot with SST secondary axis
# ─────────────────────────────────────────────────────────────────────────────

_LINE_COLS = [
    "black", "red", "royalblue", "darkorange", "forestgreen",
    "firebrick", "goldenrod", "mediumpurple", "deepskyblue", "crimson"
]


def plot_binned_line_sst_bias(
        binned_list,
        bin_centers,
        case_labels,
        sst_clim_means,
        plev_sel=500.,
        plev_avg_range=None,
        var_label='',
        var_units='',
        sst_units='K',
        xnino_reg='nino34',
        season='',
        colors=None,
        markers=None,
        figsize=(13, 6),
        dir_fig=None,
        fname=None,
):
    """
    Line plot: binned 3D variable at a selected pressure level (or vertical
    mean) vs Nino3.4 SSTA bins.  Right y-axis shows the absolute Nino3.4 SST
    (°C) for each case (clim_mean + bin_center).

    Parameters
    ----------
    binned_list    : list of xr.DataArray  dims (plev, bin), one per case
    bin_centers    : 1D array  x-axis bin centres (SSTA in K)
    case_labels    : list of str  short case names
    sst_clim_means : list of float  mean Nino3.4 TS per case (K)
    plev_sel       : float or None  pressure level to select [hPa]
    plev_avg_range : [plo, phi]  hPa vertical mean range (when plev_sel=None)
    var_label      : str  variable name for y-axis label
    var_units      : str  units for left y-axis
    sst_units      : str  SSTA units for x-axis label (default 'K')
    xnino_reg      : str  nino region name
    season         : str  season string for title
    colors         : list of str  one per case
    figsize        : tuple
    dir_fig        : str  save directory; None = no save
    fname          : str  filename; None = auto

    Returns
    -------
    fig, ax_left, ax_right
    """
    if colors is None:
        colors = _LINE_COLS
    if markers is None:
        markers = ['o'] * len(binned_list)

    bin_centers = np.asarray(bin_centers)

    # ── Pressure-level extraction helper ──────────────────────────────────
    def _get_lev(b):
        if plev_sel is not None:
            return b.sel(plev=plev_sel, method='nearest').values
        elif plev_avg_range is not None:
            plo, phi = plev_avg_range
            return b.sel(plev=slice(max(plo, phi), min(plo, phi))).mean('plev').values
        else:
            return b.mean('plev').values

    if plev_sel is not None:
        plev_lbl = f'{int(plev_sel)} hPa'
    elif plev_avg_range is not None:
        plo, phi = plev_avg_range
        plev_lbl = f'{int(min(plo,phi))}–{int(max(plo,phi))} hPa mean'
    else:
        plev_lbl = 'vertical mean'

    fig, ax = plt.subplots(figsize=figsize)
    ax_right = ax.twinx()

    for binned_item, case, sst_mean_item, col, mrk in zip(
            binned_list, case_labels, sst_clim_means, colors, markers):

        is_ens = isinstance(binned_item, (list, tuple))
        sst_mean = (float(np.mean(sst_mean_item))
                    if (is_ens and hasattr(sst_mean_item, '__iter__'))
                    else float(sst_mean_item))

        if is_ens:
            # ── Thin semi-transparent lines for each member ────────────────
            member_1ds = [_get_lev(m) for m in binned_item]
            for v1d in member_1ds:
                ax.plot(bin_centers, v1d,
                        color=col, linewidth=0.8, alpha=0.25, zorder=2)
            # ── Thick ensemble-mean line ───────────────────────────────────
            ens_mean_1d = np.stack(member_1ds).mean(axis=0)
            ax.plot(bin_centers, ens_mean_1d,
                    color=col, linewidth=2.5, marker=mrk, markersize=6,
                    label=f'{case} (N={len(binned_item)})', zorder=3)
        else:
            var_1d = _get_lev(binned_item)
            ax.plot(bin_centers, var_1d,
                    color=col, linewidth=2.5, marker=mrk, markersize=6, label=case)

        sst_abs = sst_mean + bin_centers - 273.15
        ax_right.plot(bin_centers, sst_abs,
                      color=col, linewidth=2.0, linestyle='--', alpha=0.75,
                      label=f'{case}  (mean={sst_mean - 273.15:.1f}°C)')

    ax.axhline(0., color='gray', linestyle='--', linewidth=1.2, zorder=0)
    ax.axvline(0., color='gray', linestyle='--', linewidth=1.2, zorder=0)

    ax.set_xlabel(f'{xnino_reg} SST anomaly ({sst_units})', fontsize=13)
    ax.set_ylabel(f'{var_label} anomaly ({var_units})\n@ {plev_lbl}', fontsize=12)
    ax_right.set_ylabel(f'{xnino_reg} absolute SST (°C)\n[dashed — right axis]',
                        fontsize=11, color='dimgray')
    ax_right.tick_params(axis='y', colors='dimgray')

    title = f'{var_label} binned by {xnino_reg} SSTA'
    if plev_sel is not None:
        title += f'  |  {int(plev_sel)} hPa'
    if season:
        title += f'  —  {season}'
    ax.set_title(title, fontsize=13)

    ax.legend(loc='upper left', fontsize=9, title=var_label,
              title_fontsize=9, framealpha=0.8)
    ax_right.legend(loc='upper right', fontsize=8,
                    title='Absolute SST  (dashed)', title_fontsize=8, framealpha=0.8)

    ax.grid(True, alpha=0.35)
    fig.tight_layout()

    if dir_fig is not None:
        os.makedirs(dir_fig, exist_ok=True)
        if fname is None:
            fname = f'binned_line_{var_label}_{season}.png'
        fig.savefig(os.path.join(dir_fig, fname), dpi=150, bbox_inches='tight')

    return fig, ax, ax_right


# ─────────────────────────────────────────────────────────────────────────────
# Case display style helpers
# ─────────────────────────────────────────────────────────────────────────────

def build_case_styles(enso_names, enso_cases, loop_icases):
    """Build per-case colour, marker, and display-name lists for scatter plots.

    OBS cases → black/gray shades (first OBS: circle marker, subsequent: X).
    CESM3 → always purple.  Other models → cycling colour palette.

    Parameters
    ----------
    enso_names  : list  case name strings (or lists for multi-source OBS)
    enso_cases  : list  short case-type labels (e.g. 'OBS', 'CESM2', …)
    loop_icases : int   number of cases to process

    Returns
    -------
    scat_cols, scat_markers, scat_display_names : lists, one entry per case
    """
    _obs_shades = ['black', 'darkgray', 'silver']
    _model_cols = ['red', 'royalblue', 'darkorange', 'forestgreen',
                   'firebrick', 'goldenrod', 'mediumpurple', 'deepskyblue', 'crimson']
    _iobs, _imod = 0, 0
    scat_cols, scat_markers, scat_display_names = [], [], []
    for icn, _cn in enumerate(enso_names[:loop_icases]):
        _fcn = _cn[0] if isinstance(_cn, list) else _cn
        if infer_ctype(_cn) == 'OBS':
            scat_cols.append(_obs_shades[_iobs % len(_obs_shades)])
            scat_markers.append('X' if _iobs > 0 else 'o')
            scat_display_names.append(_fcn)
            _iobs += 1
        else:
            scat_cols.append('purple' if infer_ctype(_cn) == 'cesm3'
                             else _model_cols[_imod % len(_model_cols)])
            scat_markers.append('o')
            scat_display_names.append(enso_cases[icn])
            _imod += 1
    return scat_cols, scat_markers, scat_display_names


# ─────────────────────────────────────────────────────────────────────────────
# SST lag correlations
# ─────────────────────────────────────────────────────────────────────────────

def sst_lag_corr(nino_1d, sst_np, lag):
    """Vectorised lag correlation between a 1-D Niño index and a (time, lat, lon) SST array.

    Parameters
    ----------
    nino_1d : 1-D array (time,)
    sst_np  : ndarray  (time, lat, lon)
    lag     : int  positive → Niño leads SST (SST lags); negative → SST leads Niño

    Returns
    -------
    r : ndarray (lat, lon)  Pearson correlation coefficients
    """
    n    = len(nino_1d)
    alag = abs(lag)
    if lag > 0:
        x = nino_1d[:n - alag]; y = sst_np[alag:]
    elif lag < 0:
        x = nino_1d[alag:];     y = sst_np[:n - alag]
    else:
        x = nino_1d;             y = sst_np
    xn  = x - x.mean()
    yn  = y - y.mean(axis=0)
    cov = np.einsum("t,tlm->lm", xn, yn) / len(x)
    r   = cov / (xn.std() * yn.std(axis=0) + 1e-30)
    return r


def plot_sst_lag_corr_maps(da_x_all, xnino_reg, case_name, yr0, yr1,
                            dir_fig, fig_pref_user='', fscale=1.0,
                            lags=(-6, 0, 6)):
    """Plot and save SST lag-correlation maps at the requested lags.

    Parameters
    ----------
    da_x_all     : DataArray  Monthly SST (all months, not season-filtered)
    xnino_reg    : str  Niño region key
    case_name    : str  used in plot title and filename
    yr0, yr1     : int  year range label
    dir_fig      : str  output directory (created if absent)
    fig_pref_user: str  filename prefix
    fscale       : float  font scale factor
    lags         : iterable  lag values in months
    """
    _sst_anom = monthly_anom(da_x_all)

    _nw, _ne, _nn, _ns = nino_region(xnino_reg)
    _nino_reg = _sst_anom.sel(lat=slice(_ns, _nn), lon=slice(_nw, _ne))
    _wts      = np.cos(np.deg2rad(_nino_reg.lat))
    _nino_all = _nino_reg.weighted(_wts).mean(("lat", "lon")).compute()

    _sst_sub = _sst_anom.sel(lat=slice(-45, 45))
    if float(_sst_sub.lon.min()) < 0:
        _sst_sub = _sst_sub.assign_coords(lon=(_sst_sub.lon % 360)).sortby("lon")
    _sst_sub = _sst_sub.compute()
    _nw360 = _nw % 360; _ne360 = _ne % 360

    _nino_np = _nino_all.values
    _sst_np  = _sst_sub.values
    _dpproj  = ccrs.PlateCarree()

    _lag_labels = {
        -6: "Lag −6 months (SST leads)", -3: "Lag −3 months (SST leads)",
         0: "Lag 0  (simultaneous)",
         3: "Lag +3 months (SST lags)",   6: "Lag +6 months (SST lags)",
    }

    fig_lag, axes_lag = plt.subplots(
        len(lags), 1, figsize=(12, 3.5 * len(lags)),
        subplot_kw={"projection": ccrs.PlateCarree(central_longitude=180)},
        constrained_layout=True)
    if len(lags) == 1:
        axes_lag = [axes_lag]

    _clev = np.linspace(-1, 1, 21)
    for _ilag, lag in enumerate(lags):
        ax_l = axes_lag[_ilag]
        _r   = sst_lag_corr(_nino_np, _sst_np, lag)
        cf   = ax_l.contourf(_sst_sub.lon.values, _sst_sub.lat.values, _r,
                              levels=_clev, cmap="RdBu_r", extend="both",
                              transform=_dpproj)
        ax_l.coastlines(linewidth=0.7, color="k", resolution="110m")
        ax_l.add_feature(cfeature.LAND, facecolor="lightgray", zorder=4)
        ax_l.set_extent([0, 360, -45, 45], crs=_dpproj)
        ax_l.set_title(_lag_labels.get(lag, f"Lag {lag:+d} months"), fontsize=10 * fscale)
        ax_l.plot([_nw360, _ne360, _ne360, _nw360, _nw360],
                  [_ns, _ns, _nn, _nn, _ns],
                  transform=_dpproj, color="black", linewidth=1.5,
                  linestyle="--", zorder=6)
        plt.colorbar(cf, ax=ax_l, orientation="vertical",
                     fraction=0.02, pad=0.02, label="r")

    fig_lag.suptitle(
        f"{case_name}  ({int(yr0)}–{int(yr1)})  "
        f"{xnino_reg} SSTA lag correlations — all months",
        fontsize=11 * fscale, fontweight="bold")

    os.makedirs(dir_fig, exist_ok=True)
    _fname = (fig_pref_user + "_" if fig_pref_user else "") + case_name + \
             f"_{xnino_reg}_SST_lagcorr_{int(yr0)}-{int(yr1)}.png"
    fig_lag.savefig(os.path.join(dir_fig, _fname), dpi=150, bbox_inches="tight")
    plt.close(fig_lag)


# ─────────────────────────────────────────────────────────────────────────────
# Z500 seasonal scatter helpers
# ─────────────────────────────────────────────────────────────────────────────

def calc_z500_seasonal_scatter(da_z500_anom_reg, var_x_1d):
    """Compute NDJFM seasonal means of Z500 anomaly in a region and a paired Niño index.

    Groups Nov/Dec of year Y with Jan–Mar of year Y+1 as one NDJFM season,
    then derives the spatial max/min of each seasonal mean and an EOF PC1.

    Parameters
    ----------
    da_z500_anom_reg : DataArray (time, lat, lon)  region-subset monthly anomaly
    var_x_1d         : DataArray (time,)            Niño SSTA aligned to da_z500_anom_reg

    Returns
    -------
    dict with keys: ssta_seas, z500_tmax, z500_tmin, pc1, var_exp
    """
    _times     = pd.DatetimeIndex(da_z500_anom_reg.time.values)
    _winter_yr = np.where(_times.month >= 11, _times.year + 1, _times.year)
    _uniq_yrs  = np.unique(_winter_yr)

    _z500_seas = [da_z500_anom_reg.isel(time=(_winter_yr == yr)).mean('time')
                  for yr in _uniq_yrs]
    _ssta_seas = [float(var_x_1d.isel(time=(_winter_yr == yr)).mean('time'))
                  for yr in _uniq_yrs]

    z500_seas = xr.concat(_z500_seas, dim='season')
    ssta_seas = np.array(_ssta_seas)

    z500_tmax = z500_seas.max(dim=['lat', 'lon'])
    z500_tmin = z500_seas.min(dim=['lat', 'lon'])

    # Area-weighted EOF PC1 via SVD; sign convention: PC1 > 0 during El Niño
    _lat_vals = z500_seas.lat.values
    _wts_eof  = np.sqrt(np.cos(np.deg2rad(_lat_vals)))
    _z_wt     = z500_seas.values * _wts_eof[np.newaxis, :, np.newaxis]
    _z_flat   = _z_wt.reshape(len(_uniq_yrs), -1)
    _U_eof, _S_eof, _ = np.linalg.svd(_z_flat, full_matrices=False)
    _pc1 = _U_eof[:, 0] * _S_eof[0]
    if np.corrcoef(_pc1, ssta_seas)[0, 1] < 0:
        _pc1 = -_pc1
    _var_exp = _S_eof[0] ** 2 / np.sum(_S_eof ** 2) * 100.

    return dict(ssta_seas=ssta_seas, z500_tmax=z500_tmax.values,
                z500_tmin=z500_tmin.values, pc1=_pc1, var_exp=_var_exp)
