"""
blocking_utils.py
Utility routines for the blocking frequency analysis.

  find_ens_info  – build ensemble/obs metadata DataFrame
  dataset_get    – load Z500 (or other var) daily data per ensemble
  block_z500_freq – compute 1D or 2D blocking frequency (Davini et al. 2012)
  block_file_read_write – NetCDF cache I/O
"""

import sys
import time
import importlib

import numpy as np
import pandas as pd
import xarray as xr


# Default output directory for cached blocking frequency files.
FOUT_DIR = '/glade/u/home/rneale/python/python-netcdf/blocking/'


# ─────────────────────────────────────────────────────────────────────────────
# Ensemble metadata
# ─────────────────────────────────────────────────────────────────────────────

def ens_setup(ens_names, ens_mem_num, ystart, yend):
    """Thin wrapper – build and return ensemble metadata DataFrame."""
    return find_ens_info(ens_names, ens_mem_num, ystart, yend)


def find_ens_info(ens_names, mem_num, ystart, yend):
    """
    Construct file-path templates and metadata for every ensemble/obs set.

    Returns
    -------
    pd.DataFrame  indexed by ensemble name, columns:
        ['Ensemble Type', 'Ensemble Size', 'Start Year', 'End Year',
         'Run Name', 'Run File']
    """
    import lens_simulations as sim_names
    importlib.reload(sim_names)

    OBS_SOURCES = {'ERA5', 'MERRA', 'ERAI'}
    all_info = {}

    for iens, ens_name in enumerate(ens_names):

        # Determine member list
        if ens_name in {'CESM1', 'CESM2', 'E3SMv1', 'E3SMv2', 'EAMv2', 'CAM6'}:
            run_names = sim_names.get_ens_set_names(ens_name, mem_num[iens])
        else:
            run_names = [ens_name]

        # Build file templates per source
        match ens_name:

            case 'CESM1':
                ens_type = 'model'
                base = '/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/daily/'
                file_templates = [
                    f'{base}VAR_TBD/{r}.cam.h1.VAR_TBD.19200101-20051231.nc'
                    for r in run_names
                ]
                # Member 001 starts in 1850
                r001 = 'b.e11.B20TRC5CNBDRD.f09_g16.001'
                if r001 in run_names:
                    i = run_names.index(r001)
                    file_templates[i] = file_templates[i].replace('1920', '1850', 1)

            case 'CESM2':
                ens_type = 'model'
                base = '/glade/campaign/cgd/cesm/CESM2-LE/atm/proc/tseries/day_1/'
                file_templates = [
                    f'{base}VAR_TBD/{r}.cam.h1.VAR_TBD.DATE_RANGE.nc'
                    for r in run_names
                ]

            case 'E3SMv1':
                ens_type = 'model'
                base = '/glade/campaign/cgd/amp/rneale/e3sm/'
                file_templates = [
                    f'{base}{r}/tseries/{r}_dmeans_ts_VAR_TBD.nc'
                    for r in run_names
                ]

            case ens_name if ens_name in {'E3SMv2', 'EAMv2', 'CAM6'}:
                ens_type = 'model'
                if ens_name == 'CAM6':
                    cmodel = 'cam'
                    base = '/glade/campaign/cesm/development/cvcwg/cvwg/f.e21.FHIST_FSSP370_BGC.f09_f09.ersstv5.goga/'
                else:
                    cmodel = 'eam'
                    base = '/glade/campaign/cgd/ccr/E3SMv2/FV_regridded/'

                date_ranges = {'EAMv2': '19760101-20141231',
                               'E3SMv2': '18500101-20141231',
                               'CAM6':  '18800101-20150101'}
                dr = date_ranges[ens_name]

                file_templates = [
                    f'{base}{r}/atm/proc/tseries/day_1/{r}.{cmodel}.h1.VAR_TBD.{dr}.nc'
                    for r in run_names
                ]

            case ens_name if 'b.e30' in ens_name:
                ens_type = 'model'
                base = '/glade/derecho/scratch/rneale/archive/'
                file_templates = [
                    f'{base}{r}/tseries/{r}_dmeans_ts_Z500.nc'
                    for r in run_names
                ]
                # Shorten display name
                ens_name = 'CESM3-271' if '271' in ens_name else 'CESM3-276'

            case ens_name if ens_name in OBS_SOURCES:
                ens_type = 'obs'
                base = f'/glade/work/rneale/data/{ens_name}/'
                file_templates = [f'{base}VAR_TBD.day.mean.nc']

            case _:
                print(f'  ERROR: "{ens_name}" is not a recognised ensemble or obs source.')
                sys.exit(1)

        all_info[ens_name] = [
            ens_type, mem_num[iens], ystart[iens], yend[iens],
            run_names, file_templates,
        ]

    return pd.DataFrame.from_dict(
        all_info, orient='index',
        columns=['Ensemble Type', 'Ensemble Size', 'Start Year', 'End Year',
                 'Run Name', 'Run File'],
    )


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def dataset_get(block_meta, var_name, season):
    """
    Open multi-file daily datasets for every ensemble and select season/years.

    Returns
    -------
    dict  {ens_name: xr.Dataset}   — lazy dask arrays, season-filtered
    """
    t0 = time.time()
    ens_names = list(block_meta.index)
    chunks = {'time': 365, 'lat': -1, 'lon': -1}

    print(f'dataset_get: season={season}')

    ds_ens = {}

    for ens_name in ens_names:
        run_files = [f.replace('VAR_TBD', var_name)
                     for f in block_meta.loc[ens_name]['Run File']]
        run_names = block_meta.loc[ens_name]['Run Name']
        y0 = block_meta.loc[ens_name]['Start Year']
        y1 = block_meta.loc[ens_name]['End Year']

        print(f'  {ens_name}: {len(run_names)} member(s)  {y0}–{y1}')

        ds_members = []
        for run_file in run_files:

            if ens_name == 'CESM2':
                run_file = run_file.replace('DATE_RANGE', '*')
                ds = xr.open_mfdataset(run_file, parallel=True, chunks=chunks)
            elif ens_name == 'ERA5':
                ds = xr.open_mfdataset(run_file, parallel=True, chunks=chunks)
            else:
                ds = xr.open_mfdataset(run_file, combine='nested',
                                        parallel=True, chunks=chunks)

            # Fix non-standard time axes (ERAI / MERRA use Julian day numbers)
            if ens_name in {'ERAI', 'MERRA'}:
                ds['time'] = pd.to_datetime(ds['time'].values,
                                             origin='julian', unit='D')
            if ens_name == 'ERAI':
                ds = ds.reindex(lat=ds.lat[::-1])

            # Year and season subset
            ds = ds.sel(time=slice(y0, y1))

            y_actual0 = int(ds['time'].dt.year.min())
            y_actual1 = int(ds['time'].dt.year.max())
            if y_actual0 != int(y0) or y_actual1 != int(y1):
                print(f'    Warning: {ens_name} actual years {y_actual0}–{y_actual1} '
                      f'differ from requested {y0}–{y1}')

            ds = ds.sel(time=ds['time.season'] == season)
            ds_members.append(ds)

        # Stack members along a 'name' dimension
        ds_ens_this = xr.concat(ds_members, dim='name')
        ds_ens_this = ds_ens_this.assign_coords(name=('name', run_names))

        ds_ens[ens_name] = ds_ens_this

    print(f'dataset_get: done in {time.time()-t0:.1f}s\n')
    return ds_ens


# ─────────────────────────────────────────────────────────────────────────────
# Blocking frequency
# ─────────────────────────────────────────────────────────────────────────────

# Davini et al. (2012) thresholds
_GHGN_THRESH = -5.0   # m / degree-lat
_GHGS_THRESH =  0.0

# Latitude bounds and parameters
_LAT_S  = 35.0
_LAT_N  = 75.0
_DLAT2D = 15.0        # offset for 2D gradients

# 1D fixed latitudes + ±3.75° spread
_BLAT0  = 60.00
_BLATN  = 78.85
_BLATS  = 41.25
_DELTAS = [-3.75, 0.0, 3.75]


def block_z500_freq(block_meta, ens_ds, bseason, block_diag=None, file_opts='x'):
    """
    Compute seasonal blocking frequency (Davini et al. 2012).

    Parameters
    ----------
    block_meta : pd.DataFrame  from find_ens_info / ens_setup
    ens_ds     : dict           from dataset_get
    bseason    : str            season label (for file naming)
    block_diag : '1D' or '2D'
    file_opts  : 'x' skip I/O | 'w' write | 'r' read from disk

    Returns
    -------
    dict  {ens_name: xr.DataArray}  blocking frequency [0–1]
    """
    ens_names = list(block_meta.index)
    block_freq_ens = {}

    for ens_name in ens_names:
        t0 = time.time()
        y0      = block_meta.loc[ens_name]['Start Year']
        y1      = block_meta.loc[ens_name]['End Year']
        n_mem   = block_meta.loc[ens_name]['Ensemble Size']

        if file_opts in ('w', 'x'):

            z500 = ens_ds[ens_name]['Z500'].sel(lat=slice(_LAT_S, _LAT_N))

            print(f'  block_z500_freq: {block_diag} blocking for {ens_name}')

            if block_diag == '1D':
                is_blocked = _blocking_1d(z500)

            elif block_diag == '2D':
                is_blocked = _blocking_2d(z500)

            else:
                print(f'  Unknown block_diag "{block_diag}" — use "1D" or "2D"')
                sys.exit(1)

            block_freq = is_blocked.sum('time') / is_blocked.sizes['time']

        block_freq = block_file_read_write(
            ens_name, n_mem, y0, y1, bseason, block_freq, block_diag, file_opts
        )

        pct = 100. * block_freq
        print(f'    {ens_name}: blocking {pct.min().values:.2f}–{pct.max().values:.2f}%'
              f'  ({time.time()-t0:.1f}s)')

        block_freq_ens[ens_name] = block_freq.compute()

    return block_freq_ens


def _blocking_1d(z500):
    """
    1D blocking mask: union over 3 latitude spreads (+-3.75 deg).
    Input z500 already subset to [_LAT_S, _LAT_N].
    Returns bool DataArray (name, time, lon).
    """
    lat_vals = z500.lat.values

    def _nearest_idx(target):
        return int(np.argmin(np.abs(lat_vals - target)))

    is_blocked = None

    for d in _DELTAS:
        idx_n = _nearest_idx(_BLATN + d)
        idx_0 = _nearest_idx(_BLAT0 + d)
        idx_s = _nearest_idx(_BLATS + d)

        zn = z500.isel(lat=idx_n)
        z0 = z500.isel(lat=idx_0)
        zs = z500.isel(lat=idx_s)

        dlat_n = float(lat_vals[idx_n] - lat_vals[idx_0])
        dlat_s = float(lat_vals[idx_0] - lat_vals[idx_s])

        ghgn = (zn - z0) / dlat_n
        ghgs = (z0 - zs) / dlat_s

        mask_i = (ghgs > _GHGS_THRESH) & (ghgn < _GHGN_THRESH)
        is_blocked = mask_i if is_blocked is None else (is_blocked | mask_i)

    return is_blocked


def _blocking_2d(z500):
    """
    2D blocking mask computed fully vectorized (no Python lat-loop).
    Input z500 already subset to [_LAT_S, _LAT_N].
    Returns bool DataArray (name, time, lat, lon).
    """
    lat_c = z500.lat.values                                  # center latitudes
    lat_n_tgt = np.clip(lat_c + _DLAT2D,
                        float(z500.lat.min()), float(z500.lat.max()))
    lat_s_tgt = np.clip(lat_c - _DLAT2D,
                        float(z500.lat.min()), float(z500.lat.max()))

    # Vectorized nearest-neighbour lookup for all center lats at once.
    # sel() with a plain array performs outer (vectorized) indexing.
    # assign_coords relabels the lat dimension back to center values so that
    # arithmetic with z500 (center) aligns correctly.
    # Targets beyond the subset boundary are already clipped by np.clip above,
    # matching the original code's nearest-on-boundary behaviour.
    z_n = z500.sel(lat=lat_n_tgt, method='nearest').assign_coords(lat=lat_c)
    z_s = z500.sel(lat=lat_s_tgt, method='nearest').assign_coords(lat=lat_c)

    ghgn = (z_n - z500) / _DLAT2D
    ghgs = (z500 - z_s) / _DLAT2D

    return (ghgs > _GHGS_THRESH) & (ghgn < _GHGN_THRESH)


# ─────────────────────────────────────────────────────────────────────────────
# File cache
# ─────────────────────────────────────────────────────────────────────────────

def block_file_read_write(ens_name, nens, year_start, year_end,
                          bseason, block_array_ens, block_diag, file_opts):
    """
    Read ('r'), write ('w'), or skip ('x') a cached blocking-frequency NetCDF.
    """
    fname = f'block_{block_diag}_{ens_name}_nens.{nens}_{year_start}-{year_end}_{bseason}.nc'
    fpath = FOUT_DIR + fname
    varname = f'BLOCK_{block_diag}'

    match file_opts:

        case 'w':
            print(f'  Writing {fname}')
            block_array_ens.rename(varname).to_dataset().to_netcdf(fpath)
            return block_array_ens

        case 'r':
            print(f'  Reading {fname}')
            return xr.open_dataset(fpath)[varname]

        case 'x':
            return block_array_ens

        case _:
            print(f'  Unknown file_opts "{file_opts}" — use r, w, or x')
            sys.exit(1)
