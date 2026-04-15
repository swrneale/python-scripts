"""
data_io.py  –  File discovery and lazy loading for every supported case type.

Public API
----------
load_var(case, var_name, cfg)  → xr.Dataset (lazy, trimmed to cfg lat/year range)
load_sst(case, cfg)            → xr.DataArray  (SST timeseries)
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import os
import subprocess
import xarray as xr

from config import AnalysisConfig, CaseConfig


# ── Internal helpers ──────────────────────────────────────────────────────────

# ERA5 RDA variable name maps: CAM name → (file-name fragment, in-file variable name)
_ERA5_VAR = {
    'T':     ('130_t',   'T'),
    'Q':     ('133_q',   'Q'),
    'OMEGA': ('135_w',   'W'),
    'U':     ('131_u',   'U'),
    'V':     ('132_v',   'V'),
    'Z3':    ('129_z',   'Z'),
}

_CLIMO_VAR_MAP = {
    'T': 'ta', 'Q': 'hus', 'Z3': 'hgt',
    'U': 'ua', 'V': 'va',  'OMEGA': 'omega',
}

# CF-style variable names used in CMIP6/c6_amip output
_CF_VAR = {
    'TS': 'ts', 'T': 'ta', 'Q': 'hus', 'Z3': 'hgt',
    'U': 'ua',  'V': 'va', 'OMEGA': 'wap',
}

# Paths to data collections
_DIR = {
    'rda':          Path('/glade/collections/rda/data'),
    'hadisst':      Path('/glade/p/cesmdata/cseg/inputdata/atm/cam/sst/sst_HadOIBl_bc_0.9x1.25_1850_2020_c210521.nc'),
    'lens1':        Path('/glade/campaign/cesm/collections/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/monthly'),
    'lens2':        Path('/glade/campaign/cgd/cesm/CESM2-LE/timeseries/atm/proc/tseries/month_1'),
    'lense2':       Path('/glade/campaign/cgd/ccr/E3SMv2/FV_regridded'),
    'c6_amip':      Path('/glade/collections/cdg/data/CMIP6/CMIP/NCAR/CESM2/amip'),
    'cam6_revert':  Path('/glade/campaign/cgd/amp/amwg/runs'),
    'cesm3_dev':    Path('/glade/derecho/scratch/hannay/archive'),
    'work':         Path('/glade/work/rneale/data'),
}


def _yr(y: int) -> str:
    return f'{y:04d}'


def _glob_files(directory: Path, pattern: str) -> list[Path]:
    """Return sorted list of files matching pattern in directory."""
    return sorted(directory.glob(pattern))


def _ls_filtered(directory: Path, substring: str) -> list[Path]:
    """List files in directory whose name contains substring, sorted."""
    return sorted(p for p in directory.iterdir() if substring in p.name)


# ── File-discovery per case type ─────────────────────────────────────────────

def _files_era5(var_cam: str, yr0: int, yr1: int) -> tuple[list[Path], str, bool, bool]:
    """Returns (file_list, var_name_in_file, reverse_lat, rename_coords)."""
    fname_frag, vname = _ERA5_VAR[var_cam]
    ftype = 'uv' if var_cam in ('U', 'V') else 'sc'
    cat = 'ds633.1'
    files = [
        _DIR['rda'] / cat / f'e5.moda.an.pl/{y:03d}/'
                            f'e5.moda.an.pl.128_{fname_frag}.ll025{ftype}.{y:03d}010100_{y:03d}120100.nc'
        for y in range(yr0, yr1 + 1)
    ]
    return files, vname, True, True   # reverse_lat=True, rename_coords=True


def _files_erai(var_cam: str, yr0: int, yr1: int) -> tuple[list[Path], str, bool, bool]:
    ftype_map = {'T': 'sc', 'U': 'uv', 'V': 'uv', 'OMEGA': 'uv'}
    ftype = ftype_map.get(var_cam, 'sc')
    cat = 'ds627.1'
    files = [
        _DIR['rda'] / cat / f'ei.moda.an.pl/ei.moda.an.pl.regn128{ftype}.{y:03d}{m:02d}0100.nc'
        for y in range(yr0, yr1 + 1) for m in range(1, 13)
    ]
    vname_map = {'T': 'T', 'Q': 'Q', 'OMEGA': 'w'}
    return files, vname_map.get(var_cam, var_cam), False, True


def _files_jra25(var_cam: str, yr0: int, yr1: int) -> tuple[list[Path], str, bool, bool]:
    cat = 'ds625.1'
    files = [
        _DIR['rda'] / cat / f'anl_p25/anl_p25.{y:03d}{m:02d}.nc'
        for y in range(yr0, yr1 + 1) for m in range(1, 13)
    ]
    vname_map = {'T': 'TMP_PRS', 'Q': 'Q', 'OMEGA': 'W'}
    return files, vname_map.get(var_cam, var_cam), False, False


def _files_merra2(var_cam: str, yr0: int, yr1: int) -> tuple[list[Path], str, bool, bool]:
    resn = '1.9x2.5'
    cat = 'ds313.3'
    files = [
        _DIR['rda'] / cat / f'{resn}/{y:03d}/MERRA2{y:03d}010100_{y:03d}120100.nc'
        for y in range(yr0, yr1 + 1)
    ]
    vname_map = {'T': 'T', 'Q': 'Q', 'OMEGA': 'OMEGA'}
    return files, vname_map.get(var_cam, var_cam), False, False


def _files_lens1(run_name: str, var_cam: str) -> tuple[list[Path], str, bool, bool]:
    stub = _DIR['lens1'] / var_cam / f'{run_name}.cam.h0'
    raw = subprocess.getoutput(f'ls {stub}*.nc')
    files = [Path(f) for f in raw.split('\n') if f.endswith('.nc')]
    return files, var_cam, False, False


def _files_lens2(run_name: str, var_cam: str) -> tuple[list[Path], str, bool, bool]:
    var_dir = _DIR['lens2'] / var_cam
    files = _ls_filtered(var_dir, run_name)
    return files, var_cam, False, False


def _files_lense2(run_name: str, var_cam: str) -> tuple[list[Path], str, bool, bool]:
    stub = _DIR['lense2'] / run_name / 'atm/proc/tseries/month_1' / f'{run_name}.eam.h0.{var_cam}.'
    raw = subprocess.getoutput(f'ls {stub}*.nc')
    files = [Path(f) for f in raw.split('\n') if f.endswith('.nc')]
    return files, var_cam, False, False


def _files_c6_amip(run_name: str, var_cam: str) -> tuple[list[Path], str, bool, bool]:
    cf_var = _CF_VAR[var_cam]
    var_dir = _DIR['c6_amip'] / run_name / 'Amon' / cf_var / 'gn/latest'
    files = sorted(var_dir.glob('*.nc'))
    return files, cf_var, False, True   # rename plev→lev


def _files_cam_h0(run_name: str, var_cam: str, yr0: int, yr1: int,
                  case_type: str) -> tuple[list[Path], str, bool, bool]:
    root = _DIR['cesm3_dev'] if case_type == 'cesm3_dev' else _DIR['cam6_revert']
    hist_dir = root / run_name / 'atm/hist'
    stub = f'{run_name}.cam.h0.'
    all_files = sorted(hist_dir.glob(f'{stub}*.nc'))
    # Trim to requested year range
    start_name = f'{stub}{_yr(yr0)}-01.nc'
    end_name   = f'{stub}{_yr(yr1)}-12.nc'
    names = [p.name for p in all_files]
    i0 = names.index(start_name)
    i1 = names.index(end_name)
    return all_files[i0:i1 + 1], var_cam, False, False


# ── Coordinate standardisation ────────────────────────────────────────────────

def _standardize_coords(ds: xr.Dataset, var_cam: str,
                        case_type: str, case_name: str) -> xr.Dataset:
    """Rename coordinates to the internal convention (lat, lon, lev)."""
    if var_cam == 'TS':
        return ds
    if case_name == 'ERA5':
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon', 'level': 'lev'})
    if case_type == 'c6_amip' and 'plev' in ds.dims:
        ds = ds.rename({'plev': 'lev'})
        ds = ds.assign_coords(lev=0.01 * ds.lev)  # Pa → hPa
    return ds


# ── Public loading functions ──────────────────────────────────────────────────

def load_var(case: CaseConfig, var_name: str,
             cfg: AnalysisConfig) -> tuple[xr.Dataset, str]:
    """Load a variable for the given case and return (dataset, var_name_in_ds).

    The dataset is trimmed to cfg.lats_in and cfg.years_data (time-series mode)
    or cfg.p_levs (pressure range).  Loading is lazy (dask-backed).
    """
    yr0, yr1 = cfg.years_data

    # DIV is derived from OMEGA; request the raw file
    var_cam = 'OMEGA' if var_name == 'DIV' else var_name
    reverse_lat = False
    rename_coords = False

    if case.case_type == 'reanal':
        if var_cam == 'TS':
            files   = [_DIR['hadisst']]
            vname   = 'SST_cpl'
        elif case.name == 'ERA5':
            files, vname, reverse_lat, rename_coords = _files_era5(var_cam, yr0, yr1)
        elif case.name == 'ERAI':
            files, vname, reverse_lat, rename_coords = _files_erai(var_cam, yr0, yr1)
        elif case.name == 'JRA25':
            files, vname, reverse_lat, rename_coords = _files_jra25(var_cam, yr0, yr1)
        elif case.name == 'MERRA2':
            files, vname, reverse_lat, rename_coords = _files_merra2(var_cam, yr0, yr1)
        else:
            raise ValueError(f'Unknown reanalysis: {case.name}')

    elif case.case_type == 'lens1':
        files, vname, reverse_lat, rename_coords = _files_lens1(case.run_name, var_cam)

    elif case.case_type == 'lens2':
        files, vname, reverse_lat, rename_coords = _files_lens2(case.run_name, var_cam)

    elif case.case_type == 'lense2':
        files, vname, reverse_lat, rename_coords = _files_lense2(case.run_name, var_cam)

    elif case.case_type == 'c6_amip':
        files, vname, reverse_lat, rename_coords = _files_c6_amip(case.run_name, var_cam)

    elif case.case_type in ('cam6_revert', 'cesm3_dev'):
        files, vname, reverse_lat, rename_coords = _files_cam_h0(
            case.run_name, var_cam, yr0, yr1, case.case_type)

    else:
        raise ValueError(f'Unsupported case_type: {case.case_type!r}')

    # ── Open dataset ──────────────────────────────────────────────────────────
    # Chunk along time for parallel I/O and along lat so individual chunks stay
    # under ~150 MB.  ERA5 0.25° has ~1440 lon pts; chunking lat=90 gives
    # 12×90×1440×37×4 bytes ≈ 230 MB.  CAM 1° grids are smaller so this is safe.
    print(f'    Opening {len(files)} file(s) for {case.name} / {var_cam}')
    ds = xr.open_mfdataset(
        [str(f) for f in files],
        decode_cf=True, decode_times=True,
        parallel=True,
        chunks={'time': 12, 'lat': 90, 'lon': 180},
    )

    if rename_coords:
        ds = _standardize_coords(ds, var_cam, case.case_type, case.name)

    if reverse_lat:
        ds = ds.reindex(lat=list(reversed(ds.lat)))

    # Trim spatial / temporal extent
    lat_s, lat_n = cfg.lats_in
    ds = ds.sel(lat=slice(lat_s, lat_n),
                time=slice(_yr(yr0), _yr(yr1)))
    if var_cam != 'TS':
        ds = ds.sel(lev=slice(cfg.pres_min, cfg.pres_max))

    print(f'    Time range in file: {int(ds.time.dt.year.min())}–{int(ds.time.dt.year.max())}')
    return ds, vname


def load_sst(case: CaseConfig, cfg: AnalysisConfig) -> tuple[xr.Dataset, str]:
    """Load SST for ENSO index computation.

    For reanalyses and AMIP-style runs the HadISST file is used.
    For coupled model runs the SST is taken from the same history files as the
    main variable (caller should pass back the already-opened dataset).
    """
    if case.case_type in ('cam6_revert', 'cesm3_dev'):
        # SST is TS in the h0 files – caller uses the existing files_ptr
        return None, 'TS'
    return load_var(case, 'TS', cfg)


def load_climo(case: CaseConfig, var_name: str,
               cfg: AnalysisConfig) -> tuple[xr.Dataset, str]:
    """Load pre-computed climatology / El Niño / La Niña netCDF files.

    Expects three files in /glade/work/rneale/data/<case_name>/:
        <case_name>_climo_DJF.nc
        <case_name>_nino_DJF.nc
        <case_name>_nina_DJF.nc

    The time dimension will be length-3 (climo=0, nino=1, nina=2).
    """
    var_cam = 'OMEGA' if var_name == 'DIV' else var_name
    vname   = _CLIMO_VAR_MAP.get(var_cam, var_cam)

    data_dir  = _DIR['work'] / case.name
    file_stub = data_dir / f'{case.name}_'
    files = [str(file_stub) + s for s in ('climo_DJF.nc', 'nino_DJF.nc', 'nina_DJF.nc')]

    print(f'    Loading climo files for {case.name}:')
    for f in files:
        print(f'      {f}')

    ds = xr.open_mfdataset(
        files,
        decode_cf=True, decode_times=False,
        concat_dim='time', combine='nested',
    )
    lat_s, lat_n = cfg.lats_in
    ds = ds.sel(lat=slice(lat_s, lat_n))
    return ds, vname
