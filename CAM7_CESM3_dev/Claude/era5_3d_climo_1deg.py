#!/usr/bin/env python
"""
era5_3d_climo_1deg.py

Compute monthly (time=1..12), seasonal (DJF/MAM/JJA/SON), and annual (ANN)
climatologies from ERA5 1-degree monthly mean 3D pressure-level fields.

Source : /glade/derecho/scratch/rneale/ERA5/mmean/1deg/{var}/
         Files matched: {var}_test_era5_monthly_{YEAR}_1x1.nc

Output : /glade/derecho/scratch/rneale/ERA5/climo/1deg/
         ERA5_1deg_{var}_monthly_{yr0}-{yr1}_climo.nc   (time=1..12)
         ERA5_1deg_{var}_{SEAS}_{yr0}-{yr1}_climo.nc    (SEAS = DJF/MAM/JJA/SON/ANN)

Coordinate renames in output:
  latitude       -> lat
  longitude      -> lon
  pressure_level -> lev   (hPa, positive down, 1000..20)
  valid_time     -> time  (monthly: integer 1-12; seasonal: dropped)

Run a subset of variables:
  python era5_3d_climo_1deg.py t q          # only temperature and humidity
  python era5_3d_climo_1deg.py              # all variables
"""

import xarray as xr
import numpy as np
import glob
import os
import sys
import time as timer

# ============================================================
# Configuration
# ============================================================

DIR_IN  = '/glade/derecho/scratch/rneale/ERA5/mmean/1deg'
DIR_OUT = '/glade/derecho/scratch/rneale/ERA5/climo/1deg'

# Directory short name -> ERA5 in-file variable name
VAR_MAP = {
    't':     't',    # temperature (K)
    'q':     'q',    # specific humidity (kg/kg)
    'rh':    'r',    # relative humidity (%)
    'u':     'u',    # zonal wind (m/s)
    'v':     'v',    # meridional wind (m/s)
    'z':     'z',    # geopotential (m2/s2)
    'omega': 'w',    # vertical velocity (Pa/s)
    'cloud': 'cc',   # cloud area fraction (0-1)
    'div':   'd',    # divergence (1/s)
}

SEASON_MONTHS = {
    'DJF': [12, 1, 2],
    'MAM': [3, 4, 5],
    'JJA': [6, 7, 8],
    'SON': [9, 10, 11],
    'ANN': list(range(1, 13)),
}

# NetCDF output encoding
NC_ENC_BASE = {'zlib': True, 'complevel': 4, 'dtype': 'float32'}

# Dask chunk sizes (tune to available memory)
CHUNKS = {'valid_time': 24, 'pressure_level': 16}

# ============================================================
# Helpers
# ============================================================

CLIMO_YR0 = 1979
CLIMO_YR1 = 2023


def get_files(var_short):
    """Return sorted per-year test files within CLIMO_YR0..CLIMO_YR1 only."""
    pattern = os.path.join(DIR_IN, var_short,
                           f'{var_short}_test_era5_monthly_*_1x1.nc')
    files = []
    for f in sorted(glob.glob(pattern)):
        yr_str = os.path.basename(f).split('monthly_')[1].split('_')[0]
        if yr_str.isdigit() and CLIMO_YR0 <= int(yr_str) <= CLIMO_YR1:
            files.append(f)
    return files


def preprocess(ds, var_name):
    """
    Drop variables/coords that cause open_mfdataset concat problems
    (expver string var, number scalar, etc.) keeping only what we need.
    """
    keep_dims  = {'valid_time', 'pressure_level', 'latitude', 'longitude'}
    keep_vars  = {var_name}
    keep_coord = {'valid_time', 'pressure_level', 'latitude', 'longitude',
                  'lat', 'lon', 'lev'}

    drop = [v for v in list(ds.data_vars) if v not in keep_vars]
    drop += [c for c in ds.coords
             if c not in keep_coord and c not in ds.dims]
    return ds.drop_vars(drop, errors='ignore')


def open_all_years(files, var_name):
    """
    Lazy open all per-year files, concat on valid_time, clean coordinates.
    Returns dataset with dims: time, lev, lat, lon
    """
    ds = xr.open_mfdataset(
        files,
        combine='nested',
        concat_dim='valid_time',
        join='override',
        compat='override',
        data_vars='minimal',
        coords='minimal',
        decode_times=True,
        use_cftime=True,
        chunks=CHUNKS,
        preprocess=lambda d: preprocess(d, var_name),
    )

    # Rename coordinates to CESM-compatible names
    rename_map = {}
    if 'latitude'       in ds.coords: rename_map['latitude']       = 'lat'
    if 'longitude'      in ds.coords: rename_map['longitude']      = 'lon'
    if 'pressure_level' in ds.dims:   rename_map['pressure_level'] = 'lev'
    if rename_map:
        ds = ds.rename(rename_map)

    ds = ds.rename({'valid_time': 'time'})

    # Ensure lat is ascending (ERA5 files are often N->S decreasing)
    if ds['lat'].values[0] > ds['lat'].values[-1]:
        ds = ds.isel(lat=slice(None, None, -1))

    return ds


def add_global_attrs(ds, var_short, yr0, yr1, period):
    ds.attrs.update({
        'source':       'ERA5 reanalysis (ECMWF)',
        'resolution':   '1 degree x 1 degree',
        'variable':     var_short,
        'period':       period,
        'year_range':   f'{yr0}-{yr1}',
        'created_by':   'era5_3d_climo_1deg.py',
    })
    return ds


# ============================================================
# Climatology functions
# ============================================================

def compute_monthly_climo(ds, var_name):
    """
    Compute 12-month climatology via groupby.
    Returns dataset with time=1..12 (integer month index).
    """
    da = ds[var_name].groupby('time.month').mean(dim='time')
    da = da.rename({'month': 'time'})
    da['time'].attrs = {
        'long_name': 'climatological_month',
        'units':     '1 (1=Jan, 12=Dec)',
        'axis':      'T',
    }
    return da.to_dataset(name=var_name)


def compute_seasonal_climo(ds, var_name, months):
    """
    Average over selected calendar months across all years.
    Returns dataset with no time dimension.
    """
    mask = ds['time.month'].isin(months)
    da = ds[var_name].sel(time=mask).mean(dim='time')
    return da.to_dataset(name=var_name)


# ============================================================
# I/O
# ============================================================

def save_nc(ds, var_name, fpath):
    enc = {var_name: NC_ENC_BASE.copy()}

    # Add coordinate encodings to avoid cftime write issues
    if 'time' in ds.coords:
        enc['time'] = {'dtype': 'float64'}

    ds.to_netcdf(fpath, encoding=enc)
    sz = os.path.getsize(fpath) / 1e6
    print(f'    saved  {os.path.basename(fpath)}  ({sz:.1f} MB)')


# ============================================================
# Main
# ============================================================

def process_variable(var_short, var_name):
    files = get_files(var_short)
    if not files:
        print(f'  WARNING: no *_test_* files found for {var_short}, skipping.')
        return

    yr0 = os.path.basename(files[0]).split('monthly_')[1].split('_')[0]
    yr1 = os.path.basename(files[-1]).split('monthly_')[1].split('_')[0]
    n   = len(files)

    out_dir = DIR_OUT
    os.makedirs(out_dir, exist_ok=True)

    print(f'\n{"="*62}')
    print(f'  Variable : {var_short}  (in-file name: "{var_name}")')
    print(f'  Years    : {yr0} - {yr1}  ({n} annual files)')
    print(f'  Output   : {out_dir}')
    print(f'{"="*62}')

    t0 = timer.time()

    print('  Opening files lazily...')
    ds = open_all_years(files, var_name)
    print(f'    dims: time={ds.dims["time"]}, lev={ds.dims["lev"]}, '
          f'lat={ds.dims["lat"]}, lon={ds.dims["lon"]}')

    # ------------------------------------------------------------------
    # 1. Monthly climatology  (time=12)
    # ------------------------------------------------------------------
    print('  Computing monthly climatology...')
    ds_mon = compute_monthly_climo(ds, var_name)
    ds_mon = add_global_attrs(ds_mon, var_short, yr0, yr1, 'monthly')
    fout = os.path.join(out_dir,
                        f'ERA5_1deg_{var_short}_monthly_{yr0}-{yr1}_climo.nc')
    save_nc(ds_mon, var_name, fout)

    # ------------------------------------------------------------------
    # 2. Seasonal + Annual climatologies
    # ------------------------------------------------------------------
    for seas, months in SEASON_MONTHS.items():
        print(f'  Computing {seas} climatology...')
        ds_seas = compute_seasonal_climo(ds, var_name, months)
        ds_seas = add_global_attrs(ds_seas, var_short, yr0, yr1, seas)
        fout = os.path.join(out_dir,
                            f'ERA5_1deg_{var_short}_{seas}_{yr0}-{yr1}_climo.nc')
        save_nc(ds_seas, var_name, fout)

    ds.close()
    print(f'  Done in {timer.time()-t0:.0f}s')


def main(vars_to_run=None):
    os.makedirs(DIR_OUT, exist_ok=True)

    run_map = {k: v for k, v in VAR_MAP.items()
               if vars_to_run is None or k in vars_to_run}

    if not run_map:
        print(f'ERROR: none of {vars_to_run} found in VAR_MAP ({list(VAR_MAP)})')
        sys.exit(1)

    print(f'Processing {len(run_map)} variable(s): {list(run_map)}')
    print(f'Output dir : {DIR_OUT}')

    t_total = timer.time()
    for var_short, var_name in run_map.items():
        process_variable(var_short, var_name)

    print(f'\nAll variables complete in {(timer.time()-t_total)/60:.1f} min.')


if __name__ == '__main__':
    # Optional: pass variable short names on command line to process a subset
    #   python era5_3d_climo_1deg.py t q u
    vars_requested = sys.argv[1:] if len(sys.argv) > 1 else None
    main(vars_requested)
