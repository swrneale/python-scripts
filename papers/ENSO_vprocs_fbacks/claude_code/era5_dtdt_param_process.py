"""
era5_dtdt_param_process.py
Process ERA5 parameterization heating rate from monthly GRIB files
into monthly-mean, pressure-level, regular 1°×1° NetCDF files.

Two-step approach:
  1. CDO: monmean + remapdis,r360x180   (~16 s per month, all I/O in C)
     → reduces 9 GB GRIB to 20 MB regular-grid NC on hybrid levels
  2. Python: log-p interpolation to target pressure levels (~0.1 s)

Usage:
    python era5_dtdt_param_process.py <year>

Output per year: dtdt_param_era5_monthly_YYYY_1x1.nc  in OUT_DIR
Then run era5_dtdt_param_concat.py to merge into one multi-year file.
"""

import sys
import os
import subprocess
import tempfile
import time as _time
import warnings

import numpy as np
import xarray as xr
from scipy.interpolate import interp1d

warnings.filterwarnings('ignore')

# ── Paths ─────────────────────────────────────────────────────────────────────
IN_DIR  = '/glade/derecho/scratch/rneale/ERA5/download/dtdt_param/'
OUT_DIR = '/glade/derecho/scratch/rneale/ERA5/mmean/1deg/dtdt_param/'
# CDO path: try derecho first, then casper
_CDO_PATHS = [
    '/glade/u/apps/derecho/25.10/spack/opt/spack/cdo/2.5.2/gcc/12.5.0/6rsl/bin/cdo',
    '/glade/u/apps/casper/25.10/spack/opt/spack/cdo/2.5.2/gcc/12.5.0/b6bq/bin/cdo',
]
CDO = next((p for p in _CDO_PATHS if os.path.exists(p)), 'cdo')  # fall back to PATH
os.makedirs(OUT_DIR, exist_ok=True)

# ── Target grid ───────────────────────────────────────────────────────────────
TARGET_PLEV = np.array([1000., 925., 850., 700., 600., 500.,
                        400., 300., 250., 200., 150., 100.,
                        70., 50., 30., 20.])      # hPa, stored_direction=decreasing
TARGET_LAT  = np.arange(-89.5, 90.0, 1.0)        # 180 pts, S→N
TARGET_LON  = np.arange(0.0, 360.0, 1.0)         # 360 pts
PS_REF      = 101325.0                             # Pa, standard atmosphere


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: CDO – monmean + remapdis to regular 1°×1°
# ─────────────────────────────────────────────────────────────────────────────

def cdo_step1(infile, tmpfile):
    """
    CDO: monthly mean + bilinear regrid from reduced Gaussian to 1°×1°.
    Returns True on success, False on failure.
    Runs in ~16 seconds per GRIB file.
    """
    cmd = [CDO, '-f', 'nc4', '-O',
           'remapdis,r360x180',
           '-monmean',
           infile, tmpfile]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            print(f'    CDO error: {result.stderr[:200]}')
            return False
        return True
    except subprocess.TimeoutExpired:
        print('    CDO timeout')
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Python – hybrid model levels → target pressure levels
# ─────────────────────────────────────────────────────────────────────────────

def interp_to_plevs(data_4d, p_model_hpa):
    """
    Log-pressure interpolation from (time, nlev, nlat, nlon) hybrid levels
    to TARGET_PLEV.  Returns (time, n_plev, nlat, nlon) float32.

    p_model_hpa: (nlev,) approximate pressure [hPa], top→surface (increasing).
    """
    log_p_model  = np.log(p_model_hpa)
    log_p_target = np.log(TARGET_PLEV[::-1])   # ascending order for searchsorted

    nt, nlev, nlat, nlon = data_4d.shape
    n_plev = len(TARGET_PLEV)
    result = np.full((nt, n_plev, nlat, nlon), np.nan, dtype=np.float32)

    for ip, lp in enumerate(log_p_target):
        if lp < log_p_model[0] or lp > log_p_model[-1]:
            continue
        idx = np.searchsorted(log_p_model, lp)
        out = n_plev - 1 - ip   # TARGET_PLEV is decreasing (1000→20 hPa)
        if idx == 0:
            result[:, out] = data_4d[:, 0]
        elif idx >= nlev:
            result[:, out] = data_4d[:, -1]
        else:
            w = ((lp - log_p_model[idx - 1])
                 / (log_p_model[idx] - log_p_model[idx - 1]))
            result[:, out] = (1.0 - w) * data_4d[:, idx - 1] + w * data_4d[:, idx]

    return result


def python_step2(tmpfile):
    """
    Read the CDO intermediate NC (monthly mean on hybrid levels, 1°×1° grid),
    interpolate to target pressure levels, return xr.DataArray.
    """
    ds = xr.open_dataset(tmpfile)

    # Pick the data variable (CDO names it 'avg_ttpm' from the GRIB)
    var_name = 'avg_ttpm' if 'avg_ttpm' in ds else list(ds.data_vars)[0]
    lev_dim  = 'lev' if 'lev' in ds.dims else 'hybrid'

    # Hybrid level numbers stored in the lev coordinate (e.g., 60..137)
    lev_nums = ds[lev_dim].values.astype(int)     # level numbers, sorted
    # Hybrid coefficients: CDO stores all 137 midpoint coefficients (nhym)
    hyam = ds['hyam'].values   # (137,) Pa — index 0 = level 1, …, index 136 = level 137
    hybm = ds['hybm'].values   # (137,)

    # Compute approximate pressure [hPa] at each model level that is actually present
    # lev_num 50 → hyam[49], hybm[49]  (0-based index = lev_num - 1)
    lev_idx  = lev_nums - 1
    p_model  = (hyam[lev_idx] + hybm[lev_idx] * PS_REF) / 100.0   # hPa, top→surface

    # Data: (time=1, lev, lat, lon)
    data = ds[var_name].values.astype(np.float32)   # (1, nlev, 180, 360)

    # Match lat/lon to target grid (CDO remapdis outputs -89.5..89.5)
    lat = ds['lat'].values
    lon = ds['lon'].values
    t0  = ds['time'].values   # numpy datetime64

    ds.close()

    # Level interpolation
    data_plev = interp_to_plevs(data, p_model)   # (1, 16, 180, 360)

    da = xr.DataArray(
        data_plev,
        dims=['valid_time', 'pressure_level', 'lat', 'lon'],
        coords={
            'valid_time':     t0,
            'pressure_level': TARGET_PLEV,
            'lat':            TARGET_LAT,
            'lon':            TARGET_LON,
        },
        name='mmpdt',
        attrs={
            'long_name': 'Mean temperature tendency due to parametrisations',
            'units':     'K s**-1',
        },
    )
    return da


# ─────────────────────────────────────────────────────────────────────────────
# Process one month
# ─────────────────────────────────────────────────────────────────────────────

def process_month(year, month):
    """Returns xr.DataArray (1, 16, 180, 360) or None."""
    fname = os.path.join(IN_DIR,
                         f'dtdt_param_{year}_{month:02d}_ytest_era5_modelevs.grib')
    if not os.path.exists(fname):
        # Fall back to NC (only 1979_01 exists as NC in the source)
        fname = fname.replace('.grib', '.nc')
        if not os.path.exists(fname):
            print(f'    No file: {year}-{month:02d}')
            return None

    t0 = _time.time()
    with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tf:
        tmpfile = tf.name

    try:
        # CDO: time mean + regrid
        ok = cdo_step1(fname, tmpfile)
        if not ok:
            return None
        t1 = _time.time()
        print(f'    CDO done in {t1-t0:.0f}s')

        # Python: level interpolation
        da = python_step2(tmpfile)
        print(f'    Lev interp done in {_time.time()-t1:.1f}s')

    finally:
        if os.path.exists(tmpfile):
            os.remove(tmpfile)

    # Some GRIBs bleed into adjacent months; keep only the expected month.
    expected = np.datetime64(f'{year}-{month:02d}', 'M')
    mask = da['valid_time'].values.astype('datetime64[M]') == expected
    if not mask.any():
        print(f'    Warning: no time step matches {year}-{month:02d} after CDO monmean')
        return None
    da = da.isel(valid_time=mask)

    return da


# ─────────────────────────────────────────────────────────────────────────────
# Main – one year, 12 months
# ─────────────────────────────────────────────────────────────────────────────

def main(year):
    year    = int(year)
    outfile = os.path.join(OUT_DIR, f'dtdt_param_era5_monthly_{year}_1x1.nc')

    if os.path.exists(outfile):
        print(f'Already exists: {outfile}')
        return

    t0_year = _time.time()
    monthly = []

    for month in range(1, 13):
        print(f'  {year}-{month:02d}:')
        tm = _time.time()
        da = process_month(year, month)
        if da is not None:
            monthly.append(da)
            print(f'    month done in {_time.time()-tm:.0f}s')

    if not monthly:
        print(f'No data for {year}')
        return

    # Override the CDO-set time to the first-of-month convention
    times = [np.datetime64(f'{year}-{m:02d}-01') for m in range(1, 13)
             if any(da.valid_time.values[0].astype('datetime64[M]')
                    == np.datetime64(f'{year}-{m:02d}') for da in monthly)]

    ds_year = xr.concat(monthly, dim='valid_time').to_dataset()

    # Assign first-of-month timestamps to match other ERA5 files
    new_times = [np.datetime64(str(t.astype('datetime64[M]')) + '-01')
                 for t in ds_year['valid_time'].values]
    ds_year = ds_year.assign_coords(valid_time=new_times)

    # Coordinate metadata
    ds_year['valid_time'].attrs.update(
        {'standard_name': 'time', 'long_name': 'time', 'axis': 'T'})
    ds_year['pressure_level'].attrs.update(
        {'standard_name': 'air_pressure', 'long_name': 'pressure',
         'units': 'hPa', 'positive': 'down', 'axis': 'Z',
         'stored_direction': 'decreasing'})
    ds_year['lat'].attrs.update(
        {'standard_name': 'latitude',  'long_name': 'latitude',
         'units': 'degrees_north', 'axis': 'Y'})
    ds_year['lon'].attrs.update(
        {'standard_name': 'longitude', 'long_name': 'longitude',
         'units': 'degrees_east',  'axis': 'X'})

    encoding = {'mmpdt': {'dtype': 'float32', '_FillValue': float('nan')}}
    ds_year.to_netcdf(outfile, mode='w', encoding=encoding)
    print(f'Saved: {outfile}  (total {_time.time()-t0_year:.0f}s)')


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python era5_dtdt_param_process.py <year>')
        sys.exit(1)
    main(sys.argv[1])
