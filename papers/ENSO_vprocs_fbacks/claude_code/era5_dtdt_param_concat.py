"""
era5_dtdt_param_concat.py
Concatenate annual dtdt_param files produced by era5_dtdt_param_process.py
into a single multi-year monthly file matching the format of other ERA5 1°×1° files.

Usage:
    python era5_dtdt_param_concat.py

Reads:  .../dtdt_param/dtdt_param_era5_monthly_YYYY_1x1.nc  (one per year)
Writes: .../dtdt_param/dtdt_param_era5_monthly_1x1.nc
"""

import os
import glob
import xarray as xr

DATA_DIR = '/glade/derecho/scratch/rneale/ERA5/mmean/1deg/dtdt_param/'
OUT_FILE = os.path.join(DATA_DIR, 'dtdt_param_era5_monthly_1x1.nc')

files = sorted(glob.glob(os.path.join(DATA_DIR, 'dtdt_param_era5_monthly_????_1x1.nc')))

if not files:
    print('No annual files found in', DATA_DIR)
    raise SystemExit(1)

print(f'Concatenating {len(files)} annual files:')
for f in files:
    print(' ', os.path.basename(f))

ds = xr.open_mfdataset(files, combine='by_coords', data_vars='minimal', coords='minimal')

print(f'\nTime range: {ds["valid_time"].values[[0,-1]]}')
print(f'Shape: {ds["mmpdt"].shape}')
print(f'\nWriting: {OUT_FILE}')

encoding = {
    'mmpdt': {'dtype': 'float32', '_FillValue': float('nan')},
    'valid_time': {'dtype': 'float64'},
}
ds.to_netcdf(OUT_FILE, mode='w', encoding=encoding)
ds.close()

print('Done.')
