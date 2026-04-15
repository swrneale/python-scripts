"""
merra2_div_compute.py
---------------------
Compute horizontal divergence from MERRA2 monthly-mean u and v wind files
using windspharm spherical harmonics and save to the same 1-degree directory.

Input:
    /glade/derecho/scratch/rneale/MERRA2/mmean/1deg/u/u_merra2_monthly_1x1.nc
    /glade/derecho/scratch/rneale/MERRA2/mmean/1deg/v/v_merra2_monthly_1x1.nc

Output:
    /glade/derecho/scratch/rneale/MERRA2/mmean/1deg/div/div_merra2_monthly_1x1.nc
    Variable: div  (time, plev, lat, lon)  units: s^-1
"""

import os
import numpy as np
import xarray as xr
from windspharm.xarray import VectorWind

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE   = '/glade/derecho/scratch/rneale/MERRA2/mmean/1deg'
F_U    = os.path.join(BASE, 'u', 'u_merra2_monthly_1x1.nc')
F_V    = os.path.join(BASE, 'v', 'v_merra2_monthly_1x1.nc')
DIR_OUT = os.path.join(BASE, 'div')
F_OUT  = os.path.join(DIR_OUT, 'div_merra2_monthly_1x1.nc')

os.makedirs(DIR_OUT, exist_ok=True)

# ── Load ───────────────────────────────────────────────────────────────────────
print('Loading u and v ...')
ds_u = xr.open_dataset(F_U)
ds_v = xr.open_dataset(F_V)

da_u = ds_u['u']   # (time, plev, lat, lon)
da_v = ds_v['v']

# windspharm requires lat to run N→S; sort if necessary
needs_flip = float(da_u.lat[0]) < float(da_u.lat[-1])
if needs_flip:
    da_u = da_u.sortby('lat', ascending=False)
    da_v = da_v.sortby('lat', ascending=False)

print(f'  Shape: {da_u.shape}  (sorted N→S: {needs_flip})')

# ── Compute divergence level by level ─────────────────────────────────────────
# windspharm.xarray.VectorWind handles (time, lat, lon) DataArrays
divs = []
plevs = da_u.plev.values
print(f'Processing {len(plevs)} levels: {plevs}')

for ip, plev_val in enumerate(plevs):
    u_lev = da_u.sel(plev=plev_val)   # (time, lat, lon)
    v_lev = da_v.sel(plev=plev_val)

    w = VectorWind(u_lev.fillna(0.0), v_lev.fillna(0.0))
    div_lev = w.divergence()           # (time, lat, lon), s^-1

    divs.append(div_lev)
    print(f'  plev={plev_val:6.0f} hPa  done')

# Stack along plev dimension
div_all = xr.concat(divs, dim=da_u.plev)   # (plev, time, lat, lon)
div_all = div_all.transpose('time', 'plev', 'lat', 'lon')

# Restore S→N lat ordering to match original files
if needs_flip:
    div_all = div_all.sortby('lat', ascending=True)

# ── Package and save ───────────────────────────────────────────────────────────
div_all.name = 'div'
div_all.attrs['long_name'] = 'Horizontal divergence'
div_all.attrs['units']     = 's-1'
div_all.attrs['source']    = 'Computed from MERRA2 u,v via windspharm'

ds_out = div_all.to_dataset()
ds_out['time'].attrs = ds_u['time'].attrs   # preserve time units/calendar

encoding = {'div': {'dtype': 'float32', 'zlib': True, 'complevel': 4}}

print(f'\nSaving to {F_OUT} ...')
ds_out.to_netcdf(F_OUT, encoding=encoding)
print('Done.')
