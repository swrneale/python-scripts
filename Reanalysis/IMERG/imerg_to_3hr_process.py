"""
imerg_to_3hr_process.py

Convert GPM IMERG V07 half-hourly (0.1°×0.1°) to 3-hourly products at:
  • 0.25° × 0.25°  (≈25 km)  —  50°S-50°N  →  3hrly/0.25deg/
  • 1.0°  × 1.0°   (≈100 km) —  50°S-50°N  →  3hrly/1deg/
plus _10NS variants restricted to 10°S-10°N.

Output filenames match the existing TRMM 3B42 V7 convention for drop-in
compatibility with downstream analysis.

Pipeline (per month):
  1. Day loop: read 48 HDF5 half-hourly files → average to 8 three-hourly means
  2. Write per-day 0.1° NetCDF (lat-clipped to 55°S-55°N for speed)
  3. CDO remapcon → per-day 0.25° and 1° NetCDF (weight files computed once)
  4. CDO mergetime → monthly files
  5. Python: lat-subset _10NS, add metadata, write final NetCDF

Usage:
    python imerg_to_3hr_process.py YYYY MM
"""

import sys
import os
import glob
import time as _time
import tempfile
import subprocess
import calendar

import numpy as np
import h5py
import xarray as xr

# ── Paths ──────────────────────────────────────────────────────────────────
IN_DIR = '/glade/campaign/collections/gdex/data/d731000/gpm_3imerghh_v07'
OUT25  = '/glade/derecho/scratch/rneale/IMERG/3hrly/0.25deg'
OUT1   = '/glade/derecho/scratch/rneale/IMERG/3hrly/1deg'
TMPDIR = os.environ.get('TMPDIR', '/glade/derecho/scratch/rneale/imerg_tmp')

_CDO_PATHS = [
    '/glade/u/apps/derecho/25.10/spack/opt/spack/cdo/2.5.2/gcc/12.5.0/6rsl/bin/cdo',
    '/glade/u/apps/casper/25.10/spack/opt/spack/cdo/2.5.2/gcc/12.5.0/b6bq/bin/cdo',
]
CDO = next((p for p in _CDO_PATHS if os.path.exists(p)), 'cdo')

# ── IMERG source grid ──────────────────────────────────────────────────────
# HDF5 shape: (1, 3600, 1800) → dims (time, lon, lat); transpose to (lat, lon)
IMERG_NLAT = 1800
IMERG_NLON = 3600
IMERG_LAT  = np.round(np.linspace(-89.95, 89.95, IMERG_NLAT), 10)
IMERG_LON  = np.round(np.linspace(-179.95, 179.95, IMERG_NLON), 10)

# Clip source to 55°S-55°N before CDO remap (saves I/O, buffer beyond 50°)
LAT_CLIP_S = -55.0
LAT_CLIP_N =  55.0
_lat_mask  = (IMERG_LAT >= LAT_CLIP_S) & (IMERG_LAT <= LAT_CLIP_N)
CLIP_LAT   = IMERG_LAT[_lat_mask]
CLIP_NLAT  = _lat_mask.sum()
_lat_idx_s = np.searchsorted(IMERG_LAT, LAT_CLIP_S)
_lat_idx_n = np.searchsorted(IMERG_LAT, LAT_CLIP_N, side='right')

# ── Target grids (CDO description) ────────────────────────────────────────
_GRID25_TXT = """\
gridtype = lonlat
xsize    = 1440
ysize    = 400
xfirst   = -179.875
xinc     =    0.25
yfirst   = -49.875
yinc     =    0.25
"""

_GRID1_TXT = """\
gridtype = lonlat
xsize    = 360
ysize    = 100
xfirst   = -179.5
xinc     =    1.0
yfirst   = -49.5
yinc     =    1.0
"""

# Source grid description for CDO (clipped IMERG 0.1° regular grid)
# Must be explicit so CDO doesn't mis-classify as "generic coordinates"
_IMERG_SRC_GRID_TXT = (
    f'gridtype = lonlat\n'
    f'xsize    = {IMERG_NLON}\n'
    f'ysize    = {int(_lat_mask.sum())}\n'
    f'xfirst   = {IMERG_LON[0]:.4f}\n'
    f'xinc     = 0.1\n'
    f'yfirst   = {float(IMERG_LAT[_lat_mask][0]):.4f}\n'
    f'yinc     = 0.1\n'
)

# 10°S-10°N lat bounds (matches existing _10NS files: 80 pts at 0.25°, 20 at 1°)
LAT10_S = -10.0
LAT10_N =  10.0

# ── Fill value ─────────────────────────────────────────────────────────────
FILL = -9999.9


# ─────────────────────────────────────────────────────────────────────────────
# Step 1: Read one day of IMERG half-hourly files
# ─────────────────────────────────────────────────────────────────────────────

def read_day(year, month, day):
    """
    Read 48 half-hourly HDF5 files for one day.
    Returns (48, CLIP_NLAT, 3600) float32 array in mm/hr, S→N, W→E.
    """
    day_dir = os.path.join(IN_DIR, f'{year:04d}', f'{month:02d}', f'{day:02d}')
    files   = sorted(glob.glob(
        os.path.join(day_dir,
                     f'3B-HHR.MS.MRG.3IMERG.{year:04d}{month:02d}{day:02d}-*.HDF5')))

    n = len(files)
    if n == 0:
        raise FileNotFoundError(f'No IMERG files found in {day_dir}')
    if n != 48:
        print(f'    Warning: {year}-{month:02d}-{day:02d}: {n} files (expected 48)')

    data = np.full((n, CLIP_NLAT, IMERG_NLON), np.nan, dtype=np.float32)

    for i, fpath in enumerate(files):
        with h5py.File(fpath, 'r') as hf:
            pr = hf['Grid/precipitation'][0]   # (nlon=3600, nlat=1800)
            pr = pr.T                           # → (nlat=1800, nlon=3600)
            pr[pr < 0] = np.nan                # mask fill value (-9999.9)
            data[i] = pr[_lat_idx_s:_lat_idx_n]

    return data


# ─────────────────────────────────────────────────────────────────────────────
# Step 2: Average 6 half-hourly → 3-hourly (8 per day)
# ─────────────────────────────────────────────────────────────────────────────

def to_3hr(data_hh):
    """
    (n_hh, nlat, nlon) → (n_hh//6, nlat, nlon) nanmean over groups of 6.
    """
    n_hh  = data_hh.shape[0]
    n_3hr = n_hh // 6
    return np.nanmean(
        data_hh[:n_3hr * 6].reshape(n_3hr, 6, *data_hh.shape[1:]), axis=1
    ).astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Step 3: CDO weight generation + conservative remap
# ─────────────────────────────────────────────────────────────────────────────

def _write_tmp_grid(grid_txt):
    """Write a CDO grid description to a temp file, return its path."""
    tf = tempfile.NamedTemporaryFile(suffix='.txt', mode='w', delete=False)
    tf.write(grid_txt)
    tf.close()
    return tf.name


def gen_cdo_weights(sample_nc, grid_txt, weights_nc):
    """Pre-compute CDO conservative remap weights from sample_nc."""
    tgt = _write_tmp_grid(grid_txt)
    src = _write_tmp_grid(_IMERG_SRC_GRID_TXT)
    try:
        cmd = [CDO, '-f', 'nc4', '-O',
               f'gencon,{tgt}', f'-setgrid,{src}', sample_nc, weights_nc]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f'CDO gencon failed:\n{r.stderr[:400]}')
    finally:
        os.unlink(tgt); os.unlink(src)


def cdo_remap(in_nc, grid_txt, weights_nc, out_nc):
    """Conservative remap using pre-computed weights."""
    tgt = _write_tmp_grid(grid_txt)
    src = _write_tmp_grid(_IMERG_SRC_GRID_TXT)
    try:
        cmd = [CDO, '-f', 'nc4', '-O',
               f'remap,{tgt},{weights_nc}', f'-setgrid,{src}', in_nc, out_nc]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f'CDO remap failed:\n{r.stderr[:400]}')
    finally:
        os.unlink(tgt); os.unlink(src)


# ─────────────────────────────────────────────────────────────────────────────
# Step 4: Write temp per-day 0.1° NetCDF
# ─────────────────────────────────────────────────────────────────────────────

def write_day_nc(data_3hr, times, path):
    """Write (8, CLIP_NLAT, 3600) array to NetCDF with CF coords for CDO."""
    ds = xr.Dataset(
        {'precip': xr.DataArray(data_3hr, dims=['time', 'lat', 'lon'])},
        coords={'time': times, 'lat': CLIP_LAT, 'lon': IMERG_LON},
    )
    ds['lat'].attrs = {'standard_name': 'latitude',  'units': 'degrees_north', 'axis': 'Y'}
    ds['lon'].attrs = {'standard_name': 'longitude', 'units': 'degrees_east',  'axis': 'X'}
    ds.to_netcdf(path, mode='w',
        encoding={'precip': {'dtype': 'float32', '_FillValue': FILL}})


# ─────────────────────────────────────────────────────────────────────────────
# Step 5: Write final monthly NetCDF with TRMM-compatible metadata
# ─────────────────────────────────────────────────────────────────────────────

def write_final(ds_merged, outpath, tag):
    """Add TRMM-style metadata variables and write."""
    times = ds_merged['time'].values

    # Auxiliary time variables matching TRMM 3B42 format
    dates    = np.array([int(np.datetime_as_string(t, unit='D').replace('-', ''))
                         for t in times], dtype=np.int32)
    datesec  = np.array([int((t.astype('datetime64[s]')
                               - t.astype('datetime64[D]')).astype(int))
                          for t in times], dtype=np.int32)
    yyyymmddhh = np.array([int(np.datetime_as_string(t, unit='h').replace('T','').replace('-',''))
                            for t in times], dtype=np.int32)

    ds_out = ds_merged.copy()
    ds_out['precip'].attrs = {
        'long_name':      'Precipitation rate',
        'units':          'mm/hr',
        'source':         'GPM IMERG V07B 3-hourly mean',
        'delta_t':        '0000-00-00 03:00:00',
        'time_statistic': 'mean',
    }
    ds_out['date']      = xr.DataArray(dates,     dims=['time'])
    ds_out['datesec']   = xr.DataArray(datesec,   dims=['time'],
                              attrs={'units': 'seconds since start of day'})
    ds_out['yyyymmddhh']= xr.DataArray(yyyymmddhh, dims=['time'])
    ds_out.attrs = {
        'title':         f'IMERG V07B → TRMM 3B42 format: {tag}',
        'source_data':   'GPM IMERG V07B (3IMERGHH)',
        'Conventions':   'CF-1.6',
        'creation_date': _time.strftime('%Y-%m-%d %H:%M UTC'),
    }

    enc = {'precip': {'dtype': 'float32', '_FillValue': FILL, 'zlib': True, 'complevel': 4}}
    ds_out.to_netcdf(outpath, mode='w', encoding=enc)


# ─────────────────────────────────────────────────────────────────────────────
# Main: process one month
# ─────────────────────────────────────────────────────────────────────────────

def process_month(year, month):
    import pandas as pd

    n_days = calendar.monthrange(year, month)[1]
    tag    = f'{year:04d}{month:02d}'
    t0_tot = _time.time()

    # Output file paths
    out25        = os.path.join(OUT25, f'IMERG_3hr.{tag}.0p25deg.nc')
    out25_10ns   = os.path.join(OUT25, f'IMERG_3hr.{tag}.0p25deg.10NS.nc')
    out1         = os.path.join(OUT1,  f'IMERG_3hr.{tag}.1deg.nc')
    out1_10ns    = os.path.join(OUT1,  f'IMERG_3hr.{tag}.1deg.10NS.nc')

    if all(os.path.exists(p) for p in [out25, out25_10ns, out1, out1_10ns]):
        print(f'All output files already exist for {tag}, skipping.')
        return

    os.makedirs(TMPDIR, exist_ok=True)

    # Temp file lists for CDO mergetime
    day_files_25 = []
    day_files_1  = []

    # Weight files (computed once from first day)
    wts25 = os.path.join(TMPDIR, f'weights_{tag}_25deg.nc')
    wts1  = os.path.join(TMPDIR, f'weights_{tag}_1deg.nc')
    weights_ready = False

    print(f'\n=== {tag}: processing {n_days} days ===')

    for day in range(1, n_days + 1):
        t0_day = _time.time()
        print(f'  Day {day:02d}/{n_days}:', end='', flush=True)

        # 3-hourly timestamps for this day
        day_times = [pd.Timestamp(f'{year:04d}-{month:02d}-{day:02d}')
                     + pd.Timedelta(hours=3 * i) for i in range(8)]

        # Read + average
        data_hh  = read_day(year, month, day)   # (48, CLIP_NLAT, 3600)
        data_3hr = to_3hr(data_hh)              # (8, CLIP_NLAT, 3600)
        del data_hh
        print(f' read+avg:{_time.time()-t0_day:.0f}s', end='', flush=True)

        # Write temp 0.1° day file
        tmp_in  = os.path.join(TMPDIR, f'imerg_{tag}_day{day:02d}_01deg.nc')
        tmp_25  = os.path.join(TMPDIR, f'imerg_{tag}_day{day:02d}_25deg.nc')
        tmp_1   = os.path.join(TMPDIR, f'imerg_{tag}_day{day:02d}_1deg.nc')
        write_day_nc(data_3hr, day_times, tmp_in)
        del data_3hr

        # Generate CDO weights from first day
        if not weights_ready:
            t_w = _time.time()
            gen_cdo_weights(tmp_in, _GRID25_TXT, wts25)
            gen_cdo_weights(tmp_in, _GRID1_TXT,  wts1)
            weights_ready = True
            print(f' wts:{_time.time()-t_w:.0f}s', end='', flush=True)

        # CDO remap
        t_r = _time.time()
        cdo_remap(tmp_in, _GRID25_TXT, wts25, tmp_25)
        cdo_remap(tmp_in, _GRID1_TXT,  wts1,  tmp_1)
        print(f' remap:{_time.time()-t_r:.0f}s  total:{_time.time()-t0_day:.0f}s')

        os.unlink(tmp_in)
        day_files_25.append(tmp_25)
        day_files_1.append(tmp_1)

    # ── Merge daily files into monthly ────────────────────────────────────
    print('  Merging monthly...', flush=True)

    for res, day_files, out_global, out_10ns, label in [
        ('0.25°', day_files_25, out25,   out25_10ns, ''),
        ('1°',    day_files_1,  out1,    out1_10ns,  '.1x1'),
    ]:
        # CDO mergetime
        tmp_merged = os.path.join(TMPDIR, f'imerg_{tag}_merged{label}.nc')
        r = subprocess.run(
            [CDO, '-f', 'nc4', '-O', 'mergetime'] + day_files + [tmp_merged],
            capture_output=True, text=True, timeout=300)
        if r.returncode != 0:
            raise RuntimeError(f'CDO mergetime failed ({res}):\n{r.stderr[:400]}')

        # Load, add metadata, write final files
        ds = xr.open_dataset(tmp_merged).load()
        os.unlink(tmp_merged)

        # Global file
        write_final(ds, out_global, tag)
        print(f'  Wrote {os.path.basename(out_global)}')

        # 10°S-10°N subset
        ds_10ns = ds.sel(lat=slice(LAT10_S, LAT10_N))
        write_final(ds_10ns, out_10ns, tag + '_10NS')
        print(f'  Wrote {os.path.basename(out_10ns)}')

        ds.close()

    # Clean up day files and weight files
    for f in day_files_25 + day_files_1:
        if os.path.exists(f):
            os.unlink(f)
    for f in [wts25, wts1]:
        if os.path.exists(f):
            os.unlink(f)

    print(f'=== {tag} complete in {(_time.time()-t0_tot)/60:.1f} min ===')


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python imerg_to_3hr_process.py YYYY MM')
        sys.exit(1)
    process_month(int(sys.argv[1]), int(sys.argv[2]))
