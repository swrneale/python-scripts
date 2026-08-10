"""
hindcast_blocking_utils.py
Utility routines for blocking frequency analysis on S2S hindcasts.

Data layout (under DATA_DIR):
  {year}/{month:02d}/Z3_cesm2cam6v2_{DD}{mon}{YYYY}00z_d01_d46_m{MM}.nc
  • 46 lead days (d01–d46), 11+ ensemble members (m00–m21)
  • Z3 already on pressure levels (lev_p); Z500 = Z3.sel(lev_p=500)

Workflow
--------
1.  build_z500_cache()        – extract Z500 per (member, year)
                                  Output: z500_hindcast_m{MM:02d}_{YYYY}.nc
                                  dims: (start_date, lead_day, lat, lon)
2.  block_freq_all_leaddays() – compute blocking freq per lead day per member
3.  hindcast_blocking_figs   – plot results

Blocking algorithm: Davini et al. (2012), identical thresholds to blocking_utils.py

Cache file naming
-----------------
One file per (member, year):  z500_hindcast_m{MM:02d}_{YYYY}.nc
  • Years are fully independent — adding a new year never touches existing files.
  • block_freq_all_leaddays concatenates all available (or specified) years before
    computing blocking frequency.
/glade/work/rneale/git/python-scripts/blocking/claudeblocking
Performance notes
-----------------
• Each source file holds all 46 lead days for one (start_date, member) pair.
  build_z500_cache opens every source file exactly ONCE, extracting all lead days
  in a single read.  The previous per-lead-day outer loop caused N_lead × N_starts × N_mem
  file opens; the new approach uses only N_starts × N_mem opens (~46× fewer I/Os).
• Within each year, members are processed in parallel via ThreadPoolExecutor (n_workers=8).
• block_freq_all_leaddays opens each per-year member file lazily, concatenates years,
  then iterates over lead days in-memory — no repeated file opens.
"""

from __future__ import annotations

import datetime
import getpass
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr


# ── Default paths ─────────────────────────────────────────────────────────────

DATA_DIR  = Path('/glade/campaign/cesm/development/cross-wg/S2S/CESM2/S2SHINDCASTS/Z3/')
CACHE_DIR = Path('/glade/work/rneale/python-netcdf/hindcast_blocking')

_LEV500 = 500.0   # hPa

# ── Davini et al. (2012) blocking thresholds — identical to blocking_utils.py ─

_GHGN_THRESH = -5.0
_GHGS_THRESH =  0.0

_LAT_S  = 35.0
_LAT_N  = 75.0
_DLAT2D = 15.0

_BLAT0  = 60.00
_BLATN  = 78.85
_BLATS  = 41.25
_DELTAS = [-3.75, 0.0, 3.75]

_SEASON_MONTHS = {
    'DJF': {12, 1, 2},
    'MAM': {3, 4, 5},
    'JJA': {6, 7, 8},
    'SON': {9, 10, 11},
}

_FILE_RE = re.compile(
    r'Z3_cesm2cam6v2_(\d{2}\w{3}\d{4})00z_d\d+_d\d+_m(\d{2})\.nc'
)
# Matches per-(member, year) cache files: z500_hindcast_m00_2022.nc
_CACHE_RE = re.compile(r'z500_hindcast_m(\d+)_(\d{4})\.nc')


# ─────────────────────────────────────────────────────────────────────────────
# File inventory
# ─────────────────────────────────────────────────────────────────────────────

def list_hindcast_files(data_dir: Path | str = DATA_DIR,
                        years:   list[int] | None = None,
                        months:  list[int] | None = None) -> pd.DataFrame:
    """
    Enumerate all hindcast Z3 files under *data_dir*.

    Parameters
    ----------
    data_dir : root of the Z3 hindcast tree
    years    : restrict to these calendar years (None → all)
    months   : restrict to these calendar months 1–12 (None → all)

    Returns
    -------
    pd.DataFrame  columns: year, month, start_str, member, path
    """
    data_dir = Path(data_dir)
    rows = []

    year_dirs = sorted(d for d in data_dir.iterdir() if d.is_dir())
    if years is not None:
        year_dirs = [d for d in year_dirs if int(d.name) in set(years)]

    for ydir in year_dirs:
        month_dirs = sorted(d for d in ydir.iterdir() if d.is_dir())
        if months is not None:
            month_dirs = [d for d in month_dirs if int(d.name) in set(months)]

        for mdir in month_dirs:
            for fpath in sorted(mdir.glob('Z3_*.nc')):
                m = _FILE_RE.match(fpath.name)
                if m:
                    rows.append({
                        'year':      int(ydir.name),
                        'month':     int(mdir.name),
                        'start_str': m.group(1),           # e.g. '04jan1999'
                        'member':    int(m.group(2)),       # 0–10
                        'path':      str(fpath),
                    })

    df = pd.DataFrame(rows)
    print(
        f'list_hindcast_files: {len(df)} files  '
        f'| {df["year"].nunique()} years  '
        f'| {df["start_str"].nunique()} start dates  '
        f'| {df["member"].nunique()} members'
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Z500 cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _z500_cache_path(mem: int, year: int, cache_dir: Path) -> Path:
    """One cache file per (member, year)."""
    return cache_dir / f'z500_hindcast_m{mem:02d}_{year}.nc'


def _find_member_year_files(cache_dir: Path,
                             years: list[int] | None = None
                             ) -> dict[int, list[Path]]:
    """
    Scan cache_dir for per-(member, year) files.

    Returns {member_int: [sorted list of per-year file paths]}.
    Only includes years in *years* if provided.
    """
    result: dict[int, list[Path]] = {}
    for f in sorted(cache_dir.glob('z500_hindcast_m*.nc')):
        m = _CACHE_RE.match(f.name)
        if not m:
            continue
        mem = int(m.group(1))
        yr  = int(m.group(2))
        if years is not None and yr not in set(years):
            continue
        result.setdefault(mem, []).append(f)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Z500 cache build
# ─────────────────────────────────────────────────────────────────────────────

def build_z500_cache(data_dir:   Path | str = DATA_DIR,
                     cache_dir:  Path | str = CACHE_DIR,
                     years:      list[int] | None = None,
                     months:     list[int] | None = None,
                     n_leaddays: int  = 46,
                     overwrite:  bool = False,
                     n_workers:  int  = 8) -> None:
    """
    Extract Z500 from all hindcast files and save one NetCDF per (member, year).

    Output: {cache_dir}/z500_hindcast_m{MM:02d}_{YYYY}.nc
      dims: (start_date, lead_day, lat, lon)
      coord 'start_date': datetime64, chronologically sorted
      coord 'lead_day':   1-based integer

    Years are processed independently — adding a new year never overwrites or
    invalidates existing year files.  Each source file is opened exactly ONCE
    and all lead days are extracted together.  Within each year, members are
    processed in parallel using n_workers threads.

    Parameters
    ----------
    n_leaddays : lead days to retain per start date (max 46)
    overwrite  : re-compute even if cache file already exists
    n_workers  : number of parallel threads per year
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = list_hindcast_files(data_dir, years=years, months=months)
    if df.empty:
        raise RuntimeError('No hindcast files found — check data_dir / years / months.')

    years_found = sorted(df['year'].unique())

    # Read grid from first available file
    ds0 = xr.open_dataset(df['path'].iloc[0])
    lat = ds0['lat'].values
    lon = ds0['lon'].values
    ds0.close()

    # Summary: member counts can differ between years — report upfront
    year_member_counts = {yr: sorted(df[df['year'] == yr]['member'].unique())
                          for yr in years_found}
    n_total = sum(len(m) for m in year_member_counts.values())
    print(f'\nbuild_z500_cache: {len(years_found)} year(s)  →  {n_total} output files'
          f'  (n_workers={n_workers})')
    for yr, mems in year_member_counts.items():
        print(f'  {yr}: {len(mems)} members  (m{mems[0]:02d}–m{mems[-1]:02d})')
    print()

    for year in years_found:
        year_df      = df[df['year'] == year]
        year_members = sorted(year_df['member'].unique())  # only members present this year

        # Sort start dates chronologically (pd.to_datetime handles lowercase %b)
        starts_dt  = pd.to_datetime(
            sorted(year_df['start_str'].unique()), format='%d%b%Y'
        ).sort_values()
        starts_str = [dt.strftime('%d%b%Y').lower() for dt in starts_dt]

        # Index for this year:  member → start_str → path
        file_idx: dict[int, dict[str, str]] = {m: {} for m in year_members}
        for _, row in year_df.iterrows():
            file_idx[row['member']][row['start_str']] = row['path']

        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
        user    = getpass.getuser()

        print(f'  year {year}: {len(year_members)} members  '
              f'{len(starts_str)} start dates  '
              f'({starts_str[0]} → {starts_str[-1]})')

        def _process_member(mem: int,
                             _year:       int               = year,
                             _members:    list              = year_members,
                             _starts_str: list[str]         = starts_str,
                             _starts_dt:  pd.DatetimeIndex  = starts_dt,
                             _idx:        dict              = file_idx,
                             _now:        str               = now_utc,
                             _user:       str               = user) -> None:
            out_file = _z500_cache_path(mem, _year, cache_dir)
            if out_file.exists() and not overwrite:
                print(f'    m{mem:02d}: cache exists, skipping → {out_file.name}')
                return

            t0 = time.time()
            z500_arr = np.full(
                (len(_starts_str), n_leaddays, len(lat), len(lon)),
                np.nan, dtype=np.float32,
            )

            n_starts = len(_starts_str)
            for is_, start_s in enumerate(_starts_str):
                fpath = _idx[mem].get(start_s)
                if fpath is None:
                    continue
                ds     = xr.open_dataset(fpath)
                z3     = ds['Z3'].sel(lev_p=_LEV500).values   # (time, lat, lon)
                n_days = min(z3.shape[0], n_leaddays)
                z500_arr[is_, :n_days] = z3[:n_days]
                ds.close()
                if (is_ + 1) % 10 == 0 or (is_ + 1) == n_starts:
                    print(f'    m{mem:02d}: {is_+1}/{n_starts} start dates  '
                          f'({time.time()-t0:.0f}s elapsed)')

            da = xr.DataArray(
                z500_arr,
                dims=('start_date', 'lead_day', 'lat', 'lon'),
                coords={
                    'start_date': _starts_dt.values,   # datetime64, chronological
                    'lead_day':   np.arange(1, n_leaddays + 1),
                    'lat':        lat,
                    'lon':        lon,
                },
                name='Z500',
                attrs={'units': 'm', 'long_name': 'Geopotential height at 500 hPa'},
            )
            ds_out = da.to_dataset()
            ds_out.attrs = {
                'title':            'CESM2 S2S hindcast Z500 cache',
                'created':          _now,
                'created_by':       _user,
                'source_data_dir':  str(data_dir),
                'year':             str(_year),
                'member':           f'm{mem:02d}',
                'member_range':     f'm{_members[0]:02d}–m{_members[-1]:02d}',
                'n_members_year':   str(len(_members)),
                'n_leaddays':       str(n_leaddays),
                'start_date_range': f'{_starts_str[0]} to {_starts_str[-1]}',
                'n_start_dates':    str(len(_starts_str)),
                'blocking_ref':     'Davini et al. (2012)',
            }
            ds_out.to_netcdf(out_file)
            print(f'    m{mem:02d}: saved → {out_file.name}  ({time.time()-t0:.1f}s)')

        # Submit all members in sorted order; collect in that same order so
        # progress prints and error reporting follow member sequence.
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_process_member, mem) for mem in year_members]
            for mem, fut in zip(year_members, futures):
                exc = fut.exception()
                if exc:
                    print(f'    ERROR on m{mem:02d}: {exc}')

    print('\nbuild_z500_cache: done.')


def z500_cache_status(cache_dir: Path | str = CACHE_DIR) -> None:
    """Print which (member, year) cache files exist in cache_dir."""
    cache_dir = Path(cache_dir)
    member_year_files = _find_member_year_files(cache_dir)
    all_years   = sorted({int(_CACHE_RE.match(f.name).group(2))
                          for files in member_year_files.values()
                          for f in files})
    all_members = sorted(member_year_files.keys())
    n_files = sum(len(v) for v in member_year_files.values())
    print(f'Z500 cache: {n_files} file(s) in {cache_dir}')
    print(f'  members: {all_members}   years: {all_years}')
    for mem in all_members:
        have_years = [int(_CACHE_RE.match(f.name).group(2))
                      for f in member_year_files.get(mem, [])]
        missing = [y for y in all_years if y not in have_years]
        status = 'OK' if not missing else f'MISSING {missing}'
        print(f'  m{mem:02d}:  {sorted(have_years)}  {status}')


_G = 9.80665   # m s-2 — geopotential → geopotential height conversion


def era5_block_freq_1d(era5_path:  Path | str,
                        dates,
                        season:    str | None = None) -> xr.DataArray:
    """
    Compute ERA5 1D blocking frequency matching the hindcast start-date
    calendar distribution.

    For each unique (month, day) in *dates*, every ERA5 timestep with that
    calendar date (across all available ERA5 years) is selected.  Blocking
    frequency is the fraction of those days that satisfy the 1D criterion.

    ERA5 'z' is geopotential (m² s⁻²); divide by g=9.80665 to get m.

    Parameters
    ----------
    era5_path : path to ERA5 Z500 daily-mean file (variable name 'z')
    dates     : hindcast start dates (any datetime-like: list, DatetimeIndex…)
    season    : additionally restrict ERA5 to this season
                ('DJF'|'MAM'|'JJA'|'SON'|None = all months)

    Returns
    -------
    DataArray(lon)  blocking frequency in [0, 1]
    """
    era5_path = Path(era5_path)
    ds  = xr.open_dataset(era5_path)
    # Rename time before passing to _blocking_1d, which uses hindcast as the time dim.
    # Keep lats wide enough to cover BLATN+delta_max = 78.85+3.75 = 82.6°N;
    # _blocking_1d does its own sel(lat=slice(_LAT_S, _LAT_N)) internally.
    z   = (ds['z'] / _G).rename({'time': 'hindcast'})   # geopotential → height (m)

    # Unique (month, day) pairs derived from hindcast start dates
    dti           = pd.DatetimeIndex(dates)
    month_day_set = set(zip(dti.month, dti.day))

    era5_time = pd.DatetimeIndex(z.hindcast.values)
    keep      = np.array([(t.month, t.day) in month_day_set for t in era5_time])

    if season is not None:
        valid_months = _SEASON_MONTHS[season.upper()]
        keep &= era5_time.month.isin(valid_months)

    z = z.isel(hindcast=keep)
    if z.sizes['hindcast'] == 0:
        raise ValueError('No ERA5 days matched the given dates / season filter.')

    bf = _blocking_1d(z).mean('hindcast').compute()
    ds.close()
    print(f'era5_block_freq_1d: {keep.sum()} days  '
          f'mean blocking {float(bf.mean()) * 100:.1f}%')
    return bf


def era5_block_freq_2d(era5_path:  Path | str,
                       dates,
                       season:    str | None = None) -> xr.DataArray:
    """
    Compute ERA5 2D blocking frequency matching the hindcast start-date
    calendar distribution.

    Parameters
    ----------
    era5_path : path to ERA5 Z500 daily-mean file (variable name 'z')
    dates     : hindcast start dates (any datetime-like: list, DatetimeIndex…)
    season    : additionally restrict ERA5 to this season

    Returns
    -------
    DataArray(lat, lon)  blocking frequency in [0, 1]
    """
    era5_path = Path(era5_path)
    ds  = xr.open_dataset(era5_path)
    z   = (ds['z'] / _G).rename({'time': 'hindcast'})

    dti           = pd.DatetimeIndex(dates)
    month_day_set = set(zip(dti.month, dti.day))
    era5_time     = pd.DatetimeIndex(z.hindcast.values)
    keep          = np.array([(t.month, t.day) in month_day_set for t in era5_time])

    if season is not None:
        valid_months = _SEASON_MONTHS[season.upper()]
        keep &= era5_time.month.isin(valid_months)

    z = z.isel(hindcast=keep)
    if z.sizes['hindcast'] == 0:
        raise ValueError('No ERA5 days matched the given dates / season filter.')

    bf = _blocking_2d(z).mean('hindcast').compute()
    ds.close()
    print(f'era5_block_freq_2d: {keep.sum()} days  '
          f'mean blocking {float(bf.mean()) * 100:.1f}%')
    return bf


def era5_season_clim_1d(era5_path: Path | str,
                        season:    str | None = None) -> xr.DataArray:
    """
    1D blocking frequency over ALL days in *season* across the full ERA5 record.

    Unlike era5_block_freq_1d, no calendar-date filtering is applied — every
    season day in the file contributes equally to the climatology.

    Returns DataArray(lon) blocking frequency in [0, 1].
    """
    era5_path = Path(era5_path)
    ds        = xr.open_dataset(era5_path)
    z         = (ds['z'] / _G).rename({'time': 'hindcast'})

    if season is not None:
        valid_months = _SEASON_MONTHS[season.upper()]
        era5_time    = pd.DatetimeIndex(z.hindcast.values)
        keep         = era5_time.month.isin(valid_months)
        z            = z.isel(hindcast=keep)

    if z.sizes['hindcast'] == 0:
        raise ValueError('No ERA5 days matched the season filter.')

    bf = _blocking_1d(z).mean('hindcast').compute()
    ds.close()
    n = z.sizes['hindcast']
    print(f'era5_season_clim_1d: {n} days  mean blocking {float(bf.mean()) * 100:.1f}%')
    return bf


def era5_season_clim_gradient(era5_path: Path | str,
                               season:    str | None = None,
                               diag:      str = 'GHGS') -> xr.DataArray:
    """
    Mean Z500 gradient (GHGS or GHGN) over ALL days in *season* across the
    full ERA5 record — the climatological gradient strength.

    Parameters
    ----------
    era5_path : path to ERA5 Z500 daily-mean file (variable name 'z')
    season    : 'DJF'|'MAM'|'JJA'|'SON'|None (all months)
    diag      : 'GHGS' or 'GHGN'

    Returns
    -------
    DataArray(lon)  gradient in m / degree-lat
    """
    if diag not in ('GHGS', 'GHGN'):
        raise ValueError(f'diag must be "GHGS" or "GHGN", got "{diag}"')
    era5_path = Path(era5_path)
    ds        = xr.open_dataset(era5_path)
    z         = (ds['z'] / _G).rename({'time': 'hindcast'})

    if season is not None:
        valid_months = _SEASON_MONTHS[season.upper()]
        era5_time    = pd.DatetimeIndex(z.hindcast.values)
        keep         = era5_time.month.isin(valid_months)
        z            = z.isel(hindcast=keep)

    if z.sizes['hindcast'] == 0:
        raise ValueError('No ERA5 days matched the season filter.')

    grad = (_ghgs_1d(z) if diag == 'GHGS' else _ghgn_1d(z)).mean('hindcast').compute()
    ds.close()
    n = z.sizes['hindcast']
    print(f'era5_season_clim_gradient ({diag}): {n} days  mean {float(grad.mean()):.2f} m/deg')
    return grad


def era5_season_clim_2d(era5_path: Path | str,
                        season:    str | None = None) -> xr.DataArray:
    """
    2D blocking frequency over ALL days in *season* across the full ERA5 record.

    Returns DataArray(lat, lon) blocking frequency in [0, 1].
    """
    era5_path = Path(era5_path)
    ds        = xr.open_dataset(era5_path)
    z         = (ds['z'] / _G).rename({'time': 'hindcast'})

    if season is not None:
        valid_months = _SEASON_MONTHS[season.upper()]
        era5_time    = pd.DatetimeIndex(z.hindcast.values)
        keep         = era5_time.month.isin(valid_months)
        z            = z.isel(hindcast=keep)

    if z.sizes['hindcast'] == 0:
        raise ValueError('No ERA5 days matched the season filter.')

    bf = _blocking_2d(z).mean('hindcast').compute()
    ds.close()
    n = z.sizes['hindcast']
    print(f'era5_season_clim_2d: {n} days  mean blocking {float(bf.mean()) * 100:.1f}%')
    return bf


# ─────────────────────────────────────────────────────────────────────────────
# ERA5 per-year verification helpers
# ─────────────────────────────────────────────────────────────────────────────

ERA5_VERIFY_DIR = Path('/glade/derecho/scratch/rneale/ERA5/dmean/1deg')
_ERA5_VERIFY_PAT = 'z500_test_era5_download_{year}_dmean_1x1.nc'


def _load_era5_verify_z500(era5_dir: Path, years: list[int]) -> xr.DataArray:
    """
    Load and concatenate per-year ERA5 verification files; return Z500 in m.

    Files use 'valid_time' as the time coordinate and have a pressure_level
    dimension of size 1 that is squeezed away.  'valid_time' is renamed to
    'time' for consistency with downstream helpers.
    """
    das = []
    for yr in sorted(set(years)):
        fpath = era5_dir / _ERA5_VERIFY_PAT.format(year=yr)
        if not fpath.exists():
            print(f'  WARNING: ERA5 verify file missing: {fpath.name}')
            continue
        ds = xr.open_dataset(fpath)
        z  = (ds['z'] / _G).squeeze('pressure_level', drop=True)
        z  = z.rename({'valid_time': 'time'})
        das.append(z)
        ds.close()
    if not das:
        raise FileNotFoundError(f'No ERA5 verification files found in {era5_dir}')
    return xr.concat(das, dim='time')


def era5_verify_freq_1d(era5_dir: Path | str,
                        years:    list[int],
                        season:   str | None = None) -> xr.DataArray:
    """
    ERA5 1D blocking frequency for the given hindcast years filtered to *season*.

    Loads per-year ERA5 files, selects all season days, and returns a single
    blocking-frequency line — the observed truth for comparison with hindcast lines.

    Returns DataArray(lon) blocking frequency in [0, 1].
    """
    era5_dir = Path(era5_dir)
    z_all    = _load_era5_verify_z500(era5_dir, years)

    if season is not None:
        valid_months = _SEASON_MONTHS[season.upper()]
        era5_time    = pd.DatetimeIndex(z_all.time.values)
        keep         = era5_time.month.isin(valid_months)
        z_all        = z_all.isel(time=keep)

    if z_all.sizes['time'] == 0:
        raise ValueError('No ERA5 verify days matched years / season filter.')

    z_all = z_all.rename({'time': 'hindcast'})
    bf    = _blocking_1d(z_all).mean('hindcast').compute()
    print(f'era5_verify_freq_1d: {z_all.sizes["hindcast"]} days  '
          f'mean blocking {float(bf.mean()) * 100:.1f}%')
    return bf


def era5_verify_freq_2d(era5_dir: Path | str,
                        years:    list[int],
                        season:   str | None = None) -> xr.DataArray:
    """
    ERA5 2D blocking frequency for the given hindcast years filtered to *season*.

    Returns DataArray(lat, lon) blocking frequency in [0, 1].
    """
    era5_dir = Path(era5_dir)
    z_all    = _load_era5_verify_z500(era5_dir, years)

    if season is not None:
        valid_months = _SEASON_MONTHS[season.upper()]
        era5_time    = pd.DatetimeIndex(z_all.time.values)
        keep         = era5_time.month.isin(valid_months)
        z_all        = z_all.isel(time=keep)

    if z_all.sizes['time'] == 0:
        raise ValueError('No ERA5 verify days matched years / season filter.')

    z_all = z_all.rename({'time': 'hindcast'})
    bf    = _blocking_2d(z_all).mean('hindcast').compute()
    print(f'era5_verify_freq_2d: {z_all.sizes["hindcast"]} days  '
          f'mean blocking {float(bf.mean()) * 100:.1f}%')
    return bf


def era5_verify_gradient(era5_dir: Path | str,
                          years:    list[int],
                          season:   str | None = None,
                          diag:     str = 'GHGS') -> xr.DataArray:
    """
    Mean Z500 gradient (GHGS or GHGN) for the given hindcast years filtered
    to *season* — the observed gradient strength for verification.

    Parameters
    ----------
    era5_dir : directory containing per-year ERA5 verification files
    years    : hindcast years to include
    season   : 'DJF'|'MAM'|'JJA'|'SON'|None
    diag     : 'GHGS' or 'GHGN'

    Returns
    -------
    DataArray(lon)  gradient in m / degree-lat
    """
    if diag not in ('GHGS', 'GHGN'):
        raise ValueError(f'diag must be "GHGS" or "GHGN", got "{diag}"')
    era5_dir = Path(era5_dir)
    z_all    = _load_era5_verify_z500(era5_dir, years)

    if season is not None:
        valid_months = _SEASON_MONTHS[season.upper()]
        era5_time    = pd.DatetimeIndex(z_all.time.values)
        keep         = era5_time.month.isin(valid_months)
        z_all        = z_all.isel(time=keep)

    if z_all.sizes['time'] == 0:
        raise ValueError('No ERA5 verify days matched years / season filter.')

    z_all = z_all.rename({'time': 'hindcast'})
    grad  = (_ghgs_1d(z_all) if diag == 'GHGS' else _ghgn_1d(z_all)).mean('hindcast').compute()
    print(f'era5_verify_gradient ({diag}): {z_all.sizes["hindcast"]} days  '
          f'mean {float(grad.mean()):.2f} m/deg')
    return grad


def load_z500_for_leadday(lead_day:  int,
                          cache_dir: Path | str = CACHE_DIR,
                          years:     list[int] | None = None) -> xr.DataArray:
    """
    Load cached Z500 for *lead_day* (1-based), concatenating all available years.

    Returns DataArray(hindcast, member, lat, lon).
    """
    cache_dir        = Path(cache_dir)
    member_year_map  = _find_member_year_files(cache_dir, years=years)
    if not member_year_map:
        raise FileNotFoundError(
            f'No Z500 cache files in {cache_dir}\n  → run build_z500_cache() first.'
        )

    member_das = []
    for mem, files in sorted(member_year_map.items()):
        year_das = [xr.open_dataset(f)['Z500'].sel(lead_day=lead_day)
                    for f in files]
        da = xr.concat(year_das, dim='start_date') if len(year_das) > 1 else year_das[0]
        da = da.assign_coords(member=mem).expand_dims('member')
        member_das.append(da)

    return (xr.concat(member_das, dim='member')
              .rename({'start_date': 'hindcast'})
              .transpose('hindcast', 'member', 'lat', 'lon'))


# ─────────────────────────────────────────────────────────────────────────────
# Blocking algorithms  (same metric as blocking_utils.py)
# ─────────────────────────────────────────────────────────────────────────────

def _blocking_1d(z500: xr.DataArray) -> xr.DataArray:
    """
    1D blocking mask (Davini et al. 2012) for arbitrary-leading dims.

    Slices to [_LAT_S, _LAT_N] = [35, 75]°N, matching blocking_utils.py.
    BLATN (78.85°) lies outside this range and snaps to the array maximum
    (~75°N), which is intentional — it matches the effective algorithm used
    in all climatological comparisons from blocking_utils.py.
    """
    z500_s   = z500.sel(lat=slice(_LAT_S, _LAT_N))
    lat_vals = z500_s.lat.values

    def _nidx(t: float) -> int:
        return int(np.argmin(np.abs(lat_vals - t)))

    is_blocked = None
    for d in _DELTAS:
        zn = z500_s.isel(lat=_nidx(_BLATN + d))
        z0 = z500_s.isel(lat=_nidx(_BLAT0 + d))
        zs = z500_s.isel(lat=_nidx(_BLATS + d))
        dlat_n = float(lat_vals[_nidx(_BLATN + d)] - lat_vals[_nidx(_BLAT0 + d)])
        dlat_s = float(lat_vals[_nidx(_BLAT0 + d)] - lat_vals[_nidx(_BLATS + d)])
        ghgn   = (zn - z0) / dlat_n
        ghgs   = (z0 - zs) / dlat_s
        mask_i = (ghgs > _GHGS_THRESH) & (ghgn < _GHGN_THRESH)
        is_blocked = mask_i if is_blocked is None else (is_blocked | mask_i)
    return is_blocked


def _blocking_2d(z500: xr.DataArray) -> xr.DataArray:
    """
    2D blocking mask (Davini et al. 2012) for arbitrary-leading dims.

    Input z500 must include lat in [_LAT_S, _LAT_N].
    Returns boolean DataArray with same dims as input.
    """
    z500_s  = z500.sel(lat=slice(_LAT_S, _LAT_N))
    lat_c   = z500_s.lat.values
    lat_n_t = np.clip(lat_c + _DLAT2D,
                      float(z500_s.lat.min()), float(z500_s.lat.max()))
    lat_s_t = np.clip(lat_c - _DLAT2D,
                      float(z500_s.lat.min()), float(z500_s.lat.max()))
    z_n = z500_s.sel(lat=lat_n_t, method='nearest').assign_coords(lat=lat_c)
    z_s = z500_s.sel(lat=lat_s_t, method='nearest').assign_coords(lat=lat_c)
    ghgn = (z_n - z500_s) / _DLAT2D
    ghgs = (z500_s - z_s) / _DLAT2D
    return (ghgs > _GHGS_THRESH) & (ghgn < _GHGN_THRESH)


def _ghgs_1d(z500: xr.DataArray) -> xr.DataArray:
    """
    South Z500 gradient (GHGS) at blocking latitudes averaged over 3 delta offsets.

    Returns DataArray(..., lon) in m / degree-lat.  Positive values indicate
    the normal equatorward decrease of Z500 (blocking criterion: GHGS > 0).
    """
    z500_s   = z500.sel(lat=slice(_LAT_S, _LAT_N))
    lat_vals = z500_s.lat.values

    def _nidx(t: float) -> int:
        return int(np.argmin(np.abs(lat_vals - t)))

    parts = []
    for d in _DELTAS:
        z0     = z500_s.isel(lat=_nidx(_BLAT0 + d))
        zs     = z500_s.isel(lat=_nidx(_BLATS + d))
        dlat_s = float(lat_vals[_nidx(_BLAT0 + d)] - lat_vals[_nidx(_BLATS + d)])
        parts.append((z0 - zs) / dlat_s)
    return (parts[0] + parts[1] + parts[2]) / 3.0


def _ghgn_1d(z500: xr.DataArray) -> xr.DataArray:
    """
    North Z500 gradient (GHGN) at blocking latitudes averaged over 3 delta offsets.

    Returns DataArray(..., lon) in m / degree-lat.  Negative values indicate
    the reversed poleward gradient associated with blocking (criterion: GHGN < -5).
    """
    z500_s   = z500.sel(lat=slice(_LAT_S, _LAT_N))
    lat_vals = z500_s.lat.values

    def _nidx(t: float) -> int:
        return int(np.argmin(np.abs(lat_vals - t)))

    parts = []
    for d in _DELTAS:
        zn     = z500_s.isel(lat=_nidx(_BLATN + d))
        z0     = z500_s.isel(lat=_nidx(_BLAT0 + d))
        dlat_n = float(lat_vals[_nidx(_BLATN + d)] - lat_vals[_nidx(_BLAT0 + d)])
        parts.append((zn - z0) / dlat_n)
    return (parts[0] + parts[1] + parts[2]) / 3.0


def _filter_by_season(z500: xr.DataArray,
                       season: str) -> xr.DataArray:
    """
    Filter hindcast dimension to those whose start date falls in *season*.

    Expects hindcast coordinate to be datetime64.
    """
    valid_months = _SEASON_MONTHS[season.upper()]
    months = pd.DatetimeIndex(z500.hindcast.values).month
    keep   = months.isin(valid_months)
    return z500.isel(hindcast=keep)


# ─────────────────────────────────────────────────────────────────────────────
# Per-lead-day blocking frequency
# ─────────────────────────────────────────────────────────────────────────────

def block_freq_for_leadday(z500: xr.DataArray,
                            block_diag: str) -> xr.DataArray:
    """
    Compute blocking diagnostic per ensemble member for one lead day.

    Parameters
    ----------
    z500       : DataArray(hindcast, member, lat, lon)
    block_diag : '1D'  – 1D blocking frequency [0, 1]
                 '2D'  – 2D blocking frequency [0, 1]
                 'GHGS' – mean south gradient strength (m / degree-lat)
                 'GHGN' – mean north gradient strength (m / degree-lat)

    Returns
    -------
    DataArray(member, lon)       for '1D', 'GHGS', 'GHGN'
    DataArray(member, lat, lon)  for '2D'
    """
    if block_diag == '1D':
        return _blocking_1d(z500).mean('hindcast').compute()
    elif block_diag == '2D':
        return _blocking_2d(z500).mean('hindcast').compute()
    elif block_diag == 'GHGS':
        return _ghgs_1d(z500).mean('hindcast').compute()
    elif block_diag == 'GHGN':
        return _ghgn_1d(z500).mean('hindcast').compute()
    else:
        raise ValueError(f'block_diag must be "1D", "2D", "GHGS", or "GHGN"; got "{block_diag}"')


def block_freq_all_leaddays(n_leaddays:        int  = 46,
                             block_diag:        str  = '1D',
                             cache_dir:         Path | str = CACHE_DIR,
                             years:             list[int] | None = None,
                             members:           list[int] | None = None,
                             save_block_cache:  bool = True,
                             overwrite_block:   bool = False,
                             season:            str | None = None) -> dict:
    """
    Compute blocking frequency for every lead day.

    Lazily opens all per-(member, year) cache files once, concatenates years
    along start_date, then iterates over lead days in-memory — no repeated
    file opens across lead days.

    Parameters
    ----------
    n_leaddays       : number of lead days to process (max 46)
    block_diag       : '1D' or '2D'
    cache_dir        : directory containing z500 cache files
    years            : restrict to these years (None → all cached years)
    members          : restrict to these member numbers (None → all available).
                       Use this to avoid mixing members with different year coverage,
                       e.g. members=list(range(22)) for m00–m21 only.
    save_block_cache : save blocking-freq arrays alongside Z500 cache
    overwrite_block  : re-compute even if blocking cache file exists
    season           : 'DJF' | 'MAM' | 'JJA' | 'SON' | None (all months)

    Returns
    -------
    dict  { lead_day (int) : DataArray(member, [lat,] lon) }
    """
    cache_dir = Path(cache_dir)

    member_year_map = _find_member_year_files(cache_dir, years=years)
    if not member_year_map:
        raise RuntimeError(
            f'No Z500 cache files found in {cache_dir}\n  → run build_z500_cache() first.'
        )

    all_years = sorted({int(_CACHE_RE.match(f.name).group(2))
                        for files in member_year_map.values() for f in files})

    # Apply explicit member filter if requested.
    if members is not None:
        members_set = set(members)
        dropped = [m for m in sorted(member_year_map) if m not in members_set]
        member_year_map = {m: v for m, v in member_year_map.items() if m in members_set}
        if not member_year_map:
            raise RuntimeError('No cache files match the requested members list.')
        if dropped:
            print(f'  NOTE: restricting to {len(member_year_map)} member(s) '
                  f'(m{min(members_set):02d}–m{max(members_set):02d}); '
                  f'excluded {len(dropped)} member(s) outside requested range.')

    # Open per-(member, year) files with dask chunked by lead_day.
    # Without chunks=, xr.concat triggers an immediate numpy load of ALL data —
    # 51 members × 5 years × ~400 MB per file easily exceeds kernel memory.
    # With chunks={'lead_day': 1}, concat builds a dask graph; data is read
    # only when .compute() is called, one lead day at a time (~57 MB peak per member).
    member_ds: dict[int, xr.DataArray] = {}
    for mem, files in sorted(member_year_map.items()):
        year_das = [xr.open_dataset(f, chunks={'lead_day': 1})['Z500'] for f in files]
        member_ds[mem] = (xr.concat(year_das, dim='start_date')
                          if len(year_das) > 1 else year_das[0])

    members  = sorted(member_ds.keys())
    year_tag = (f'{all_years[0]}' if len(all_years) == 1
                else f'{all_years[0]}-{all_years[-1]}')
    varname  = f'BLOCK_{block_diag}'
    seas_tag = f'_{season}' if season else ''

    _VALID_DIAGS = ('1D', '2D', 'GHGS', 'GHGN')
    if block_diag not in _VALID_DIAGS:
        raise ValueError(f'block_diag must be one of {_VALID_DIAGS}; got "{block_diag}"')

    print(
        f'block_freq_all_leaddays: block_diag={block_diag}  '
        f'season={season or "all"}  n_leaddays={n_leaddays}  '
        f'years={all_years}  members={members}\n'
    )

    block_freq: dict[int, xr.DataArray] = {}

    for lead in range(1, n_leaddays + 1):
        block_file = (cache_dir /
                      f'block_{block_diag}_leadday_{lead:03d}_{year_tag}{seas_tag}.nc')

        if block_file.exists() and not overwrite_block:
            print(f'  lead day {lead:3d}: reading block cache')
            block_freq[lead] = xr.open_dataset(block_file)[varname]
            continue

        t0 = time.time()

        # Process one member at a time to avoid loading all members simultaneously.
        # Concatenating all members at once (old approach) created a
        # (n_hindcast_max × n_members × lat × lon) array that OOMs when members
        # span different year subsets (xarray NaN-pads to the longest member).
        member_bfs = []
        n_hind_max = 0

        for mem in members:
            da = (member_ds[mem]
                  .sel(lead_day=lead)
                  .rename({'start_date': 'hindcast'}))

            if season is not None:
                da = _filter_by_season(da, season)
                if da.sizes['hindcast'] == 0:
                    continue

            n_hind_max = max(n_hind_max, da.sizes['hindcast'])

            if block_diag == '1D':
                bf_mem = _blocking_1d(da).mean('hindcast').compute()
            elif block_diag == '2D':
                bf_mem = _blocking_2d(da).mean('hindcast').compute()
            elif block_diag == 'GHGS':
                bf_mem = _ghgs_1d(da).mean('hindcast').compute()
            else:  # GHGN
                bf_mem = _ghgn_1d(da).mean('hindcast').compute()
            member_bfs.append(bf_mem.assign_coords(member=mem).expand_dims('member'))

        if not member_bfs:
            print(f'  lead day {lead:3d}: no hindcasts for season {season} — skip')
            continue

        bf = xr.concat(member_bfs, dim='member')
        block_freq[lead] = bf

        if save_block_cache:
            bf.rename(varname).to_dataset().to_netcdf(block_file)

        if block_diag in ('1D', '2D'):
            val_str = f'mean blocking {bf.values.mean() * 100.:.1f}%'
        else:
            val_str = f'mean {block_diag} {bf.values.mean():.2f} m/deg'
        print(f'  lead day {lead:3d}: {n_hind_max} hindcasts  {val_str}  ({time.time()-t0:.1f}s)')

    for ds in member_ds.values():
        ds.close()

    print('\nblock_freq_all_leaddays: done.')
    return block_freq
