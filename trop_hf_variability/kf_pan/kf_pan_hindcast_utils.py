"""
kf_pan_hindcast_utils.py
Wheeler-Kiladis spectral analysis of CESM2 S2S PRECT hindcasts as a function of
lead day.  Companion to kf_pan_utils.py (free-running) — modeled on
hindcast_blocking_utils.py.

Data layout
-----------
Source files under DATA_DIR (flat directory of daily-mean h2 output):
  cesm2cam6v2.{MM}.{YYYY-MM-DD}.{EE}.cam.h2.{YYYY-MM-DD}-00000.nc
  • MM  — start-month tag (2-digit)
  • EE  — ensemble member (2-digit)
  • Each file: 46 lead days × 192 lat × 288 lon of PRECC + PRECL (m/s)

Workflow
--------
1.  build_prect_cache()          — extract PRECT = (PRECC+PRECL) × 86.4e6 (mm/day)
                                    Output: prect_hindcast_m{EE:02d}_{YYYY}.nc
                                    dims: (start_date, lead_day, lat, lon), lat
                                    clipped to ±LAT_BOUND_CACHE (25° by default).
2.  load_prect_segments()        — build a list of (N, nlat, mlon) segments, one
                                    per (start_date × member), starting at a
                                    requested lead day and running n_day_win days.
3.  wk_spectrum_lead_day()       — pool those segments and delegate to
                                    kf_pan_utils.compute_wk_spectrum_from_segments.
4.  wk_spectra_lead_days()       — convenience wrapper for a list of lead days.
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

# Silence HDF5's low-level C-library error traces to stderr.  netCDF probes
# non-HDF5 files during format detection and HDF5 dumps a diag stack every
# time; the Python-level FileNotFoundError still surfaces cleanly.
try:
    import h5py as _h5py
    _h5py._errors.silence_errors()
except Exception:
    pass

import kf_pan_utils as kfp


# ── Default paths ─────────────────────────────────────────────────────────────

DATA_DIR        = Path('/glade/campaign/cesm/development/cross-wg/S2S/CESM2/S2SHINDCASTS/daily')
CACHE_DIR       = Path('/glade/work/rneale/python-netcdf/hindcast_kf_pan')
IMERG_DAILY_DIR = Path('/glade/derecho/scratch/rneale/IMERG/daily/1deg')

# Lat window kept in the cache — wider than the standard KF ±15 belt so
# that lat_bound can be tightened at analysis time without rebuilding.
LAT_BOUND_CACHE = 25.0

# PRECT unit conversion: (PRECC+PRECL) m/s → mm/day
_MPS_TO_MMDAY = 86400.0 * 1000.0

_SEASON_MONTHS = {
    'DJF': {12, 1, 2},
    'MAM': {3, 4, 5},
    'JJA': {6, 7, 8},
    'SON': {9, 10, 11},
}

# Match daily h2 hindcast files (skip the older 'cesm2cam6' variant with no v2).
# Two naming conventions exist in the archive:
#   pre-2002: cesm2cam6v2.MM.YYYY-MM-DD.EE.cam.h2.…   (2-digit month prefix)
#   2005+   : cesm2cam6v2.YYYY-MM-DD.EE.cam.h2.…      (no prefix)
_FILE_RE = re.compile(
    r'cesm2cam6v2\.(?:\d{2}\.)?(\d{4}-\d{2}-\d{2})\.(\d{2})\.cam\.h2\.'
    r'\d{4}-\d{2}-\d{2}-\d+\.nc$'
)
_CACHE_RE = re.compile(r'prect_hindcast_m(\d{2})_(\d{4})\.nc$')


# ─────────────────────────────────────────────────────────────────────────────
# File inventory
# ─────────────────────────────────────────────────────────────────────────────

def list_hindcast_files(data_dir: Path | str = DATA_DIR,
                        years:   list[int] | None = None,
                        months:  list[int] | None = None) -> pd.DataFrame:
    """Enumerate cesm2cam6v2 daily h2 files in *data_dir*.

    Returns a DataFrame with columns: year, month, start_date, member, path.
    """
    data_dir = Path(data_dir)
    rows = []
    for f in sorted(data_dir.glob('cesm2cam6v2.*.cam.h2.*.nc')):
        m = _FILE_RE.match(f.name)
        if not m:
            continue
        start_date = pd.to_datetime(m.group(1))
        rows.append({
            'year':       start_date.year,
            'month':      start_date.month,
            'start_date': start_date,
            'member':     int(m.group(2)),
            'path':       str(f),
        })
    df = pd.DataFrame(rows)
    if years is not None:
        df = df[df['year'].isin(set(years))]
    if months is not None:
        df = df[df['month'].isin(set(months))]
    df = df.sort_values(['year', 'start_date', 'member']).reset_index(drop=True)
    print(
        f'list_hindcast_files: {len(df)} files  '
        f'| {df["year"].nunique()} years  '
        f'| {df["start_date"].nunique()} start dates  '
        f'| {df["member"].nunique()} members'
    )
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PRECT cache helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cache_path(mem: int, year: int, cache_dir: Path) -> Path:
    return cache_dir / f'prect_hindcast_m{mem:02d}_{year}.nc'


def _find_member_year_files(cache_dir: Path,
                             years: list[int] | None = None,
                             members: list[int] | None = None
                             ) -> dict[int, list[Path]]:
    """{member → sorted list of per-year cache paths}, optionally filtered."""
    result: dict[int, list[Path]] = {}
    year_set   = set(years)   if years   is not None else None
    member_set = set(members) if members is not None else None
    for f in sorted(cache_dir.glob('prect_hindcast_m*.nc')):
        m = _CACHE_RE.match(f.name)
        if not m:
            continue
        mem, yr = int(m.group(1)), int(m.group(2))
        if member_set is not None and mem not in member_set:
            continue
        if year_set   is not None and yr  not in year_set:
            continue
        result.setdefault(mem, []).append(f)
    return result


def prect_cache_status(cache_dir: Path | str = CACHE_DIR) -> None:
    """Print (member, year) coverage of the PRECT cache."""
    cache_dir = Path(cache_dir)
    member_year_files = _find_member_year_files(cache_dir)
    all_years   = sorted({int(_CACHE_RE.match(f.name).group(2))
                          for files in member_year_files.values() for f in files})
    all_members = sorted(member_year_files.keys())
    n_files = sum(len(v) for v in member_year_files.values())
    print(f'PRECT cache: {n_files} file(s) in {cache_dir}')
    print(f'  members: {all_members}   years: {all_years}')
    for mem in all_members:
        have = sorted(int(_CACHE_RE.match(f.name).group(2))
                      for f in member_year_files.get(mem, []))
        missing = [y for y in all_years if y not in have]
        tag = 'OK' if not missing else f'MISSING {missing}'
        print(f'  m{mem:02d}: {have} {tag}')


# ─────────────────────────────────────────────────────────────────────────────
# Cache build
# ─────────────────────────────────────────────────────────────────────────────

def build_prect_cache(data_dir:        Path | str = DATA_DIR,
                      cache_dir:       Path | str = CACHE_DIR,
                      years:           list[int] | None = None,
                      months:          list[int] | None = None,
                      n_leaddays:      int  = 46,
                      lat_bound_cache: float = LAT_BOUND_CACHE,
                      overwrite:       bool = False,
                      n_workers:       int  = 8) -> None:
    """Build per-(member, year) PRECT caches from raw h2 hindcast files.

    Output: {cache_dir}/prect_hindcast_m{EE:02d}_{YYYY}.nc
      dims: (start_date, lead_day, lat, lon)
      PRECT in mm/day, latitude clipped to ±lat_bound_cache.

    Parameters
    ----------
    n_leaddays      : lead days to retain per start date (max = file's time length)
    lat_bound_cache : symmetric equatorial lat clip applied at cache time
    overwrite       : rebuild even if cache file exists
    n_workers       : parallel members per year (ThreadPoolExecutor)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    df = list_hindcast_files(data_dir, years=years, months=months)
    if df.empty:
        raise RuntimeError('No hindcast files found — check data_dir / years / months.')

    # Grid & lat clip from the first file
    ds0 = xr.open_dataset(df['path'].iloc[0])
    lat_all = ds0['lat'].values
    lon     = ds0['lon'].values
    lat_mask = np.abs(lat_all) <= lat_bound_cache
    lat      = lat_all[lat_mask]
    ds0.close()

    years_found = sorted(df['year'].unique())
    year_member_counts = {yr: sorted(df[df['year'] == yr]['member'].unique())
                          for yr in years_found}
    n_total = sum(len(m) for m in year_member_counts.values())
    print(f'\nbuild_prect_cache: {len(years_found)} year(s)  →  {n_total} output files'
          f'  (n_workers={n_workers}, lat_bound_cache=±{lat_bound_cache}°, '
          f'nlat={len(lat)}, nlon={len(lon)})')
    for yr, mems in year_member_counts.items():
        print(f'  {yr}: {len(mems)} members  (m{mems[0]:02d}–m{mems[-1]:02d})')
    print()

    for year in years_found:
        year_df      = df[df['year'] == year]
        year_members = sorted(year_df['member'].unique())

        starts_dt  = pd.DatetimeIndex(sorted(year_df['start_date'].unique()))
        starts_key = list(starts_dt)

        # member → start_date → path
        file_idx: dict[int, dict[pd.Timestamp, str]] = {m: {} for m in year_members}
        for _, row in year_df.iterrows():
            file_idx[row['member']][row['start_date']] = row['path']

        now_utc = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')
        user    = getpass.getuser()

        print(f'  year {year}: {len(year_members)} members  '
              f'{len(starts_key)} start dates  '
              f'({starts_dt[0].date()} → {starts_dt[-1].date()})')

        def _process_member(mem: int,
                             _year:    int              = year,
                             _members: list             = year_members,
                             _starts:  list             = starts_key,
                             _dti:     pd.DatetimeIndex = starts_dt,
                             _idx:     dict             = file_idx,
                             _now:     str              = now_utc,
                             _user:    str              = user) -> None:
            out_file = _cache_path(mem, _year, cache_dir)
            if out_file.exists() and not overwrite:
                print(f'    m{mem:02d}: cache exists, skipping → {out_file.name}')
                return

            t0 = time.time()
            arr = np.full(
                (len(_starts), n_leaddays, len(lat), len(lon)),
                np.nan, dtype=np.float32,
            )

            n_starts = len(_starts)
            for is_, s in enumerate(_starts):
                fpath = _idx[mem].get(s)
                if fpath is None:
                    continue
                ds = xr.open_dataset(fpath)
                # (time, lat, lon), lat_mask applied via .isel
                p = (ds['PRECC'] + ds['PRECL']).isel(lat=np.where(lat_mask)[0]).values
                n_days = min(p.shape[0], n_leaddays)
                arr[is_, :n_days] = p[:n_days] * _MPS_TO_MMDAY
                ds.close()
                if (is_ + 1) % 20 == 0 or (is_ + 1) == n_starts:
                    print(f'    m{mem:02d}: {is_+1}/{n_starts} start dates  '
                          f'({time.time()-t0:.0f}s)')

            da = xr.DataArray(
                arr,
                dims=('start_date', 'lead_day', 'lat', 'lon'),
                coords={
                    'start_date': _dti.values,
                    'lead_day':   np.arange(1, n_leaddays + 1),
                    'lat':        lat,
                    'lon':        lon,
                },
                name='PRECT',
                attrs={
                    'units':       'mm/day',
                    'long_name':   'Total precipitation (PRECC + PRECL)',
                    'derivation':  '(PRECC + PRECL) * 86400 * 1000',
                },
            )
            ds_out = da.to_dataset()
            ds_out.attrs = {
                'title':            'CESM2 S2S hindcast PRECT cache',
                'created':          _now,
                'created_by':       _user,
                'source_data_dir':  str(data_dir),
                'year':             str(_year),
                'member':           f'm{mem:02d}',
                'n_members_year':   str(len(_members)),
                'n_leaddays':       str(n_leaddays),
                'n_start_dates':    str(len(_starts)),
                'lat_bound_cache':  f'±{lat_bound_cache}°',
            }
            ds_out.to_netcdf(out_file)
            print(f'    m{mem:02d}: saved → {out_file.name}  ({time.time()-t0:.1f}s)')

        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = [pool.submit(_process_member, mem) for mem in year_members]
            for mem, fut in zip(year_members, futures):
                exc = fut.exception()
                if exc:
                    print(f'    ERROR on m{mem:02d}: {exc}')

    print('\nbuild_prect_cache: done.')


# ─────────────────────────────────────────────────────────────────────────────
# Segment loader
# ─────────────────────────────────────────────────────────────────────────────

def _filter_starts_by_season(starts: pd.DatetimeIndex,
                              season: str | None) -> np.ndarray:
    """Boolean mask over *starts*: True where start month is in *season*."""
    if season is None:
        return np.ones(len(starts), dtype=bool)
    valid = _SEASON_MONTHS[season.upper()]
    return np.array([m in valid for m in starts.month])


def load_prect_segments(lead_day:   int,
                        n_day_win:  int,
                        cache_dir:  Path | str = CACHE_DIR,
                        years:      list[int] | None = None,
                        members:    list[int] | None = None,
                        season:     str | None = None,
                        lat_bound:  float | None = None,
                        verbose:    bool = True
                        ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Build a list of hindcast PRECT segments starting at *lead_day*.

    One segment per (member × start_date) that has valid data for lead days
    lead_day .. lead_day + n_day_win - 1.  Segments are returned as
    (n_day_win, nlat, nlon) float32 arrays in mm/day.

    Parameters
    ----------
    lead_day  : 1-based lead day at which each segment starts
    n_day_win : segment length in days (must be ≤ n_leaddays_cache - lead_day + 1)
    years     : restrict to these years (None → all cached years)
    members   : restrict to these members (None → all cached members)
    season    : filter start_dates by season ('DJF'|'MAM'|'JJA'|'SON'|None)
    lat_bound : further trim cached lat to ±lat_bound (None → keep as cached)

    Returns
    -------
    (segments, lat, lon)
    """
    cache_dir = Path(cache_dir)
    member_year_files = _find_member_year_files(cache_dir, years=years, members=members)
    if not member_year_files:
        raise FileNotFoundError(
            f'No PRECT cache files in {cache_dir} — run build_prect_cache() first.'
        )

    segments: list[np.ndarray] = []
    lat_out = lon_out = None
    n_dropped_nan = 0

    for mem, files in sorted(member_year_files.items()):
        for f in files:
            ds = xr.open_dataset(f)
            n_lead_cache = ds.sizes['lead_day']
            if lead_day + n_day_win - 1 > n_lead_cache:
                ds.close()
                raise ValueError(
                    f'lead_day={lead_day} + n_day_win={n_day_win} exceeds cache '
                    f'n_leaddays={n_lead_cache} in {f.name}'
                )

            da = ds['PRECT'].sel(
                lead_day=slice(lead_day, lead_day + n_day_win - 1)
            )
            if lat_bound is not None:
                da = da.sel(lat=slice(-lat_bound, lat_bound))
            if lat_out is None:
                lat_out = da['lat'].values
                lon_out = da['lon'].values

            arr    = da.values                    # (n_start, n_day_win, nlat, nlon)
            starts = pd.DatetimeIndex(da['start_date'].values)
            keep   = _filter_starts_by_season(starts, season)

            for is_, ok in enumerate(keep):
                if not ok:
                    continue
                seg = arr[is_]                    # (n_day_win, nlat, nlon)
                if np.isnan(seg).any():
                    n_dropped_nan += 1
                    continue
                segments.append(seg)

            ds.close()

    if not segments:
        raise RuntimeError(
            f'No segments loaded for lead_day={lead_day}, season={season}. '
            f'Check cache coverage & filters.'
        )
    if verbose:
        print(f'  load_prect_segments: lead_day={lead_day}  n_day_win={n_day_win}  '
              f'segments={len(segments)}  (dropped {n_dropped_nan} NaN)')

    return segments, lat_out, lon_out


# ─────────────────────────────────────────────────────────────────────────────
# Spectrum drivers
# ─────────────────────────────────────────────────────────────────────────────

def wk_spectrum_lead_day(lead_day:  int,
                         n_day_win: int = 32,
                         cache_dir: Path | str = CACHE_DIR,
                         years:     list[int] | None = None,
                         members:   list[int] | None = None,
                         season:    str | None = None,
                         lat_bound: float = 15.0,
                         label:     str | None = None) -> dict:
    """WK spectrum pooled over all (member × start_date) at *lead_day*.

    Returns a results dict compatible with kfp.plot_wk_panel — same keys as
    kfp.compute_wk_spectrum, plus 'label' and 'n_segments'.
    """
    segments, lat, lon = load_prect_segments(
        lead_day=lead_day, n_day_win=n_day_win,
        cache_dir=cache_dir, years=years, members=members,
        season=season, lat_bound=lat_bound,
    )
    t0 = time.time()
    res = kfp.compute_wk_spectrum_from_segments(segments, spd=1, vscale=1.0)
    res['label']       = label or f'Day {lead_day}'
    res['lead_day']    = lead_day
    res['n_segments']  = len(segments)
    res['n_day_win']   = n_day_win
    print(f'  wk_spectrum_lead_day({lead_day}): {len(segments)} segments '
          f'→ spectrum in {time.time()-t0:.1f}s')
    return res


def wk_spectra_lead_days(lead_days: list[int],
                         n_day_win: int = 32,
                         cache_dir: Path | str = CACHE_DIR,
                         years:     list[int] | None = None,
                         members:   list[int] | None = None,
                         season:    str | None = None,
                         lat_bound: float = 15.0) -> list[dict]:
    """Convenience: WK spectra for a list of lead days.

    Returns list of results dicts, one per lead day, ordered as given.
    """
    results = []
    print(f'wk_spectra_lead_days: lead_days={lead_days}  n_day_win={n_day_win}  '
          f'season={season or "all"}  years={years or "all"}  members={members or "all"}')
    for ld in lead_days:
        results.append(wk_spectrum_lead_day(
            lead_day=ld, n_day_win=n_day_win,
            cache_dir=cache_dir, years=years, members=members,
            season=season, lat_bound=lat_bound,
        ))
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Hindcast start-date discovery
# ─────────────────────────────────────────────────────────────────────────────

def hindcast_start_dates(cache_dir: Path | str = CACHE_DIR,
                         years:     list[int] | None = None,
                         members:   list[int] | None = None,
                         season:    str | None = None) -> pd.DatetimeIndex:
    """Sorted unique start dates across the selected cache subset.

    Members are unioned — a start date is included if ANY selected member has
    a hindcast on that date.  Used to drive the IMERG matched-start pool.
    """
    cache_dir = Path(cache_dir)
    m = _find_member_year_files(cache_dir, years=years, members=members)
    if not m:
        raise FileNotFoundError(
            f'No PRECT cache files in {cache_dir} — run build_prect_cache() first.'
        )
    all_dates: set = set()
    for files in m.values():
        for f in files:
            ds = xr.open_dataset(f)
            starts = pd.DatetimeIndex(ds['start_date'].values)
            if season is not None:
                mask = _filter_starts_by_season(starts, season)
                starts = starts[mask]
            all_dates.update(starts.tolist())
            ds.close()
    return pd.DatetimeIndex(sorted(all_dates))


# ─────────────────────────────────────────────────────────────────────────────
# IMERG loaders (daily 1° observations)
# ─────────────────────────────────────────────────────────────────────────────

def _open_imerg_daily_range(yr0: int, yr1: int,
                             imerg_dir: Path | str = IMERG_DAILY_DIR
                             ) -> xr.Dataset:
    """Open IMERG daily monthly files spanning yr0..yr1 (inclusive)."""
    imerg_dir = Path(imerg_dir)
    files: list[Path] = []
    for yr in range(yr0, yr1 + 1):
        for mo in range(1, 13):
            f = imerg_dir / f'IMERG_daily.{yr}{mo:02d}.1deg.nc'
            if f.exists():
                files.append(f)
    if not files:
        raise FileNotFoundError(f'No IMERG monthly files in {imerg_dir} for {yr0}-{yr1}')
    return xr.open_mfdataset(files, combine='by_coords')


def _imerg_precip(ds: xr.Dataset, lat_bound: float) -> xr.DataArray:
    """Slice IMERG precip to ±lat_bound and ensure S→N latitude order."""
    da = ds['precip']
    if da['lat'].values[0] > da['lat'].values[-1]:
        da = da.sortby('lat')
    return da.sel(lat=slice(-lat_bound, lat_bound))


def imerg_wk_spectrum_climatology(yr0: int,
                                   yr1: int,
                                   n_day_win: int = 32,
                                   lat_bound: float = 15.0,
                                   n_day_skip: int = 0,
                                   imerg_dir: Path | str = IMERG_DAILY_DIR,
                                   label: str | None = None) -> dict:
    """WK spectrum of IMERG cut into fixed-length segments over yr0..yr1.

    Segments of length *n_day_win* are cut across the concatenated IMERG record
    with *n_day_skip* days between window starts (0 → non-overlapping, negative
    → overlap).  Segments with any NaN are dropped.

    Returns a results dict compatible with kfp.plot_wk_panel.
    """
    ds = _open_imerg_daily_range(yr0, yr1, imerg_dir=imerg_dir)
    da = _imerg_precip(ds, lat_bound)
    x  = da.values.astype(np.float32)              # (time, nlat, nlon), mm/day
    ds.close()

    ntim = x.shape[0]
    stride = n_day_win + n_day_skip                # 0 skip → back-to-back
    if stride < 1:
        raise ValueError('n_day_skip must satisfy n_day_win + n_day_skip >= 1')

    segments: list[np.ndarray] = []
    n_dropped = 0
    for i0 in range(0, ntim - n_day_win + 1, stride):
        seg = x[i0:i0 + n_day_win]
        if np.isnan(seg).any():
            n_dropped += 1
            continue
        segments.append(seg)
    if not segments:
        raise RuntimeError(f'No valid IMERG segments produced for {yr0}-{yr1}')
    print(f'  imerg_wk_spectrum_climatology({yr0}-{yr1}): '
          f'{len(segments)} segments  (dropped {n_dropped} NaN)  '
          f'n_day_win={n_day_win} skip={n_day_skip}')

    res = kfp.compute_wk_spectrum_from_segments(segments, spd=1, vscale=1.0)
    res['label']      = label or f'IMERG clim {yr0}-{yr1}'
    res['n_segments'] = len(segments)
    res['n_day_win']  = n_day_win
    return res


def imerg_wk_spectrum_matched(start_dates,
                               n_day_win: int = 32,
                               lat_bound: float = 15.0,
                               imerg_dir: Path | str = IMERG_DAILY_DIR,
                               label: str | None = None) -> dict:
    """WK spectrum from IMERG segments matching the hindcast start dates.

    For each date in *start_dates* pull the n_day_win-day segment starting on
    that calendar day.  Missing / NaN / out-of-range dates are dropped.
    """
    dti = pd.DatetimeIndex(start_dates).sort_values()
    if len(dti) == 0:
        raise ValueError('start_dates is empty')

    yr0 = int(dti.min().year)
    yr1 = int((dti.max() + pd.Timedelta(days=n_day_win)).year)
    ds  = _open_imerg_daily_range(yr0, yr1, imerg_dir=imerg_dir)
    da  = _imerg_precip(ds, lat_bound)
    da.load()                                       # load once; slice many times

    segments: list[np.ndarray] = []
    n_missing = n_nan = 0
    for s in dti:
        end = s + pd.Timedelta(days=n_day_win - 1)
        try:
            seg = da.sel(time=slice(s, end)).values
        except KeyError:
            n_missing += 1
            continue
        if seg.shape[0] != n_day_win:
            n_missing += 1
            continue
        if np.isnan(seg).any():
            n_nan += 1
            continue
        segments.append(seg)
    ds.close()

    if not segments:
        raise RuntimeError('No valid IMERG matched-start segments produced.')
    print(f'  imerg_wk_spectrum_matched: {len(segments)} segments  '
          f'(from {len(dti)} start dates; dropped {n_missing} missing, {n_nan} NaN)')

    res = kfp.compute_wk_spectrum_from_segments(segments, spd=1, vscale=1.0)
    res['label']      = label or 'IMERG matched'
    res['n_segments'] = len(segments)
    res['n_day_win']  = n_day_win
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Lead-time BUCKET analysis
# For each (member, year) build a chained daily series that uses
#   lead days [b*D-D+1 .. b*D]   where b = bucket index (1-based), D = days_per_bucket.
# When start dates are spaced D days apart these chunks stitch end-to-end into
# an uninterrupted daily series across the year — proper WK cadence, propagation
# preserved.  Segments are then cut and pooled for the WK spectrum.
# ─────────────────────────────────────────────────────────────────────────────

def _bucket_lead_range(bucket_idx: int, days_per_bucket: int) -> tuple[int, int]:
    """1-based (lead_start, lead_end) inclusive for the given bucket."""
    if bucket_idx < 1:
        raise ValueError(f'bucket_idx must be ≥ 1; got {bucket_idx}')
    lead_start = (bucket_idx - 1) * days_per_bucket + 1
    lead_end   = bucket_idx * days_per_bucket
    return lead_start, lead_end


def _chain_bucket_series_per_memyear(bucket_idx:      int,
                                     days_per_bucket: int,
                                     cache_dir:       Path | str,
                                     years:           list[int] | None,
                                     members:         list[int] | None,
                                     season:          str | None,
                                     lat_bound:       float
                                     ) -> tuple[list[np.ndarray], np.ndarray, np.ndarray]:
    """Build one chained daily series per (member, year).

    For each (member, year), pull lead days [lead_start..lead_end] from every
    start date (sorted by start_date), concat time-wise → single (time, lat, lon)
    array.  NaN start dates are skipped.  Returns a list of these arrays plus
    (lat, lon) coordinates.
    """
    cache_dir = Path(cache_dir)
    my_files = _find_member_year_files(cache_dir, years=years, members=members)
    if not my_files:
        raise FileNotFoundError(
            f'No PRECT cache files in {cache_dir} — run build_prect_cache() first.'
        )
    lead_start, lead_end = _bucket_lead_range(bucket_idx, days_per_bucket)

    chains: list[np.ndarray] = []
    lat_out = lon_out = None

    for mem, files in sorted(my_files.items()):
        for f in files:
            ds = xr.open_dataset(f)
            n_lead_cache = ds.sizes['lead_day']
            if lead_end > n_lead_cache:
                ds.close()
                raise ValueError(
                    f'bucket {bucket_idx} needs lead days {lead_start}-{lead_end} '
                    f'but cache only has {n_lead_cache} lead days in {f.name}'
                )
            da = ds['PRECT'].sel(lead_day=slice(lead_start, lead_end))
            if lat_bound is not None:
                da = da.sel(lat=slice(-lat_bound, lat_bound))
            if lat_out is None:
                lat_out = da['lat'].values
                lon_out = da['lon'].values

            starts = pd.DatetimeIndex(da['start_date'].values)
            order  = np.argsort(starts)                       # chronological
            keep   = _filter_starts_by_season(starts[order], season)
            arr    = da.values[order]                         # (n_start, D, lat, lon)

            per_start = []
            for i, ok in enumerate(keep):
                if not ok:
                    continue
                block = arr[i]                                # (D, lat, lon)
                if np.isnan(block).any():
                    continue
                per_start.append(block)
            ds.close()
            if not per_start:
                continue

            chain = np.concatenate(per_start, axis=0)         # (n_time, lat, lon)
            chains.append(chain)

    if not chains:
        raise RuntimeError(
            f'No lead-bucket chains built for bucket {bucket_idx} '
            f'({lead_start}-{lead_end}).  Check cache/filters.'
        )
    return chains, lat_out, lon_out


def _cut_chains_to_segments(chains:     list[np.ndarray],
                             n_day_win:  int,
                             n_day_skip: int = 0) -> list[np.ndarray]:
    """Cut each chain into fixed-length segments (default non-overlapping)."""
    stride = n_day_win + n_day_skip
    if stride < 1:
        raise ValueError('n_day_win + n_day_skip must be ≥ 1')
    segments: list[np.ndarray] = []
    for c in chains:
        ntim = c.shape[0]
        for i0 in range(0, ntim - n_day_win + 1, stride):
            seg = c[i0:i0 + n_day_win]
            if np.isnan(seg).any():
                continue
            segments.append(seg)
    return segments


def wk_spectrum_lead_bucket(bucket_idx:      int,
                             days_per_bucket: int = 7,
                             n_day_win:       int = 96,
                             n_day_skip:      int = 0,
                             cache_dir:       Path | str = CACHE_DIR,
                             years:           list[int] | None = None,
                             members:         list[int] | None = None,
                             season:          str | None = None,
                             lat_bound:       float = 15.0,
                             label:           str | None = None) -> dict:
    """WK spectrum from chained daily series over lead-bucket B (1-based).

    Each (member, year) contributes one chained (time, lat, lon) series built
    from lead days [(b-1)*D+1 .. b*D] of every start date, sorted chronologically.
    Chains are cut into n_day_win segments (non-overlapping by default) and
    pooled for the spectrum.
    """
    chains, lat, lon = _chain_bucket_series_per_memyear(
        bucket_idx=bucket_idx, days_per_bucket=days_per_bucket,
        cache_dir=cache_dir, years=years, members=members,
        season=season, lat_bound=lat_bound,
    )
    segments = _cut_chains_to_segments(chains, n_day_win, n_day_skip)
    if not segments:
        raise RuntimeError(
            f'No {n_day_win}-day segments could be cut from bucket {bucket_idx} '
            f'chains (max chain length = {max(c.shape[0] for c in chains)}).'
        )
    lead_start, lead_end = _bucket_lead_range(bucket_idx, days_per_bucket)
    default_label = f'Week {bucket_idx}' if days_per_bucket == 7 else \
                     f'Lead {lead_start}-{lead_end}'
    t0 = time.time()
    res = kfp.compute_wk_spectrum_from_segments(segments, spd=1, vscale=1.0)
    res['label']       = label or default_label
    res['bucket_idx']  = bucket_idx
    res['lead_range']  = (lead_start, lead_end)
    res['n_chains']    = len(chains)
    res['n_segments']  = len(segments)
    res['n_day_win']   = n_day_win
    print(f'  wk_spectrum_lead_bucket({bucket_idx}) [days {lead_start}-{lead_end}]: '
          f'{len(chains)} chains → {len(segments)} segments  ({time.time()-t0:.1f}s)')
    return res


def wk_spectra_lead_buckets(bucket_idxs:     list[int],
                             days_per_bucket: int = 7,
                             n_day_win:       int = 96,
                             n_day_skip:      int = 0,
                             cache_dir:       Path | str = CACHE_DIR,
                             years:           list[int] | None = None,
                             members:         list[int] | None = None,
                             season:          str | None = None,
                             lat_bound:       float = 15.0) -> list[dict]:
    """WK spectra for a list of lead buckets, returned in the given order."""
    print(f'wk_spectra_lead_buckets: buckets={bucket_idxs}  D={days_per_bucket}  '
          f'n_day_win={n_day_win}  n_day_skip={n_day_skip}  '
          f'season={season or "all"}  years={years or "all"}  members={members or "all"}')
    return [wk_spectrum_lead_bucket(
                bucket_idx=b, days_per_bucket=days_per_bucket,
                n_day_win=n_day_win, n_day_skip=n_day_skip,
                cache_dir=cache_dir, years=years, members=members,
                season=season, lat_bound=lat_bound,
            ) for b in bucket_idxs]


def imerg_wk_spectrum_matched_bucket(start_dates,
                                      bucket_idx:      int,
                                      days_per_bucket: int = 7,
                                      n_day_win:       int = 96,
                                      n_day_skip:      int = 0,
                                      lat_bound:       float = 15.0,
                                      imerg_dir:       Path | str = IMERG_DAILY_DIR,
                                      label:           str | None = None) -> dict:
    """IMERG matched-verification WK for a lead bucket.

    For each hindcast start date s, extract IMERG days
    s + (lead_start-1)..s + (lead_end-1) — the calendar window the hindcast
    bucket verifies against.  Chain those blocks chronologically → one long
    daily series → cut into n_day_win segments → WK spectrum.
    """
    dti = pd.DatetimeIndex(start_dates).sort_values()
    if len(dti) == 0:
        raise ValueError('start_dates is empty')

    lead_start, lead_end = _bucket_lead_range(bucket_idx, days_per_bucket)
    yr0 = int(dti.min().year)
    yr1 = int((dti.max() + pd.Timedelta(days=lead_end)).year)

    ds  = _open_imerg_daily_range(yr0, yr1, imerg_dir=imerg_dir)
    da  = _imerg_precip(ds, lat_bound)
    da.load()

    per_start = []
    n_missing = n_nan = 0
    for s in dti:
        d0 = s + pd.Timedelta(days=lead_start - 1)
        d1 = s + pd.Timedelta(days=lead_end   - 1)
        block = da.sel(time=slice(d0, d1)).values
        if block.shape[0] != days_per_bucket:
            n_missing += 1
            continue
        if np.isnan(block).any():
            n_nan += 1
            continue
        per_start.append(block)
    ds.close()

    if not per_start:
        raise RuntimeError('No valid IMERG bucket blocks — check dates & IMERG coverage.')

    chain = np.concatenate(per_start, axis=0)                # (n_time, lat, lon)
    segments = _cut_chains_to_segments([chain], n_day_win, n_day_skip)
    if not segments:
        raise RuntimeError(
            f'IMERG chain has {chain.shape[0]} days; too short for {n_day_win}-day window.'
        )

    default_label = f'IMERG week {bucket_idx}' if days_per_bucket == 7 else \
                     f'IMERG lead {lead_start}-{lead_end}'
    res = kfp.compute_wk_spectrum_from_segments(segments, spd=1, vscale=1.0)
    res['label']       = label or default_label
    res['bucket_idx']  = bucket_idx
    res['lead_range']  = (lead_start, lead_end)
    res['n_segments']  = len(segments)
    res['n_day_win']   = n_day_win
    print(f'  imerg_wk_spectrum_matched_bucket({bucket_idx}): chain={chain.shape[0]} d '
          f'→ {len(segments)} segments  (dropped {n_missing} missing, {n_nan} NaN)')
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Temporal-standard-deviation maps
# Same data-selection knobs as the WK path, but pool values at a single lead
# day (or across the OBS record / OBS matched days), then compute time-std
# per grid point.
# ─────────────────────────────────────────────────────────────────────────────

def _std_result(std_map: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                label: str, **extra) -> dict:
    """Bundle a std map with coord arrays and a label."""
    return {'std': std_map, 'lat': lat, 'lon': lon, 'label': label, **extra}


def hindcast_std_map_lead_day(lead_day:  int,
                              cache_dir: Path | str = CACHE_DIR,
                              years:     list[int] | None = None,
                              members:   list[int] | None = None,
                              season:    str | None = None,
                              lat_bound: float = 15.0,
                              label:     str | None = None) -> dict:
    """Per-grid-point temporal std of PRECT at a single lead day, pooled over
    all (member × year × start_date) samples that pass the filters.
    """
    cache_dir = Path(cache_dir)
    my_files = _find_member_year_files(cache_dir, years=years, members=members)
    if not my_files:
        raise FileNotFoundError(
            f'No PRECT cache files in {cache_dir} — run build_prect_cache() first.'
        )
    pool: list[np.ndarray] = []
    lat_out = lon_out = None
    for mem, files in sorted(my_files.items()):
        for f in files:
            ds = xr.open_dataset(f)
            if lead_day > ds.sizes['lead_day']:
                ds.close()
                raise ValueError(f'lead_day={lead_day} exceeds cache in {f.name}')
            da = ds['PRECT'].sel(lead_day=lead_day)
            da = da.sel(lat=slice(-lat_bound, lat_bound))
            if lat_out is None:
                lat_out = da['lat'].values
                lon_out = da['lon'].values
            starts = pd.DatetimeIndex(da['start_date'].values)
            keep   = _filter_starts_by_season(starts, season)
            arr    = da.values[keep]                          # (n_start_kept, lat, lon)
            arr    = arr[~np.isnan(arr).any(axis=(1, 2))]
            if arr.size:
                pool.append(arr)
            ds.close()
    if not pool:
        raise RuntimeError(f'No samples for lead_day={lead_day} std.')
    stacked = np.concatenate(pool, axis=0)                    # (n_samples, lat, lon)
    std_map = stacked.std(axis=0)
    print(f'  hindcast_std_map_lead_day({lead_day}): pooled {stacked.shape[0]} samples')
    return _std_result(std_map, lat_out, lon_out,
                       label=label or f'Day {lead_day}',
                       lead_day=lead_day, n_samples=stacked.shape[0])


def hindcast_std_maps_lead_days(lead_days: list[int], **kw) -> list[dict]:
    """Std maps for a list of single lead days."""
    return [hindcast_std_map_lead_day(lead_day=ld, **kw) for ld in lead_days]


def imerg_std_map_climatology(yr0: int, yr1: int,
                              lat_bound: float = 15.0,
                              imerg_dir: Path | str = IMERG_DAILY_DIR,
                              label: str | None = None) -> dict:
    """Per-grid-point temporal std of IMERG over yr0..yr1 (all days)."""
    ds = _open_imerg_daily_range(yr0, yr1, imerg_dir=imerg_dir)
    da = _imerg_precip(ds, lat_bound)
    arr = da.values.astype(np.float32)                        # (time, lat, lon)
    ds.close()
    valid = ~np.isnan(arr).any(axis=(1, 2))
    arr = arr[valid]
    std_map = arr.std(axis=0)
    print(f'  imerg_std_map_climatology({yr0}-{yr1}): {arr.shape[0]} days')
    return _std_result(std_map, da['lat'].values, da['lon'].values,
                       label=label or f'IMERG clim {yr0}-{yr1}',
                       n_samples=arr.shape[0])


def imerg_std_map_matched(start_dates,
                          lat_bound: float = 15.0,
                          imerg_dir: Path | str = IMERG_DAILY_DIR,
                          label: str | None = None) -> dict:
    """Per-grid-point temporal std of IMERG restricted to hindcast start dates."""
    dti = pd.DatetimeIndex(start_dates).sort_values()
    if len(dti) == 0:
        raise ValueError('start_dates is empty')
    ds = _open_imerg_daily_range(int(dti.min().year), int(dti.max().year),
                                  imerg_dir=imerg_dir)
    da = _imerg_precip(ds, lat_bound)
    da.load()
    picks = []
    n_missing = 0
    for s in dti:
        try:
            v = da.sel(time=s).values
        except KeyError:
            n_missing += 1
            continue
        if np.isnan(v).any():
            n_missing += 1
            continue
        picks.append(v)
    ds.close()
    if not picks:
        raise RuntimeError('No IMERG matched-start days available.')
    stacked = np.stack(picks, axis=0)
    std_map = stacked.std(axis=0)
    print(f'  imerg_std_map_matched: {stacked.shape[0]}/{len(dti)} start dates '
          f'(dropped {n_missing})')
    return _std_result(std_map, da['lat'].values, da['lon'].values,
                       label=label or 'IMERG matched',
                       n_samples=stacked.shape[0])


def plot_std_panel(results:    list[dict],
                   var_name:   str = 'PRECT',
                   units:      str = 'mm/day',
                   nx:         int = 3,
                   levels:     np.ndarray | None = None,
                   cmap:       str = 'YlGnBu',
                   save_path:  Path | str | None = None,
                   fig_title:  str | None = None):
    """Grid of temporal-std lat/lon maps, one panel per result dict.

    Each result is expected to have keys: 'std' (nlat, nlon), 'lat', 'lon',
    'label'.  Layout: nx columns, ceil(len/nx) rows.
    """
    import matplotlib.pyplot as plt

    ncases = len(results)
    ny = int(np.ceil(ncases / nx))
    fig, axes = plt.subplots(ny, nx, figsize=(4.5 * nx, 2.5 * ny), sharey=True)
    axes = np.atleast_2d(axes).ravel()
    fig.suptitle(fig_title or f'{var_name} temporal std ({units})', fontsize=11)

    if levels is None:
        finite_max = max(float(np.nanmax(r['std'])) for r in results)
        levels = np.linspace(0, finite_max, 15)

    for i, r in enumerate(results):
        ax = axes[i]
        cf = ax.contourf(r['lon'], r['lat'], r['std'],
                          levels=levels, cmap=cmap, extend='max')
        ax.set_title(f"{r['label']}  (n={r.get('n_samples', '?')})", fontsize=9)
        ax.set_xlabel('Longitude', fontsize=8)
        if i % nx == 0:
            ax.set_ylabel('Latitude', fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(ncases, len(axes)):
        axes[j].set_visible(False)

    cbar = fig.colorbar(cf, ax=axes.tolist(), shrink=0.7, pad=0.02, aspect=40)
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label(units, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'Saved: {save_path}')
    return fig
