---
title: "S2S archive comparison — NOAA SFS beta1 vs CESM2 S2SHINDCASTS"
author: "Compiled from live S3 / Glade contents"
date: \today
geometry: landscape, margin=1cm
fontsize: 10pt
colorlinks: true
---

## Overview

Three co-existing archives feed the current S2S / blocking / kf_pan analyses:

- **SFS reforecast** — full ensemble × init × lead structure kept intact; right feed for
  lead-day and probabilistic analyses.
- **SFS MDTF** — ensemble and lead axes discarded to produce long chained daily records
  that behave like a free-running run; the natural feed for MDTF PODs, WK spectra, and
  long-record climatologies.
- **CESM2 S2SHINDCASTS** — CESM2/CAM6 subseasonal hindcast archive on Glade
  (weekly cadence, 22 members), split across two companion NetCDF trees: `daily/`
  (CAM h2 output) and `Z3/` (geopotential height on pressure levels).

## Format comparison

| Property                     | SFS `reforecast/{MM}/atm_daily.zarr`                                                 | SFS `MDTF/daily_mem000_{MM}.zarr`                                                            | **CESM2 S2SHINDCASTS**                                                                             |
|:-----------------------------|:-------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------|
| **Purpose**                  | Probabilistic S2S hindcast — lead-day scoring, ensemble spread                       | MDTF POD framework — feed as if free-running run                                             | Weekly CESM2 S2S hindcast — lead-day skill, blocking, S2S predictability                          |
| **Path root**                | `s3://noaa-oar-sfsdev-pds/experiments/beta1/reforecast/`                             | `s3://noaa-oar-sfsdev-pds/experiments/beta1/MDTF/`                                            | `/glade/campaign/cesm/development/cross-wg/S2S/CESM2/S2SHINDCASTS/`                                 |
| **Sub-trees**                | One store per init month: `{MM}/atm_daily.zarr` (+ ocn/ice, monthly, clim)           | One store per init month: `daily_mem000_{MM}.zarr`                                           | Two co-registered trees: `daily/*.nc` (CAM h2 daily) + `Z3/{YYYY}/{MM}/*.nc`                        |
| **Format**                   | Consolidated Zarr v2, blosc(lz4), anon-readable S3                                   | Consolidated Zarr v2, blosc(lz4), anon-readable S3                                            | NetCDF-4, local POSIX (one file per start × member)                                                |
| **Init months available**    | 03, 04, 05, 06, 07, 08, 11 (no 09 daily)                                             | 03, 04, 05, 06, 07, 08, **09**, 11                                                            | All 12 months                                                                                      |
| **Init cadence**             | Monthly — 1st of MM only                                                             | Monthly (already chained)                                                                    | **Weekly** — 7-day delta, ~52 starts/year (98 % Mondays)                                           |
| **Starts per year, per member** | 7 (one per available month)                                                       | folded into single `time`                                                                    | 52–53                                                                                              |
| **Year coverage**            | 1991 → 2025 (35 y)                                                                   | 1991 → 2025 chained (35 y in one series)                                                     | 1999 → 2023 (25 y)                                                                                 |
| **Total unique starts**      | 245 per member (7 × 35)                                                              | (none exposed)                                                                               | 1,316                                                                                              |
| **Members**                  | 11 (`'000'..'010'` strings)                                                          | 1 (`mem000` baked in)                                                                        | up to 22 (`m00..m21`, some years >22)                                                              |
| **Lead length**              | 47 days                                                                              | folded into `time`                                                                           | 46 days                                                                                            |
| **Dims (analysis vars)**     | `(member=11, init=35, lead=47, lat=181, lon=360)`                                    | `(time=12784, lat=181, lon=360)`                                                             | `daily/`: `(time=46, lat=192, lon=288)`, plus `(ilev=33, zlon=1)` for zonal-mean TEM vars. `Z3/`: `(time=46, lev_p=14, lat=192, lon=288)` |
| **`init` / start encoding**  | int64 `days since 1991-{MM}-01`, one per year                                        | none                                                                                          | Encoded in **filename**: `Z3_cesm2cam6v2_{DD}{mon}{YYYY}00z_..._m{MM}.nc`; also in `time[0]`         |
| **`lead` encoding**          | int64 `days`, 0..46                                                                  | none                                                                                          | `time` is verification-date; lead = day index 1..46                                                |
| **`time` encoding**          | none (init × lead)                                                                   | float64 `seconds since 1970-01-01`, verification-time, chained 1991-03-01 → ~2026-02-28      | float64 CF datetime, calendar=noleap, 46 daily verification dates per file                        |
| **Grid**                     | 1° global regular, `lat=181` (−90..90 asc), `lon=360` (0..359°E)                     | same                                                                                          | CAM FV **~0.9° × 1.25°**, `lat=192` (−90..90 asc), `lon=288` (0..358.75°E)                          |
| **Fill / missing**           | `9.999e+20` (float32)                                                                | same                                                                                          | CF `_FillValue`, per-var                                                                            |
| **Chunking (HGT/Z500 var)**  | `[1, 1, 47, 181, 360]` — one (member, init) at a time                                | `[10, 181, 360]` — 10 days at a time                                                          | One file per (start × member) — no internal chunking                                                |
| **Total files / stores**     | 7 stores (one per MM)                                                                | 8 stores (one per MM)                                                                        | 16,746 `daily/*.nc` files; ~similar count under `Z3/`                                              |
| **Local cache format**       | `{cache_dir}/z500_hindcast_m{MM:02d}_{YYYY}.nc`, dims `(start_date, lead_day, lat, lon)`, built by `build_z500_cache_from_sfs` | (not cached — read directly from S3)                                                          | `{cache_dir}/z500_hindcast_m{MM:02d}_{YYYY}.nc` (built by `build_z500_cache`), same layout, plus `prect_hindcast_m{MM:02d}_{YYYY}.nc` for kf_pan |
| **Time semantics**           | Forecast state at (member × init × lead)                                             | Verification-date state (single member)                                                       | Forecast state at (member × start_date × lead_day)                                                  |
| **# variables**              | ~80                                                                                  | 75                                                                                            | 27 (h2 daily on lat/lon) + 8 (zonal-mean TEM on ilev/zlon/lat) + 1 (`Z3` on 14 pressure levels)     |
| **Best for**                 | Arbitrary (member, init) slicing; lead-day pools; ensemble stats                     | Streaming a long daily series; MDTF PODs; WK / spectra / climatology                          | Weekly lead-day analyses; blocking-by-lead; PRECT kf_pan; large ensemble variance                   |

## Variable content — where do the fields overlap?

CESM `daily/` (CAM h2) uses CAM/CMIP-style names, not GRIB. The table below shows
the *rough* equivalences for the fields that come up in these analyses. `Z3` sits
in a separate archive (`Z3/{YYYY}/{MM}/*.nc`) and holds heights on 14 pressure
levels: 5, 10, 20, 30, 50, 70, 100, 200, 300, 500, 700, 850, 925, 1000 hPa.

| Category                        | SFS reforecast                                                            | SFS MDTF                       | CESM S2SHINDCASTS                                                                    |
|:--------------------------------|:--------------------------------------------------------------------------|:-------------------------------|:-------------------------------------------------------------------------------------|
| Precipitation                   | `PRATE_surface` (kg m⁻² s⁻¹), `APCP_surface` (accum)                      | `PRATE_surface` only           | `PRECC` + `PRECL` (m/s); `PRECT = (PRECC+PRECL) × 86.4e6` mm/day                     |
| Radiation surface (down)        | `DSWRF_surface`, `DLWRF_surface`                                          | none                           | `FSDS`, `FLDS` (both W m⁻²)                                                          |
| Radiation surface (up)          | `USWRF_surface`, `ULWRF_surface`                                          | none                           | not in h2                                                                            |
| Radiation TOA                   | `DSWRF_topofatmosphere`, `USWRF_topofatmosphere`, `ULWRF_topofatmosphere` | `ULWRF_topofatmosphere` only   | `FSNT` (net SW TOA), `FLNT` (net LW TOA)                                             |
| Cloud (total)                   | `TCDC_entireatmosphere_consideredasasinglelayer_`                         | none                           | `CLDTOT`                                                                             |
| Cloud water/ice column          | none                                                                      | none                           | `TGCLDLWP`, `TGCLDIWP` (column liquid/ice)                                            |
| Column moisture                 | none                                                                      | none                           | `TMQ` (precipitable water, kg m⁻²)                                                    |
| Sea-level pressure              | `PRMSL_meansealevel`                                                      | none                           | `PSL`                                                                                |
| Surface pressure                | `PRES_surface`                                                            | `PRES_surface`                 | `PS`                                                                                 |
| Heat fluxes                     | `LHTFL_surface`, `SHTFL_surface`                                          | `LHTFL_surface` only           | `QFLX` (moisture flux, kg m⁻² s⁻¹) — no SHFLX in h2                                   |
| Sea ice / SST                   | none                                                                      | `ICEC_surface`                 | `SST` (K); no ice conc                                                                |
| Potential evap                  | none                                                                      | `PEVPR_surface`                | not in h2                                                                             |
| Vertical wind                   | `VVEL_500mb`                                                              | `VVEL_500mb`                   | zonal-mean `Wzm` only (no 3-D field in h2)                                            |
| Winds (pressure levels)         | `UGRD`/`VGRD` @ 50–1000 mb (11 levels)                                    | same                           | **not in h2** (only surface `U10`, `WSPDSRFAV`, `WSPDSRFMX`; zonal-mean `Uzm`,`Vzm`)   |
| Winds (10 m)                    | `UGRD_10maboveground`, `VGRD_10maboveground`                              | same                           | `U10` (10 m wind speed only)                                                          |
| Heights (pressure levels)       | `HGT` @ 50, 100, 200, 500, 700, 850, 1000 mb + surface                    | same                           | **`Z3` archive** — 14 lev_p, includes 500 mb → `Z500 = Z3.sel(lev_p=500)`             |
| Temperatures (pressure levels)  | `TMP` @ 50, 100, 200, 250, 300, 500, 600, 700, 850, 925, 1000 mb + 2 m + surface | same                    | not in h2 (only `TROP_T` at tropopause; zonal-mean `THzm`)                            |
| 2 m T, RH, Q                    | `TMP_2maboveground`, `SPFH_2maboveground`                                 | same                           | `QREFHT` (2 m Q), `RHREFHT` (2 m RH), `RH600` (600 mb RH)                             |
| Humidity (pressure levels)      | `SPFH` @ 100–1000 mb                                                      | same (no 50 mb)                | not in h2                                                                             |
| Tropopause                      | none                                                                      | none                           | `TROP_P`, `TROP_T`                                                                    |
| Zonal-mean TEM diagnostics      | none                                                                      | none                           | `Uzm`, `Vzm`, `Wzm`, `THzm`, `UVzm`, `UWzm`, `VTHzm`, `WTHzm` (on ilev × lat)         |
| Snow                            | none                                                                      | none                           | `SNOWHICE`, `SNOWHLND` (snow depth)                                                   |
| Soil moisture / temperature     | `SOILL`, `SOILW` (4 layers), `CISOILM_0M2mbelowground`                    | same                           | not in h2                                                                             |
| Land / topo                     | `LAND_surface`                                                            | `LAND_surface`                 | `LANDFRAC`, `OCNFRAC`, `PHIS` (surface geopotential)                                  |

**Key CESM-specific gaps to note.** The h2 daily archive is deliberately
lightweight — full 3-D fields on model or pressure levels are *not* included
outside of `Z3` (heights). For anything requiring winds, temperature, or
humidity on pressure levels (e.g. jet diagnostics, MJO composites), the SFS
reforecast is the *only* one of these three archives that ships all three in a
single store.

## Minimal readers

```python
import fsspec, xarray as xr

# 1. SFS reforecast — 5-D probabilistic slice
mp = fsspec.get_mapper(
    's3://noaa-oar-sfsdev-pds/experiments/beta1/reforecast/03/atm_daily.zarr',
    anon=True)
ds_r = xr.decode_cf(xr.open_zarr(mp, consolidated=True), decode_timedelta=False)
z_r  = ds_r['HGT_500mb']         # dims (member, init, lead, lat, lon)

# 2. SFS MDTF — chained verification-time series (member 0 only)
mp = fsspec.get_mapper(
    's3://noaa-oar-sfsdev-pds/experiments/beta1/MDTF/daily_mem000_03.zarr',
    anon=True)
ds_m = xr.decode_cf(xr.open_zarr(mp, consolidated=True))
z_m  = ds_m['HGT_500mb']         # dims (time, lat, lon)

# 3. CESM S2SHINDCASTS — one NetCDF per (start, member); use hindcast_blocking_utils
import hindcast_blocking_utils as hbu
df    = hbu.list_hindcast_files(hbu.DATA_DIR, years=[2015])   # 22 members × 52 starts
z500  = hbu.load_z500_for_leadday(lead_day=10, cache_dir=hbu.CACHE_DIR)
                                  # dims (hindcast, member, lat, lon)
```

## Rules of thumb

- **Lead-day + full 3-D atmosphere** → SFS reforecast (winds, T, Q on 9–11 pressure levels).
- **Lead-day + geopotential blocking** → either SFS reforecast *or* CESM (both ship `Z500`).
- **Weekly-cadence lead-day skill, PRECT kf_pan** → CESM (weekly starts give the 7-day cadence chaining works out of the box).
- **Long chained daily record for MDTF PODs, WK spectra, climatology** → SFS MDTF.
- **Ensemble spread** → SFS reforecast (11) or CESM (22), depending on which variables you need.

## Cache layout produced by these analyses

All three ingest paths converge on the same per-(member, year) NetCDF layout
so the downstream blocking / kf_pan code is source-agnostic:

- Filename : `z500_hindcast_m{MM:02d}_{YYYY}.nc` or `prect_hindcast_m{MM:02d}_{YYYY}.nc`
- Dims     : `(start_date, lead_day, lat, lon)`
- Var      : `Z500` in m or `PRECT` in mm/day
- Builder  : `hindcast_blocking_utils.build_z500_cache()` for CESM,
             `hindcast_blocking_utils.build_z500_cache_from_sfs()` for SFS reforecast

Point `CACHE_DIR` at the SFS or CESM cache and every downstream helper works
unchanged.
