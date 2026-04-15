"""
raobs_utils.py
==============
Python utility module for reading and plotting RAOB sounding diagnostics.
Converted from NCL raobs_utils.ncl.

Key data notes:
  - RAOB files: raob_soundings_{station}.cdf
  - Time variable `synTime` uses units "Seconds since (1970-1-1 00:00:0.0)"
    and must be decoded manually with pd.to_datetime(..., unit='s', origin='unix')
  - Open RAOB files with decode_times=False
  - 2D mandatory-level variables are shaped (ntime, 22): index 0 = surface,
    indices 1:nlev+1 = 22 mandatory levels

Author: converted from NCL by Claude Code, 2026
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from scipy.interpolate import interp1d

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

STATION_NAMES = [
    "Albuquerque", "Churchill", "Davenport", "Denver", "Egedesminde",
    "KeyWest", "LakeCharles", "Minneapolis", "Narassarssuaq", "Norman",
    "OklahomaCity", "Omaha", "SantaTeresa", "Slidell", "Springfield",
    "Tallahassee", "Thule", "Topeka", "Tucson", "Valparaiso", "Kingston",
    "EasterIsland", "SanCristobal", "Hilo", "Tarawa", "GrandCayman",
]

MAN_PLEVELS = np.array(
    [1000., 925., 850., 700., 500., 400., 300., 250., 200., 150., 100., 70., 50., 30., 20., 10.],
    dtype=np.float64,
)

# Month x-axis tick positions (day-of-year) and labels
_MONTH_TICKS = -15.5 + np.array([31, 61, 92, 122, 153, 183, 214, 245, 275, 305, 333, 366])
_MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Variable name mapping from CAM/common names -> RAOB file variable names
_CNAMES = ["PS", "T", "U", "V", "VMAG", "Q", "DPD", "RELHUM"]
_RNAMES_M = {
    "PS": "prMan", "T": "tpMan", "U": "U", "V": "V",
    "VMAG": "wsMan", "Q": "Q", "DPD": "tdMan", "RELHUM": "RELHUM",
}
_RNAMES_T = {
    "PS": "prSigT", "T": "tpSigT", "U": "U", "V": "V",
    "VMAG": "wsSigT", "Q": "Q", "DPD": "tdSigT", "RELHUM": "RELHUM",
}
_RNAMES_W = {
    "PS": "prSigT", "T": "tpSigW", "U": "U", "V": "V",
    "VMAG": "wsSigW", "Q": "Q", "DPD": "tdSigW", "RELHUM": "RELHUM",
}

# Obs precip dataset paths and variable names
_OBS_PRECIP = {
    "CPC":    ("/glade/work/rneale/data/NOAA_CPC_USA/precip.V1.0.day.ltm.nc", "precip"),
    "TRMM":   ("/glade/work/rneale/data/TRMM/3B42.1998-2009.nc",              "precip"),
    "GPCP":   ("/glade/work/rneale/data/GPCP/GPCP_1DD_v1.2_199610-201407.nc", "PREC"),
    "Livneh": ("/glade/work/rneale/data/Livneh/prec.day.ltm.nc",              "prec"),
    "PRISM":  ("/glade/work/rneale/data/PRISM/daily/PRISM_day_1981-2014.nc",  "PRECT"),
}


# ===========================================================================
# Helper / private functions
# ===========================================================================

def _running_mean(arr, n):
    """
    Centered running mean of a 1-D array over n points.

    Uses np.convolve with mode='same' and a uniform kernel of length n.
    Values near the edges are divided by the actual number of valid points
    (like NCL's runave with the circular flag False).

    Parameters
    ----------
    arr : array-like, shape (N,)
    n   : int, window width

    Returns
    -------
    numpy.ndarray, same shape as arr
    """
    arr = np.asarray(arr, dtype=float)
    kernel = np.ones(n)
    out = np.convolve(arr, kernel, mode='same')
    # normalise by actual overlap at edges
    norm = np.convolve(np.ones_like(arr), kernel, mode='same')
    return out / norm


def _doy_from_times(times):
    """
    Compute day-of-year (1-based) from a pandas DatetimeIndex.

    Parameters
    ----------
    times : pandas.DatetimeIndex

    Returns
    -------
    numpy.ndarray of int, day_of_year values 1..366
    """
    return times.day_of_year.values


# ===========================================================================
# Public utility functions
# ===========================================================================

def is_3d_var(varname):
    """
    Return True if *varname* is a 3-D (time × level) variable.

    The 3-D variables are: U, V, T, Q.

    Parameters
    ----------
    varname : str

    Returns
    -------
    bool
    """
    return varname in ("U", "V", "T", "Q")


def raob_file(station, rdir):
    """
    Return the full path to a RAOB sounding file for *station*.

    Parameters
    ----------
    station : str   Station name (must be in STATION_NAMES)
    rdir    : str   Directory containing raob_soundings_*.cdf files

    Returns
    -------
    str  Full path to the .cdf file

    Raises
    ------
    ValueError  If station is not in STATION_NAMES
    FileNotFoundError  If the file does not exist on disk
    """
    if station not in STATION_NAMES:
        raise ValueError(
            f"Station '{station}' is not available.\n"
            f"Valid stations: {STATION_NAMES}"
        )
    fpath = os.path.join(rdir, f"raob_soundings_{station}.cdf")
    if not os.path.isfile(fpath):
        raise FileNotFoundError(
            f"RAOB file not found: {fpath}"
        )
    print(f"  Station '{station}' file found: {fpath}")
    return fpath


def cam_file(crun_name, cdir, cvarname, sdata):
    """
    Locate the CAM output file for *cvarname* and populate *sdata* with
    processing-control keys.

    Looks first for a pre-processed single-point file at:
        {cdir}/{cvarname}/{crun_name}.cam.{chan}.{cvarname}.{station}.nc

    If that exists, sets ``sdata['lcproc'] = False`` and returns the path.
    Otherwise sets ``sdata['lcproc'] = True``, globs for raw files in the
    campaign directory, and if the variable is 3-D also locates the PS file.

    The following keys in *sdata* are written/updated:
        lcproc    : bool
        pfile     : str (expected processed output path)
        pdir      : str (cdir)
        ps_file   : str (only for 3-D variables when lcproc=True)

    Parameters
    ----------
    crun_name : str
    cdir      : str   Base directory (e.g. /glade/work/rneale/large_ens/)
    cvarname  : str   CAM variable name
    sdata     : dict  Shared data dictionary (modified in place)

    Returns
    -------
    str or list of str
        Path(s) to the file(s) to open.
    """
    chan = sdata.get('chan', 'h2')
    station = sdata['station']

    fstub = f"{cvarname}/{crun_name}.cam.{chan}.{cvarname}"
    pfile = os.path.join(cdir, fstub + f".{station}.nc")

    print(f"\n  CAM simulation: {crun_name}")

    sdata['pfile'] = pfile
    sdata['pdir']  = cdir

    if os.path.isfile(pfile):
        print(f"  Processed CAM file exists for {cvarname}: {pfile}")
        sdata['lcproc'] = False
        return pfile
    else:
        print(f"  Processed CAM file not found for {cvarname} – will process raw files.")
        sdata['lcproc'] = True

        # Raw files live under the campaign archive
        cdir_raw = "/glade/campaign/cgd/amp/rneale/large_ens/"
        raw_pattern = os.path.join(cdir_raw, fstub + "*.nc")
        raw_files = sorted(glob.glob(raw_pattern))
        if not raw_files:
            # fallback to cgd collection
            cdir_raw2 = "/glade/collections/cdg/data/cesmLE/CESM-CAM5-BGC-LE/atm/proc/tseries/hourly6/"
            raw_pattern = os.path.join(cdir_raw2, fstub + "*.nc")
            raw_files = sorted(glob.glob(raw_pattern))
        print(f"  Raw file(s): {raw_files}")

        if is_3d_var(cvarname):
            # We also need the PS processed file for vertical interpolation
            ps_stub = f"PS/{crun_name}.cam.{chan}.PS"
            ps_file = os.path.join(cdir, ps_stub + f".{station}.nc")
            sdata['ps_file'] = ps_file
            print(f"  PS file: {ps_file}")

        return raw_files


def get_raob_times(ds):
    """
    Convert the ``synTime`` variable from a RAOB dataset into a
    pandas DatetimeIndex.

    The RAOB files store time as integer seconds since the Unix epoch
    (1970-01-01), but with a non-standard unit string that xarray cannot
    decode automatically. This function performs the conversion manually.

    Parameters
    ----------
    ds : xarray.Dataset  Opened with ``decode_times=False``

    Returns
    -------
    pandas.DatetimeIndex
    """
    print("  RAOB: decoding synTime coordinate...")
    raw = ds['synTime'].values.astype(np.float64)
    # Mask NCL fill values (default ~9.97e36) and any other out-of-range entries
    valid = np.isfinite(raw) & (raw > 0) & (raw < 4e9)
    result = np.full(len(raw), np.datetime64('NaT', 's'))
    result[valid] = raw[valid].astype(np.int64).astype('datetime64[s]')
    return pd.DatetimeIndex(result)


def raob_info(station, ds, times, sdata):
    """
    Read station metadata from the RAOB dataset and populate *sdata*.

    Sets: sdata['lat'], sdata['lon'], sdata['elev'], sdata['stwid'],
          sdata['man_plevels'].

    Also prints a summary of station identity and time range.

    Parameters
    ----------
    station : str
    ds      : xarray.Dataset  RAOB file opened with decode_times=False
    times   : pandas.DatetimeIndex  from get_raob_times
    sdata   : dict  Modified in place
    """
    sdata['lat']   = float(ds['staLat'].values.flat[0])
    sdata['lon']   = float(ds['staLon'].values.flat[0])
    sdata['elev']  = float(ds['staElev'].values.flat[0])
    sdata['stwid'] = int(ds['wmoStat'].values.flat[0])

    if sdata.get('lev_type', 'M') == 'M':
        sdata['man_plevels'] = MAN_PLEVELS.copy()
    else:
        # Significant level grid: 50 mb steps from 1000 to 50 mb
        dplev = 50.
        pmax, pmin = 1000., 50.
        nplevs = int((pmax - pmin) / dplev)
        sdata['man_plevels'] = np.linspace(1000., 10., nplevs + 1)

    print()
    print("*" * 65)
    print("  INFO FROM RAOB FILE")
    print("*" * 65)
    print(f"  Station : {station}  (WMO id: {sdata['stwid']})")
    print(f"  Lat: {sdata['lat']:.2f}  Lon: {sdata['lon']:.2f}  Elev: {sdata['elev']:.0f} m")

    valid = times[~pd.isnull(times)]
    if len(valid):
        t0, t1 = valid[0], valid[-1]
        print(f"  Time range: {t0.strftime('%HZ %d %b %Y')} to {t1.strftime('%HZ %d %b %Y')}")

    yf, yl = sdata.get('year_first', '?'), sdata.get('year_last', '?')
    print(f"  Requested year range: {yf} – {yl}")
    print(f"  Output levels: {sdata['man_plevels']} mb")
    print()


def raob_read(ds, cname, times, sdata):
    """
    Read a variable from the RAOB dataset and return it as a DataArray.

    Handles derived variables:
      - U / V   : computed from wind speed (wsMan) and direction (wdMan)
      - Q       : computed from dew-point depression using Tetens formula
      - RELHUM  : relative humidity from temp and dew-point
      - PS      : surface pressure from index 0 of mandatory-level pressure

    For 2-D variables on mandatory levels, strips off the surface level
    (index 0) and assigns the standard mandatory pressure coordinate.

    Parameters
    ----------
    ds     : xarray.Dataset  RAOB file (decode_times=False)
    cname  : str             Variable name (CAM-style)
    times  : pandas.DatetimeIndex
    sdata  : dict

    Returns
    -------
    xarray.DataArray  (time,) or (time, lev)
    """
    lev_type = sdata.get('lev_type', 'M')
    mplevs   = sdata['man_plevels']
    nmplevs  = len(mplevs)
    pi       = np.pi

    print()
    print("*" * 65)
    print(f"  RAOB: reading {cname}")
    print("*" * 65)

    # ------------------------------------------------------------------
    # Derived: zonal / meridional wind
    # ------------------------------------------------------------------
    if cname in ("U", "V"):
        if lev_type == 'M':
            spd_name, dir_name = "wsMan", "wdMan"
        elif lev_type == 'T':
            spd_name, dir_name = "wsSigT", "wdSigT"
        else:
            spd_name, dir_name = "wsSigW", "wdSigW"

        spd = ds[spd_name].values.astype(float)
        wdir = ds[dir_name].values.astype(float)
        # Replace fill/flagged values
        fill = 99999.
        spd  = np.where(np.abs(spd)  >= fill, np.nan, spd)
        wdir = np.where(np.abs(wdir) >= fill, np.nan, wdir)

        if cname == "U":
            data = -spd * np.sin(wdir * pi / 180.)
            long_name = "Zonal Wind"
        else:
            data = -spd * np.cos(wdir * pi / 180.)
            long_name = "Meridional Wind"
        units = "m/s"
        var_out_raw = data

    # ------------------------------------------------------------------
    # Derived: specific humidity Q
    # ------------------------------------------------------------------
    elif cname == "Q":
        if lev_type == 'M':
            dpd_name, t_name, p_name = "tdMan", "tpMan", "prMan"
        elif lev_type == 'T':
            dpd_name, t_name, p_name = "tdSigT", "tpSigT", "prSigT"
        else:
            dpd_name, t_name, p_name = "tdSigW", "tpSigW", "prSigW"

        dpd_raw  = ds[dpd_name].values.astype(float)
        temp_raw = ds[t_name].values.astype(float)
        pres_raw = ds[p_name].values.astype(float)

        fill = 99999.
        dpd_raw  = np.where(np.abs(dpd_raw)  >= fill, np.nan, dpd_raw)
        temp_raw = np.where(np.abs(temp_raw) >= fill, np.nan, temp_raw)
        pres_raw = np.where(np.abs(pres_raw) >= fill, np.nan, pres_raw)

        # Convert: temp in K -> Celsius; pressure already in Pa -> convert to hPa
        temp_c = temp_raw - 273.15    # K -> degC
        dpd_c  = dpd_raw              # DPD in K = DPD in degC
        pres_hpa = pres_raw * 0.01    # Pa -> hPa

        dpt_c = temp_c - dpd_c       # dew-point in degC

        # Tetens formula: sat vapour pressure over water
        es = 6.112 * np.exp(17.67 * dpt_c / (dpt_c + 243.5))  # hPa
        # Specific humidity in g/kg
        data = 622. * es / (pres_hpa - 0.378 * es)
        data = np.where(pres_hpa > 0., data, np.nan)

        long_name = "Specific Humidity"
        units     = "g/kg"
        var_out_raw = data

    # ------------------------------------------------------------------
    # Derived: relative humidity
    # ------------------------------------------------------------------
    elif cname == "RELHUM":
        if lev_type == 'M':
            dpd_name, t_name = "tdMan", "tpMan"
        elif lev_type == 'T':
            dpd_name, t_name = "tdSigT", "tpSigT"
        else:
            dpd_name, t_name = "tdSigW", "tpSigW"

        dpd_raw  = ds[dpd_name].values.astype(float)
        temp_raw = ds[t_name].values.astype(float)

        fill = 99999.
        dpd_raw  = np.where(np.abs(dpd_raw)  >= fill, np.nan, dpd_raw)
        temp_raw = np.where(np.abs(temp_raw) >= fill, np.nan, temp_raw)

        temp_c = temp_raw - 273.15
        dpd_c  = dpd_raw
        dpt_c  = temp_c - dpd_c

        e_sat = 6.112 * np.exp(17.67 * temp_c / (temp_c + 243.5))
        e_dew = 6.112 * np.exp(17.67 * dpt_c  / (dpt_c  + 243.5))

        data = 100. * e_dew / e_sat
        data = np.clip(data, 0., 110.)   # physical range guard

        long_name = "Relative Humidity"
        units     = "%"
        var_out_raw = data

    # ------------------------------------------------------------------
    # PRECT: not available in RAOB files; return NaN array
    # (PRECT is only meaningful for CAM model data; station_precip_plot
    #  is only called when cam=True, so this result is never plotted.)
    # ------------------------------------------------------------------
    elif cname == "PRECT":
        n = ds.dims.get('recNum', ds.dims.get('time', len(times)))
        var_out_raw = np.full(n, np.nan)
        long_name = "Precipitation Rate"
        units     = "mm/day"

    # ------------------------------------------------------------------
    # Direct read: PS (surface pressure from index 0 of mandatory array)
    # ------------------------------------------------------------------
    elif cname == "PS":
        pres_raw = ds["prMan"].values.astype(float)
        fill = 99999.
        pres_raw = np.where(np.abs(pres_raw) >= fill, np.nan, pres_raw)
        # index 0 is surface pressure; convert Pa -> mb
        data = pres_raw[:, 0] * 0.01
        long_name = "Surface Pressure"
        units     = "mb"

        # QC: mask bogus values for high-altitude stations
        bad_stations = ("Albuquerque", "Tucson", "Denver", "Minneapolis", "SantaTeresa")
        if sdata.get('station', '') in bad_stations:
            for bad_val in (1000., 700., 819.):
                data = np.where(data == bad_val, np.nan, data)

        var_out_raw = data

    # ------------------------------------------------------------------
    # Direct read: other named RAOB variable (VMAG, DPD, T, etc.)
    # ------------------------------------------------------------------
    else:
        if lev_type == 'M':
            rmap = _RNAMES_M
        elif lev_type == 'T':
            rmap = _RNAMES_T
        else:
            rmap = _RNAMES_W

        rname = rmap.get(cname)
        if rname is None:
            raise ValueError(f"raob_read: no RAOB variable mapping for '{cname}'")

        raw = ds[rname].values.astype(float)
        fill = 99999.
        raw = np.where(np.abs(raw) >= fill, np.nan, raw)
        var_out_raw = raw
        long_name = cname
        units     = ""

    # ------------------------------------------------------------------
    # Assign pandas times as coordinate and drop missing-time rows
    # ------------------------------------------------------------------
    # Attach time coordinate
    n_times_file = var_out_raw.shape[0]
    times_trunc = times[:n_times_file]
    good = ~pd.isnull(times_trunc)
    good_idx = np.where(good)[0]

    if var_out_raw.ndim == 1:
        data_trimmed = var_out_raw[good_idx]
        dims = ['time']
        coords = {'time': times_trunc[good_idx]}
        da = xr.DataArray(data_trimmed, dims=dims, coords=coords)
    else:
        # 2-D: shape (ntime, nlev_file)
        # For mandatory levels: trim surface level (index 0), keep 1:nmplevs+1
        if lev_type == 'M' and cname not in ("U", "V", "Q", "RELHUM"):
            data2d = var_out_raw[:, 1:nmplevs + 1]
        elif cname in ("U", "V"):
            data2d = var_out_raw[:, 1:nmplevs + 1]
        elif cname in ("Q", "RELHUM"):
            data2d = var_out_raw[:, 1:nmplevs + 1]
        else:
            data2d = var_out_raw[:, 1:nmplevs + 1]

        data_trimmed = data2d[good_idx, :]
        dims = ['time', 'lev']
        coords = {
            'time': times_trunc[good_idx],
            'lev':  mplevs,
        }
        da = xr.DataArray(data_trimmed, dims=dims, coords=coords)
        da['lev'].attrs['units'] = 'mb'

    da.attrs['long_name'] = long_name
    da.attrs['units']     = units
    da.attrs['cname']     = cname
    da.attrs['stname']    = sdata.get('station', '')
    da.attrs['lat']       = sdata.get('lat', np.nan)
    da.attrs['lon']       = sdata.get('lon', np.nan)
    da.attrs['elev']      = sdata.get('elev', np.nan)

    return da


def cam_read(files_or_ds, cname, sdata):
    """
    Read a CAM variable at the station location.

    If ``sdata['lcproc']`` is True: opens the raw file(s), finds the nearest
    grid point by minimum distance to station lat/lon, extracts a single-point
    time series, performs vertical interpolation to MAN_PLEVELS (for 3-D
    variables) using geocat.comp.interp_hybrid_to_pressure, applies unit
    scaling, and writes a processed file to ``sdata['pfile']``.

    If ``sdata['lcproc']`` is False: simply reads the variable from the
    pre-processed single-point file.

    Unit scaling applied on write:
      PS     : Pa  → mb     (×0.01)
      PRECT  : m/s → mm/day (×86400000)
      DTCOND : K/s → K/day  (×86400)
      Q      : kg/kg → g/kg (×1000)

    Parameters
    ----------
    files_or_ds : str | list of str | xarray.Dataset
    cname       : str  CAM variable name
    sdata       : dict

    Returns
    -------
    xarray.DataArray
    """
    import xarray as xr
    import numpy as np

    slat = sdata['lat']
    slon = sdata['lon']
    lcproc = sdata.get('lcproc', False)
    pfile  = sdata.get('pfile', '')
    pdir   = sdata.get('pdir', '')
    mplevs = sdata.get('man_plevels', MAN_PLEVELS)

    print()
    print("*" * 65)
    print(f"  CAM: reading {cname}")
    print("*" * 65)
    print(f"  Station location: {slat:.2f} N, {slon:.2f} E")

    if lcproc:
        print("  Extracting nearest grid-point from raw files...")

        # Open file(s)
        if isinstance(files_or_ds, xr.Dataset):
            ds = files_or_ds
            close_ds = False
        elif isinstance(files_or_ds, (list, tuple)):
            ds = xr.open_mfdataset(files_or_ds, combine='by_coords')
            close_ds = True
        else:
            ds = xr.open_dataset(files_or_ds)
            close_ds = True

        clat = ds['lat'].values
        clon = ds['lon'].values

        # Find nearest grid point: broadcast to 2D and minimise distance
        clat2d, clon2d = np.meshgrid(clat, clon, indexing='ij')
        dist2d = (clat2d - slat) ** 2 + (clon2d - slon) ** 2
        ij = np.unravel_index(np.argmin(dist2d), dist2d.shape)
        ilat_gp, ilon_gp = ij
        lat_gp = float(clat[ilat_gp])
        lon_gp = float(clon[ilon_gp])
        print(f"  Nearest CAM grid point: {lat_gp:.2f} N, {lon_gp:.2f} E")

        # Determine if variable is 3-D (time, lev, lat, lon)
        var_dims = ds[cname].dims
        is_3d = (len(var_dims) == 4)

        if is_3d:
            print("  3-D variable: interpolating hybrid levels -> pressure levels")
            try:
                import geocat.comp as gc
            except ImportError:
                raise ImportError(
                    "geocat.comp is required for CAM hybrid-to-pressure interpolation. "
                    "Install with: conda install -c conda-forge geocat-comp"
                )

            # Extract single column: (time, lev, 1, 1) shape for geocat
            var_col = ds[cname][:, :, ilat_gp, ilon_gp]  # (time, lev)

            # Read PS from pre-processed PS file
            ps_file = sdata.get('ps_file', '')
            if not ps_file or not os.path.isfile(ps_file):
                raise FileNotFoundError(
                    f"PS file needed for 3-D interpolation but not found: {ps_file}"
                )
            ds_ps = xr.open_dataset(ps_file)
            ps_col = ds_ps['PS'].values  # (time,) in Pa

            hyam = ds['hyam'].values
            hybm = ds['hybm'].values
            p0   = float(ds['P0'].values)

            # geocat wants pressure levels in Pa
            plevs_pa = mplevs * 100.

            # Reshape for geocat: needs (time, lev, lat, lon)
            nt = var_col.shape[0]
            nlev_in = var_col.shape[1]
            var_4d = var_col.values.reshape(nt, nlev_in, 1, 1)
            ps_3d  = ps_col.reshape(nt, 1, 1)

            var_interp = gc.interp_hybrid_to_pressure(
                var_4d,
                ps_3d,
                hyam,
                hybm,
                p0=p0,
                new_levels=plevs_pa,
                method='log',
            )
            # Output shape: (time, new_lev, lat=1, lon=1)
            data_out = var_interp[:, :, 0, 0]

            times_out = pd.to_datetime(ds['time'].values)
            da = xr.DataArray(
                data_out,
                dims=['time', 'lev'],
                coords={'time': times_out, 'lev': mplevs},
            )
            da['lev'].attrs['units'] = 'mb'
            ds_ps.close()

        else:
            # 1-D surface variable: (time, lat, lon)
            var_sfc = ds[cname][:, ilat_gp, ilon_gp]
            times_out = pd.to_datetime(ds['time'].values)
            da = xr.DataArray(
                var_sfc.values,
                dims=['time'],
                coords={'time': times_out},
            )

        if close_ds:
            ds.close()

        # Unit scaling
        if cname == 'PS':
            da = da * 0.01
            da.attrs['units']     = 'mb'
            da.attrs['long_name'] = 'Surface Pressure'
        elif cname == 'PRECT':
            da = da * 86400. * 1000.
            da.attrs['units']     = 'mm/day'
            da.attrs['long_name'] = 'Total Precipitation'
        elif cname == 'DTCOND':
            da = da * 86400.
            da.attrs['units']     = 'K/day'
            da.attrs['long_name'] = 'Moist Physics dT/dt'
        elif cname == 'Q':
            da = da * 1000.
            da.attrs['units']     = 'g/kg'
            da.attrs['long_name'] = 'Specific Humidity'

        da.attrs['lat'] = lat_gp
        da.attrs['lon'] = lon_gp - 360. if lon_gp > 180. else lon_gp

        # Write processed file
        pdir_var = os.path.join(pdir, cname)
        os.makedirs(pdir_var, exist_ok=True)
        ds_write = da.to_dataset(name=cname)
        ds_write.attrs['station']    = sdata.get('station', '')
        ds_write.attrs['station_id'] = str(sdata.get('stwid', ''))
        ds_write.attrs['lat']        = float(lat_gp)
        ds_write.attrs['lon']        = float(da.attrs['lon'])
        print(f"  Writing processed file: {pfile}")
        ds_write.to_netcdf(pfile)
        print("  Done writing.")

    else:
        print("  Reading from pre-processed file...")
        if isinstance(files_or_ds, xr.Dataset):
            ds = files_or_ds
        else:
            ds = xr.open_dataset(files_or_ds)
        da = ds[cname]
        # Ensure time is datetime (handle cftime objects, e.g. NoLeap calendar)
        if not np.issubdtype(da['time'].dtype, np.datetime64):
            cft = da['time'].values
            times_std = np.array([np.datetime64(t.isoformat(), 'ns') for t in cft])
            da = da.assign_coords(time=pd.DatetimeIndex(times_std))

    # Attach common metadata
    da.attrs.setdefault('cname',   cname)
    da.attrs.setdefault('stname',  sdata.get('station', ''))
    da.attrs.setdefault('elev',    sdata.get('elev', np.nan))

    # Time range report
    times_da = pd.to_datetime(da['time'].values)
    t0, t1 = times_da[0], times_da[-1]
    print(f"  Time range: {t0.strftime('%HZ %d %b %Y')} to {t1.strftime('%HZ %d %b %Y')}")

    yf = sdata.get('year_first')
    yl = sdata.get('year_last')
    avail_y0 = t0.year
    avail_y1 = t1.year
    if yf and yl:
        if yf < avail_y0 or yl > avail_y1:
            warnings.warn(
                f"Requested year range {yf}-{yl} is outside available "
                f"{avail_y0}-{avail_y1} in CAM file."
            )

    return da


def raob_sig2p(var_sig, splevs_Pa, mplevs_mb):
    """
    Interpolate a variable from significant levels to mandatory pressure levels.

    Uses log-pressure interpolation via scipy.interpolate.interp1d.

    Parameters
    ----------
    var_sig   : numpy.ndarray, shape (ntime, nsig)
                Variable on significant levels
    splevs_Pa : numpy.ndarray, shape (ntime, nsig)
                Pressure at significant levels in Pa
    mplevs_mb : numpy.ndarray, shape (nmlev,)
                Target mandatory pressure levels in mb

    Returns
    -------
    numpy.ndarray, shape (ntime, nmlev)
    """
    splevs_mb = splevs_Pa * 0.01  # Pa -> mb
    ntime, nmlev = var_sig.shape[0], len(mplevs_mb)
    out = np.full((ntime, nmlev), np.nan)

    log_mplevs = np.log(mplevs_mb)

    print("  Interpolating from significant levels to mandatory levels...")
    for it in range(ntime):
        p_prof = splevs_mb[it, :]
        v_prof = var_sig[it, :]
        mask   = np.isfinite(p_prof) & np.isfinite(v_prof) & (p_prof > 0.)
        if mask.sum() < 2:
            continue
        log_p = np.log(p_prof[mask])
        v_m   = v_prof[mask]
        # sort by descending pressure (ascending log-p for interp)
        sort_idx = np.argsort(log_p)
        log_p_s  = log_p[sort_idx]
        v_s      = v_m[sort_idx]
        try:
            f_interp = interp1d(log_p_s, v_s, kind='linear',
                                bounds_error=False, fill_value=np.nan)
            out[it, :] = f_interp(log_mplevs)
        except Exception:
            pass
    print("  Done.")
    return out


def get_obs_precip(pdset, sdata):
    """
    Read daily precipitation from an observational dataset at the nearest
    grid point to the station.

    Supported datasets: 'CPC', 'TRMM', 'GPCP', 'Livneh', 'PRISM'.

    Parameters
    ----------
    pdset : str   Dataset name
    sdata : dict  Must contain 'lat' and 'lon'

    Returns
    -------
    xarray.DataArray  Precipitation at nearest grid point
    """
    if pdset not in _OBS_PRECIP:
        raise ValueError(f"Unknown obs precip dataset: {pdset}. "
                         f"Choose from {list(_OBS_PRECIP.keys())}")

    fpath, vname = _OBS_PRECIP[pdset]
    if not os.path.isfile(fpath):
        raise FileNotFoundError(f"Obs precip file not found: {fpath}")

    ds = xr.open_dataset(fpath)
    slat = sdata['lat']
    slon = sdata['lon']

    clat = ds['lat'].values
    clon = ds['lon'].values

    # Match station longitude to the dataset's convention (0–360 or –180–180)
    if clon.max() > 180.:
        slon_match = slon % 360.   # convert –180–180  →  0–360
    else:
        slon_match = (slon + 180.) % 360. - 180.   # convert 0–360  →  –180–180

    clat2d, clon2d = np.meshgrid(clat, clon, indexing='ij')
    dist2d = (clat2d - slat) ** 2 + (clon2d - slon_match) ** 2
    ij = np.unravel_index(np.argmin(dist2d), dist2d.shape)
    ilat_gp, ilon_gp = ij
    lat_gp = float(clat[ilat_gp])
    lon_gp = float(clon[ilon_gp])
    print(f"  Nearest {pdset} grid point: {lat_gp:.2f} N, {lon_gp:.2f} E")

    da = ds[vname][:, ilat_gp, ilon_gp]
    ds.close()
    return da


# ===========================================================================
# Contour level helper
# ===========================================================================

def get_contour_levels(cname, nomean=False):
    """
    Return (vmin, vmax, step) for filled contour plots of *cname*.

    Parameters
    ----------
    cname   : str   Variable name
    nomean  : bool  If True, return anomaly range instead of mean range

    Returns
    -------
    tuple (vmin, vmax, step)
    """
    _levels = {
        # (mean_range, anom_range)
        'T':       ((220., 300., 5.),  (-5.,  5.,  0.5)),
        'DTCOND':  ((-10., 10., 1.),   (-10., 10., 1.)),
        'Q':       ((0.5,  20., 0.5),  (-5.,  5.,  0.5)),
        'U':       ((-10., 10., 0.5),  (-10., 10., 0.5)),
        'VMAG':    ((-10., 10., 0.5),  (-10., 10., 0.5)),
        'V':       ((-6.,   6., 0.5),  (-6.,   6., 0.5)),
        'DPD':     ((-6.,   6., 0.5),  (-6.,   6., 0.5)),
        'RELHUM':  ((10.,  100., 5.),  (-20., 20., 2.)),
    }
    entry = _levels.get(cname, ((-10., 10., 1.), (-10., 10., 1.)))
    return entry[1] if nomean else entry[0]


# ===========================================================================
# Axis setup helper
# ===========================================================================

def setup_doy_xaxis(ax):
    """
    Configure *ax* to display day-of-year on the x-axis with month labels.

    Tick positions are centred in each calendar month (using the
    -15.5 + cumulative-day-count convention from the NCL original).
    X range is set to 1–366.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    """
    ax.set_xlim(1., 366.)
    ax.set_xticks(_MONTH_TICKS)
    ax.set_xticklabels(_MONTH_LABELS)
    ax.tick_params(axis='x', which='major', length=0)

    # Minor ticks at month boundaries (for vertical grid lines)
    minor_ticks = [1, 32, 61, 92, 122, 153, 183, 214, 245, 275, 305, 333, 366]
    ax.set_xticks(minor_ticks, minor=True)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.xaxis.grid(True, which='minor', linestyle='--', linewidth=0.5, alpha=0.7)
    ax.set_xlabel("Month")


# ===========================================================================
# Plotting functions
# ===========================================================================

def station_1d_plot(ax, da, sdata):
    """
    1-D annual cycle scatter + running-mean line plot.

    Plots all individual sounding values as small grey dots, then overlays
    the running mean ± 1 std (if ``sdata['lp1d_av_std']``) and separate
    00Z / 12Z (and model 06Z / 18Z) lines (if ``sdata['lp1d_0012Z']``).

    Parameters
    ----------
    ax    : matplotlib.axes.Axes
    da    : xarray.DataArray  (time,) with a datetime-compatible 'time' coord
    sdata : dict
    """
    year0    = sdata.get('year_first', 1970)
    year1    = sdata.get('year_last',  2100)
    drunave  = sdata.get('drunave', 60)
    station  = sdata.get('station', '')
    lat      = sdata.get('lat', np.nan)
    lon      = sdata.get('lon', np.nan)
    elev     = sdata.get('elev', np.nan)
    is_cam   = sdata.get('cam', False)

    times = pd.to_datetime(da['time'].values)
    doy   = _doy_from_times(times)
    years = times.year
    vals  = da.values.astype(float)

    # Drop NaN times
    good = np.isfinite(doy.astype(float))
    doy, years, vals = doy[good], years[good], vals[good]

    # Scatter: all sondes
    ax.scatter(doy, vals, s=0.5, color='k', alpha=0.2, zorder=1)

    # Mask to requested year range
    in_range = (years >= year0) & (years <= year1)
    vals_yr  = np.where(in_range, vals, np.nan)

    # Build per-DOY annual cycle arrays
    ndoys = int(np.nanmax(doy)) if len(doy) > 0 else 366
    ac_doy = np.arange(1, ndoys + 1, dtype=float)

    if sdata.get('lp1d_av_std', True):
        ac_mean = np.full(ndoys, np.nan)
        ac_msd  = np.full(ndoys, np.nan)
        ac_psd  = np.full(ndoys, np.nan)
        for id_ in range(1, ndoys + 1):
            idx = np.where(doy == id_)[0]
            if len(idx) == 0:
                continue
            v = vals_yr[idx]
            v = v[np.isfinite(v)]
            if len(v) == 0:
                continue
            ac_mean[id_ - 1] = np.nanmean(v)
            s = np.nanstd(v, ddof=1)
            ac_msd[id_ - 1] = ac_mean[id_ - 1] - s
            ac_psd[id_ - 1] = ac_mean[id_ - 1] + s

        smooth_mean = _running_mean(ac_mean, drunave)
        smooth_msd  = _running_mean(ac_msd,  drunave)
        smooth_psd  = _running_mean(ac_psd,  drunave)

        ax.plot(ac_doy, smooth_mean, 'k-',  lw=2, zorder=3, label='mean')
        ax.plot(ac_doy, smooth_msd,  'k--', lw=1, zorder=3, label='mean−1σ')
        ax.plot(ac_doy, smooth_psd,  'k--', lw=1, zorder=3, label='mean+1σ')

    if sdata.get('lp1d_0012Z', False):
        if is_cam:
            z_hours  = [0, 6, 12, 18]
            colors   = ['blue', 'darkgreen', 'red', 'brown']
            labels   = ['00Z', '06Z', '12Z', '18Z']
        else:
            z_hours  = [0, 12]
            colors   = ['blue', 'red']
            labels   = ['00Z', '12Z']

        hour_vals = times.hour.values[good]
        for zh, col, lbl in zip(z_hours, colors, labels):
            ac_zh = np.full(ndoys, np.nan)
            for id_ in range(1, ndoys + 1):
                idx = np.where((doy == id_) & (hour_vals == zh))[0]
                if len(idx) == 0:
                    continue
                v = vals_yr[idx]
                v = v[np.isfinite(v)]
                if len(v) > 0:
                    ac_zh[id_ - 1] = np.nanmean(v)
            smooth = _running_mean(ac_zh, drunave)
            ax.plot(ac_doy, smooth, color=col, lw=2, zorder=4, label=lbl)

        ax.legend(fontsize=8, loc='upper right')

    # Title and labels
    long_name = da.attrs.get('long_name', da.attrs.get('cname', ''))
    units     = da.attrs.get('units', '')
    ax.set_title(f"{station}  ({lat:.2f}\u00b0N, {lon:.2f}\u00b0E, {elev:.0f} m elev.)",
                 fontsize=10)
    ax.set_ylabel(f"{long_name}  [{units}]")

    # CAM grid-point annotation
    if is_cam:
        cam_lat = da.attrs.get('lat', np.nan)
        cam_lon = da.attrs.get('lon', np.nan)
        cam_elev = da.attrs.get('cam_elev', np.nan)
        ax.annotate(
            f"CAM grid-point: {cam_lat:.2f}N, {cam_lon:.2f}E, {cam_elev:.0f}m",
            xy=(0.02, 0.02), xycoords='axes fraction', fontsize=7, color='grey',
        )

    setup_doy_xaxis(ax)


def station_precip_plot(ax, da, sdata, obs_datasets=('GPCP', 'TRMM')):
    """
    Precipitation annual-cycle plot with observational dataset overlays.

    Plots the model (or obs) running-mean annual cycle plus mean ± std or
    hour-separated lines, then overlays specified obs datasets.

    Parameters
    ----------
    ax           : matplotlib.axes.Axes
    da           : xarray.DataArray  (time,) precipitation in mm/day
    sdata        : dict
    obs_datasets : tuple of str  Dataset names to overlay (default: GPCP, TRMM)
    """
    year0   = sdata.get('year_first', 1970)
    year1   = sdata.get('year_last',  2100)
    drunave = sdata.get('drunave', 60)
    station = sdata.get('station', '')
    lat     = sdata.get('lat', np.nan)
    lon     = sdata.get('lon', np.nan)
    elev    = sdata.get('elev', np.nan)
    is_cam  = sdata.get('cam', False)

    times = pd.to_datetime(da['time'].values)
    doy   = _doy_from_times(times)
    years = times.year
    vals  = da.values.astype(float)

    good = np.isfinite(doy.astype(float))
    doy, years, vals = doy[good], years[good], vals[good]
    hour_vals = times.hour.values[good]

    in_range = (years >= year0) & (years <= year1)
    vals_yr  = np.where(in_range, vals, np.nan)

    ndoys  = int(np.nanmax(doy)) if len(doy) > 0 else 366
    ac_doy = np.arange(1, ndoys + 1, dtype=float)

    ax.set_ylim(0., 10.)

    def _build_mean(data, doy_arr, hours_arr, zh=None):
        ac = np.full(ndoys, np.nan)
        for id_ in range(1, ndoys + 1):
            if zh is not None:
                idx = np.where((doy_arr == id_) & (hours_arr == zh))[0]
            else:
                idx = np.where(doy_arr == id_)[0]
            v = data[idx]
            v = v[np.isfinite(v)]
            if len(v) > 0:
                ac[id_ - 1] = np.nanmean(v)
        return ac

    if sdata.get('lp1d_av_std', True):
        ac_mean = _build_mean(vals_yr, doy, hour_vals)
        std_arr = np.full(ndoys, np.nan)
        for id_ in range(1, ndoys + 1):
            idx = np.where(doy == id_)[0]
            v = vals_yr[idx]
            v = v[np.isfinite(v)]
            if len(v) > 1:
                std_arr[id_ - 1] = np.nanstd(v, ddof=1)

        ax.plot(ac_doy, _running_mean(ac_mean, drunave), 'k-', lw=2.5, zorder=5, label='Model mean')

    if sdata.get('lp1d_0012Z', False):
        z_hours = [0, 6, 12, 18]
        cols    = ['blue', 'darkgreen', 'red', 'brown']
        lbls    = ['00Z', '06Z', '12Z', '18Z']
        for zh, col, lbl in zip(z_hours, cols, lbls):
            ac_zh = _build_mean(vals_yr, doy, hour_vals, zh=zh)
            ax.plot(ac_doy, _running_mean(ac_zh, drunave),
                    color=col, lw=2, zorder=4, label=lbl)

    # Obs datasets
    obs_styles = {
        'GPCP':   ('gray', '-'),
        'TRMM':   ('gray', '--'),
        'CPC':    ('dimgrey', ':'),
        'Livneh': ('dimgrey', '-.'),
        'PRISM':  ('dimgrey', (0, (3, 1, 1, 1))),
    }
    for dset in obs_datasets:
        try:
            obs_da = get_obs_precip(dset, sdata)
        except (FileNotFoundError, Exception) as e:
            print(f"  Warning: could not load {dset}: {e}")
            continue
        obs_times = pd.to_datetime(obs_da['time'].values)
        obs_doy   = obs_times.day_of_year.values
        obs_vals  = obs_da.values.astype(float)
        ndoys_obs = int(np.nanmax(obs_doy))
        ac_obs    = np.full(ndoys_obs, np.nan)
        for id_ in range(1, ndoys_obs + 1):
            idx = np.where(obs_doy == id_)[0]
            v = obs_vals[idx]
            v = v[np.isfinite(v)]
            if len(v) > 0:
                ac_obs[id_ - 1] = np.nanmean(v)
        col, ls = obs_styles.get(dset, ('grey', '-'))
        ax.plot(np.arange(1, ndoys_obs + 1),
                _running_mean(ac_obs, drunave),
                color=col, ls=ls, lw=2, zorder=3, label=dset)

    ax.legend(fontsize=8, loc='upper right')
    ax.set_title(f"{station}  ({lat:.2f}\u00b0N, {lon:.2f}\u00b0E, {elev:.0f} m elev.)",
                 fontsize=10)
    ax.set_ylabel("Precipitation [mm/day]")
    setup_doy_xaxis(ax)

    if is_cam:
        cam_lat  = da.attrs.get('lat', np.nan)
        cam_lon  = da.attrs.get('lon', np.nan)
        cam_elev = da.attrs.get('cam_elev', np.nan)
        ax.annotate(
            f"CAM: {cam_lat:.2f}N, {cam_lon:.2f}E, {cam_elev:.0f}m",
            xy=(0.02, 0.02), xycoords='axes fraction', fontsize=7, color='grey',
        )


def station_tp_plot(ax, da, sdata, fig=None):
    """
    Time–pressure filled contour plot of a 2-D RAOB or CAM variable.

    The plot shows the running-mean annual cycle on a day-of-year x-axis
    with an inverted log-pressure y-axis.

    Behaviour is controlled by flags in *sdata*:
      lp2d_mean    : plot mean annual cycle (default True)
      lp2d_nomean  : subtract median annual cycle before plotting
      lp2d_std     : overlay blue contours of std dev
      lp2d_0012Z   : return smoothed 00Z, 12Z, and difference arrays for
                     3-panel plotting (does NOT render to *ax*; the caller
                     must handle multiple axes)

    Parameters
    ----------
    ax    : matplotlib.axes.Axes  (primary/single panel axes)
    da    : xarray.DataArray  (time, lev)
    sdata : dict
    fig   : matplotlib.figure.Figure, optional

    Returns
    -------
    None  (or dict with keys 'ac00_smooth', 'ac12_smooth', 'acd_smooth'
           when lp2d_0012Z is True — shapes (lev, doy))
    """
    year0    = sdata.get('year_first', 1970)
    year1    = sdata.get('year_last',  2100)
    drunave  = sdata.get('drunave', 60)
    station  = sdata.get('station', '')
    lat      = sdata.get('lat', np.nan)
    lon      = sdata.get('lon', np.nan)
    elev     = sdata.get('elev', np.nan)
    is_cam   = sdata.get('cam', False)
    cname    = da.attrs.get('cname', '')
    long_name = da.attrs.get('long_name', cname)
    units     = da.attrs.get('units', '')

    times = pd.to_datetime(da['time'].values)
    doy   = _doy_from_times(times)
    years = times.year
    vals  = da.values.astype(float)   # (ntime, nlev)
    levs  = da['lev'].values           # mandatory pressure levels in mb

    # Year mask
    in_range = (years >= year0) & (years <= year1)
    vals_yr  = np.where(in_range[:, np.newaxis], vals, np.nan)

    hour_vals = times.hour.values
    ndoys     = int(np.nanmax(doy)) if len(doy) > 0 else 366
    nlevs     = len(levs)
    ac_doy    = np.arange(1, ndoys + 1, dtype=float)

    vmin, vmax, vstep = get_contour_levels(cname, nomean=sdata.get('lp2d_nomean', False))
    levels_filled = np.arange(vmin, vmax + vstep, vstep)

    # ------------------------------------------------------------------
    # Helper: build (lev, doy) annual cycle array
    # ------------------------------------------------------------------
    def _build_ac(mask_extra=None, stat='mean'):
        """Build (nlevs, ndoys) annual cycle. mask_extra is boolean (ntime,)."""
        ac = np.full((nlevs, ndoys), np.nan)
        for id_ in range(1, ndoys + 1):
            if mask_extra is not None:
                idx = np.where((doy == id_) & mask_extra)[0]
            else:
                idx = np.where((doy == id_) &
                               ((hour_vals == 0) | (hour_vals == 12)))[0]
            if len(idx) == 0:
                continue
            v = vals_yr[idx, :]  # (nsel, nlev)
            if stat == 'mean':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    ac[:, id_ - 1] = np.nanmean(v, axis=0)
            elif stat == 'std':
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    ac[:, id_ - 1] = np.nanstd(v, axis=0, ddof=1)
        return ac

    # ------------------------------------------------------------------
    # 3-panel 00Z / 12Z / diff mode
    # ------------------------------------------------------------------
    if sdata.get('lp2d_0012Z', False):
        mask00  = hour_vals == 0
        mask12  = hour_vals == 12
        ac00    = _build_ac(mask_extra=mask00)
        ac12    = _build_ac(mask_extra=mask12)
        acd     = ac00 - ac12

        # Apply running mean along doy axis
        ac00_sm = np.apply_along_axis(_running_mean, 1, ac00, drunave)
        ac12_sm = np.apply_along_axis(_running_mean, 1, ac12, drunave)
        acd_sm  = np.apply_along_axis(_running_mean, 1, acd,  drunave)

        return {
            'ac00_smooth': ac00_sm,
            'ac12_smooth': ac12_sm,
            'acd_smooth':  acd_sm,
        }

    # ------------------------------------------------------------------
    # Single-panel mean plot
    # ------------------------------------------------------------------
    ac_mean = _build_ac()

    if sdata.get('lp2d_nomean', False):
        # Subtract the median along the doy axis for each level
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            median_ac = np.nanmedian(ac_mean, axis=1, keepdims=True)
        ac_mean = ac_mean - median_ac

    # Running mean along doy axis
    ac_smooth = np.apply_along_axis(_running_mean, 1, ac_mean, drunave)

    # Filled contour
    cf = ax.contourf(
        ac_doy, levs, ac_smooth,
        levels=levels_filled,
        cmap='RdBu_r',
        extend='both',
    )
    if fig is not None:
        fig.colorbar(cf, ax=ax, orientation='vertical', pad=0.02,
                     label=f"{long_name} [{units}]")

    # Black contour lines
    cs = ax.contour(
        ac_doy, levs, ac_smooth,
        levels=levels_filled,
        colors='k',
        linewidths=0.5,
    )
    # Negative dashed
    neg_levs = levels_filled[levels_filled < 0.]
    if len(neg_levs):
        ax.contour(
            ac_doy, levs, ac_smooth,
            levels=neg_levs,
            colors='k',
            linewidths=0.5,
            linestyles='dashed',
        )

    # Std dev overlay
    if sdata.get('lp2d_std', False):
        ac_std = _build_ac(stat='std')
        ac_std_smooth = np.apply_along_axis(_running_mean, 1, ac_std, drunave)
        ax.contour(
            ac_doy, levs, ac_std_smooth,
            colors='blue',
            linewidths=0.8,
        )

    # Y-axis: log scale, inverted, 1000–400 mb
    ax.set_yscale('log')
    ax.set_ylim(1000., 400.)
    ax.invert_yaxis()
    ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
    ax.yaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_ylabel("Pressure (mb)")

    # Title
    cam_info = ""
    if is_cam:
        cam_lat  = da.attrs.get('lat', np.nan)
        cam_lon  = da.attrs.get('lon', np.nan)
        cam_elev = da.attrs.get('cam_elev', np.nan)
        cam_info = f" | CAM: {cam_lat:.2f}N {cam_lon:.2f}E {cam_elev:.0f}m"
    ax.set_title(
        f"{station}  ({lat:.2f}\u00b0N, {lon:.2f}\u00b0E, {elev:.0f} m){cam_info}",
        fontsize=9,
    )
    ax.text(0.5, -0.08, f"{long_name} ({units})",
            ha='center', transform=ax.transAxes, fontsize=9)

    setup_doy_xaxis(ax)
