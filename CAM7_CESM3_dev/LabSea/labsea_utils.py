"""
Utilities for Labrador Sea CESM3 pre-industrial development diagnostics.

Two runs:
  388 (b.e30_alpha09e_...) - remains frozen in Labrador Sea
  377 (b.e30_alpha09d_...) - melts in Labrador Sea
Both start from the same slightly frozen state.

Provides:
  - CASES : dict of run metadata (path, label, colour)
  - REGIONS : dict of lat/lon boxes for Labrador Sea and SE sub-region
  - VARS : plotting metadata (units, scale, cmap, contour levels) for the six diagnostic fields
  - load_run(...)          : open multi-year monthly h0a stream as one xarray dataset
  - seasonal_mean(...)     : composite season mean (JFM/JAS/etc) over selected years
  - regional_mean_ts(...)  : area-weighted regional monthly timeseries
  - to_seasonal_ts(...)    : collapse monthly ts to seasonal means (year, season)
  - plot_pair_map(...)     : side-by-side maps 388 vs 377 for one field/season
  - plot_seasonal_ts(...)  : line plot of seasonal timeseries for both runs
"""

import glob
import os

import numpy as np
import xarray as xr

import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

ARCHIVE_ROOT = "/glade/derecho/scratch/hannay/archive"

GPCP_FILE = "/glade/work/rneale/data/GPCP/gpcp.mon.mean.197901-201607.nc"

CASES = {
    "388": {
        "case_name": "b.e30_alpha09e_m.B1850C_MTso_Gris_Marbl.ne30_t233_wgx3.388",
        "label":     "388 (frozen)",
        "color":     "royalblue",
        "kind":      "model",
    },
    "377": {
        "case_name": "b.e30_alpha09d_m.B1850C_MTso_Gris_Marbl.ne30_t233_wgx3.377",
        "label":     "377 (melts)",
        "color":     "firebrick",
        "kind":      "model",
    },
    "GPCP": {
        "label":  "GPCP obs",
        "color":  "black",
        "kind":   "obs",
    },
}

# Longitudes are 0-360 in the h0a files.
REGIONS = {
    # Map plot domain — a Labrador Sea window that also shows Greenland and
    # the eastern Canadian coast for context.
    "LabSea": {
        "lat_min": 45.0, "lat_max": 75.0,
        "lon_min": 280.0, "lon_max": 340.0,
    },
    # Area-average box for seasonal timeseries — the SE side of the
    # Labrador Sea where 388 freezes in JFM.
    "LabSea_SE": {
        "lat_min": 55.0, "lat_max": 62.0,
        "lon_min": 305.0, "lon_max": 318.0,
    },
}

# Season -> list of month integers
SEASONS = {
    "JFM": [1, 2, 3],
    "AMJ": [4, 5, 6],
    "JAS": [7, 8, 9],
    "OND": [10, 11, 12],
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
    "ANN": list(range(1, 13)),
}


# Plotting metadata for the six diagnostic fields.
# 'derive' is a function of the dataset that returns the raw field (for PRECT).
VARS = {
    "ICEFRAC": {
        "long_name": "Sea-ice fraction",
        "units":     "fraction",
        "scale":     1.0,
        "offset":    0.0,
        "cmap":      "Blues",
        "levels_jfm": np.linspace(0.0, 1.0, 11),
        "levels_jas": np.linspace(0.0, 1.0, 11),
    },
    "SHFLX": {
        "long_name": "Sensible heat flux (up)",
        "units":     "W m$^{-2}$",
        "scale":     1.0,
        "offset":    0.0,
        "cmap":      "RdBu_r",
        "levels_jfm": np.arange(-50, 351, 25),
        "levels_jas": np.arange(-50, 101, 10),
    },
    "LHFLX": {
        "long_name": "Latent heat flux (up)",
        "units":     "W m$^{-2}$",
        "scale":     1.0,
        "offset":    0.0,
        "cmap":      "RdBu_r",
        "levels_jfm": np.arange(-50, 351, 25),
        "levels_jas": np.arange(0, 251, 20),
    },
    "TS": {
        "long_name": "Surface temperature",
        "units":     "$^{\\circ}$C",
        "scale":     1.0,
        "offset":    -273.15,
        "cmap":      "RdBu_r",
        "levels_jfm": np.arange(-30, 21, 2),
        "levels_jas": np.arange(-5, 21, 1),
    },
    "PRECT": {
        "long_name": "Total precipitation",
        "units":     "mm day$^{-1}$",
        "scale":     86400.0 * 1000.0,
        "offset":    0.0,
        "cmap":      "YlGnBu",
        "levels_jfm": np.arange(0, 11, 1),
        "levels_jas": np.arange(0, 11, 1),
    },
    "CLDLOW": {
        "long_name": "Low cloud fraction",
        "units":     "%",
        "scale":     100.0,
        "offset":    0.0,
        "cmap":      "Blues",
        "levels_jfm": np.arange(0, 101, 10),
        "levels_jas": np.arange(0, 101, 10),
    },
    "FSNS": {
        "long_name": "Net solar flux at surface",
        "units":     "W m$^{-2}$",
        "scale":     1.0,
        "offset":    0.0,
        "cmap":      "YlOrRd",
        "levels_jfm": np.arange(0, 151, 10),
        "levels_jas": np.arange(0, 301, 20),
    },
    "FLNS": {
        "long_name": "Net longwave flux at surface (up)",
        "units":     "W m$^{-2}$",
        "scale":     1.0,
        "offset":    0.0,
        "cmap":      "RdBu_r",
        "levels_jfm": np.arange(-20, 121, 10),
        "levels_jas": np.arange(-20, 121, 10),
    },
    "FSDS": {
        "long_name": "Downwelling solar flux at surface",
        "units":     "W m$^{-2}$",
        "scale":     1.0,
        "offset":    0.0,
        "cmap":      "YlOrRd",
        "levels_jfm": np.arange(0, 201, 20),
        "levels_jas": np.arange(0, 401, 25),
    },
    "RESSURF": {
        # Net downward energy flux at the surface, derived from CAM output as
        #     RESSURF = FSNS - FLNS - SHFLX - LHFLX
        # (FSNS is net solar down; FLNS/SHFLX/LHFLX positive up.)  A positive
        # value means the surface is gaining energy, negative losing.
        "long_name": "Net surface energy flux (down)",
        "units":     "W m$^{-2}$",
        "scale":     1.0,
        "offset":    0.0,
        "cmap":      "RdBu_r",
        "levels_jfm": np.arange(-300, 301, 30),
        "levels_jas": np.arange(-200, 201, 20),
    },
}

VAR_LIST = ["ICEFRAC", "SHFLX", "LHFLX", "TS", "PRECT", "CLDLOW",
            "FSNS", "FLNS", "FSDS", "RESSURF"]

# ---------------------------------------------------------------------------
# MOM ocean variables (curvilinear grid, kept out of VAR_LIST above so the
# atm-side loops don't touch them).  The notebook drives MOM plots via a
# separate loop that uses the MOM-aware helpers below.
# ---------------------------------------------------------------------------

MOM_HIST_SUBDIR = "ocn/hist"
MOM_STREAM      = "sfc"   # daily sfc-stream, contains mlotst
MOM_STATIC_STREAM = "static"

# Map internal name → MOM variable in the sfc stream.
MOM_VAR_MAP = {
    "MLD": "mlotst",   # Ocean mixed-layer thickness (sigma-T, m)
}
MOM_VAR_LIST = list(MOM_VAR_MAP.keys())

VARS["MLD"] = {
    "long_name": "Mixed-layer depth (σ$_T$)",
    "units":     "m",
    "scale":     1.0,
    "offset":    0.0,
    "cmap":      "cmo.deep" if False else "viridis",
    # Deep winter mixing in Lab Sea reaches 500-2000 m in obs; sub-tropical
    # summer values are 20-80 m.  Use widely-spaced levels.
    "levels_jfm": np.array([25, 50, 100, 200, 400, 600, 800, 1200, 1600, 2000, 2500]),
    "levels_jas": np.array([10, 20, 30, 40, 60, 80, 100, 150, 200, 300, 400]),
}


def _hist_dir(run_id):
    case = CASES[run_id]["case_name"]
    return os.path.join(ARCHIVE_ROOT, case, "atm", "hist")


def list_h0a_files(run_id, year_range=None):
    """Return sorted list of h0a monthly files for a run, optionally restricted to year_range = (y0, y1) inclusive."""
    d = _hist_dir(run_id)
    case = CASES[run_id]["case_name"]
    files = sorted(glob.glob(os.path.join(d, f"{case}.cam.h0a.*.nc")))

    if year_range is not None:
        y0, y1 = year_range
        keep = []
        for f in files:
            # trailing pattern .YYYY-MM.nc
            stem = os.path.splitext(os.path.basename(f))[0]   # strip .nc
            ym = stem.rsplit(".", 1)[-1]                       # YYYY-MM
            try:
                yy = int(ym.split("-")[0])
            except ValueError:
                continue
            if y0 <= yy <= y1:
                keep.append(f)
        files = keep
    return files


def _preprocess(ds, vars_keep):
    """Keep only wanted variables plus a couple of ancillary fields."""
    aux = [v for v in ("LANDFRAC", "OCNFRAC") if v in ds.variables]
    keep = [v for v in vars_keep if v in ds.variables] + aux
    return ds[keep]


def load_run(run_id, vars_needed, year_range=None, parallel=False):
    """
    Open the full monthly stream for a run as a single xarray dataset.

    vars_needed  - iterable of CAM variable names to keep on disk
                   (PRECC/PRECL are added automatically when PRECT is requested).
    year_range   - (y0, y1) inclusive, or None for all years.
    parallel     - passed to xr.open_mfdataset (requires dask).
    """
    vars_needed = list(vars_needed)
    if "PRECT" in vars_needed:
        for extra in ("PRECC", "PRECL"):
            if extra not in vars_needed:
                vars_needed.append(extra)
    if "RESSURF" in vars_needed:
        for extra in ("FSNS", "FLNS", "SHFLX", "LHFLX"):
            if extra not in vars_needed:
                vars_needed.append(extra)

    files = list_h0a_files(run_id, year_range=year_range)
    if not files:
        raise FileNotFoundError(f"No h0a files found for run {run_id} in {_hist_dir(run_id)}")

    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        parallel=parallel,
        preprocess=lambda d: _preprocess(d, vars_needed),
        decode_times=True,
    )

    # Add convenience year/month coordinates from the time bounds midpoint if available.
    if "time" in ds.coords:
        ds = ds.assign_coords(
            year=("time", ds["time"].dt.year.values),
            month=("time", ds["time"].dt.month.values),
        )
    return ds


# ---------------------------------------------------------------------------
# MOM (ocean) loading — curvilinear grid
# ---------------------------------------------------------------------------

def _mom_hist_dir(run_id):
    case = CASES[run_id]["case_name"]
    return os.path.join(ARCHIVE_ROOT, case, MOM_HIST_SUBDIR)


def load_mom_static(run_id):
    """Return the MOM static grid (geolat, geolon, wet, areacello) as a Dataset."""
    case = CASES[run_id]["case_name"]
    fpath = os.path.join(_mom_hist_dir(run_id),
                         f"{case}.mom6.h.{MOM_STATIC_STREAM}.nc")
    ds = xr.open_dataset(fpath, decode_times=False)
    keep = [v for v in ("geolat", "geolon", "wet", "areacello") if v in ds]
    return ds[keep]


def list_mom_files(run_id, stream=None, year_range=None):
    """Return sorted list of MOM history files for a stream (default 'sfc')."""
    stream = stream or MOM_STREAM
    case = CASES[run_id]["case_name"]
    files = sorted(glob.glob(os.path.join(
        _mom_hist_dir(run_id),
        f"{case}.mom6.h.{stream}.*.nc")))
    # skip tiled entries (trailing '.NNNN' after .nc) — keep the concatenated file
    files = [f for f in files if f.endswith(".nc")]

    if year_range is not None:
        y0, y1 = year_range
        keep = []
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            ym = stem.rsplit(".", 1)[-1]
            try:
                yy = int(ym.split("-")[0])
            except ValueError:
                continue
            if y0 <= yy <= y1:
                keep.append(f)
        files = keep
    return files


def load_run_mom(run_id, mom_vars=None, year_range=None, parallel=False):
    """
    Open the MOM sfc daily stream, aggregate daily → monthly means, and attach
    the static grid.  Returns an xarray dataset with keys named per MOM_VAR_MAP
    (e.g. 'MLD'), plus 2D 'geolat', 'geolon' coordinates and 'wet', 'areacello'.
    """
    mom_vars = list(mom_vars or MOM_VAR_LIST)
    file_vars = [MOM_VAR_MAP[v] for v in mom_vars if v in MOM_VAR_MAP]

    files = list_mom_files(run_id, stream=MOM_STREAM, year_range=year_range)
    if not files:
        raise FileNotFoundError(
            f"No MOM {MOM_STREAM} files for run {run_id} in {_mom_hist_dir(run_id)}"
        )

    def _pp(ds):
        # Keep only the requested data variables — this drops MOM6's
        # `average_T1`, `average_T2`, `average_DT`, `time_bounds`, etc. which
        # can carry inconsistent time metadata between files and cause
        # xarray to invent a spurious `time1` dimension on combine.
        keep = [v for v in file_vars if v in ds.variables]
        out = ds[keep]
        # Belt-and-braces: strip any bounds-like coords that survived.
        drop = [c for c in out.coords
                if c not in out.dims and c not in ("time",)]
        return out.reset_coords(drop, drop=True) if drop else out

    ds = xr.open_mfdataset(
        files, combine="nested", concat_dim="time",
        parallel=parallel,
        preprocess=_pp,
        data_vars="minimal", coords="minimal", compat="override",
        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True),
        decode_timedelta=False,
    )

    # Rename to canonical names (mlotst → MLD, etc.)
    rename = {MOM_VAR_MAP[v]: v for v in mom_vars if MOM_VAR_MAP[v] in ds}
    ds = ds.rename(rename)

    # Daily → monthly means.  Use resample which preserves the cftime index
    # (groupby-by-strftime would collapse it to strings and break downstream
    # time arithmetic).
    ds_mon = ds.resample(time="1MS").mean()
    ds_mon = ds_mon.assign_coords(
        year=("time", ds_mon["time"].dt.year.values),
        month=("time", ds_mon["time"].dt.month.values),
    )

    # Attach static grid
    stat = load_mom_static(run_id)
    for c in stat.data_vars:
        ds_mon[c] = stat[c]

    return ds_mon


# ---------------------------------------------------------------------------
# Variable extraction + scaling
# ---------------------------------------------------------------------------

def get_var(ds, var):
    """Return the raw DataArray for a variable, computing derived fields if needed."""
    if var == "PRECT":
        if "PRECT" in ds.variables:
            da = ds["PRECT"]
        else:
            da = ds["PRECC"] + ds["PRECL"]
    elif var == "RESSURF":
        # Net surface energy flux (down):  FSNS - FLNS - SHFLX - LHFLX
        da = ds["FSNS"] - ds["FLNS"] - ds["SHFLX"] - ds["LHFLX"]
    else:
        da = ds[var]
    return da


def apply_scale(da, var):
    info = VARS[var]
    return da * info["scale"] + info["offset"]


# ---------------------------------------------------------------------------
# Observations (GPCP for PRECT)
# ---------------------------------------------------------------------------

def load_gpcp():
    """
    Open GPCP monthly precip as an xarray dataset with variable 'PRECT' already
    in mm/day, plus year/month convenience coordinates.
    """
    ds = xr.open_dataset(GPCP_FILE, decode_times=True)
    da = ds["precip"].rename("PRECT")
    # Latitudes in GPCP go north->south; make ascending so .sel(slice) works
    if float(da["lat"][0]) > float(da["lat"][-1]):
        da = da.reindex(lat=da["lat"][::-1])
    out = da.to_dataset()
    out = out.assign_coords(
        year=("time", out["time"].dt.year.values),
        month=("time", out["time"].dt.month.values),
    )
    out.attrs["kind"] = "obs"
    out.attrs["already_in_plot_units"] = 1
    return out


def obs_seasonal_mean(ds_obs, var, season):
    """Composite mean of the season's months for an obs dataset already in plot units."""
    months = SEASONS[season]
    da = ds_obs[var]
    da = da.where(da["month"].isin(months), drop=True)
    return da.mean(dim="time")


def obs_regional_mean_ts(ds_obs, var, region):
    """Area-weighted monthly regional-mean timeseries from obs dataset already in plot units."""
    da = ds_obs[var]
    da = regional_slice(da, region)
    w = _area_weights(da)
    return da.weighted(w).mean(dim=("lat", "lon"))


# ---------------------------------------------------------------------------
# Seasonal / regional operations
# ---------------------------------------------------------------------------

def seasonal_mean(ds, var, season, year_range=None):
    """
    Composite mean of the season's months across the year range.
    Returns a (lat, lon) DataArray in plot units.
    """
    months = SEASONS[season]
    da = get_var(ds, var)
    if year_range is not None:
        y0, y1 = year_range
        da = da.where((da["year"] >= y0) & (da["year"] <= y1), drop=True)
    da = da.where(da["month"].isin(months), drop=True)
    da_mean = da.mean(dim="time")
    return apply_scale(da_mean, var)


def _area_weights(da):
    w = np.cos(np.deg2rad(da["lat"]))
    w.name = "area_weights"
    return w


def regional_slice(da, region):
    r = REGIONS[region]
    return da.sel(
        lat=slice(r["lat_min"], r["lat_max"]),
        lon=slice(r["lon_min"], r["lon_max"]),
    )


def regional_mean_ts(ds, var, region):
    """Area-weighted monthly regional-mean timeseries in plot units."""
    da = get_var(ds, var)
    da = regional_slice(da, region)
    w = _area_weights(da)
    ts = da.weighted(w).mean(dim=("lat", "lon"))
    return apply_scale(ts, var)


def to_seasonal_ts(monthly_ts, season):
    """
    Collapse a monthly timeseries to one value per year for the given season.
    For DJF the December is assigned to the following JF-year (standard convention).
    Returns a DataArray with a 'year' coordinate.
    """
    months = SEASONS[season]
    da = monthly_ts

    if season == "DJF":
        yr = da["year"].values.copy()
        mo = da["month"].values
        yr[mo == 12] = yr[mo == 12] + 1
        da = da.assign_coords(sea_year=("time", yr))
    else:
        da = da.assign_coords(sea_year=("time", da["year"].values))

    da = da.where(da["month"].isin(months), drop=True)
    grouped = da.groupby("sea_year").mean(dim="time")
    grouped = grouped.rename({"sea_year": "year"})
    return grouped


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _setup_map_axes(ax, region):
    r = REGIONS[region]
    ax.set_extent([r["lon_min"], r["lon_max"], r["lat_min"], r["lat_max"]],
                  crs=ccrs.PlateCarree())
    ax.coastlines(resolution="50m", linewidth=0.8, color="black")
    ax.add_feature(cfeature.LAND, facecolor="lightgray", zorder=0)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray", alpha=0.6)
    gl.top_labels = False
    gl.right_labels = False


def _region_box(ax, region, edge_color="k"):
    r = REGIONS[region]
    rect = mpatches.Rectangle(
        (r["lon_min"], r["lat_min"]),
        r["lon_max"] - r["lon_min"],
        r["lat_max"] - r["lat_min"],
        linewidth=1.5, edgecolor=edge_color, facecolor="none",
        transform=ccrs.PlateCarree(), zorder=10,
    )
    ax.add_patch(rect)


def plot_pair_map(fields, var, season, region="LabSea",
                  save_dir=None, extra_box=None):
    """
    Side-by-side maps of a field for two or more cases.

    fields : dict {case_id: 2D DataArray in plot units} — order preserved.
             case_id must exist in CASES (models or obs, e.g. "GPCP").
    var    : variable key in VARS
    season : season key (used for title + level-set selection)
    region : REGIONS key for map extent
    extra_box : optional REGIONS key to overlay as a red box (e.g., 'LabSea_SE')
    """
    info = VARS[var]
    key = "levels_jfm" if season == "JFM" else "levels_jas"
    levels = info.get(key, None)
    if levels is None:
        vmin = min(float(f.min()) for f in fields.values())
        vmax = max(float(f.max()) for f in fields.values())
        levels = np.linspace(vmin, vmax, 15)

    run_ids = list(fields.keys())
    ncol = len(run_ids)
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1, ncol, figsize=(6.5 * ncol, 5.5),
        subplot_kw={"projection": proj},
    )
    if ncol == 1:
        axes = [axes]

    ims = []
    for ax, rid in zip(axes, run_ids):
        da = regional_slice(fields[rid], region)
        _setup_map_axes(ax, region)
        im = ax.contourf(
            da["lon"], da["lat"], da.values,
            levels=levels, cmap=info["cmap"], extend="both",
            transform=proj,
        )
        ims.append(im)
        if extra_box is not None:
            _region_box(ax, extra_box, edge_color="red")

        w = _area_weights(da)
        m = da.weighted(w).mean(dim=("lat", "lon")).values
        label = CASES.get(rid, {}).get("label", rid)
        ax.set_title(f"{label}   mean={float(m):.2f}",
                     fontsize=12, fontweight="bold")

    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    cbar = fig.colorbar(ims[-1], cax=cbar_ax, orientation="horizontal", extend="both")
    cbar.set_label(f"{info['long_name']} ({info['units']})", fontsize=12)

    fig.suptitle(f"{var} — {info['long_name']} — {season}",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.90, bottom=0.16, wspace=0.08)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"map_{var}_{season}_{region}.png")
        fig.savefig(fname, dpi=140, bbox_inches="tight")
        print(f"  saved {fname}")

    return fig, axes


def plot_seasonal_ts(ts_by_run, var, season, region,
                     save_dir=None, roll=None):
    """
    Overlay seasonal-mean timeseries for all cases on one panel.

    ts_by_run : dict {case_id: 1D DataArray with 'year' coord}
                Obs cases (CASES[id]['kind']=='obs') are drawn along the model
                x-axis starting at the first model year — real years disregarded.
    roll      : optional int, apply a centered rolling mean of this length
    """
    info = VARS[var]
    fig, ax = plt.subplots(figsize=(10, 4.5))

    # Determine model start year for obs axis alignment.
    model_start = None
    for rid, ts in ts_by_run.items():
        if CASES.get(rid, {}).get("kind", "model") == "model":
            model_start = int(ts["year"].values.min())
            break

    for rid, ts in ts_by_run.items():
        info_c = CASES.get(rid, {})
        kind = info_c.get("kind", "model")
        c = info_c.get("color", "gray")
        label = info_c.get("label", rid)

        if kind == "obs" and model_start is not None:
            x = model_start + np.arange(ts.sizes["year"])
        else:
            x = ts["year"].values
        v = ts.values

        style = {"color": c}
        if kind == "obs":
            style["linestyle"] = "--"

        ax.plot(x, v, alpha=0.35, linewidth=1.0, **style)
        if roll is not None and len(v) >= roll:
            s = ts.rolling(year=roll, center=True).mean()
            ax.plot(x, s.values, linewidth=2.2,
                    label=f"{label} ({roll}-yr smoothed)", **style)
        else:
            ax.plot(x, v, linewidth=2.0, label=label, **style)

    ax.set_xlabel("Model year (obs plotted along same axis)", fontsize=12)
    ax.set_ylabel(f"{info['long_name']} ({info['units']})", fontsize=12)
    ax.set_title(f"{var} — {season} regional mean over {region}",
                 fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.4)
    ax.legend(fontsize=11, loc="best")
    fig.tight_layout()

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"ts_{var}_{season}_{region}.png")
        fig.savefig(fname, dpi=140, bbox_inches="tight")
        print(f"  saved {fname}")

    return fig, ax


# ---------------------------------------------------------------------------
# Convenience: overlap year range across runs
# ---------------------------------------------------------------------------

def year_range(run_id):
    """Return (y0, y1) inclusive from the file names."""
    files = list_h0a_files(run_id)
    years = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        ym = stem.rsplit(".", 1)[-1]
        try:
            years.append(int(ym.split("-")[0]))
        except ValueError:
            pass
    return min(years), max(years)


def overlap_years(run_ids):
    ranges = [year_range(r) for r in run_ids]
    y0 = max(r[0] for r in ranges)
    y1 = min(r[1] for r in ranges)
    return y0, y1


# ---------------------------------------------------------------------------
# MOM curvilinear-grid seasonal-mean / regional-mean / mapping
# ---------------------------------------------------------------------------

def _mom_geobox_mask(ds, region):
    """Boolean mask on the MOM curvilinear grid for the given lat/lon box.

    Uses geolat + normalised geolon (0-360) so the box is expressed in the
    same convention as REGIONS.
    """
    r = REGIONS[region]
    glon = (ds["geolon"] % 360.0)
    glat = ds["geolat"]
    m = (glat >= r["lat_min"]) & (glat <= r["lat_max"]) & \
        (glon >= r["lon_min"]) & (glon <= r["lon_max"])
    if "wet" in ds:
        m = m & (ds["wet"] > 0.5)
    return m


def mom_seasonal_mean(ds, var, season, year_range=None):
    """Season composite of a MOM variable in plot units, over ocean cells only."""
    months = SEASONS[season]
    da = ds[var]
    if year_range is not None:
        y0, y1 = year_range
        da = da.where((ds["year"] >= y0) & (ds["year"] <= y1), drop=True)
    da = da.where(ds["month"].isin(months), drop=True)
    da_mean = da.mean(dim="time")
    if "wet" in ds:
        da_mean = da_mean.where(ds["wet"] > 0.5)
    # Attach geolat/geolon so downstream plotting can find them
    da_mean = da_mean.assign_coords(geolat=ds["geolat"],
                                     geolon=ds["geolon"])
    return apply_scale(da_mean, var)


def mom_regional_mean_ts(ds, var, region):
    """Area-weighted monthly ocean regional-mean timeseries on the MOM grid."""
    da = ds[var]
    mask = _mom_geobox_mask(ds, region)
    if "areacello" in ds:
        w = ds["areacello"].where(mask)
    else:
        w = mask.astype(float)
    w = w.fillna(0.0)
    total_w = w.sum(dim=("yh", "xh"))
    ts = (da.where(mask) * w).sum(dim=("yh", "xh")) / total_w
    return apply_scale(ts, var)


def plot_pair_map_mom(fields, var, season, region="LabSea",
                       save_dir=None, extra_box=None):
    """Side-by-side maps for MOM curvilinear-grid fields (e.g. MLD).

    fields : dict {run_id: DataArray with geolat, geolon coords} — order kept.
    """
    info = VARS[var]
    key = "levels_jfm" if season == "JFM" else "levels_jas"
    levels = info.get(key)
    print(levels)
   
    if levels is None:
        vmin = min(float(np.nanmin(f)) for f in fields.values())
        vmax = max(float(np.nanmax(f)) for f in fields.values())
        levels = np.linspace(vmin, vmax, 15)
       
        

    r = REGIONS[region]
    run_ids = list(fields.keys())
    ncol = len(run_ids)
    proj = ccrs.PlateCarree()
    fig, axes = plt.subplots(
        1, ncol, figsize=(6.5 * ncol, 5.5),
        subplot_kw={"projection": proj},
    )
    if ncol == 1:
        axes = [axes]

    ims = []
    for ax, rid in zip(axes, run_ids):
        da = fields[rid]
        glat = da["geolat"].values
        glon = (da["geolon"].values % 360.0)
        # Use contourf directly on the 2D geographic mesh (cartopy handles the
        # curvilinear transform); mask cells outside the plot box for cleaner
        # panel-mean statistics.
        _setup_map_axes(ax, region)
        im = ax.contourf(
            glon, glat, da.values,
            levels=levels, cmap=info["cmap"], extend="both",
            transform=proj,
        )
        ims.append(im)
        if extra_box is not None:
            _region_box(ax, extra_box, edge_color="red")

        # Area-weighted panel mean over the box
        m_box = (glat >= r["lat_min"]) & (glat <= r["lat_max"]) & \
                (glon >= r["lon_min"]) & (glon <= r["lon_max"])
        vals = da.values
        panel_mean = float(np.nanmean(vals[m_box])) if m_box.any() else float("nan")
        label = CASES.get(rid, {}).get("label", rid)
        ax.set_title(f"{label}   mean={panel_mean:.1f}",
                     fontsize=12, fontweight="bold")

    cbar_ax = fig.add_axes([0.15, 0.06, 0.7, 0.03])
    cbar = fig.colorbar(ims[-1], cax=cbar_ax, orientation="horizontal", extend="both")
    cbar.set_label(f"{info['long_name']} ({info['units']})", fontsize=12)

    fig.suptitle(f"{var} — {info['long_name']} — {season}",
                 fontsize=15, fontweight="bold", y=0.98)
    fig.subplots_adjust(top=0.90, bottom=0.16, wspace=0.08)

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        fname = os.path.join(save_dir, f"map_{var}_{season}_{region}.png")
        fig.savefig(fname, dpi=140, bbox_inches="tight")
        print(f"  saved {fname}")

    return fig, axes


def year_range_mom(run_id, stream=None):
    """Return (y0, y1) inclusive from MOM sfc file names."""
    stream = stream or MOM_STREAM
    files = list_mom_files(run_id, stream=stream)
    years = []
    for f in files:
        stem = os.path.splitext(os.path.basename(f))[0]
        ym = stem.rsplit(".", 1)[-1]
        try:
            years.append(int(ym.split("-")[0]))
        except ValueError:
            pass
    if not years:
        return None
    return min(years), max(years)
