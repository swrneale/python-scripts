"""
cam_sortcontour.py

Contour plot of a 3D CAM field ordered (binned) by a 2D surface field.

For each case the data in a specified lat/lon region is stacked across
(time, lat, lon) and every column is assigned to a PRECT (or other 2D field)
percentile bin.  The mean 3D profile in each bin is plotted as a vertical
column; together the bins form a contourf plot with the 2D field on the x-axis
and pressure on the (inverted) y-axis.

Four cases are plotted as sub-panels:
  1. f.e30.FLTHIST.CAM7.L58.000a
  2. f.e30.FLTHIST.CAM7.L32.000a
  3. f.e30.FLTHIST.CAM7.L58.001a
  4. f.e30.FLTHIST.CAM7.L32.001a

Usage
-----
python cam_sortcontour.py                        # defaults
python cam_sortcontour.py -v3 ZMMU -v2 PRECT
python cam_sortcontour.py --lat1 -15 --lat2 15 --lon1 150 --lon2 280
python cam_sortcontour.py --year-start 1980 --year-end 1985
python cam_sortcontour.py --nbins 30 -o my_plot.png
"""

import os
import glob
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

# ── configuration ─────────────────────────────────────────────────────────────

ARCHIVE_DIR = "/glade/derecho/scratch/rneale/archive"

CASES = [
    "f.e30.FLTHIST.CAM7.L58.000a",
    "f.e30.FLTHIST.CAM7.L32.000a",
    "f.e30.FLTHIST.CAM7.L58.001a",
    "f.e30.FLTHIST.CAM7.L32.001a",
]

CASE_LABELS = [
    "L58  000a",
    "L32  000a",
    "L58  001a",
    "L32  001a",
]

# unit conversions: (scale, display_label)
UNIT_SCALE = {
    "PRECT":  (86_400 * 1000, "mm day⁻¹"),
    "PRECC":  (86_400 * 1000, "mm day⁻¹"),
    "ZMDT":   (86_400,        "K day⁻¹"),
    "ZMDQ":   (86_400 * 1000, "g kg⁻¹ day⁻¹"),
    "ZMMU":   (86_400,        "kg m⁻² day⁻¹"),
    "ZMMD":   (86_400,        "kg m⁻² day⁻¹"),
}

P_TOP_HPA = 100.0   # mask levels above this pressure (lower p = higher altitude)

# ── helpers ───────────────────────────────────────────────────────────────────

def get_files(case, year_start, year_end):
    """Return sorted list of h1a files for the given year range."""
    hist_dir = os.path.join(ARCHIVE_DIR, case, "atm", "hist")
    all_files = []
    for year in range(year_start, year_end + 1):
        pattern = os.path.join(hist_dir, f"{case}.cam.h1a.{year}-*-*.nc")
        all_files.extend(glob.glob(pattern))
    return sorted(all_files)


def load_region(files, var2d, var3d, lat1, lat2, lon1, lon2):
    """
    Open files with open_mfdataset and return area-selected DataArrays.

    Returns
    -------
    da2d : DataArray (time, lat, lon)
    da3d : DataArray (time, lev, lat, lon)
    lev  : 1-D pressure array in hPa
    """
    keep = [var2d, var3d, "lev"]
    ds = xr.open_mfdataset(
        files,
        combine="by_coords",
        data_vars="minimal",
        coords="minimal",
        compat="override",
        join="override",
    )
    # lat selection (handle any ordering)
    lat_mask = (ds.lat >= lat1) & (ds.lat <= lat2)
    ds = ds.sel(lat=lat_mask)

    # lon selection – allow wrapping (lon1 > lon2) if needed
    if lon1 <= lon2:
        lon_mask = (ds.lon >= lon1) & (ds.lon <= lon2)
    else:
        lon_mask = (ds.lon >= lon1) | (ds.lon <= lon2)
    ds = ds.sel(lon=lon_mask)

    da2d = ds[var2d]
    da3d = ds[var3d]
    lev  = ds["lev"].values   # hybrid levels stored in hPa
    ds.close()
    return da2d, da3d, lev


def build_bin_profiles(da2d, da3d, lev, nbins, p_top=P_TOP_HPA, pct_max=99.0):
    """
    Flatten (time, lat, lon) and bin by da2d values.

    Bin edges span the 0–pct_max percentile range; all data above pct_max
    is lumped into the last bin.  Bin centres are the actual data mean within
    each bin (more representative than edge midpoints).

    Returns
    -------
    bin_centers : (nbins,) – bin-mean values of the 2D field
    profiles    : (nbins, nlev_used) – bin-mean of the 3D field
    lev_used    : pressure levels kept (>= p_top)
    """
    lev_mask = lev >= p_top
    lev_used = lev[lev_mask]

    # load and flatten
    data2d = da2d.values.ravel()                     # (N,)
    data3d = da3d.values[:, lev_mask, :, :]          # (time, lev, lat, lon)
    ntime, nlev, nlat, nlon = data3d.shape
    data3d = data3d.reshape(ntime, nlev, -1)          # (time, lev, N_space)
    data3d = data3d.transpose(0, 2, 1)               # (time, N_space, lev)
    data3d = data3d.reshape(-1, nlev)                 # (N, lev)

    # remove NaN columns in 2D field
    valid = np.isfinite(data2d)
    data2d = data2d[valid]
    data3d = data3d[valid]

    # percentile-based bin edges; last edge = inf to absorb the tail
    bin_edges = np.percentile(data2d, np.linspace(0, pct_max, nbins + 1))
    bin_edges[-1] = np.inf

    bin_centers = np.full(nbins, np.nan)
    profiles    = np.full((nbins, nlev), np.nan)

    for i in range(nbins):
        mask = (data2d >= bin_edges[i]) & (data2d < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers[i] = np.mean(data2d[mask])
            profiles[i]    = np.nanmean(data3d[mask], axis=0)

    return bin_centers, profiles, lev_used


def unit_scale(varname):
    if varname in UNIT_SCALE:
        return UNIT_SCALE[varname]
    return (1.0, varname)


# ── main plot ─────────────────────────────────────────────────────────────────

def make_plot(var2d, var3d, lat1, lat2, lon1, lon2,
              year_start, year_end, nbins, pct_max, outfile):

    scale2d, units2d = unit_scale(var2d)
    scale3d, units3d = unit_scale(var3d)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharey=True)
    axes = axes.ravel()

    all_profiles = []   # collect for common color scale

    case_data = []
    for case in CASES:
        files = get_files(case, year_start, year_end)
        if not files:
            print(f"WARNING: no files found for {case} {year_start}-{year_end}")
            case_data.append(None)
            continue
        print(f"Loading {len(files)} files for {case} ...")
        da2d, da3d, lev = load_region(files, var2d, var3d, lat1, lat2, lon1, lon2)
        bin_centers, profiles, lev_used = build_bin_profiles(da2d, da3d, lev, nbins, pct_max=pct_max)
        bin_centers_disp = bin_centers * scale2d
        profiles_disp    = profiles   * scale3d
        case_data.append((bin_centers_disp, profiles_disp, lev_used))
        all_profiles.append(profiles_disp)

    # symmetric color limits centred on zero (robust percentile)
    all_vals = np.concatenate([p.ravel() for p in all_profiles if p is not None])
    vmax = np.nanpercentile(np.abs(all_vals), 98)
    vmin = -vmax

    for ax, data, label in zip(axes, case_data, CASE_LABELS):
        if data is None:
            ax.set_visible(False)
            continue
        bin_centers_disp, profiles_disp, lev_used = data

        # contourf: x = bins (PRECT), y = pressure
        cf = ax.contourf(
            bin_centers_disp,
            lev_used,
            profiles_disp.T,          # (lev, bins) → y first
            levels=20,
            cmap="RdBu_r",
            vmin=vmin,
            vmax=vmax,
            extend="both",
        )
        ax.contour(
            bin_centers_disp,
            lev_used,
            profiles_disp.T,
            levels=[0],
            colors="k",
            linewidths=0.8,
        )

        ax.set_yscale("log")
        ax.set_ylim(lev_used.max(), P_TOP_HPA)   # inverted pressure axis
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.set_yticks([1000, 850, 700, 500, 300, 200, 100])

        ax.set_xlabel(f"{var2d}  [{units2d}]", fontsize=10)
        ax.set_ylabel("Pressure (hPa)",        fontsize=10)
        ax.set_title(label, fontsize=11, fontweight="bold")

    # shared colorbar
    fig.subplots_adjust(right=0.88, hspace=0.35, wspace=0.18)
    cbar_ax = fig.add_axes([0.91, 0.12, 0.025, 0.76])
    fig.colorbar(cf, cax=cbar_ax, label=f"{var3d}  [{units3d}]")

    region_str = (f"lat [{lat1}°, {lat2}°]  lon [{lon1}°, {lon2}°]  "
                  f"{year_start}–{year_end}")
    fig.suptitle(
        f"{var3d} sorted by {var2d} — {region_str}\n"
        f"({nbins} percentile bins, 0–{pct_max:.0f}th pct)",
        fontsize=12,
    )

    if outfile is None:
        outfile = (f"cam_sortcontour_{var3d}_by_{var2d}"
                   f"_{year_start}-{year_end}_{nbins}bins.png")
    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved → {outfile}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v3", "--var3d",  default="ZMDT",
                   help="3D CAM variable (vertical field, default: ZMDT)")
    p.add_argument("-v2", "--var2d",  default="PRECT",
                   help="2D CAM variable used for ordering (default: PRECT)")
    p.add_argument("--lat1",  type=float, default=-30.0, help="South lat bound (default: -30)")
    p.add_argument("--lat2",  type=float, default=30.0,  help="North lat bound (default:  30)")
    p.add_argument("--lon1",  type=float, default=0.0,   help="West lon bound  (default:   0)")
    p.add_argument("--lon2",  type=float, default=360.0, help="East lon bound  (default: 360)")
    p.add_argument("--year-start", type=int, default=1981, help="First year (default: 1981)")
    p.add_argument("--year-end",   type=int, default=1982, help="Last year  (default: 1982)")
    p.add_argument("--nbins", type=int, default=20,
                   help="Number of percentile bins for the 2D field (default: 20)")
    p.add_argument("--pct-max", type=float, default=99.0,
                   help="Upper percentile cap for binning; tail above is lumped into last bin (default: 99)")
    p.add_argument("-o", "--outfile", default=None, help="Output PNG path")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    make_plot(
        var2d      = args.var2d.upper(),
        var3d      = args.var3d.upper(),
        lat1       = args.lat1,
        lat2       = args.lat2,
        lon1       = args.lon1,
        lon2       = args.lon2,
        year_start = args.year_start,
        year_end   = args.year_end,
        nbins      = args.nbins,
        pct_max    = args.pct_max,
        outfile    = args.outfile,
    )
