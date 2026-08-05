"""
scam_composite_scatter.py

Composite scatter plot of SCAM IOP case means across run types and vertical resolutions.

X-axis : IOP groups (togaII, gateIII, arm95, arm97)
Y-axis : time-mean of a chosen variable
Color  : run type  (CAM6 | CAM7 | CAM7 as CAM6)
Marker : resolution (L32 | L48 | L58 | L256)

Usage
-----qf
python scam_composite_scatter.py                      # plots PRECT (mm/day)
python scam_composite_scatter.py -v LHFLX             # surface LH flux
python scam_composite_scatter.py -v T -p 850          # T at ~850 hPa
python scam_composite_scatter.py -v PRECT --obs       # add observed IOP stars
python scam_composite_scatter.py -v PRECT -o out.png
"""

import os
import glob
import argparse
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import xarray as xr

warnings.filterwarnings("ignore", category=FutureWarning)

# ── configuration ────────────────────────────────────────────────────────────

BASE_DIR = "/glade/work/rneale/scam_cases/cases/ZM_PBL_paper"
IOP_DIR  = "/glade/work/rneale/scam_cases/iops"

IOPS       = ["togaII",     "gateIII",  "arm95",    "arm97"]
IOP_LABELS = ["TOGA-COARE", "GATE III", "ARM 1995", "ARM 1997"]

IOP_OBS_FILES = {
    "togaII":  "TOGAII_4scam.nc",
    "gateIII": "GATEIII_4scam.nc",
    "arm95":   "ARM95_4scam.nc",
    "arm97":   "ARM97_4scam.nc",
}

# Map from model variable name → observed variable name in the IOP file.
OBS_VAR_MAP = {
    "PRECT": "Prec",
    "LHFLX": "lhflx",
    "SHFLX": "shflx",
    "T":     "T",
    "Q":     "q",
    "U":     "u",
    "V":     "v",
    "OMEGA": "omega",
    "RELHUM": "relhum",
}

RUN_TYPES  = ["cam6-phys.000", "cam7-phys.000", "cam7-phys.001"]
RUN_LABELS = ["CAM6",          "CAM7",          "CAM7 as CAM6"]
RUN_COLORS = ["#1f77b4",       "#ff7f0e",        "#2ca02c"]

LEVELS        = ["L32", "L48", "L58", "L256"]
LEVEL_MARKERS = {"L32": "o", "L48": "s", "L58": "^", "L256": "D"}
MARKER_SIZE   = 90
MARKER_EDGE   = 0.6

IOP_STEP        = 3.5
RUNTYPE_OFFSETS = [-0.55, 0.0, 0.55]

UNIT_CONVERSIONS = {
    "PRECT": (86_400 * 1000, "mm day⁻¹"),
    "PRECC": (86_400 * 1000, "mm day⁻¹"),
    "PRECL": (86_400 * 1000, "mm day⁻¹"),
    # CLUBB static energy tendency [J/(kg s)] → K/day: divide by Cp (1004 J/kg/K) × 86400 s/day
    "STEND_CLUBB": (86_400 / 1004.0, "K day⁻¹"),
}

AUTO_UNIT_CONVERSIONS = {
    "K/s":      (86_400,             "K day⁻¹"),
    "kg/kg":    (1_000,              "g kg⁻¹"),
    "kg/kg/s":  (1_000 * 86_400,     "g kg⁻¹ day⁻¹"),
    "kg/kg /s": (1_000 * 86_400,     "g kg⁻¹ day⁻¹"),
}


# ── helpers ──────────────────────────────────────────────────────────────────

def get_display_scale(varname):
    """Return (scale, display_units) for varname.

    Priority: UNIT_CONVERSIONS > AUTO_UNIT_CONVERSIONS (from file units attr) > fallback.
    """
    if varname in UNIT_CONVERSIONS:
        return UNIT_CONVERSIONS[varname]
    for iop in IOPS:
        for rt in RUN_TYPES:
            for lv in LEVELS:
                fpath = find_h0i_file(iop, rt, lv)
                if fpath is None:
                    continue
                ds = xr.open_dataset(fpath, decode_times=False)
                if varname in ds:
                    raw_units = ds[varname].attrs.get("units", "")
                    ds.close()
                    return AUTO_UNIT_CONVERSIONS.get(raw_units,
                                                     (1.0, raw_units or "model units"))
                ds.close()
    return (1.0, "model units")


def find_h0i_file(iop, run_type, level):
    case_dir = os.path.join(
        BASE_DIR, f"FSCAM.T42_T42.{iop}.{run_type}.{level}", "run"
    )
    matches = glob.glob(os.path.join(case_dir, "*.cam.h0i.*.nc"))
    return matches[0] if matches else None


def find_pressure_level_idx(fpath, plev):
    """Return the 0-based model level index nearest to plev (hPa)."""
    ds = xr.open_dataset(fpath, decode_times=False)
    ps_mean  = float(ds["PS"].mean().values) if "PS" in ds else 101325.0
    p0       = float(ds["P0"].values)        if "P0" in ds else 100000.0
    pres_hpa = (ds["hyam"].values * p0 + ds["hybm"].values * ps_mean) / 100.0
    ds.close()
    return int(np.argmin(np.abs(pres_hpa - plev)))


def compute_stats(fpath, varname, plev=None, lev_idx=None):
    """Return dict of {mean, std, min, max} for varname (all NaN if variable missing).

    lev_idx : 0-based level index (takes priority over plev if both given)
    plev    : target pressure in hPa; nearest level is selected
    """
    nan_result = dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan)
    ds = xr.open_dataset(fpath, decode_times=False)
    if varname not in ds:
        ds.close()
        return nan_result
    da = ds[varname].squeeze()
    if "lev" in da.dims:
        if lev_idx is not None:
            da = da.isel(lev=lev_idx)
        elif plev is not None:
            ps_mean  = float(ds["PS"].mean().values) if "PS" in ds else 101325.0
            p0       = float(ds["P0"].values)        if "P0" in ds else 100000.0
            pres_hpa = (ds["hyam"].values * p0 + ds["hybm"].values * ps_mean) / 100.0
            idx      = int(np.argmin(np.abs(pres_hpa - plev)))
            da       = da.isel(lev=idx)
        else:
            da = da.mean(dim="lev")
    ds.close()
    return dict(
        mean=float(da.mean().values),
        std =float(da.std().values),
        min =float(da.min().values),
        max =float(da.max().values),
    )


def compute_mean(fpath, varname, plev=None, lev_idx=None):
    return compute_stats(fpath, varname, plev=plev, lev_idx=lev_idx)["mean"]


def get_obs_mean(iop, varname, plev=None, lev_idx=None):
    """Return time-mean observed value in display units, or np.nan if unavailable."""
    obs_varname = OBS_VAR_MAP.get(varname)
    fpath = os.path.join(IOP_DIR, IOP_OBS_FILES.get(iop, ""))
    if obs_varname is None or not os.path.exists(fpath):
        return np.nan
    ds = xr.open_dataset(fpath, decode_times=False)
    if obs_varname not in ds:
        ds.close()
        return np.nan
    da = ds[obs_varname].squeeze()
    obs_units = ds[obs_varname].attrs.get("units", "").lower().strip()
    if "lev" in da.dims:
        if lev_idx is not None:
            da = da.isel(lev=lev_idx)
        elif plev is not None:
            pres_hpa = ds["lev"].values / 100.0   # IOP lev is in Pa
            idx = int(np.argmin(np.abs(pres_hpa - plev)))
            da = da.isel(lev=idx)
        else:
            da = da.mean(dim="lev")
    raw_val = float(da.mean().values)
    ds.close()
    scale, _ = get_display_scale(varname)
    # Some IOP files store Prec in mm/s rather than m/s
    if varname in ("PRECT", "PRECC", "PRECL") and "mm" in obs_units:
        return raw_val * 86_400        # mm/s → mm/day
    return raw_val * scale


# ── main plot ─────────────────────────────────────────────────────────────────

STAT_LABELS = {
    "mean": "time-mean",
    "min":  "minimum",
    "max":  "maximum",
    "std":  "std dev",
}


def _auto_outfile(varname, plev, lev_idx, stat):
    """Build an output filename from the plot parameters."""
    parts = [varname]
    if lev_idx is not None:
        parts.append("levSFC" if lev_idx == -1 else f"lev{lev_idx}")
    elif plev is not None:
        parts.append(f"{int(plev)}hPa")
    parts.append(stat)
    return "_".join(parts) + ".png"


def make_plot(varname, plev=None, lev_idx=None, outfile=None, obs=False,
              stat="mean"):
    """
    varname : CAM variable name (e.g. "PRECT", "LHFLX", "T")
    plev    : pressure level in hPa — selects nearest model level; level index printed
    lev_idx : 0-based model level index (takes priority over plev if both given)
    outfile : output path (None = auto-generate as <VAR>[_lev][_hPa]_<stat>.png)
    obs     : if True, overplot observed IOP mean as a black star where available
    stat    : which statistic to plot — "mean" | "min" | "max" | "std"
    """
    if stat not in STAT_LABELS:
        raise ValueError(f"stat must be one of {list(STAT_LABELS)}; got {stat!r}")
    scale, units = get_display_scale(varname)
    outfile = outfile or _auto_outfile(varname, plev, lev_idx, stat)

    # ── determine display string for level selection ──────────────────────────
    if lev_idx == -1:
        lev_str = " @ surface level"
    elif lev_idx is not None:
        lev_str = f" @ lev {lev_idx}"
    elif plev is not None:
        # Find representative level index (may differ across resolutions)
        ref_idx = None
        for iop in IOPS:
            for rt in RUN_TYPES:
                for lv in LEVELS:
                    fpath = find_h0i_file(iop, rt, lv)
                    if fpath:
                        ref_idx = find_pressure_level_idx(fpath, plev)
                        print(f"  {lv}: nearest level to {plev} hPa → lev {ref_idx}")
                # only report one run type per resolution
                break
        idx_str = f" (lev {ref_idx})" if ref_idx is not None else ""
        lev_str = f" @ {plev} hPa{idx_str}"
    else:
        lev_str = ""

    fig, ax = plt.subplots(figsize=(12, 5))

    # ── model scatter ─────────────────────────────────────────────────────────
    for i_iop, iop in enumerate(IOPS):
        x_iop = i_iop * IOP_STEP
        for rt, rt_color, rt_offset in zip(RUN_TYPES, RUN_COLORS, RUNTYPE_OFFSETS):
            x_pos = x_iop + rt_offset
            for level in LEVELS:
                fpath = find_h0i_file(iop, rt, level)
                if fpath is None:
                    continue
                st = compute_stats(fpath, varname, plev=plev, lev_idx=lev_idx)
                val = st[stat] * scale
                if np.isnan(val):
                    continue
                ax.scatter(x_pos, val, s=MARKER_SIZE, c=rt_color,
                           marker=LEVEL_MARKERS[level],
                           linewidths=MARKER_EDGE, edgecolors="k", zorder=4)

    # ── observed means (black stars) ──────────────────────────────────────────
    iop_centers = [i * IOP_STEP for i in range(len(IOPS))]
    obs_plotted = False
    if obs:
        for iop, xc in zip(IOPS, iop_centers):
            obs_val = get_obs_mean(iop, varname, plev=plev, lev_idx=lev_idx)
            if not np.isnan(obs_val):
                ax.scatter(xc, obs_val, s=220, c="k", marker="*",
                           linewidths=0.5, edgecolors="k", zorder=5)
                obs_plotted = True

    # ── x-axis ────────────────────────────────────────────────────────────────
    ax.set_xticks(iop_centers)
    ax.set_xticklabels(IOP_LABELS, fontsize=11)
    for ic in iop_centers[1:]:
        ax.axvline(ic - IOP_STEP / 2, color="0.8", lw=0.8, zorder=1)
    ax.set_xlim(iop_centers[0] - IOP_STEP * 0.6, iop_centers[-1] + IOP_STEP * 0.6)

    # ── y-axis ────────────────────────────────────────────────────────────────
    ax.set_ylabel(f"{varname}{lev_str}  [{units}]", fontsize=11)
    ax.set_title(f"SCAM IOP ensemble — {STAT_LABELS[stat]} {varname}{lev_str}", fontsize=12)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5, zorder=0)
    ax.set_axisbelow(True)

    # ── legends (figure-level to avoid axes-bbox clipping) ───────────────────
    color_handles = [
        mlines.Line2D([], [], color=c, marker="o", markersize=8,
                      markeredgecolor="k", markeredgewidth=0.6,
                      linestyle="none", label=lbl)
        for c, lbl in zip(RUN_COLORS, RUN_LABELS)
    ]
    if obs and obs_plotted:
        color_handles.append(
            mlines.Line2D([], [], color="k", marker="*", markersize=11,
                          linestyle="none", label="Observed")
        )
    marker_handles = [
        mlines.Line2D([], [], color="0.4", marker=LEVEL_MARKERS[lv], markersize=8,
                      markeredgecolor="k", markeredgewidth=0.6,
                      linestyle="none", label=lv)
        for lv in LEVELS
    ]

    fig.tight_layout(rect=[0, 0, 0.74, 1])
    fig.canvas.draw()
    rx = ax.get_position().x1 + 0.01

    fig.legend(handles=color_handles, title="Run type",
               bbox_to_anchor=(rx, 0.98), loc="upper left",
               fontsize=9, title_fontsize=9, framealpha=0.9, borderaxespad=0)
    fig.legend(handles=marker_handles, title="Vert. resolution",
               bbox_to_anchor=(rx, 0.38), loc="upper left",
               fontsize=9, title_fontsize=9, framealpha=0.9, borderaxespad=0)

    fig.savefig(outfile, dpi=150, bbox_inches="tight")
    print(f"Saved → {outfile}")
    plt.close(fig)


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-v", "--var",    default="PRECT",
                   help="CAM variable name (default: PRECT)")
    p.add_argument("-p", "--plev",   default=None, type=float,
                   help="Pressure level in hPa — selects nearest model level (prints level index)")
    p.add_argument("-l", "--lev-idx", default=None, type=int,
                   help="0-based model level index (overrides --plev)")
    p.add_argument("-o", "--outfile", default=None,
                   help="Output filename (default: scam_scatter_<VAR>.png)")
    p.add_argument("-s", "--stat", default="mean",
                   choices=["mean", "min", "max", "std"],
                   help="Statistic to plot: mean | min | max | std  (default: mean)")
    p.add_argument("--obs", action="store_true",
                   help="Overplot observed IOP mean as black star")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    varname = args.var.upper()
    make_plot(varname, plev=args.plev, lev_idx=args.lev_idx,
              outfile=args.outfile, obs=args.obs, stat=args.stat)
