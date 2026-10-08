"""Q1/Q2 helpers for SCAM IOP comparisons.

Observed Q1/Q2 are already stored in the SCAM IOP forcing files at
/glade/work/rneale/scam_cases/iops (variables Q1, Q2, units K/s), derived
from the Yanai (1973) budget analyses that produced the forcing.

CAM does not output Q1/Q2 directly, so this module builds them from the
physics tendencies present in the h0i history:

    Q1_model = DTCOND + DTV + QRL + QRS                     [K/s]
    Q2_model = -(Lv / Cp) * (DCQ + VD01)                    [K/s equivalent]

DTCOND already contains ZM + CLUBB + microphysics moist heating, and DCQ
already contains the corresponding moisture tendencies, so summing the
component tendencies (ZMDT + STEND_CLUBB/Cp + MPDT, etc.) would double
count. VD01 carries the vertical-diffusion water-vapor tendency for
tracer 1 (Q); DTV carries the vertical-diffusion T tendency.
"""

import os
import numpy as np
import xarray as xr

LV = 2.501e6      # J/kg, latent heat of vaporization
CP = 1004.64      # J/kg/K, dry-air specific heat at constant pressure
LV_OVER_CP = LV / CP

Q1_COMPONENTS = ("DTCOND", "DTV", "QRL", "QRS")
Q2_Q_COMPONENTS = ("DCQ", "VD01")


def _pres_hpa_from_hybrid(ds):
    """Compute mid-level pressure (hPa) using time-mean PS."""
    ps_mean = float(ds["PS"].mean().values) if "PS" in ds else 101325.0
    p0 = float(ds["P0"].values) if "P0" in ds else 100000.0
    return (ds["hyam"].values * p0 + ds["hybm"].values * ps_mean) / 100.0


def _squeeze_time_lev(da):
    """Drop non (time, lev) singleton dims and force (time, lev) order."""
    da = da.squeeze()
    for d in list(da.dims):
        if d not in ("time", "lev"):
            da = da.isel({d: 0})
    if set(da.dims) == {"time", "lev"}:
        da = da.transpose("time", "lev")
    return da


def compute_model_q1q2(fpath):
    """Return (Q1[time,lev], Q2[time,lev], pres_hpa[lev]) from a CAM h0i file.

    Any missing tendency component is treated as zero and a note is added
    to the returned info dict, so ARM/GATE cases that lack a variable
    still produce a plot.
    """
    ds = xr.open_dataset(fpath, decode_times=False)
    info = {"missing_q1": [], "missing_q2": []}

    def _get(name, missing_list):
        if name not in ds:
            missing_list.append(name)
            return None
        return _squeeze_time_lev(ds[name]).values

    q1_parts = []
    for name in Q1_COMPONENTS:
        arr = _get(name, info["missing_q1"])
        if arr is not None:
            q1_parts.append(arr)
    if not q1_parts:
        ds.close()
        raise RuntimeError(f"No Q1 tendency components found in {fpath}")
    q1 = np.sum(q1_parts, axis=0)   # K/s

    q2_parts = []
    for name in Q2_Q_COMPONENTS:
        arr = _get(name, info["missing_q2"])
        if arr is not None:
            q2_parts.append(arr)
    if not q2_parts:
        ds.close()
        raise RuntimeError(f"No Q2 tendency components found in {fpath}")
    dqdt_phys = np.sum(q2_parts, axis=0)   # kg/kg/s
    q2 = -LV_OVER_CP * dqdt_phys           # K/s equivalent

    pres_hpa = _pres_hpa_from_hybrid(ds)
    ds.close()
    return q1, q2, pres_hpa, info


def load_obs_q1q2(iop_dir, iop_file):
    """Return (Q1[time,lev], Q2[time,lev], pres_hpa[lev]) from an IOP file."""
    fpath = os.path.join(iop_dir, iop_file)
    if not os.path.exists(fpath):
        return None, None, None
    ds = xr.open_dataset(fpath, decode_times=False)
    if "Q1" not in ds or "Q2" not in ds:
        ds.close()
        return None, None, None
    q1 = _squeeze_time_lev(ds["Q1"]).values     # K/s
    q2 = _squeeze_time_lev(ds["Q2"]).values     # K/s
    pres_hpa = ds["lev"].values / 100.0         # IOP lev is in Pa
    ds.close()
    return q1, q2, pres_hpa
