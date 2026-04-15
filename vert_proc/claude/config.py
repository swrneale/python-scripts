"""
config.py  –  All run configuration for the vertical-processes analysis.

Edit this file to change cases, variables, regions, and analysis settings.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd


# ── Output directory ──────────────────────────────────────────────────────────
DIR_FIGURES = Path('/glade/u/home/rneale/python/python-figs/vert_proc/')


# ── Analysis parameters ───────────────────────────────────────────────────────
@dataclass
class AnalysisConfig:
    """Top-level settings for a vproc analysis run."""
    nino_region:   str            = 'nino34'
    ssta_thresh:   float          = 0.5          # K threshold for El Niño / La Niña
    seas_mons:     tuple[str,...] = ('Jan','Feb','Dec')
    years_data:    tuple[int,int] = (1979, 2005)
    lats_in:       tuple[float,float] = (-45., 45.)
    lon_w:         float          = 0.
    lon_e:         float          = 360.
    pres_min:      float          = 50.          # hPa
    pres_max:      float          = 1050.        # hPa
    pres_step:     float          = 50.          # hPa

    # Which optional plots to produce
    plot_sst_nino:    bool = True
    plot_div_level:   bool = True
    plot_2var_scatter:bool = True
    plot_vprofiles:   bool = True

    # Primary variable and optional second variable for scatter plot
    var_plot:      str = 'DIV'
    var_plot_scat: str = 'OMEGA'

    @property
    def p_levs(self) -> np.ndarray:
        return np.arange(self.pres_min, self.pres_max, self.pres_step)


# ── Case configuration ────────────────────────────────────────────────────────
@dataclass
class CaseConfig:
    """Describes one model or reanalysis case."""
    name:          str          # Short label, e.g. 'ERA5', 'CE1.E1'
    case_type:     str          # One of CASE_TYPES
    run_name:      str          # Full run identifier used to find files on disk
    use_climo:     bool = False # True  → load pre-computed climo/nino/nina files
                                # False → derive composites from the full time-series


CASE_TYPES = {
    'reanal',
    'lens1',
    'lens2',
    'lense2',
    'c6_amip',
    'cam6_revert',
    'cesm3_dev',
}


# ── Variable metadata ─────────────────────────────────────────────────────────
def get_var_meta() -> pd.DataFrame:
    """Return a DataFrame of variable plotting metadata.

    Columns: long_name, vscale, ovscale, xmin, xmax, axmin, axmax, vunits
    vscale  – multiply raw model output by this to get plotted units
    ovscale – scale factor for obs/climo files (different format)
    xmin/xmax  – x-axis limits for climatology panel
    axmin/axmax – x-axis limits for anomaly panels
    """
    meta = {
        'DTCOND':       ['dT/dt Total',          86400.,       1.,   -5.,  5., -2.,  2., 'K/day'],
        'DCQ':          ['dq/dt Total',           86400.*1000., 1.,   -2.,  2., -2.,  2., 'g/kg/day'],
        'ZMDT':         ['dT/dt Convection',      86400.,       1.,   -5.,  5., -2.,  2., 'K/day'],
        'ZMDQ':         ['dq/dt Convection',      86400.*1000., 1.,   -4.,  4., -4.,  4., 'g/kg/day'],
        'MPDT':         ['dT/dt Microphysics',    86400./1004., 1.,   -5.,  5., -2.,  2., 'K/day'],
        'STEND_CLUBB':  ['dT/dt turbulence',      86400./1004., 1.,   -2.,  8., -2.,  8., 'K/day'],
        'OMEGA':        ['Vertical Velocity',     1.,           -1.,  -0.06, 0.06, -0.06, 0.06, 'pa/s'],
        'DIV':          ['Divergence',            1.,           100./86400., -4e-4, 4e-4, -4e-4, 4e-4, 's^-1'],
        'T':            ['Temperature',           1.,           1.,  -10., 10., -10., 10., 'K'],
        'Q':            ['Specific Humidity',     1000.,        1000., 0., 20.,  -2.,  2., 'g/kg'],
        'U':            ['Zonal Wind',            1.,           1.,  -60., 60., -10., 10., 'm/s'],
    }
    cols = ['long_name', 'vscale', 'ovscale', 'xmin', 'xmax', 'axmin', 'axmax', 'vunits']
    return pd.DataFrame.from_dict(meta, orient='index', columns=cols)


# ── Region definitions ────────────────────────────────────────────────────────
def get_regions() -> pd.DataFrame:
    """Return a DataFrame of named analysis regions.

    Columns: long_name, lat_s, lat_n, lon_w, lon_e
    Regions follow Anna Kochanov's ENSO forcing/response locations.
    """
    regs = {
        # 1. Positive precip. anomalies – equatorial central Pacific (main tropical forcing)
        'Nino Wet': ['C. Pacific Nino Wet',  -10.,  0., 160., 220.],
        # 2. Divergence anomalies – subtropical North Pacific (RWS generation region)
        'Conv U':   ['Convergence Min',       25.,  40., 150., 200.],
        # 3. Negative precip. anomalies – western Pacific (additional RWS contribution)
        'WP Dry':   ['West Pac. Nino Dry.',    0.,  15., 110., 150.],
    }
    cols = ['long_name', 'lat_s', 'lat_n', 'lon_w', 'lon_e']
    return pd.DataFrame.from_dict(regs, orient='index', columns=cols)


# ── Case run-name lookup ──────────────────────────────────────────────────────
def build_run_name_table() -> pd.DataFrame:
    """Map short case names → full run identifiers.

    Returns a DataFrame with index = short name, column 'run name'.
    """
    rl: dict[str, list[str]] = {}

    # Reanalyses
    for name in ('ERA5', 'ERAI', 'JRA25', 'CFSR', 'MERRA2'):
        rl[name] = [name]

    # CAM releases
    rl['C4']  = ['f40.1979_amip.track1.1deg.001']
    rl['C5']  = ['30L_cam5301_FAMIP.001']
    rl['C6']  = ['f.e20.FHIST.f09_f09.cesm2_1.001']
    rl['CC4'] = ['b40.20th.track1.1deg.012']
    rl['CE1'] = ['b.e11.B20TRC5CNBDRD.f09_g16.001']
    rl['CE2'] = ['b.e21.BHIST.f09_g17.CMIP6-historical.001']

    # CAM6 revert experiments
    revert_pairs = [
        ('rC5now',   'f.e20.FHIST.f09_f09.cesm2_1_cam5.001'),
        ('rC5',      'f.e20.FHIST.f09_f09.cesm2_1_true-cam5.001'),
        ('rC5t',     'f.e20.FHIST.f09_f09.cesm2_1_true-cam5_param_topo.001'),
        ('rUWold',   'f.e20.FHIST.f09_f09.cesm2_1_uw.001'),
        ('rGW',      'f.e20.FHIST.f09_f09.cesm2_1_iogw.001'),
        ('rZMc',     'f.e20.FHIST.f09_f09.cesm2_1_capeten.001'),
        ('rMG1',     'f.e20.FHIST.f09_f09.cesm2_1_mg1.002'),
        ('rSB',      'f.e20.FHIST.f09_f09.cesm2_1_sb.002'),
        ('rTMS',     'f.e20.FHIST.f09_f09.cesm2_1_tms.001'),
        ('rCE2i',    'f.e20.FHIST.f09_f09.cesm2_1_revert125.001'),
        ('rC5p',     'f.e20.FHIST.f09_f09.cesm2_1_revertcam5param.001'),
        ('rC5pm',    'f.e20.FHIST.f09_f09.cesm2_1_revertcam5param.002'),
        ('rZMp',     'f.e20.FHIST.f09_f09.cesm2_1_cam5_zmconv.001'),
        ('rM3',      'f.e20.FHIST.f09_f09.cesm2_1_mam3.001'),
        ('rUW',      'f.e20.FHIST.f09_f09.cesm2_1_uw.002'),
        ('rUWp',     'f.e20.FHIST.f09_f09.cesm2_1_uw.003'),
        ('rice',     'f.e20.FHIST.f09_f09.cesm2_1_ice-micro.001'),
        ('rpfrac',   'f.e20.FHIST.f09_f09.cesm2_1_precip_frac_method.001'),
        ('rpremit',  'f.e20.FHIST.f09_f09.cesm2_1_cld_premit.001'),
        ('rnohertz', 'f.e20.FHIST.f09_f09.cesm2_1_hetfrz-off.001'),
        ('rC5psalt', 'f.e20.FHIST.f09_f09.cesm2_1_revertc5seasalt.001'),
        ('rC5pdust', 'f.e20.FHIST.f09_f09.cesm2_1_revertc5dust.001'),
        ('rL30',     'f.e20.FHIST.f09_f09.cesm2_1_L30.001'),
        ('rclm4',    'f.e20.FHIST.f09_f09.cesm2_1_clm4.001'),
        ('CE2sst',   'f.e20.FHIST.f09_f09.cesm2_1_coupled-sst-amip.001'),
        ('CE2sstd',  'f.e20.FHIST.f09_f09.cesm2_1_coupled-sst-amip_daily.001'),
        ('REYsstd',  'f.e20.FHIST.f09_f09.cesm2_1_reynolds_daily_sst.006'),
        ('W110',     'f.e21.FWscHIST_BCG.f09_f09_mg17_110L.001'),
        ('W121',     'f.e21.FWscHIST_BCG.f09_f09_mg17_121L_DZ_400m_80kmTop.001'),
        ('L32',      'f.e21.FWscHIST.ne30_L32_cam6_3_019_plus_CESM2.2.001.hf'),
        ('L48',      'f.e21.FWscHIST.ne30_L48_cam6_3_019_plus_CESM2.2.001.hf'),
        ('L58',      'f.e21.FWscHIST.ne30_L48_BL10_cam6_3_019_plus_CESM2.2.001.hf'),
    ]
    for short, long in revert_pairs:
        rl[short] = [long]

    # CESM3 dev b-cases
    for tag in ('54', '64', '78b', '82b', '83b', '90b', '92', '98'):
        prefix = 'b.e23_alpha16b' if tag == '54' else \
                 'b.e23_alpha16g' if tag in ('64','78b','82b','83b','90b') else \
                 'b.e23_alpha17f'
        rl[tag] = [f'{prefix}.BLT1850.ne30_t232.{tag}']

    # CESM1 LENS (30 members)
    for i in range(1, 31):
        rl[f'CE1.E{i}'] = [f'b.e11.B20TRC5CNBDRD.f09_g16.{i:03d}']

    # CESM2 LENS (40 members across 4 macro years)
    macro_yrs = [1231, 1251, 1281, 1301]
    idx = 1
    for y in macro_yrs:
        for n in range(1, 11):
            rl[f'CE2.E{idx}'] = [f'b.e21.BHISTcmip6.f09_g17.LE2-{y:03d}.{n:03d}']
            idx += 1

    # CESM2 AMIP ensemble
    for i in range(1, 11):
        rl[f'C6.E{i}'] = [f'r{i}i1p1f1']

    # E3SMv2 LENS
    ensemble_nums = np.arange(101, 502, 10)
    for j, en in enumerate(ensemble_nums, start=1):
        rl[f'E3SM2.E{j}'] = [f'v2.FV1.historical_{en:04d}']

    return pd.DataFrame.from_dict(rl, orient='index', columns=['run name'])


def get_run_name(case_name: str) -> str:
    """Return the full run identifier for a short case name."""
    table = build_run_name_table()
    return table.loc[case_name, 'run name']


# ── Convenience: build ensemble case lists ───────────────────────────────────

# Maximum available members per ensemble type
_LENS_MAX = {'lens1': 30, 'lens2': 40, 'lense2': 41}
_LENS_PREFIX = {'lens1': 'CE1', 'lens2': 'CE2', 'lense2': 'E3SM2'}


def lens_cases(lens_type: str, n_members: int) -> list[CaseConfig]:
    """Return CaseConfig objects for the first n_members of a large ensemble.

    Parameters
    ----------
    lens_type : 'lens1' (CESM1 LE), 'lens2' (CESM2 LE), or 'lense2' (E3SMv2 LE)
    n_members : how many members to include (counted from E1)

    Example
    -------
    >>> cases = [CaseConfig('ERA5', 'reanal', 'ERA5', use_climo=True),
    ...          *lens_cases('lens1', 10),
    ...          *lens_cases('lens2', 5)]
    """
    if lens_type not in _LENS_MAX:
        raise ValueError(f'lens_type must be one of {list(_LENS_MAX)}. Got {lens_type!r}')
    max_m = _LENS_MAX[lens_type]
    if n_members > max_m:
        raise ValueError(f'{lens_type} only has {max_m} members; requested {n_members}')

    prefix = _LENS_PREFIX[lens_type]
    table  = build_run_name_table()
    return [
        CaseConfig(f'{prefix}.E{i}', lens_type, table.loc[f'{prefix}.E{i}', 'run name'])
        for i in range(1, n_members + 1)
    ]


# ── Convenience: build a default case list ───────────────────────────────────
def default_cases() -> list[CaseConfig]:
    """Return the default list of cases for a quick analysis run.

    Modify this function (or build your own list) to change which cases run.
    """
    return [
        CaseConfig('ERA5',   'reanal', 'ERA5',  use_climo=True),
    ]
