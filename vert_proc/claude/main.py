"""
main.py  –  Orchestration script for the vertical-processes ENSO analysis.

Edit config.py to change cases, variables, regions, and analysis parameters.
Run with:  python main.py
"""

from __future__ import annotations

import importlib

import xarray as xr

import config
import data_io
import enso
import regions
import plots

# Reload modules so changes take effect without restarting the kernel
for _mod in (config, data_io, enso, regions, plots):
    importlib.reload(_mod)

from config import AnalysisConfig, default_cases, get_var_meta, get_regions
from data_io import load_var, load_sst, load_climo
from enso import compute_enso_index, derive_composites, derive_composites_climo
from regions import build_regional_profiles
from plots import (plot_sst_index, plot_div_pressure_level,
                   plot_scatter_2vars, plot_region_boxes,
                   plot_vertical_profiles)


def run(cfg: AnalysisConfig | None = None,
        cases: list | None = None) -> None:
    if cfg is None:
        cfg = AnalysisConfig()
    if cases is None:
        cases = default_cases()
    var_meta = get_var_meta()
    reg_df   = get_regions()
    p_levs   = cfg.p_levs

    # Show region map once
    plot_region_boxes(reg_df, cfg)

    # Accumulate per-case regional profiles for the final multi-panel plot
    all_case_profiles: dict[str, xr.DataArray] = {}

    for case in cases:
        print()
        print('=' * 60)
        print(f'Case: {case.name}  [{case.case_type}]  climo={case.use_climo}')
        print('=' * 60)

        # ── Load primary variable ─────────────────────────────────────────
        if case.use_climo:
            ds_var, vname = load_climo(case, cfg.var_plot, cfg)
            climo, nino_anom, nina_anom = derive_composites_climo(
                cfg.var_plot, ds_var, vname, var_meta, p_levs)   # pass full var_meta
            ps_comp = None
            ds_ptr  = ds_var    # needed for hybrid→pressure (not used for climo)

        else:
            ds_var, vname = load_var(case, cfg.var_plot, cfg)

            # ── SST for ENSO index ────────────────────────────────────────
            if case.case_type in ('cam6_revert', 'cesm3_dev'):
                sst_ds, sst_vname = ds_var, 'TS'
            else:
                sst_ds, sst_vname = load_sst(case, cfg)

            inino, inina = compute_enso_index(
                sst_ds[sst_vname], cfg.nino_region, cfg.ssta_thresh)

            if cfg.plot_sst_nino:
                plot_sst_index(case.name, sst_ds[sst_vname],
                               cfg.nino_region, inino, inina, cfg)

            # ── PS for hybrid→pressure interpolation ──────────────────────
            ps_ds = None
            if case.case_type not in ('cam6_revert', 'cesm3_dev'):
                ps_ds, _ = load_var(case, 'PS', cfg)
            else:
                ps_ds = ds_var    # PS is in the same h0 files

            vm_row = var_meta.loc[cfg.var_plot]
            climo, nino_anom, nina_anom, ps_comp = derive_composites(
                case, cfg.var_plot, ds_var, vname, cfg, vm_row,
                inino, inina, ps_ds=ps_ds)

            ds_ptr = ds_var

        composites = (climo, nino_anom, nina_anom)

        # ── Optional: max/min pressure-level map ──────────────────────────
        if cfg.plot_div_level:
            plot_div_pressure_level(case, cfg.var_plot, composites, ps_comp, ds_ptr, cfg)

        # ── Optional: 2-variable scatter plot ─────────────────────────────
        if cfg.plot_2var_scatter:
            if case.use_climo:
                ds_var2, vname2 = load_climo(case, cfg.var_plot_scat, cfg)
                comp2 = derive_composites_climo(
                    cfg.var_plot_scat, ds_var2, vname2,
                    var_meta, p_levs)   # pass full var_meta
            else:
                ds_var2, vname2 = load_var(case, cfg.var_plot_scat, cfg)
                vm2_row = var_meta.loc[cfg.var_plot_scat]
                _, nino2, nina2, _ = derive_composites(
                    case, cfg.var_plot_scat, ds_var2, vname2, cfg, vm2_row,
                    inino, inina, ps_ds=ps_ds)
                climo2, _ = derive_composites(
                    case, cfg.var_plot_scat, ds_var2, vname2, cfg, vm2_row,
                    inino, inina)[:2]
                comp2 = (climo2, nino2, nina2)

            plot_scatter_2vars(case, cfg.var_plot, cfg.var_plot_scat,
                               composites, comp2, ps_comp, reg_df, ds_ptr,
                               var_meta, cfg)

        # ── Regional vertical profiles ────────────────────────────────────
        all_case_profiles[case.name] = build_regional_profiles(composites, reg_df)

    # ── Final multi-panel profile figure ─────────────────────────────────────
    if cfg.plot_vprofiles:
        pref_out = 'vproc_analysis'
        plot_vertical_profiles(
            all_case_profiles, p_levs, cfg.var_plot, reg_df,
            var_meta, cases, cfg.years_data, pref_out, cfg)

    print()
    print('-- Done --')


if __name__ == '__main__':
    run()
