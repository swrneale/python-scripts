"""Generate ZMPBL_simulations.pdf — configuration summary of 8 CESM-CAM cases."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.table import Table
from datetime import date
from pathlib import Path

OUT = Path('/glade/work/rneale/git/python-scripts/papers/ZM_PBL/cam6_4_032/claude/ZMPBL_simulations.pdf')

# ── Page helpers ────────────────────────────────────────────────────────
def new_page(pdf, title=None, subtitle=None):
    fig = plt.figure(figsize=(11, 8.5))   # landscape US letter
    fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.05)
    ax = fig.add_subplot(111)
    ax.axis('off')
    if title:
        ax.text(0.0, 1.02, title, transform=ax.transAxes,
                fontsize=14, fontweight='bold', va='bottom')
    if subtitle:
        ax.text(0.0, 0.985, subtitle, transform=ax.transAxes,
                fontsize=9, style='italic', color='#555', va='top')
    return fig, ax


def draw_table(ax, columns, rows, col_widths=None, y_top=0.94, row_h=0.045,
               header_fc='#e5e5e5', row_alt_fc='#f7f7f7', fontsize=8):
    n_cols = len(columns)
    if col_widths is None:
        col_widths = [1.0 / n_cols] * n_cols
    assert abs(sum(col_widths) - 1.0) < 1e-6, 'col_widths must sum to 1'

    # header
    x_positions = [0.0]
    for w in col_widths:
        x_positions.append(x_positions[-1] + w)

    def cell(x0, y0, w, h, text, **kw):
        fc = kw.pop('facecolor', 'none')
        ec = kw.pop('edgecolor', '#888')
        weight = kw.pop('weight', 'normal')
        ha = kw.pop('ha', 'left')
        family = kw.pop('family', 'sans-serif')
        rect = plt.Rectangle((x0, y0 - h), w, h, transform=ax.transAxes,
                             facecolor=fc, edgecolor=ec, linewidth=0.4, zorder=1)
        ax.add_patch(rect)
        pad = 0.004
        tx = x0 + pad if ha == 'left' else (x0 + w/2 if ha == 'center' else x0 + w - pad)
        ax.text(tx, y0 - h/2, text, transform=ax.transAxes,
                fontsize=fontsize, ha=ha, va='center', weight=weight,
                family=family, zorder=2, wrap=True)

    # header row
    y = y_top
    for i, col in enumerate(columns):
        cell(x_positions[i], y, col_widths[i], row_h,
             col, facecolor=header_fc, weight='bold', ha='center')
    y -= row_h
    for r, row in enumerate(rows):
        fc = row_alt_fc if (r % 2) else 'white'
        for i, val in enumerate(row):
            ha = 'center' if i == 0 else 'left'
            fam = 'monospace' if isinstance(val, str) and ('.' in val or '=' in val or '_' in val) else 'sans-serif'
            cell(x_positions[i], y, col_widths[i], row_h,
                 str(val), facecolor=fc, ha=ha, family=fam)
        y -= row_h


# ── Content ─────────────────────────────────────────────────────────────
today = date.today().isoformat()

pdf = PdfPages(str(OUT))

# ─── Page 1: title + overview ────────────────────────────────────────────
fig, ax = new_page(pdf,
    title='ZM–PBL Convection: CESM-CAM Case Configuration Summary',
    subtitle=f'Generated {today}   |   R. Neale')
overview = (
    "Overview\n"
    "--------\n"
    "This document summarises the eight CESM-CAM simulations used in the ZM–PBL\n"
    "convection study. All are AMIP-style (F-compset) integrations starting\n"
    "1979-01-01 with 30-minute coupling (ATM_NCPL=48).\n\n"
    "The experiment matrix probes three factors:\n"
    "   1. CAM6 vs CAM7 physics (compsets HIST_CAM60 vs HIST_CAM70%LT)\n"
    "   2. Vertical resolution in CAM7 (L32 / L48 / L58)\n"
    "   3. Zhang-McFarlane PBL-parcel source on (000a) vs off (001a)\n"
    "      controlled by namelist  zmconv_parcel_pbl\n\n"
    "Cases 1-7 share a single source tree (CAM_ZMKE, tag cam6_4_032-50,\n"
    "branch cam6_ke_zm); case 8 is the legacy CESM2.2 CAM6/FV baseline.\n\n"
    "SourceMods:  all cases contain only README files in SourceMods/src.cam\n"
    "             and SourceMods/src.clm — no code overrides."
)
ax.text(0.02, 0.90, overview, transform=ax.transAxes,
        fontsize=10, family='monospace', va='top')
pdf.savefig(fig); plt.close(fig)

# ─── Page 2: family / grid / levels / tag ────────────────────────────────
fig, ax = new_page(pdf, title='Table 1.  Case family, grid, vertical resolution, tag')
cols  = ['#', 'Case', 'Compset', 'Grid', 'Levels', 'CAM/CESM tag']
rows = [
    (1, 'f.e30.FLTHIST.CAM7.L32.000a',  'HIST_CAM70%LT', 'ne30pg3',     32, 'cam6_4_032-50-gd4e4b939'),
    (2, 'f.e30.FLTHIST.CAM7.L48.000a',  'HIST_CAM70%LT', 'ne30pg3',     48, 'cam6_4_032-50-gd4e4b939'),
    (3, 'f.e30.FLTHIST.CAM7.L58.000a',  'HIST_CAM70%LT', 'ne30pg3',     58, 'cam6_4_032-50-gd4e4b939'),
    (4, 'f.e30.FLTHIST.CAM7.L32.001a',  'HIST_CAM70%LT', 'ne30pg3',     32, 'cam6_4_032-50-gd4e4b939'),
    (5, 'f.e30.FLTHIST.CAM7.L48.001a',  'HIST_CAM70%LT', 'ne30pg3',     48, 'cam6_4_032-50-gd4e4b939'),
    (6, 'f.e30.FLTHIST.CAM7.L58.001a',  'HIST_CAM70%LT', 'ne30pg3',     58, 'cam6_4_032-50-gd4e4b939'),
    (7, 'f.e30.FHIST.CAM6-phys.L32.000a','HIST_CAM60',    'ne30pg3',     32, 'cam6_4_032-50-gd4e4b939'),
    (8, 'f.e22.FHIST.f09_f09.CAM6.L32.000','HIST_CAM60 (+SIAC)','f09_f09 (FV 0.9x1.25)', 32, 'cam_cesm2_2_rel_09-1-gc47c4d74'),
]
draw_table(ax, cols, rows,
           col_widths=[0.03, 0.29, 0.14, 0.19, 0.06, 0.29],
           y_top=0.93, row_h=0.058, fontsize=8)
ax.text(0.0, 0.02,
        'CAM7 compset resolves to  HIST_CAM70%LT_CLM60%SP_CICE%PRES_DOCN%DOM_MOSART_SGLC_SWAV_SESP;\n'
        'CAM6 compset resolves to  HIST_CAM60_CLM50%SP_CICE%PRES_DOCN%DOM_MOSART_SGLC_SWAV_SESP  (+SIAC for case 8).',
        transform=ax.transAxes, fontsize=7.5, family='monospace', style='italic', color='#333')
pdf.savefig(fig); plt.close(fig)

# ─── Page 3: physics options (differences only) ──────────────────────────
fig, ax = new_page(pdf,
    title='Table 2.  Physics options (differences from compset defaults)',
    subtitle='Non-listed physics options are the compset default and identical across the group.')
cols = ['Cases', 'Deep conv', 'Shallow conv', 'Microphysics', 'ZM PBL parcel']
rows = [
    ('1-3   CAM7 *.000a  (L32/L48/L58)',       'ZM', 'CLUBB_SGS', 'MG3 (+graupel)', 'on (default)'),
    ('4-6   CAM7 *.001a  (L32/L48/L58)',       'ZM', 'CLUBB_SGS', 'MG3 (+graupel)', 'OFF'),
    ('7     CAM6-phys L32 (ne30, e30 tag)',    'ZM', 'CLUBB_SGS', 'MG2 (no graupel)', 'on'),
    ('8     CAM6 f09 (CESM2.2)',               'ZM', 'CLUBB_SGS', 'MG2 (no graupel)', 'on'),
]
draw_table(ax, cols, rows,
           col_widths=[0.42, 0.10, 0.14, 0.20, 0.14],
           y_top=0.93, row_h=0.065, fontsize=9)
pdf.savefig(fig); plt.close(fig)

# ─── Page 4: user_nl_cam entries ─────────────────────────────────────────
fig, ax = new_page(pdf, title='Table 3.  user_nl_cam entries')
cols = ['#', 'Case', 'ncdata (initial conditions)', 'Physics tuning']
rows = [
    (1, 'L32.000a',       'FLT_L32_ne30pg3_IC_c220623.nc', '(none)'),
    (2, 'L48.000a',       'FLT_L48_ne30pg3_IC_c220623.nc', '(none)'),
    (3, 'L58.000a',       '(compset default)',             '(none)'),
    (4, 'L32.001a',       'FLT_L32_ne30pg3_IC_c220623.nc', 'zmconv_parcel_pbl = .false.'),
    (5, 'L48.001a',       'FLT_L48_ne30pg3_IC_c220623.nc', 'zmconv_parcel_pbl = .false.'),
    (6, 'L58.001a',       '(compset default)',             'zmconv_parcel_pbl = .false.'),
    (7, 'CAM6-phys L32',  '(compset default)',             '(none)'),
    (8, 'CAM6 f09',       '(compset default)',             '(none)'),
]
draw_table(ax, cols, rows,
           col_widths=[0.04, 0.16, 0.44, 0.36],
           y_top=0.93, row_h=0.055, fontsize=9)
diag = (
    "Common diagnostic tape settings (all cases):\n"
    "  mfilt  = 0, 10, 60          nhtfrq = 0, -24, -3         ndens = 2, 2, 2\n"
    "  interpolate_output = .true. (cases 1-7 only; case 8 already on FV)\n"
    "  interpolate_nlat, interpolate_nlon = 192, 288\n"
    "\n"
    "fincl history-tape variables include ZM convection diagnostics:\n"
    "  ZMDT, ZMDQ, ZMMU, ZMMD, DTCOND, DCQ, CAPE, LCL/PLCL/TLCL/LEL, TKE, PCONVT, TROP_*, ...\n"
    "Case 8 (CESM2.2) omits LCL / PLCL / TLCL / LEL / BUOY / DMPDZ (variables only in newer CAM7 code)."
)
ax.text(0.0, 0.30, diag, transform=ax.transAxes,
        fontsize=8.5, family='monospace', va='top', color='#222')
pdf.savefig(fig); plt.close(fig)

# ─── Page 5: run length + key takeaway ───────────────────────────────────
fig, ax = new_page(pdf, title='Table 4.  Run length')
cols = ['Case', 'STOP_N (years)', 'ATM_NCPL', 'Start date']
rows = [
    ('L32.000a',        4,  48, '1979-01-01'),
    ('L48.000a',        5,  48, '1979-01-01'),
    ('L58.000a',        2,  48, '1979-01-01'),
    ('L32.001a',        5,  48, '1979-01-01'),
    ('L48.001a',        5,  48, '1979-01-01'),
    ('L58.001a',        3,  48, '1979-01-01'),
    ('CAM6-phys L32',  10,  48, '1979-01-01'),
    ('CAM6 f09',        3,  48, '1979-01-01'),
]
draw_table(ax, cols, rows,
           col_widths=[0.36, 0.22, 0.20, 0.22],
           y_top=0.90, row_h=0.055, fontsize=9)

takeaway = (
    "Key takeaway\n"
    "------------\n"
    "Cases 1-7 share a single source tree (CAM_ZMKE @ cam6_4_032-50, branch\n"
    "cam6_ke_zm).  Differences are limited to:\n"
    "    - compset          (CAM7 physics vs CAM6 physics)\n"
    "    - vertical grid    (L32 / L48 / L58 for CAM7)\n"
    "    - one namelist knob   zmconv_parcel_pbl   (on for 000a, off for 001a)\n\n"
    "This isolates the effect of the ZM deep-convection PBL-parcel source at\n"
    "three vertical resolutions, with CAM6 physics runs (cases 7-8) providing\n"
    "the pre-ZMKE baseline.  Case 8 additionally tests the legacy CESM2.2 /\n"
    "finite-volume dycore combination against modern spectral-element ne30pg3."
)
ax.text(0.0, 0.34, takeaway, transform=ax.transAxes,
        fontsize=10, family='monospace', va='top')
pdf.savefig(fig); plt.close(fig)

pdf.close()
print(f'wrote {OUT}   ({OUT.stat().st_size} bytes)')
