#!/usr/bin/env python3
"""
Interpolate CAM spectral-element (SE) grid output to a regular lat/lon grid.

The variable list and dimension ordering from the source file are preserved.
The ncol dimension is replaced by (lat, lon) at the same position in each variable.

Usage
-----
  # Single file, linear interpolation at 1-degree (default)
  se_to_latlon.py input.nc --outdir /path/to/output

  # All h0a files, area-weighted at 0.5-degree
  se_to_latlon.py /path/se_grid/*.nc --outdir /path/hist \\
      --method area_weighted --nlat 360 --nlon 720

  # FV2 resolution (1.9x2.5-degree)
  se_to_latlon.py input.nc --nlat 96 --nlon 144 --suffix _fv2

Interpolation methods
---------------------
  linear        Barycentric interpolation on 2-D Delaunay triangulation in
                (lat, lon) space with boundary point augmentation to handle
                the 0/360 longitude seam.  Default.
  area_weighted Area-weighted average of k nearest source points, using the
                areawt field from the source file.
  nearest       Nearest-neighbour (fastest; useful for categorical fields).

Requirements: numpy, scipy, netCDF4
"""

import argparse
import os
import sys
import glob as glob_module
import time
import numpy as np
import netCDF4 as nc
from scipy.spatial import Delaunay, cKDTree
import scipy.sparse as sparse


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Interpolate CAM SE grid NetCDF files to regular lat/lon",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("input", nargs="+",
                   help="Input SE-grid NetCDF file(s) (shell globs are fine)")
    p.add_argument("--outdir", default=None,
                   help="Output directory (default: parent dir of each input file)")
    p.add_argument("--nlat", type=int, default=180,
                   help="Output latitude count (default: 180 → 1-degree)")
    p.add_argument("--nlon", type=int, default=360,
                   help="Output longitude count (default: 360 → 1-degree)")
    p.add_argument("--method", choices=["linear", "area_weighted", "nearest"],
                   default="linear",
                   help="Interpolation method (default: linear)")
    p.add_argument("--k-neighbors", type=int, default=4, dest="k",
                   help="Nearest neighbors for area_weighted/nearest (default: 4)")
    p.add_argument("--fill-threshold", type=float, default=0.5, dest="fill_thresh",
                   help="Min valid-weight fraction to produce output (default: 0.5)")
    p.add_argument("--suffix", default="_latlon",
                   help="Suffix inserted before .nc in output name (default: _latlon)")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite existing output files")
    p.add_argument("--compress", type=int, default=4, choices=range(10), metavar="0-9",
                   help="zlib compression level for data variables (0=off, default: 4)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------

def make_target_grid(nlat, nlon):
    """Cell-centred lat/lon axes."""
    dlat = 180.0 / nlat
    dlon = 360.0 / nlon
    lats = np.linspace(-90.0 + dlat / 2, 90.0 - dlat / 2, nlat)
    lons = np.linspace(dlon / 2, 360.0 - dlon / 2, nlon)
    return lats, lons


def to_xyz(lat_deg, lon_deg):
    """Lat/lon (degrees) → unit-sphere Cartesian, shape (n, 3)."""
    lat = np.radians(np.asarray(lat_deg, dtype=np.float64).ravel())
    lon = np.radians(np.asarray(lon_deg, dtype=np.float64).ravel())
    return np.column_stack([
        np.cos(lat) * np.cos(lon),
        np.cos(lat) * np.sin(lon),
        np.sin(lat),
    ])


# ---------------------------------------------------------------------------
# Regridder
# ---------------------------------------------------------------------------

class Regridder:
    """
    Pre-compute sparse weight matrix W of shape (nlat*nlon, ncol).
    Apply with self.regrid(data) where data.shape[-1] == ncol.

    The linear method uses 2-D Delaunay triangulation in (lat, lon) space
    with source-point augmentation near the 0/360 longitude boundary to
    handle the wrap correctly.  This is much faster than a 3-D triangulation
    while giving exact barycentric interpolation.
    """

    FILL = 1.0e36

    def __init__(self, src_lat, src_lon, src_area, tgt_lat, tgt_lon,
                 method="linear", k=4, fill_thresh=0.5):
        self.nlat = len(tgt_lat)
        self.nlon = len(tgt_lon)
        self.nout = self.nlat * self.nlon
        self.nsrc = len(src_lat)
        self.method = method
        self.fill_thresh = fill_thresh

        # Ensure plain float64 arrays (source vars come as masked arrays)
        src_lat = np.asarray(src_lat, dtype=np.float64)
        src_lon = np.asarray(src_lon, dtype=np.float64)
        src_area = np.asarray(src_area, dtype=np.float64)

        tgt_lat2d, tgt_lon2d = np.meshgrid(tgt_lat, tgt_lon, indexing="ij")
        self._tgt_lat_flat = tgt_lat2d.ravel()
        self._tgt_lon_flat = tgt_lon2d.ravel()

        print(f"  Building {method} weights: {self.nsrc} src → {self.nlat}×{self.nlon} tgt ...",
              end="", flush=True)
        t0 = time.perf_counter()

        if method == "linear":
            self.W = self._linear(src_lat, src_lon)
        elif method == "area_weighted":
            self.W = self._area_weighted(src_lat, src_lon, src_area, k)
        elif method == "nearest":
            self.W = self._nearest(src_lat, src_lon)

        print(f" {time.perf_counter()-t0:.1f}s", flush=True)

    # ------------------------------------------------------------------
    # Weight builders
    # ------------------------------------------------------------------

    def _linear(self, src_lat, src_lon):
        """
        Barycentric weights via 2-D Delaunay in (lat, lon) space.
        Source points near the 0/360 boundary are duplicated at ±360 so
        that triangles can span the seam without gaps.
        """
        dlon = 360.0 / self.nlon
        margin = max(3.0 * dlon, 5.0)   # augment within this many degrees of boundaries

        near_r = src_lon > (360.0 - margin)
        near_l = src_lon < margin

        aug_lat = np.concatenate([src_lat, src_lat[near_r], src_lat[near_l]])
        aug_lon = np.concatenate([src_lon, src_lon[near_r] - 360.0, src_lon[near_l] + 360.0])
        aug_idx = np.concatenate([
            np.arange(self.nsrc),
            np.where(near_r)[0],
            np.where(near_l)[0],
        ])

        src_pts = np.column_stack([aug_lat, aug_lon])
        tgt_pts = np.column_stack([self._tgt_lat_flat, self._tgt_lon_flat])

        tri = Delaunay(src_pts)
        sid = tri.find_simplex(tgt_pts)          # (nout,)

        rows, cols, vals = [], [], []

        # Outside convex hull → nearest-neighbour fallback (usually poles)
        out = sid < 0
        if out.any():
            tree = cKDTree(to_xyz(src_lat, src_lon))
            tgt_out_xyz = to_xyz(self._tgt_lat_flat[out], self._tgt_lon_flat[out])
            _, nn = tree.query(tgt_out_xyz, k=1)
            rows.extend(np.where(out)[0].tolist())
            cols.extend(nn.ravel().tolist())
            vals.extend(np.ones(int(out.sum())).tolist())

        # Barycentric interpolation (vectorised)
        ins = ~out
        if ins.any():
            ins_idx = np.where(ins)[0]
            # Retrieve pre-computed affine transform from the Delaunay object
            # tri.transform[k] = [[T (2×2)], [origin (2,)]] for simplex k
            T_mats = tri.transform[sid[ins], :2]       # (n, 2, 2)
            origins = tri.transform[sid[ins], 2]       # (n, 2)
            b = tgt_pts[ins] - origins                 # (n, 2)
            bary2 = np.einsum("...ij,...j->...i", T_mats, b)   # (n, 2)

            bary3 = np.empty((len(ins_idx), 3), dtype=np.float64)
            bary3[:, :2] = bary2
            bary3[:, 2] = 1.0 - bary2.sum(axis=1)
            bary3 = np.clip(bary3, 0.0, None)
            bary3 /= bary3.sum(axis=1, keepdims=True)

            # Map augmented simplex vertex indices back to original source indices
            simplices_src = aug_idx[tri.simplices[sid[ins]]]    # (n, 3)

            rows.extend(np.repeat(ins_idx, 3).tolist())
            cols.extend(simplices_src.ravel().tolist())
            vals.extend(bary3.ravel().tolist())

        return sparse.csr_matrix((vals, (rows, cols)),
                                 shape=(self.nout, self.nsrc))

    def _area_weighted(self, src_lat, src_lon, src_area, k):
        """k-nearest-neighbour average weighted by source grid-cell area."""
        tree = cKDTree(to_xyz(src_lat, src_lon))
        tgt_xyz = to_xyz(self._tgt_lat_flat, self._tgt_lon_flat)
        _, idx = tree.query(tgt_xyz, k=k)           # (nout, k)
        if k == 1:
            idx = idx[:, np.newaxis]

        w = src_area[idx]                            # (nout, k)
        wsum = w.sum(axis=1, keepdims=True)
        wsum = np.where(wsum == 0, 1.0, wsum)
        w /= wsum

        rows = np.repeat(np.arange(self.nout), k)
        return sparse.csr_matrix((w.ravel(), (rows, idx.ravel())),
                                 shape=(self.nout, self.nsrc))

    def _nearest(self, src_lat, src_lon):
        tree = cKDTree(to_xyz(src_lat, src_lon))
        tgt_xyz = to_xyz(self._tgt_lat_flat, self._tgt_lon_flat)
        _, idx = tree.query(tgt_xyz, k=1)
        return sparse.csr_matrix(
            (np.ones(self.nout), (np.arange(self.nout), idx.ravel())),
            shape=(self.nout, self.nsrc),
        )

    # ------------------------------------------------------------------
    # Apply weights
    # ------------------------------------------------------------------

    def regrid(self, data):
        """
        Parameters
        ----------
        data : ndarray whose last axis has length ncol

        Returns
        -------
        ndarray, last two axes = (nlat, nlon), dtype float32
        """
        shape = data.shape
        assert shape[-1] == self.nsrc, \
            f"Last dim {shape[-1]} != expected ncol {self.nsrc}"

        d = np.asarray(data, dtype=np.float64).reshape(-1, self.nsrc)

        # Separate fill/NaN from valid data
        fill_mask = (np.abs(d) >= 1.0e35) | ~np.isfinite(d)
        d_clean = np.where(fill_mask, 0.0, d)
        valid = (~fill_mask).astype(np.float64)

        # Matrix multiply: CSR × dense is faster than dense × CSC
        # W is (nout, nsrc), result shape is (nflat, nout)
        num = (self.W @ d_clean.T).T  # (nout, nflat).T = (nflat, nout)
        den = (self.W @ valid.T).T

        # Where den is below threshold, mark as fill (NaN here, converted later)
        out = np.where(den >= self.fill_thresh, num / np.where(den == 0, 1.0, den),
                       np.nan)
        return out.reshape(shape[:-1] + (self.nlat, self.nlon)).astype(np.float32)


# ---------------------------------------------------------------------------
# NetCDF helpers
# ---------------------------------------------------------------------------

SKIP_VARS = {"lat", "lon", "areawt", "area"}    # replaced by regular-grid versions


def _copy_ncattrs(src, dst):
    for a in src.ncattrs():
        if not a.startswith("_"):
            dst.setncattr(a, src.getncattr(a))


def _copy_var_meta(src_var, dst_var):
    skip = {"_FillValue"}
    for a in src_var.ncattrs():
        if a not in skip:
            dst_var.setncattr(a, src_var.getncattr(a))


def output_path(input_path, outdir, suffix):
    base = os.path.basename(input_path)
    root, ext = os.path.splitext(base)
    name = root + suffix + ext
    d = outdir if outdir else os.path.dirname(os.path.abspath(input_path))
    return os.path.join(d, name)


# ---------------------------------------------------------------------------
# Per-file processor
# ---------------------------------------------------------------------------

def process_file(input_path, out_path, regridder, tgt_lat, tgt_lon, compress):
    print(f"  Input : {input_path}", flush=True)
    print(f"  Output: {out_path}", flush=True)
    t0 = time.perf_counter()

    with nc.Dataset(input_path, "r") as src:
        fmt = src.file_format
        with nc.Dataset(out_path, "w", format=fmt) as dst:

            # Global attributes
            _copy_ncattrs(src, dst)
            dst.se_regrid_method = regridder.method
            dst.se_regrid_script = "se_to_latlon.py"
            dst.se_regrid_nlat = regridder.nlat
            dst.se_regrid_nlon = regridder.nlon

            # Dimensions: swap ncol → lat + lon
            for dname, dobj in src.dimensions.items():
                if dname == "ncol":
                    dst.createDimension("lat", regridder.nlat)
                    dst.createDimension("lon", regridder.nlon)
                else:
                    dst.createDimension(dname, None if dobj.isunlimited() else len(dobj))

            # Write target coordinate variables
            lv = dst.createVariable("lat", "f8", ("lat",))
            lv.units = "degrees_north"; lv.long_name = "latitude"
            lv[:] = tgt_lat

            lnv = dst.createVariable("lon", "f8", ("lon",))
            lnv.units = "degrees_east"; lnv.long_name = "longitude"
            lnv[:] = tgt_lon

            # Process all variables in source order
            n_interp = n_copy = 0
            for vname in src.variables:
                if vname in SKIP_VARS:
                    continue

                sv = src.variables[vname]
                src_dims = sv.dimensions

                if "ncol" not in src_dims:
                    # No spatial content → copy verbatim
                    dv = dst.createVariable(vname, sv.dtype, src_dims,
                                            zlib=(compress > 0),
                                            complevel=compress)
                    _copy_var_meta(sv, dv)
                    dv[:] = sv[:]
                    n_copy += 1
                else:
                    # Replace ncol with lat, lon (preserves position in dim list)
                    out_dims = tuple(
                        d for orig in src_dims
                        for d in (("lat", "lon") if orig == "ncol" else (orig,))
                    )
                    fv = float(getattr(sv, "_FillValue", 1.0e36))
                    dv = dst.createVariable(vname, "f4", out_dims,
                                            fill_value=fv,
                                            zlib=(compress > 0),
                                            complevel=compress)
                    _copy_var_meta(sv, dv)

                    regridded = regridder.regrid(sv[:])
                    dv[:] = np.where(np.isnan(regridded), fv, regridded)
                    n_interp += 1

    elapsed = time.perf_counter() - t0
    print(f"  Interpolated {n_interp} vars, copied {n_copy} vars  ({elapsed:.1f}s)", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Expand glob patterns
    input_files = []
    for pat in args.input:
        expanded = sorted(glob_module.glob(pat))
        if expanded:
            input_files.extend(expanded)
        elif os.path.exists(pat):
            input_files.append(pat)
        else:
            print(f"Warning: no files matched '{pat}'", file=sys.stderr)

    if not input_files:
        sys.exit("No input files found.")

    if args.outdir:
        os.makedirs(args.outdir, exist_ok=True)

    tgt_lat, tgt_lon = make_target_grid(args.nlat, args.nlon)

    # Build regridder once from the first file's source grid
    print(f"\nReading source grid from: {input_files[0]}")
    with nc.Dataset(input_files[0], "r") as ds:
        src_lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
        src_lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
        src_area = np.asarray(ds.variables["areawt"][:], dtype=np.float64)

    regridder = Regridder(
        src_lat, src_lon, src_area, tgt_lat, tgt_lon,
        method=args.method,
        k=args.k,
        fill_thresh=args.fill_thresh,
    )

    total = len(input_files)
    for i, fpath in enumerate(input_files, 1):
        opath = output_path(fpath, args.outdir, args.suffix)
        print(f"\n[{i}/{total}]", flush=True)

        if os.path.exists(opath) and not args.overwrite:
            print(f"  Skipping (exists): {opath}")
            continue

        try:
            process_file(fpath, opath, regridder, tgt_lat, tgt_lon, args.compress)
        except Exception as exc:
            print(f"  ERROR: {exc}", file=sys.stderr)
            if os.path.exists(opath):
                os.remove(opath)
            raise

    print("\nAll done.", flush=True)


if __name__ == "__main__":
    main()
