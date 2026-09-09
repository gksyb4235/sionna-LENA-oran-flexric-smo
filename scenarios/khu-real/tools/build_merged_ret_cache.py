#!/usr/bin/env python3
"""Assemble a per-cell-independent RET cache by copying gNB groups out of the
existing single-tilt (all-3-cells-same-tilt) caches.

Each gNB's ray-traced channel data in ret_caches/tilt_{T}deg/*.h5 was computed
with only that one gNB present in the Sionna scene (see build_sionna_rt_cache.py,
which adds one "tx" transmitter, solves, then removes it before the next gNB).
So a gNB's cached data depends only on its own tilt, never on the other two
cells' tilts, and groups can be freely recombined across different tilt_*deg
source files -- no new ray tracing needed.

Band layout in this scenario: the 3.5GHz cache holds only gNB_5G, and the
1.8GHz cache holds only gNB_4G_1 + gNB_4G_2. So:
  - gNB_5G's tilt never requires a merge -- just point --sionnaCacheFile35 at
    the existing ret_caches/tilt_{T}deg/sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5.
  - gNB_4G_1/gNB_4G_2 only need a merge when their tilts *differ*; if equal,
    the existing tilt_{T}deg/sionna_rt_cache_1p8ghz_2x2_1x1_lzf.h5 already has
    both groups at that tilt.

Usage:
    build_merged_ret_cache.py --tilt-4g1 3 --tilt-4g2 9 --out PATH
"""
import argparse
import os
import sys

import h5py

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
RET_CACHE_ROOT = os.path.join(REPO_ROOT, "scenarios", "khu-real", "ret_caches")
VALID_TILTS = (3, 5, 7, 9, 12)
BAND_1P8_FILE = "sionna_rt_cache_1p8ghz_2x2_1x1_lzf.h5"
BAND_3P5_FILE = "sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5"


def source_path(tilt, band_file):
    return os.path.join(RET_CACHE_ROOT, f"tilt_{tilt}deg", band_file)


def resolve_5g_cache(tilt_5g):
    """gNB_5G never needs merging -- the existing per-tilt 3.5GHz file already
    contains exactly (and only) its group."""
    if tilt_5g not in VALID_TILTS:
        raise ValueError(f"tilt_5g must be one of {VALID_TILTS}, got {tilt_5g}")
    path = source_path(tilt_5g, BAND_3P5_FILE)
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    return path


def build_or_reuse_1p8_cache(tilt_4g1, tilt_4g2, merged_dir):
    """Return a path to a 1.8GHz cache file containing gNB_4G_1 at tilt_4g1
    and gNB_4G_2 at tilt_4g2. Reuses the existing same-tilt file when the two
    match; otherwise builds (or reuses a previously-built) merged file."""
    for t in (tilt_4g1, tilt_4g2):
        if t not in VALID_TILTS:
            raise ValueError(f"tilt must be one of {VALID_TILTS}, got {t}")

    if tilt_4g1 == tilt_4g2:
        path = source_path(tilt_4g1, BAND_1P8_FILE)
        if not os.path.isfile(path):
            raise FileNotFoundError(path)
        return path

    os.makedirs(merged_dir, exist_ok=True)
    out_path = os.path.join(merged_dir, f"merged_1p8ghz_4g1_{tilt_4g1}deg_4g2_{tilt_4g2}deg.h5")
    done_marker = out_path + ".done"
    if os.path.isfile(done_marker) and os.path.isfile(out_path):
        return out_path

    # Build under a temp name then atomically rename, so concurrent sweep
    # workers racing to build the same pair never see a half-written file.
    tmp_path = out_path + f".tmp{os.getpid()}"
    src_4g1 = source_path(tilt_4g1, BAND_1P8_FILE)
    src_4g2 = source_path(tilt_4g2, BAND_1P8_FILE)
    if not os.path.isfile(src_4g1):
        raise FileNotFoundError(src_4g1)
    if not os.path.isfile(src_4g2):
        raise FileNotFoundError(src_4g2)

    with h5py.File(tmp_path, "w") as out_f:
        with h5py.File(src_4g1, "r") as f1:
            # File-level attrs (frequency_hz etc.) are identical across all
            # tilt_*deg builds for this band; ValidateSionnaCacheTopology()
            # in the scenario requires frequency_hz at the file level.
            for key, value in f1.attrs.items():
                out_f.attrs[key] = value
            out_f.copy(f1["gNB_4G_1"], "gNB_4G_1")
        with h5py.File(src_4g2, "r") as f2:
            out_f.copy(f2["gNB_4G_2"], "gNB_4G_2")

    os.replace(tmp_path, out_path)
    with open(done_marker, "w") as marker:
        marker.write("ok\n")
    return out_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tilt-5g", type=int, required=True, choices=VALID_TILTS)
    parser.add_argument("--tilt-4g1", type=int, required=True, choices=VALID_TILTS)
    parser.add_argument("--tilt-4g2", type=int, required=True, choices=VALID_TILTS)
    parser.add_argument(
        "--merged-dir",
        default=os.path.join(RET_CACHE_ROOT, "merged_percell"),
        help="where to store newly-assembled 1.8GHz merges",
    )
    args = parser.parse_args()

    path_35 = resolve_5g_cache(args.tilt_5g)
    path_18 = build_or_reuse_1p8_cache(args.tilt_4g1, args.tilt_4g2, args.merged_dir)

    # Emit shell-sourceable output for the caller.
    print(f"SIONNA_CACHE_18={path_18}")
    print(f"SIONNA_CACHE_35={path_35}")


if __name__ == "__main__":
    main()
