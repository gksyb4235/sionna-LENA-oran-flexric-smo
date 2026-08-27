#!/usr/bin/env python3
"""Offline Sionna RT lookup-table cache builder.

Precomputes the ray-traced channel (CIR + per-path angles) for every gNB in
gnbs-ret.csv against every distinct UE position that appears across all
ue_positions_seed*.csv traces, at the ns-3 side's own tr38901/isotropic
antenna configuration and current (fixed, non-swept) bearing/tilt values.
This is meant to let a future ns-3 SionnaLookupChannelModel skip the live
PathSolver call entirely and just look the result up.

Position resolution: x/y rounded to 1m, z rounded to 0.1m. Chosen after
checking (see conversation/scratch analysis, not reproduced here) that
z varies almost entirely as a function of (x, y) terrain height -- 1m (x,y)
cells have a mean z-spread of ~7cm, so z needs its own finer rounding only
to avoid the rare case (steep terrain) where a coarse (x,y) cell straddles
a large elevation change and a naive shared z would clip through the mesh.

Usage:
    python build_sionna_rt_cache.py [--out PATH] [--frequency HZ]
        [--gnb-rows N] [--gnb-cols N] [--ue-rows N] [--ue-cols N]
        [--gnb-prefix PREFIX] [--batch-size N] [--max-depth N]
        [--diffuse-reflection] [--refraction]
        [--zero-path-policy {error,nearest}] [--limit N]

Output: an HDF5 file with one group per gNB (by name from gnbs-ret.csv),
containing datasets:
    positions   (N, 3) float64      -- the actual (rounded) query positions
    a           (N, num_rx_ant, num_tx_ant, num_paths) complex64
    tau         (N, num_rx_ant, num_tx_ant, num_paths) float32
    theta_t, phi_t, theta_r, phi_r  (N, num_paths) float32 -- per-path angles
plus attrs: gnb position, bearing_deg, tilt_deg, frequency, antenna config.
"""
import argparse
import csv
import glob
import math
import os
import sys
import time

import numpy as np

SCENARIOS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(SCENARIOS_DIR))
SCENE_XML = os.path.join(REPO_ROOT, "scenes", "khu-real", "KHU_Cropped_Sionna_RT.xml")
GNB_CSV = os.path.join(SCENARIOS_DIR, "gnbs-ret.csv")
UE_TRACE_GLOB = os.path.join(SCENARIOS_DIR, "ue_positions_seed*.csv")

DEFAULT_CENTRAL_FREQUENCY_HZ = 3.5e9
DEFAULT_GNB_ROWS, DEFAULT_GNB_COLS = 4, 8
DEFAULT_UE_ROWS, DEFAULT_UE_COLS = 2, 4
MAX_DEPTH = 3
MAX_PATHS = 128  # generous cap; observed counts in this scene are ~30-100


def lround(value):
    """Match C++ std::lround(): nearest integer, halfway away from zero."""
    return math.floor(value + 0.5) if value >= 0 else math.ceil(value - 0.5)


def position_key(x, y, z):
    """Return the exact integer key used by SionnaLookupChannelModel."""
    return lround(x), lround(y), lround(z * 10.0)


def load_gnb_positions(path):
    gnbs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = [x.strip() for x in line.split(",")]
            if fields[0] in ("gnb_id", "id"):
                continue
            if len(fields) not in (4, 6):
                raise ValueError(f"bad gNB row: {line!r}")
            name, x, y, z = fields[0], float(fields[1]), float(fields[2]), float(fields[3])
            bearing = float(fields[4]) if len(fields) == 6 else 0.0
            tilt = float(fields[5]) if len(fields) == 6 else 0.0
            gnbs.append({"name": name, "pos": (x, y, z), "bearing_deg": bearing, "tilt_deg": tilt})
    return gnbs


def collect_unique_positions():
    """Round x,y to 1m and z to 0.1m across every seed trace; keep one
    representative exact (x,y,z) per bucket (first one seen)."""
    files = sorted(glob.glob(UE_TRACE_GLOB))
    if not files:
        raise RuntimeError(f"no UE trace files matched {UE_TRACE_GLOB}")
    buckets = {}
    for path in files:
        with open(path) as f:
            lines = f.readlines()
        start = 0
        for i, line in enumerate(lines):
            if line.startswith("time_s"):
                start = i
                break
        reader = csv.reader(lines[start + 1:])
        for row in reader:
            if len(row) < 5:
                continue
            x, y, z = float(row[2]), float(row[3]), float(row[4])
            key = position_key(x, y, z)
            if key not in buckets:
                buckets[key] = (x, y, z)
    return list(buckets.values())


def build_scene(frequency_hz, gnb_rows, gnb_cols, ue_rows, ue_cols):
    import sionna.rt as rt

    scene = rt.load_scene(filename=SCENE_XML, merge_shapes=True)
    scene.tx_array = rt.PlanarArray(
        num_rows=gnb_rows, num_cols=gnb_cols, pattern="tr38901", polarization="V"
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=ue_rows, num_cols=ue_cols, pattern="iso", polarization="V"
    )
    scene.frequency = frequency_hz
    return scene, rt


def run_gnb(scene, rt, solver, gnb, positions, batch_size, h5group, log_prefix,
            frequency_hz, gnb_rows, gnb_cols, ue_rows, ue_cols, max_paths,
            max_depth, diffuse_reflection, refraction, zero_path_policy):
    tx = rt.Transmitter(
        name="tx",
        position=gnb["pos"],
        orientation=(
            float(np.deg2rad(gnb["bearing_deg"])),
            float(np.deg2rad(gnb["tilt_deg"])),
            0.0,
        ),
        velocity=(0.0, 0.0, 0.0),
    )
    scene.add(tx)

    n = len(positions)
    all_pos = np.array(positions, dtype=np.float64)
    a_out, tau_out = None, None
    theta_t_out = theta_r_out = phi_t_out = phi_r_out = num_paths_out = None

    t0 = time.time()
    done = 0
    for start in range(0, n, batch_size):
        chunk = positions[start:start + batch_size]
        rx_names = [f"rx{start + k}" for k in range(len(chunk))]
        for name, pos in zip(rx_names, chunk):
            scene.add(rt.Receiver(name=name, position=tuple(pos),
                                  orientation=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0)))

        paths = solver(
            scene=scene, max_depth=max_depth, los=True, specular_reflection=True,
            diffuse_reflection=diffuse_reflection, diffraction=True, edge_diffraction=True,
            refraction=refraction, synthetic_array=False, seed=49,
        )
        a, tau = paths.cir(normalize_delays=True, out_type="numpy")
        # theta_t/phi_t/theta_r/phi_r come back with the same per-antenna-
        # element dims as a/tau ([num_rx, num_rx_ant, num_tx, num_tx_ant,
        # num_paths]), but a given path's angle of departure/arrival is a
        # geometric property of the path, not the antenna -- far-field, so
        # it's (numerically) identical across every element. Keep just one
        # representative element's copy instead of storing the same numbers
        # num_rx_ant*num_tx_ant times over.
        def path_angle(arr):
            arr = np.asarray(arr)
            return arr[:, 0, 0, 0, :]

        theta_t = path_angle(paths.theta_t)
        phi_t = path_angle(paths.phi_t)
        theta_r = path_angle(paths.theta_r)
        phi_r = path_angle(paths.phi_r)

        def squeeze_tx_and_time(arr):
            # [num_rx, num_rx_ant, num_tx=1, num_tx_ant, num_paths(, num_time_steps=1)]
            # -> [num_rx, num_rx_ant, num_tx_ant, num_paths]
            arr = np.asarray(arr)
            if arr.ndim == 6:
                arr = arr[..., 0]  # drop trailing time-step axis
            arr = np.squeeze(arr, axis=2)  # drop num_tx axis (always 1: one gNB per call)
            return arr

        a = squeeze_tx_and_time(a)
        tau = squeeze_tx_and_time(tau)

        # Sionna pads every receiver in a batch to the batch-wide path
        # dimension. Invalid receiver/path entries have zero coefficients and
        # must not be counted as real paths. Compact each receiver's valid
        # paths independently before writing the fixed-size HDF5 datasets.
        batch_num_paths = a.shape[3]
        num_rx_ant = a.shape[1]
        num_tx_ant = a.shape[2]

        if a_out is None:
            a_out = h5group.create_dataset(
                "a", shape=(n, num_rx_ant, num_tx_ant, max_paths),
                dtype=np.complex64, chunks=True, compression="lzf",
            )
            tau_out = h5group.create_dataset(
                "tau", shape=(n, num_rx_ant, num_tx_ant, max_paths),
                dtype=np.float32, chunks=True, compression="lzf",
            )
            theta_t_out = h5group.create_dataset(
                "theta_t", shape=(n, max_paths), dtype=np.float32,
                chunks=True, compression="lzf",
            )
            phi_t_out = h5group.create_dataset(
                "phi_t", shape=(n, max_paths), dtype=np.float32,
                chunks=True, compression="lzf",
            )
            theta_r_out = h5group.create_dataset(
                "theta_r", shape=(n, max_paths), dtype=np.float32,
                chunks=True, compression="lzf",
            )
            phi_r_out = h5group.create_dataset(
                "phi_r", shape=(n, max_paths), dtype=np.float32,
                chunks=True, compression="lzf",
            )
            num_paths_out = h5group.create_dataset(
                "num_paths", shape=(n,), dtype=np.int32
            )

        if batch_num_paths > max_paths:
            print(f"{log_prefix} WARNING: batch has {batch_num_paths} paths, "
                  f"truncating to max_paths={max_paths}", flush=True)
        compact_a = np.zeros((len(chunk), num_rx_ant, num_tx_ant, max_paths), np.complex64)
        compact_tau = np.zeros((len(chunk), num_rx_ant, num_tx_ant, max_paths), np.float32)
        compact_theta_t = np.zeros((len(chunk), max_paths), np.float32)
        compact_phi_t = np.zeros((len(chunk), max_paths), np.float32)
        compact_theta_r = np.zeros((len(chunk), max_paths), np.float32)
        compact_phi_r = np.zeros((len(chunk), max_paths), np.float32)
        path_counts = np.zeros(len(chunk), np.int32)

        valid_paths = np.any(np.abs(a) > 0.0, axis=(1, 2))
        for receiver in range(len(chunk)):
            indices = np.flatnonzero(valid_paths[receiver])[:max_paths]
            count = len(indices)
            path_counts[receiver] = count
            if count == 0:
                continue
            compact_a[receiver, :, :, :count] = a[receiver, :, :, indices].transpose(1, 2, 0)
            compact_tau[receiver, :, :, :count] = tau[receiver, :, :, indices].transpose(1, 2, 0)
            compact_theta_t[receiver, :count] = theta_t[receiver, indices]
            compact_phi_t[receiver, :count] = phi_t[receiver, indices]
            compact_theta_r[receiver, :count] = theta_r[receiver, indices]
            compact_phi_r[receiver, :count] = phi_r[receiver, indices]

        rows = slice(start, start + len(chunk))
        a_out[rows] = compact_a
        tau_out[rows] = compact_tau
        theta_t_out[rows] = compact_theta_t
        phi_t_out[rows] = compact_phi_t
        theta_r_out[rows] = compact_theta_r
        phi_r_out[rows] = compact_phi_r
        num_paths_out[rows] = path_counts

        for name in rx_names:
            scene.remove(name)

        done += len(chunk)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else float("inf")
        print(f"{log_prefix} {done}/{n}  ({elapsed:.1f}s elapsed, "
              f"{rate:.1f} pos/s, ETA {eta:.0f}s)", flush=True)

    h5group.create_dataset("positions", data=all_pos)

    num_paths = num_paths_out[:]
    zero_rows = np.flatnonzero(num_paths == 0)
    h5group.attrs["zero_path_rows_original"] = len(zero_rows)
    h5group.attrs["zero_path_policy"] = zero_path_policy
    if len(zero_rows) and zero_path_policy == "error":
        raise RuntimeError(
            f"{log_prefix} {len(zero_rows)} receiver positions have no non-zero paths; "
            "increase --max-depth, enable --diffuse-reflection/--refraction, or explicitly "
            "select --zero-path-policy nearest"
        )
    if len(zero_rows) and zero_path_policy == "nearest":
        valid_rows = np.flatnonzero(num_paths > 0)
        if not len(valid_rows):
            raise RuntimeError(f"{log_prefix} every receiver position has zero paths")
        repaired_from = np.full(n, -1, dtype=np.int64)
        for row in zero_rows:
            distances = np.linalg.norm(all_pos[valid_rows] - all_pos[row], axis=1)
            source = int(valid_rows[np.argmin(distances)])
            repaired_from[row] = source
            count = int(num_paths_out[source])
            a_out[row] = a_out[source]
            tau_out[row] = tau_out[source]
            theta_t_out[row] = theta_t_out[source]
            phi_t_out[row] = phi_t_out[source]
            theta_r_out[row] = theta_r_out[source]
            phi_r_out[row] = phi_r_out[source]
            num_paths_out[row] = count
        h5group.create_dataset("repaired_from_index", data=repaired_from)
        print(f"{log_prefix} repaired {len(zero_rows)} zero-path rows from nearest valid rows",
              flush=True)
    h5group.attrs["gnb_position"] = gnb["pos"]
    h5group.attrs["bearing_deg"] = gnb["bearing_deg"]
    h5group.attrs["tilt_deg"] = gnb["tilt_deg"]
    h5group.attrs["frequency_hz"] = frequency_hz
    h5group.attrs["gnb_array"] = f"{gnb_rows}x{gnb_cols} tr38901"
    h5group.attrs["ue_array"] = f"{ue_rows}x{ue_cols} iso"

    scene.remove("tx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(SCENARIOS_DIR, "sionna_rt_cache.h5"))
    parser.add_argument("--frequency", type=float, default=DEFAULT_CENTRAL_FREQUENCY_HZ)
    parser.add_argument("--gnb-rows", type=int, default=DEFAULT_GNB_ROWS)
    parser.add_argument("--gnb-cols", type=int, default=DEFAULT_GNB_COLS)
    parser.add_argument("--ue-rows", type=int, default=DEFAULT_UE_ROWS)
    parser.add_argument("--ue-cols", type=int, default=DEFAULT_UE_COLS)
    parser.add_argument("--gnb-prefix", default="",
                        help="only build gNB IDs beginning with this prefix")
    parser.add_argument("--max-paths", type=int, default=MAX_PATHS)
    parser.add_argument("--max-depth", type=int, default=MAX_DEPTH)
    parser.add_argument("--diffuse-reflection", action="store_true",
                        help="enable diffuse reflections in the Sionna path solver")
    parser.add_argument("--refraction", action="store_true",
                        help="enable refraction in the Sionna path solver")
    parser.add_argument("--zero-path-policy", choices=("error", "nearest"), default="error",
                        help="reject zero-path rows (default) or copy the nearest valid row")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="cap #positions (debug)")
    args = parser.parse_args()

    import h5py

    print("Loading gNB positions...")
    gnbs = load_gnb_positions(GNB_CSV)
    if args.gnb_prefix:
        gnbs = [g for g in gnbs if g["name"].startswith(args.gnb_prefix)]
    if not gnbs:
        raise RuntimeError(f"no gNB IDs matched prefix {args.gnb_prefix!r}")
    for g in gnbs:
        print(f"  {g['name']}: pos={g['pos']} bearing={g['bearing_deg']} tilt={g['tilt_deg']}")

    print("Collecting unique UE positions across all seed traces...")
    positions = collect_unique_positions()
    if args.limit:
        positions = positions[:args.limit]
    print(f"  {len(positions)} unique positions "
          f"(x/y @1m, z @0.1m grid) x {len(gnbs)} gNBs "
          f"= {len(positions) * len(gnbs)} total RT queries")

    if min(args.gnb_rows, args.gnb_cols, args.ue_rows, args.ue_cols,
           args.batch_size, args.max_paths, args.max_depth) <= 0:
        raise ValueError("array dimensions, batch size and max paths must be positive")
    if args.frequency <= 0:
        raise ValueError("frequency must be positive")

    print(f"Configuration: frequency={args.frequency:g} Hz, "
          f"gNB={args.gnb_rows}x{args.gnb_cols}, UE={args.ue_rows}x{args.ue_cols}, "
          f"max_paths={args.max_paths}, max_depth={args.max_depth}, "
          f"diffuse_reflection={args.diffuse_reflection}, refraction={args.refraction}, "
          f"zero_path_policy={args.zero_path_policy}")
    scene, rt = build_scene(args.frequency, args.gnb_rows, args.gnb_cols,
                            args.ue_rows, args.ue_cols)
    solver = rt.PathSolver()

    with h5py.File(args.out, "w") as hf:
        hf.attrs["scene"] = SCENE_XML
        hf.attrs["position_resolution"] = "x,y @1m, z @0.1m"
        hf.attrs["frequency_hz"] = args.frequency
        hf.attrs["gnb_array"] = f"{args.gnb_rows}x{args.gnb_cols} tr38901"
        hf.attrs["ue_array"] = f"{args.ue_rows}x{args.ue_cols} iso"
        hf.attrs["max_paths"] = args.max_paths
        hf.attrs["max_depth"] = args.max_depth
        hf.attrs["diffuse_reflection"] = args.diffuse_reflection
        hf.attrs["refraction"] = args.refraction
        hf.attrs["zero_path_policy"] = args.zero_path_policy
        hf.attrs["gnb_prefix"] = args.gnb_prefix
        for gnb in gnbs:
            print(f"\n=== {gnb['name']} ===")
            group = hf.create_group(gnb["name"])
            run_gnb(scene, rt, solver, gnb, positions, args.batch_size, group,
                    log_prefix=f"[{gnb['name']}]", frequency_hz=args.frequency,
                    gnb_rows=args.gnb_rows, gnb_cols=args.gnb_cols,
                    ue_rows=args.ue_rows, ue_cols=args.ue_cols,
                    max_paths=args.max_paths, max_depth=args.max_depth,
                    diffuse_reflection=args.diffuse_reflection,
                    refraction=args.refraction,
                    zero_path_policy=args.zero_path_policy)

    print(f"\nDone. Cache written to {args.out} "
          f"({os.path.getsize(args.out) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
