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
    python build_sionna_rt_cache.py [--out PATH] [--batch-size N] [--limit N]

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
import os
import sys
import time

import numpy as np

SCENARIOS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(SCENARIOS_DIR))
SCENE_XML = os.path.join(REPO_ROOT, "scenes", "khu-real", "KHU_Cropped_Sionna_RT.xml")
GNB_CSV = os.path.join(SCENARIOS_DIR, "gnbs-ret.csv")
UE_TRACE_GLOB = os.path.join(SCENARIOS_DIR, "ue_positions_seed*.csv")

CENTRAL_FREQUENCY_HZ = 3.5e9  # matches khu-real-nr-sionna-pooled.cc's centralFrequencyBand1
GNB_ROWS, GNB_COLS = 4, 8
UE_ROWS, UE_COLS = 2, 4
MAX_DEPTH = 3
MAX_PATHS = 128  # generous cap; observed counts in this scene are ~30-100


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
            key = (round(x), round(y), round(z, 1))
            if key not in buckets:
                buckets[key] = (x, y, z)
    return list(buckets.values())


def build_scene():
    import sionna.rt as rt

    scene = rt.load_scene(filename=SCENE_XML, merge_shapes=True)
    scene.tx_array = rt.PlanarArray(
        num_rows=GNB_ROWS, num_cols=GNB_COLS, pattern="tr38901", polarization="V"
    )
    scene.rx_array = rt.PlanarArray(
        num_rows=UE_ROWS, num_cols=UE_COLS, pattern="iso", polarization="V"
    )
    scene.frequency = CENTRAL_FREQUENCY_HZ
    return scene, rt


def run_gnb(scene, rt, solver, gnb, positions, batch_size, h5group, log_prefix):
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
            scene=scene, max_depth=MAX_DEPTH, los=True, specular_reflection=True,
            diffuse_reflection=False, diffraction=True, edge_diffraction=True,
            refraction=False, synthetic_array=False, seed=49,
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

        # Different position batches produce different numbers of resolved
        # paths (Sionna returns exactly as many as that batch needed), so
        # every dataset here is padded/truncated to a fixed MAX_PATHS on
        # write -- the true count is preserved separately (num_paths, below)
        # so the consumer knows which trailing entries are real.
        batch_num_paths = a.shape[3]
        num_rx_ant = a.shape[1]
        num_tx_ant = a.shape[2]

        if a_out is None:
            a_out = h5group.create_dataset(
                "a", shape=(n, num_rx_ant, num_tx_ant, MAX_PATHS),
                dtype=np.complex64, chunks=True, compression="gzip", compression_opts=4,
            )
            tau_out = h5group.create_dataset(
                "tau", shape=(n, num_rx_ant, num_tx_ant, MAX_PATHS),
                dtype=np.float32, chunks=True, compression="gzip", compression_opts=4,
            )
            theta_t_out = h5group.create_dataset(
                "theta_t", shape=(n, MAX_PATHS), dtype=np.float32,
                chunks=True, compression="gzip", compression_opts=4,
            )
            phi_t_out = h5group.create_dataset(
                "phi_t", shape=(n, MAX_PATHS), dtype=np.float32,
                chunks=True, compression="gzip", compression_opts=4,
            )
            theta_r_out = h5group.create_dataset(
                "theta_r", shape=(n, MAX_PATHS), dtype=np.float32,
                chunks=True, compression="gzip", compression_opts=4,
            )
            phi_r_out = h5group.create_dataset(
                "phi_r", shape=(n, MAX_PATHS), dtype=np.float32,
                chunks=True, compression="gzip", compression_opts=4,
            )
            num_paths_out = h5group.create_dataset(
                "num_paths", shape=(n,), dtype=np.int32
            )

        if batch_num_paths > MAX_PATHS:
            print(f"{log_prefix} WARNING: batch has {batch_num_paths} paths, "
                  f"truncating to MAX_PATHS={MAX_PATHS}", flush=True)
        keep = min(batch_num_paths, MAX_PATHS)

        a_out[start:start + len(chunk), :, :, :keep] = a[..., :keep].astype(np.complex64)
        tau_out[start:start + len(chunk), :, :, :keep] = tau[..., :keep].astype(np.float32)
        theta_t_out[start:start + len(chunk), :keep] = theta_t[:, :keep].astype(np.float32)
        phi_t_out[start:start + len(chunk), :keep] = phi_t[:, :keep].astype(np.float32)
        theta_r_out[start:start + len(chunk), :keep] = theta_r[:, :keep].astype(np.float32)
        phi_r_out[start:start + len(chunk), :keep] = phi_r[:, :keep].astype(np.float32)
        num_paths_out[start:start + len(chunk)] = keep

        for name in rx_names:
            scene.remove(name)

        done += len(chunk)
        elapsed = time.time() - t0
        rate = done / elapsed if elapsed > 0 else 0
        eta = (n - done) / rate if rate > 0 else float("inf")
        print(f"{log_prefix} {done}/{n}  ({elapsed:.1f}s elapsed, "
              f"{rate:.1f} pos/s, ETA {eta:.0f}s)", flush=True)

    h5group.create_dataset("positions", data=all_pos)
    h5group.attrs["gnb_position"] = gnb["pos"]
    h5group.attrs["bearing_deg"] = gnb["bearing_deg"]
    h5group.attrs["tilt_deg"] = gnb["tilt_deg"]
    h5group.attrs["frequency_hz"] = CENTRAL_FREQUENCY_HZ
    h5group.attrs["gnb_array"] = f"{GNB_ROWS}x{GNB_COLS} tr38901"
    h5group.attrs["ue_array"] = f"{UE_ROWS}x{UE_COLS} iso"

    scene.remove("tx")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(SCENARIOS_DIR, "sionna_rt_cache.h5"))
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0, help="cap #positions (debug)")
    args = parser.parse_args()

    import h5py

    print("Loading gNB positions...")
    gnbs = load_gnb_positions(GNB_CSV)
    for g in gnbs:
        print(f"  {g['name']}: pos={g['pos']} bearing={g['bearing_deg']} tilt={g['tilt_deg']}")

    print("Collecting unique UE positions across all seed traces...")
    positions = collect_unique_positions()
    if args.limit:
        positions = positions[:args.limit]
    print(f"  {len(positions)} unique positions "
          f"(x/y @1m, z @0.1m grid) x {len(gnbs)} gNBs "
          f"= {len(positions) * len(gnbs)} total RT queries")

    scene, rt = build_scene()
    solver = rt.PathSolver()

    with h5py.File(args.out, "w") as hf:
        hf.attrs["scene"] = SCENE_XML
        hf.attrs["position_resolution"] = "x,y @1m, z @0.1m"
        for gnb in gnbs:
            print(f"\n=== {gnb['name']} ===")
            group = hf.create_group(gnb["name"])
            run_gnb(scene, rt, solver, gnb, positions, args.batch_size, group,
                    log_prefix=f"[{gnb['name']}]")

    print(f"\nDone. Cache written to {args.out} "
          f"({os.path.getsize(args.out) / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
