#!/usr/bin/env python3
"""Repair per-receiver path metadata in an existing Sionna HDF5 cache.

The original cache builder stored the batch-wide padded path dimension as
``num_paths`` for every receiver. This tool compacts each receiver's non-zero
paths, writes the actual per-row count, and optionally replaces a zero-path
row with the nearest spatially valid row. The input is copied first; it is
never modified in place.
"""

import argparse
import os
import shutil

import h5py
import numpy as np


ANGLE_DATASETS = ("theta_t", "phi_t", "theta_r", "phi_r")


def compact_group(group, chunk_size, zero_path_policy):
    num_rows, num_rx_ant, num_tx_ant, max_paths = group["a"].shape
    path_counts = np.zeros(num_rows, dtype=np.int32)

    for start in range(0, num_rows, chunk_size):
        stop = min(start + chunk_size, num_rows)
        a = group["a"][start:stop]
        tau = group["tau"][start:stop]
        angles = {name: group[name][start:stop] for name in ANGLE_DATASETS}

        compact_a = np.zeros_like(a)
        compact_tau = np.zeros_like(tau)
        compact_angles = {name: np.zeros_like(values) for name, values in angles.items()}
        valid_paths = np.any(np.abs(a) > 0.0, axis=(1, 2))

        for receiver in range(stop - start):
            indices = np.flatnonzero(valid_paths[receiver])[:max_paths]
            count = len(indices)
            path_counts[start + receiver] = count
            if count == 0:
                continue
            compact_a[receiver, :, :, :count] = a[receiver, :, :, indices].transpose(1, 2, 0)
            compact_tau[receiver, :, :, :count] = tau[receiver, :, :, indices].transpose(1, 2, 0)
            for name in ANGLE_DATASETS:
                compact_angles[name][receiver, :count] = angles[name][receiver, indices]

        group["a"][start:stop] = compact_a
        group["tau"][start:stop] = compact_tau
        for name in ANGLE_DATASETS:
            group[name][start:stop] = compact_angles[name]

    zero_rows = np.flatnonzero(path_counts == 0)
    group.attrs["zero_path_rows_original"] = len(zero_rows)
    group.attrs["zero_path_policy"] = zero_path_policy

    if len(zero_rows) and zero_path_policy == "error":
        raise RuntimeError(
            f"{group.name}: {len(zero_rows)} zero-path rows; rerun with "
            "--zero-path-policy nearest to repair them"
        )

    repaired_from = np.full(num_rows, -1, dtype=np.int64)
    if len(zero_rows):
        valid_rows = np.flatnonzero(path_counts > 0)
        if not len(valid_rows):
            raise RuntimeError(f"{group.name}: every row has zero paths")
        positions = group["positions"][:]
        for row in zero_rows:
            distances = np.linalg.norm(positions[valid_rows] - positions[row], axis=1)
            source = int(valid_rows[np.argmin(distances)])
            repaired_from[row] = source
            path_counts[row] = path_counts[source]
            for name in ("a", "tau", *ANGLE_DATASETS):
                group[name][row] = group[name][source]

    group["num_paths"][:] = path_counts
    if "repaired_from_index" in group:
        del group["repaired_from_index"]
    group.create_dataset("repaired_from_index", data=repaired_from)
    return len(zero_rows), int(path_counts.min()), int(path_counts.max())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("cache", help="source HDF5 cache (left unchanged)")
    parser.add_argument("--out", required=True, help="repaired output HDF5 cache")
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--zero-path-policy", choices=("error", "nearest"), default="error")
    args = parser.parse_args()

    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")
    if os.path.abspath(args.cache) == os.path.abspath(args.out):
        raise ValueError("--out must differ from the source cache")

    shutil.copy2(args.cache, args.out)
    with h5py.File(args.out, "r+") as cache:
        cache.attrs["path_metadata_repaired"] = True
        cache.attrs["zero_path_policy"] = args.zero_path_policy
        for name in cache:
            group = cache[name]
            if not isinstance(group, h5py.Group) or "a" not in group:
                continue
            zero_rows, min_paths, max_paths = compact_group(
                group, args.chunk_size, args.zero_path_policy
            )
            print(
                f"{name}: zero rows={zero_rows}, repaired path range={min_paths}..{max_paths}",
                flush=True,
            )

    print(f"Repaired cache written to {args.out}", flush=True)


if __name__ == "__main__":
    main()
