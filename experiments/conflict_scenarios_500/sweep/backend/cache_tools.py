"""Read-only exact RET cache metadata audit; run with the testbed venv (h5py/numpy)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np

ANGLES = (3, 5, 7, 9, 12)
GROUPS = {"1p8ghz": ("gNB_4G_1", "gNB_4G_2"), "3p5ghz": ("gNB_5G",)}


def audit(cache_root: Path, gnb_positions: Path) -> dict:
    cache_root = cache_root.resolve(strict=True)
    with gnb_positions.open(encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(line for line in handle if not line.lstrip().startswith("#")))
    sites = {row["gnb_id"]: row for row in rows}
    if set(sites) != {"gNB_5G", "gNB_4G_1", "gNB_4G_2"}:
        raise ValueError("Expected the three reviewed gNB labels")
    expected = []
    for angle in ANGLES:
        for band, groups in GROUPS.items():
            path = cache_root / f"tilt_{angle}deg/sionna_rt_cache_{band}_2x2_1x1_lzf.h5"
            expected.append((path, {group: angle for group in groups}, band))
    for first in ANGLES:
        for second in ANGLES:
            if first != second:
                path = cache_root / f"merged_percell/merged_1p8ghz_4g1_{first}deg_4g2_{second}deg.h5"
                expected.append((path, {"gNB_4G_1": first, "gNB_4G_2": second}, "1p8ghz"))
    issues = []
    files = []
    grid_digest = None
    for path, groups, band in expected:
        item = {"path": str(path), "groups": [], "passed": False}
        try:
            if not path.is_file():
                raise ValueError("Missing exact cache")
            item["bytes"] = path.stat().st_size
            with h5py.File(path, "r") as handle:
                if set(handle.keys()) != set(groups):
                    raise ValueError("Cache groups do not match the band")
                for name, tilt in groups.items():
                    if not isinstance(handle.get(name, getlink=True), h5py.HardLink):
                        raise ValueError(f"External/soft group link forbidden: {name}")
                    group = handle[name]
                    for child in group:
                        if not isinstance(group.get(child, getlink=True), h5py.HardLink):
                            raise ValueError(f"External/soft dataset link forbidden: {name}/{child}")
                    attrs = group.attrs
                    expected_frequency = 1.8e9 if band == "1p8ghz" else 3.5e9
                    if abs(float(attrs["tilt_deg"]) - tilt) > 1e-6:
                        raise ValueError(f"Wrong tilt: {name}")
                    if abs(float(attrs["frequency_hz"]) - expected_frequency) > 1:
                        raise ValueError(f"Wrong band: {name}")
                    site = sites[name]
                    position = np.array([float(site[key]) for key in ("x", "y", "z")])
                    if np.linalg.norm(np.asarray(attrs["gnb_position"]) - position) >= 0.01:
                        raise ValueError(f"Wrong site: {name}")
                    difference = (float(attrs["bearing_deg"]) - float(site["bearing_deg"]) + 180) % 360 - 180
                    if abs(difference) > 1e-6:
                        raise ValueError(f"Wrong bearing: {name}")
                    if attrs["gnb_array"] != "2x2 tr38901" or attrs["ue_array"] != "1x1 iso":
                        raise ValueError(f"Wrong antenna array: {name}")
                    for dataset in ("positions", "a", "tau", "num_paths", "phi_r", "phi_t", "theta_r", "theta_t"):
                        if dataset not in group:
                            raise ValueError(f"Missing dataset: {name}/{dataset}")
                    positions = np.asarray(group["positions"][:], dtype="<f8")
                    if positions.ndim != 2 or positions.shape[1] != 3 or not np.isfinite(positions).all():
                        raise ValueError(f"Invalid position grid: {name}")
                    actual_digest = hashlib.sha256(positions.tobytes()).hexdigest()
                    if grid_digest is None:
                        grid_digest = actual_digest
                    if actual_digest != grid_digest:
                        raise ValueError(f"Position grid mismatch: {name}")
                    for dataset in ("a", "tau", "num_paths", "phi_r", "phi_t", "theta_r", "theta_t"):
                        if group[dataset].shape[0] != len(positions):
                            raise ValueError(f"Dataset row count differs from grid: {name}/{dataset}")
                    item["groups"].append({"name": name, "tilt_deg": tilt,
                                           "frequency_hz": expected_frequency, "positions": len(positions),
                                           "position_grid_sha256": actual_digest})
            item["passed"] = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            item["error"] = str(error)
            issues.append({"path": str(path), "error": str(error)})
        files.append(item)
    return {"schema": "agora-ret-cache-audit-v1", "passed": not issues, "files": files,
            "files_checked": len(files), "issues": issues, "position_grid_sha256": grid_digest,
            "scope": "all_five_tilts_both_bands_and_twenty_mixed_lowband_caches_metadata_grid_no_channel_data_read",
            "trace_coverage_checked": False}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--gnb-positions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.cache_root, args.gnb_positions)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
        handle.write("\n")
    print(json.dumps({"passed": result["passed"], "files_checked": result["files_checked"],
                      "issues": result["issues"]}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
