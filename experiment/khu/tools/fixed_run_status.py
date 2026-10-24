#!/usr/bin/env python3
"""Validate fixed-parameter sweep inputs and retain evidence for completed runs."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import uuid

FIELDS = (
    "name", "txp_5g", "txp_4g1", "txp_4g2", "ret_5g", "ret_4g1", "ret_4g2",
    "cio_5g", "cio_4g1", "cio_4g2", "ttt_ms", "hys_db",
)
ERROR = re.compile(
    r"NS_FATAL|\[FATAL\]|NS_ASSERT|aborted|malloc\(\)|free\(\):|"
    r"invalid pointer|double free|corrupted size|corrupted double-linked|segmentation fault|bad_alloc|"
    r"AddressSanitizer:|terminate called", re.IGNORECASE
)
TRAFFIC_START = 0.5


def now():
    return datetime.now(timezone.utc).isoformat()


def metadata(row, duration, interval):
    values = row.split(",")
    if len(values) != len(FIELDS) or not re.fullmatch(r"gnn_[0-9]{4,}", values[0]):
        raise ValueError("Invalid fixed-parameter row")
    return {
        "schema_version": 1, "experiment_kind": "fixed", "run_id": values[0],
        "simulation_time_s": duration, "kpi_report_interval_s": interval,
        "traffic_start_s": TRAFFIC_START,
        "parameters": {key: float(value) for key, value in zip(FIELDS[1:], values[1:])},
    }


def validate_outputs(directory, duration, interval):
    """Require every complete reporting window for all three KHU cells."""
    errors = []
    evidence = {}
    logs = [directory / "run.log", directory.parent / f"{directory.name}.launcher.log"]
    if not logs[0].is_file():
        errors.append("missing run.log")
    for logfile in logs:
        if logfile.is_file():
            with logfile.open(errors="replace") as stream:
                if any(ERROR.search(line) for line in stream):
                    errors.append(f"failure marker in {logfile.name}")
    expected_ticks = max(0, math.ceil((duration - TRAFFIC_START) / interval - 1e-10) - 1)
    if expected_ticks < 1:
        errors.append("duration does not contain a full KPI window")
    ticks = {}
    for filename in ("cell_kpi.csv", "ue_kpi.csv"):
        path = directory / filename
        try:
            with path.open(newline="") as stream:
                reader = csv.DictReader(stream)
                required = {"time_s", "interval_s"}
                if filename == "cell_kpi.csv":
                    required.add("cell_id")
                if not required <= set(reader.fieldnames or ()):
                    raise ValueError("missing required columns")
                count, final = 0, 0.0
                for row in reader:
                    t, step = float(row["time_s"]), float(row["interval_s"])
                    if not math.isfinite(t) or not math.isclose(step, interval, abs_tol=1e-6):
                        raise ValueError("nonfinite time or unexpected interval")
                    if t >= duration + 1e-6 or t < TRAFFIC_START + interval - 1e-6:
                        raise ValueError("time outside complete reporting windows")
                    tick = round((t - TRAFFIC_START) / interval)
                    if not math.isclose(t, TRAFFIC_START + tick * interval, abs_tol=1e-6):
                        raise ValueError("unexpected KPI timestamp")
                    if filename == "cell_kpi.csv":
                        cells = ticks.setdefault(tick, set())
                        if row["cell_id"] in cells:
                            raise ValueError("duplicate cell row")
                        cells.add(row["cell_id"])
                    count += 1
                    final = max(final, t)
                if not count:
                    raise ValueError("no data rows")
                evidence[filename] = {"rows": count, "final_time_s": final}
        except (OSError, ValueError, TypeError, KeyError) as error:
            errors.append(f"{filename}: {error}")
    if set(ticks) != set(range(1, expected_ticks + 1)) or any(len(c) != 3 for c in ticks.values()):
        errors.append("cell KPI windows are incomplete (expected all 3 cells at each tick)")
    return errors, evidence


def completed(directory, expected):
    try:
        saved = json.loads((directory / "run_status.json").read_text())
        if not isinstance(saved, dict):
            return False
        if saved.get("status") != "complete" or saved.get("exit_code") != 0:
            return False
        if any(saved.get(key) != value for key, value in expected.items()):
            return False
        errors, _ = validate_outputs(directory, expected["simulation_time_s"], expected["kpi_report_interval_s"])
        return not errors
    except (OSError, ValueError, TypeError):
        return False


def finish(directory, expected, exit_code, cpu_ids):
    errors, evidence = validate_outputs(directory, expected["simulation_time_s"], expected["kpi_report_interval_s"])
    saved = dict(expected, exit_code=exit_code, finished_at=now(), cpu_ids=cpu_ids,
                 status="complete" if exit_code == 0 and not errors else "failed",
                 validation_errors=errors, artifacts=evidence)
    if exit_code in (130, 143, -2, -15):
        saved["status"] = "interrupted"
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f".run_status.{os.getpid()}.json"
    temporary.write_text(json.dumps(saved, indent=2) + "\n")
    temporary.replace(directory / "run_status.json")
    return saved


def archive(directory):
    paths = [directory, directory.parent / f"{directory.name}.launcher.log"]
    existing = [path for path in paths if path.exists() or path.is_symlink()]
    if not existing:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid.uuid4().hex[:8]
    destination = directory.parent / "_attempts" / directory.name / stamp
    destination.mkdir(parents=True)
    for path in existing:
        shutil.move(str(path), str(destination / path.name))
    print(f"[archive] previous attempt retained: {destination}")


def select_rows(args):
    if args.start < 0 or args.end < args.start or args.cores < 1 or args.workers < 1:
        raise ValueError("Invalid range or parallelism")
    if args.cores * args.workers > len(os.sched_getaffinity(0)):
        raise ValueError("Requested CPUs exceed the process's allowed CPU affinity")
    if not math.isfinite(args.duration) or not math.isfinite(args.interval) or args.interval <= 0:
        raise ValueError("Simulation duration and KPI interval must be finite and positive")
    if args.duration <= TRAFFIC_START + args.interval or args.memory < 1:
        raise ValueError("Duration must contain a full KPI window; memory limit must be positive")
    by_index = {}
    with args.csv.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if not re.fullmatch(r"gnn_[0-9]{4,}", row["name"]):
                raise ValueError(f"Invalid run name: {row['name']}")
            index = int(row["name"].split("_")[1])
            if index in by_index:
                raise ValueError(f"Duplicate scenario index: {index}")
            by_index[index] = row
    missing = [index for index in range(args.start, args.end + 1) if index not in by_index]
    if missing:
        raise ValueError(f"Requested indices absent from fixed manifest: {missing[:12]}")
    selected = []
    for index in range(args.start, args.end + 1):
        row = by_index[index]
        numbers = {key: float(row[key]) for key in FIELDS[1:]}
        if not all(math.isfinite(value) for value in numbers.values()):
            raise ValueError(f"Nonfinite parameters in {row['name']}")
        for key in ("ret_5g", "ret_4g1", "ret_4g2"):
            if numbers[key] not in (3, 5, 7, 9, 12):
                raise ValueError(f"Unsupported RET tilt: {row[key]}")
        required = [args.traces / f"ue_positions_seed{index}.csv"]
        for key, band in (("ret_5g", "3p5ghz"), ("ret_4g1", "1p8ghz"), ("ret_4g2", "1p8ghz")):
            required.append(args.caches / f"tilt_{int(numbers[key])}deg/sionna_rt_cache_{band}_2x2_1x1_lzf.h5")
        for path in required:
            if not path.is_file():
                raise ValueError(f"Missing input: {path}")
        selected.append(",".join(row[key] for key in FIELDS))
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)
    plan = commands.add_parser("rows")
    plan.add_argument("csv", type=Path)
    for name in ("start", "end", "cores", "workers"):
        plan.add_argument(name, type=int)
    plan.add_argument("--traces", type=Path, required=True)
    plan.add_argument("--caches", type=Path, required=True)
    plan.add_argument("--duration", type=float, required=True)
    plan.add_argument("--interval", type=float, required=True)
    plan.add_argument("--memory", type=int, required=True)
    for name in ("check", "finish", "archive"):
        sub = commands.add_parser(name)
        sub.add_argument("directory", type=Path)
        if name == "archive":
            continue
        sub.add_argument("row")
        sub.add_argument("duration", type=float)
        sub.add_argument("interval", type=float)
        if name == "finish":
            sub.add_argument("exit_code", type=int)
            sub.add_argument("cpu_ids")
    args = parser.parse_args()
    try:
        if args.action == "rows":
            print("\n".join(select_rows(args)))
        elif args.action == "archive":
            archive(args.directory)
        else:
            expected = metadata(args.row, args.duration, args.interval)
            if args.action == "check":
                return 0 if completed(args.directory, expected) else 1
            saved = finish(args.directory, expected, args.exit_code, args.cpu_ids)
            if saved["status"] != "complete":
                print(f"[failed] {args.directory.name}: exit={args.exit_code}; " + "; ".join(saved["validation_errors"]), file=sys.stderr)
                return 1
    except (OSError, ValueError, KeyError) as error:
        print(f"[fixed sweep] {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
