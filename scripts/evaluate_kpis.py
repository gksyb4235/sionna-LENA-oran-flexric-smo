#!/usr/bin/env python3
"""Aggregate recorded cell KPI intervals without mixing experiment regimes.

Use --collection with a conflict sweep `collect` JSON report, or --run with a
fixed-parameter run directory. Raw CSVs require --include-unverified. This is
an offline evaluator: it does not launch simulations or change input files.
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
COUNTERS = (
    "tx_packets_interval", "tx_bytes_interval", "rx_packets_interval",
    "rx_bytes_interval", "confirmed_lost_packets_interval", "prb_used_reg",
    "prb_capacity_reg", "tb_total_interval", "tb_corrupt_interval",
    "tb_retx_interval", "tb_bytes_interval", "delay_samples_interval",
    "ho_attempts_interval", "ho_successes_interval", "ho_failures_interval",
    "pingpongs_interval", "rlfs_interval",
)
SUMS = ("delay_sum_ms_interval", "interval_energy_j")
REQUIRED = {"time_s", "interval_s", "cell_id", "rx_bytes_interval",
            "tx_bytes_interval", "tx_packets_interval", "rx_packets_interval"}
NOTES = [
    "Fixed and scheduled experiments must be evaluated separately; no cross-run pooling is performed.",
    "Durations are sums of CSV interval_s, not inferred simulation wall time or packet active time.",
    "Windows are nominal intervals anchored at the earliest report time minus its interval_s; "
    "an interval crossing a window boundary is rejected, never split into invented packet counts.",
    "The first nominal interval can start before simulation time zero; the producer may record "
    "a full interval_s for a partial startup interval. No unobserved terminal interval is added.",
    "Cell counts use the producer's serving-cell attribution at report time; they are not "
    "a reconstruction of the cell that transmitted each packet during handover.",
    "Delivery/loss and handover ratios use interval counts, not matched packet or handover cohorts; "
    "in-flight events may cross a window boundary. Ratios are not clamped.",
    "Missing or zero denominators produce null. Cumulative counters are never summed.",
    "Per-report percentiles are not pooled: no combined RSRP/SINR P5 or delay P95/P99 is fabricated.",
    "Energy efficiency uses simulated BS energy, not measured host energy. PRB utilization "
    "uses the producer's counted RB-symbol opportunities, not a new full-slot capacity model.",
]


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def number(value, field, integral=False):
    try:
        result = int(value) if integral else float(value)
    except (ValueError, TypeError) as error:
        raise ValueError(f"Invalid {field}: {value!r}") from error
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"Nonfinite or negative {field}: {value!r}")
    return result


def ratio(numerator, denominator):
    return numerator / denominator if numerator is not None and denominator else None


def new_group(cell, row, columns):
    result = {"cell_id": cell, "cell": row.get("cell", ""), "rows": 0,
              "report_first_s": float(row["time_s"]), "report_last_s": float(row["time_s"]),
              "nominal_duration_s": 0.0}
    result.update({name: 0 for name in COUNTERS + SUMS if name in columns})
    if "num_ues" in columns:
        result["ue_seconds"] = 0.0
    if "scheduled_offered_load_mbps" in columns:
        result["scheduled_offered_megabits"] = 0.0
    return result


def add_row(group, row, interval):
    group["rows"] += 1
    group["report_last_s"] = float(row["time_s"])
    group["nominal_duration_s"] += interval
    for name in COUNTERS + SUMS:
        if name in group:
            group[name] += number(row[name], name, name in COUNTERS)
    if "ue_seconds" in group:
        group["ue_seconds"] += number(row["num_ues"], "num_ues", True) * interval
    if "scheduled_offered_megabits" in group:
        group["scheduled_offered_megabits"] += number(
            row["scheduled_offered_load_mbps"], "scheduled_offered_load_mbps") * interval


def finish_group(group):
    result = dict(group)
    result["rx_megabits"] = 8 * result["rx_bytes_interval"] / 1e6
    result["goodput_mbps"] = ratio(result["rx_megabits"], result["nominal_duration_s"])
    result["generated_load_mbps"] = ratio(8 * result["tx_bytes_interval"] / 1e6,
                                          result["nominal_duration_s"])
    formulas = {
        "byte_delivery_ratio": ("rx_bytes_interval", "tx_bytes_interval"),
        "packet_delivery_ratio": ("rx_packets_interval", "tx_packets_interval"),
        "confirmed_loss_to_tx_ratio": ("confirmed_lost_packets_interval", "tx_packets_interval"),
        "prb_utilization_fraction": ("prb_used_reg", "prb_capacity_reg"),
        "dl_bler": ("tb_corrupt_interval", "tb_total_interval"),
        "harq_retx_ratio": ("tb_retx_interval", "tb_total_interval"),
        "delay_mean_ms": ("delay_sum_ms_interval", "delay_samples_interval"),
        "ho_success_to_attempt_ratio": ("ho_successes_interval", "ho_attempts_interval"),
        "ho_failure_to_attempt_ratio": ("ho_failures_interval", "ho_attempts_interval"),
        "pingpong_to_success_ratio": ("pingpongs_interval", "ho_successes_interval"),
        "mean_num_ues": ("ue_seconds", "nominal_duration_s"),
        "energy_efficiency_mbit_per_j": ("rx_megabits", "interval_energy_j"),
        "mean_bs_power_w": ("interval_energy_j", "nominal_duration_s"),
    }
    for name, (numerator, denominator) in formulas.items():
        result[name] = ratio(result.get(numerator), result.get(denominator))
    return result


def aggregate_cell_csv(path, window_seconds=None):
    """Return per-cell totals and optional unsplit, nominal-interval windows."""
    cells, windows, last_time, origin = {}, {}, {}, None
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)) or not REQUIRED.issubset(columns):
            raise ValueError(f"Invalid KPI header; missing {sorted(REQUIRED - set(columns))}")
        for line, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"Malformed CSV row {line}")
            when = number(row["time_s"], "time_s")
            interval = number(row["interval_s"], "interval_s")
            cell = number(row["cell_id"], "cell_id", True)
            if interval == 0:
                raise ValueError(f"Zero interval at row {line}")
            if cell in last_time and not math.isclose(
                    when - last_time[cell], interval, rel_tol=0, abs_tol=1e-6):
                raise ValueError(f"Duplicate, nonmonotonic, or missing interval for cell {cell} at {when}")
            last_time[cell] = when
            if origin is None:
                origin = when - interval
            if when - interval < origin - 1e-6:
                raise ValueError("CSV must begin with its earliest nominal interval")
            if cell not in cells:
                cells[cell] = new_group(cell, row, columns)
            elif cells[cell]["cell"] != row.get("cell", ""):
                raise ValueError(f"Changing cell identity for cell_id={cell}")
            add_row(cells[cell], row, interval)
            if window_seconds:
                index = math.floor((when - origin - 1e-8) / window_seconds)
                start = origin + index * window_seconds
                if when - interval < start - 1e-6:
                    raise ValueError(f"Interval ending {when} crosses a {window_seconds}s window boundary")
                key = cell, index
                if key not in windows:
                    windows[key] = new_group(cell, row, columns)
                    windows[key].update(window_index=index, window_start_s=start,
                                        window_end_s=start + window_seconds)
                add_row(windows[key], row, interval)
    if not cells:
        raise ValueError("No KPI data rows")
    return ([finish_group(cells[key]) for key in sorted(cells)],
            [finish_group(windows[key]) for key in sorted(windows)])


def local_run_path(value):
    """Resolve this checkout's old result paths without changing recorded JSON."""
    path = Path(value).expanduser().absolute()
    studies = ROOT / "experiment/khu/studies"
    if studies.is_dir() and path.is_relative_to(ROOT / "experiments"):
        path = studies / path.relative_to(ROOT / "experiments")
    return path.resolve()


def raw_run(path, include_unverified):
    directory = local_run_path(path)
    item = {"run_id": directory.name, "directory": str(directory), "included": False,
            "verification": "raw_unverified", "issues": []}
    receipt = directory / "run_status.json"
    if receipt.is_file():
        try:
            state = read_json(receipt)
            if state.get("status") == "complete" and state.get("exit_code") == 0:
                item.update(verification="completion_receipt_only", included=True)
            else:
                item["issues"].append("Receipt does not record a successful completion")
                item["verification"] = "failed_or_incomplete"
        except (OSError, ValueError, AttributeError) as error:
            item["issues"].append(f"Invalid completion receipt: {error}")
    elif include_unverified:
        item["included"] = True
    else:
        item["issues"].append("No completion receipt; use --include-unverified to inspect raw data")
    return item


def collection_runs(path):
    """Require a successful collect receipt and recheck its artifact hashes."""
    report = read_json(local_run_path(path))
    if not isinstance(report.get("jobs"), list):
        raise ValueError("Collection report needs a jobs list")
    items = []
    for job in report["jobs"]:
        item = {"run_id": job["job_id"], "directory": job.get("output", ""),
                "included": False, "verification": "failed_or_unverified", "issues": [],
                "trace_seed": job.get("trace_seed"), "manifest_hash": report.get("manifest_hash")}
        verification = job.get("verification", {})
        if not (job.get("verified_complete") is True and job.get("status") == "complete"
                and job.get("returncode") == 0 and verification.get("passed") is True):
            item["issues"].append("Collect did not verify successful completion")
        else:
            directory = local_run_path(item["directory"])
            item["directory"] = str(directory)
            artifacts = verification.get("artifacts", {})
            for name in ("cell_kpi.csv", "ue_kpi.csv", "parameter_changes.csv"):
                expected = artifacts.get(name, {})
                try:
                    if not expected.get("sha256") or digest(directory / name) != expected["sha256"]:
                        item["issues"].append(f"Missing or changed collect hash: {name}")
                except OSError as error:
                    item["issues"].append(str(error))
            if not item["issues"]:
                item.update(included=True, verification="verified_complete")
        items.append(item)
    return items


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-kind", choices=("fixed", "scheduled"), required=True)
    parser.add_argument("--run", type=Path, action="append", default=[], help="Directory containing cell_kpi.csv")
    parser.add_argument("--collection", type=Path, action="append", default=[], help="Conflict collect JSON report")
    parser.add_argument("--include-unverified", action="store_true")
    parser.add_argument("--window-seconds", type=int, choices=(60, 180))
    parser.add_argument("--output-dir", type=Path, required=True, help="New evaluation directory, normally experiment/khu/results/evaluation/NAME")
    args = parser.parse_args(argv)
    if not args.run and not args.collection:
        parser.error("Provide --run and/or --collection")
    if args.collection and args.experiment_kind != "scheduled":
        parser.error("Conflict collection reports describe scheduled experiments")
    items = [raw_run(path, args.include_unverified) for path in args.run]
    try:
        for path in args.collection:
            items.extend(collection_runs(path))
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.error(str(error))
    totals, windows, seen = [], [], set()
    for item in items:
        if not item["included"]:
            continue
        try:
            directory = Path(item["directory"])
            key = str(directory.resolve())
            if key in seen:
                raise ValueError("Duplicate run directory in inputs")
            seen.add(key)
            cell_rows, window_rows = aggregate_cell_csv(directory / "cell_kpi.csv", args.window_seconds)
            identity = {"experiment_kind": args.experiment_kind, "run_id": item["run_id"],
                        "verification": item["verification"], "directory": key}
            totals.extend({**identity, **row} for row in cell_rows)
            windows.extend({**identity, **row} for row in window_rows)
            item["csv_sha256"] = digest(directory / "cell_kpi.csv")
        except (OSError, ValueError, KeyError) as error:
            item["included"] = False
            item["issues"].append(str(error))
    output = args.output_dir.expanduser().resolve()
    try:
        output.mkdir(parents=True, exist_ok=False)
        write_csv(output / "cell_summary.csv", totals)
        if args.window_seconds:
            write_csv(output / "cell_windows.csv", windows)
        report = {"schema_version": 1, "experiment_kind": args.experiment_kind,
                  "window_seconds": args.window_seconds, "runs": items,
                  "included_runs": sum(item["included"] for item in items),
                  "notes": NOTES}
        with (output / "evaluation.json").open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, allow_nan=False)
            handle.write("\n")
    except OSError as error:
        parser.error(str(error))
    print(json.dumps({"output": str(output), "included_runs": report["included_runs"],
                      "excluded_runs": len(items) - report["included_runs"]}))
    return 0 if items and all(item["included"] for item in items) else 1


if __name__ == "__main__":
    sys.exit(main())
