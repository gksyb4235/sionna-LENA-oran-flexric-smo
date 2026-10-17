"""Read completed DRL/ns-3 runs without changing their raw outputs.

The public ``score_run(run_dir, app)`` function is for validation selection only.
The CLI reports test arms without choosing a best test policy. Dependencies are
the Python standard library and numpy; no pandas or spreadsheet runtime needed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


WINDOWS = [(s, s + 180) for s in range(0, 900, 180)]
REQUIRED = (
    "time_s", "num_ues", "delivered_megabits_interval", "generated_load_mbps",
    "prb_used_reg", "prb_capacity_reg", "ho_attempts_interval",
    "ho_successes_interval", "ho_failures_interval", "pingpongs_interval",
    "rlfs_interval", "delay_sum_ms_interval", "delay_samples_interval",
)
COUNT_FIELDS = {
    "ho_attempts": "ho_attempts_interval", "ho_successes": "ho_successes_interval",
    "ho_failures": "ho_failures_interval", "pingpongs": "pingpongs_interval",
    "rlf": "rlfs_interval",
}
INTERACTION_METRICS = (
    "prb_jain_of_interval_means", "mean_per_second_prb_jain",
    "pp_per_attempt_pct", "pp_per_success_pct", "rlf",
    "rlf_per_1000_ue_s", "ho_failure_per_attempt_pct",
    "goodput_mbps", "delivery_pct", "delay_ms", "mean_ues",
)
METHOD = {
    "windows": "Fixed half-open 180-second windows; complete 1-second three-cell rows required.",
    "ratios": "Ratio of summed numerators and denominators, never averages of per-second ratios.",
    "pp_support": "PP/HO ratios are null when the matching denominator is below 10 events.",
    "pp_definition": "Successful reverse A-B-A handover within strictly less than 1 second in the simulator.",
    "delivery": "Delivered megabits / generated megabits in the same window; not packet-cohort loss or same-future-baseline preservation.",
    "prb": "Jain of three cell aggregate PRB utilizations; mean instantaneous Jain is reported separately. Zero total utilization is undefined.",
    "delay": "Packet/sample-weighted mean: sum delay_sum_ms_interval / sum delay_samples_interval.",
    "factorial": "both - LB - MO + none at the same trace and time; descriptive pilot interaction of the adaptive policies as executed, not an isolated fixed-action effect or a significance test.",
    "causality": "Matched inputs and prefix checks do not guarantee bitwise-identical exogenous RNG draws after controls diverge.",
    "selection": "Validation only, common guarded score per app. No held-out test score is used to select policy weights or checkpoints.",
}


def ratio(numerator, denominator, scale=1.0, minimum=0.0):
    if denominator <= 0 or denominator < minimum:
        return None
    return float(scale * numerator / denominator)


def jain(values):
    values = np.asarray(values, dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        return None
    return ratio(float(values.sum() ** 2), float(len(values) * np.square(values).sum()))


def _load_json(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def execution_metadata(run_dir):
    execution = _load_json(Path(run_dir) / "execution.json")
    job = execution.get("job", execution.get("spec", execution))
    if not isinstance(job, dict):
        raise ValueError("execution job/spec must be an object")
    job = {**job}
    for name in ("phase", "arm", "mode", "reward", "policy", "trace_seed",
                 "learning_seed", "lb_start_s", "mo_start_s", "job_id"):
        if name not in job and name in execution:
            job[name] = execution[name]
    job.setdefault("job_id", Path(run_dir).name)
    return execution, job


def read_cells(path):
    rows = []
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        absent = set(REQUIRED + ("cell",)) - set(reader.fieldnames or ())
        if absent:
            raise ValueError(f"Missing cell KPI columns: {sorted(absent)}")
        for index, raw in enumerate(reader, start=2):
            try:
                row = {key: float(raw[key]) for key in REQUIRED}
                row["interval_s"] = float(raw.get("interval_s", 1.0))
                if not all(math.isfinite(value) for value in row.values()):
                    raise ValueError("non-finite mandatory value")
                if not math.isclose(row["interval_s"], 1.0, abs_tol=1e-9):
                    raise ValueError("this experiment requires one-second reports")
                if any(row[key] < 0 for key in REQUIRED if key != "time_s"):
                    raise ValueError("negative count, exposure, load, or delay")
                row["cell"] = raw["cell"]
                row["_raw"] = raw
                for optional in ("interval_energy_j", "tb_total_interval", "tb_corrupt_interval"):
                    value = raw.get(optional)
                    row[optional] = float(value) if value not in (None, "") else None
                    if row[optional] is not None and not math.isfinite(row[optional]):
                        row[optional] = None
                rows.append(row)
            except (TypeError, ValueError, KeyError) as exc:
                raise ValueError(f"Invalid cell KPI row {index} in {path}: {exc}") from exc
    if not rows:
        raise ValueError(f"No cell KPI data: {path}")
    return rows


def aggregate(rows, start_s, end_s):
    duration = float(end_s - start_s)
    if duration <= 0 or not duration.is_integer():
        raise ValueError("Aggregation requires a positive integer number of seconds")
    selected = [r for r in rows if start_s <= r["time_s"] < end_s]
    groups = defaultdict(dict)
    for row in selected:
        if row["cell"] in groups[row["time_s"]]:
            raise ValueError(f"Duplicate cell timestamp {row['cell']} at {row['time_s']}")
        groups[row["time_s"]][row["cell"]] = row
    timestamps = sorted(groups)
    names = sorted({r["cell"] for r in selected})
    if (len(names) != 3 or len(timestamps) != int(duration)
            or any(set(group) != set(names) for group in groups.values())):
        raise ValueError(f"Incomplete three-cell KPI coverage for [{start_s}, {end_s})")
    if len(timestamps) > 1 and not np.allclose(np.diff(timestamps), 1.0, rtol=0, atol=1e-8):
        raise ValueError("Missing/non-contiguous one-second KPI observations")
    total = lambda key: sum(r[key] for r in selected)
    ue_seconds = sum(r["num_ues"] * r["interval_s"] for r in selected)
    generated_mbits = sum(r["generated_load_mbps"] * r["interval_s"] for r in selected)
    delivered_mbits = total("delivered_megabits_interval")
    result = {"start_s": float(start_s), "end_s": float(end_s), "duration_s": duration,
              "cell_rows": len(selected), "first_time_s": timestamps[0], "last_time_s": timestamps[-1],
              "ue_seconds": ue_seconds, "mean_ues": ue_seconds / duration,
              "generated_mbits": generated_mbits, "delivered_mbits": delivered_mbits,
              "generated_mbps": generated_mbits / duration, "goodput_mbps": delivered_mbits / duration,
              "delivery_pct": ratio(delivered_mbits, generated_mbits, 100),
              "delay_samples": total("delay_samples_interval"),
              "delay_ms": ratio(total("delay_sum_ms_interval"), total("delay_samples_interval"))}
    result.update({name: int(total(column)) for name, column in COUNT_FIELDS.items()})
    result["pp_attempt_supported"] = result["ho_attempts"] >= 10
    result["pp_success_supported"] = result["ho_successes"] >= 10
    result["pp_per_attempt_pct"] = ratio(result["pingpongs"], result["ho_attempts"], 100, 10)
    result["pp_per_success_pct"] = ratio(result["pingpongs"], result["ho_successes"], 100, 10)
    result["ho_failure_per_attempt_pct"] = ratio(result["ho_failures"], result["ho_attempts"], 100, 10)
    result["rlf_per_1000_ue_s"] = ratio(result["rlf"], ue_seconds, 1000)
    result["pingpong_per_1000_ue_s"] = ratio(result["pingpongs"], ue_seconds, 1000)
    result["cells"] = {}
    for name in names:
        subset = [r for r in selected if r["cell"] == name]
        subtotal = lambda key: sum(r[key] for r in subset)
        result["cells"][name] = {
            "prb_pct": ratio(subtotal("prb_used_reg"), subtotal("prb_capacity_reg"), 100),
            "mean_ues": sum(r["num_ues"] * r["interval_s"] for r in subset) / duration,
            "goodput_mbps": subtotal("delivered_megabits_interval") / duration,
            "rlf": int(subtotal("rlfs_interval")), "pingpongs": int(subtotal("pingpongs_interval")),
        }
    utilizations = [result["cells"][name]["prb_pct"] for name in names]
    result["prb_jain_of_interval_means"] = jain(utilizations) if all(v is not None for v in utilizations) else None
    result["prb_gap_pp"] = max(utilizations) - min(utilizations) if all(v is not None for v in utilizations) else None
    instantaneous = []
    for group in groups.values():
        prb = [ratio(group[name]["prb_used_reg"], group[name]["prb_capacity_reg"]) for name in names]
        value = jain(prb) if all(v is not None for v in prb) else None
        if value is not None:
            instantaneous.append(value)
    result["mean_per_second_prb_jain"] = float(np.mean(instantaneous)) if instantaneous else None
    result["instantaneous_jain_valid_seconds"] = len(instantaneous)
    for optional in ("interval_energy_j", "tb_total_interval", "tb_corrupt_interval"):
        result[optional] = total(optional) if all(r[optional] is not None for r in selected) else None
    result["power_w"] = result["interval_energy_j"] / duration if result["interval_energy_j"] is not None else None
    result["bler_pct"] = (ratio(result["tb_corrupt_interval"], result["tb_total_interval"], 100)
                          if result["tb_corrupt_interval"] is not None and result["tb_total_interval"] is not None else None)
    return result


def score_metrics(metrics, app):
    """Fixed validation utility, shared between local and guarded variants.

    This deliberately omits the training reward's temporal-improvement bonus and
    action movement term. It measures achieved service over the entire evaluation
    interval. The Beta(1,9) posterior is explicitly estimated, never a reported raw
    PP rate; confidence is zero for zero attempts. Test metrics are not selected.
    """
    if app not in ("lb", "mo"):
        raise ValueError("app must be 'lb' or 'mo'")
    jain_value = metrics["prb_jain_of_interval_means"]
    if jain_value is None or metrics["delivery_pct"] is None or metrics["rlf_per_1000_ue_s"] is None:
        return {"score": None, "eligible": False, "reason": "No measurable load/exposure", "components": {}}
    attempts, pp = metrics["ho_attempts"], metrics["pingpongs"]
    posterior, confidence = (pp + 1.0) / (attempts + 10.0), attempts / (attempts + 10.0)
    successful_confidence = min(metrics["ho_successes"], attempts) / (attempts + 10.0)
    objective = 2 * (jain_value - .7) if app == "lb" else 5 * successful_confidence * (.1 - posterior)
    rlf_penalty = .8 * metrics["rlf_per_1000_ue_s"]
    delivery_penalty = 4 * max(0, .95 - metrics["delivery_pct"] / 100)
    failure_penalty = 2 * metrics["ho_failures"] / (attempts + 10.0)
    cross_penalty = (3 * confidence * max(0, posterior - .1) if app == "lb"
                     else 2 * max(0, .9 - jain_value))
    return {"score": float(objective - rlf_penalty - delivery_penalty - cross_penalty - failure_penalty), "eligible": True,
            "components": {"objective": objective, "rlf_penalty": rlf_penalty,
                           "delivery_penalty": delivery_penalty, "cross_penalty": cross_penalty,
                           "ho_failure_penalty": failure_penalty,
                           "pp_beta_1_9_posterior": posterior, "ho_confidence": confidence,
                           "successful_ho_confidence": successful_confidence}}


def score_run(run_dir, app):
    """Return a JSON-serializable validation score; refuse train/test/partial runs."""
    run_dir = Path(run_dir)
    execution, job = execution_metadata(run_dir)
    if execution.get("status") != "complete":
        raise ValueError(f"Cannot score incomplete run: {run_dir}")
    if job.get("phase") != "validation":
        raise ValueError("score_run is restricted to held-out validation runs; never select on test")
    start = float(job.get(f"{app}_start_s", 180))
    end = float(job.get("duration_s", 900))
    metrics = aggregate(read_cells(run_dir / "cell_kpi.csv"), start, end)
    return {**score_metrics(metrics, app), "app": app, "start_s": start, "end_s": end,
            "metrics": metrics, "job_id": job["job_id"], "phase": "validation",
            "selection_rule": "common-guarded-validation-utility-v1; higher is better; no temporal bonus or movement term"}


def _prefix(rows, end):
    selected = sorted((r for r in rows if 0 <= r["time_s"] < end), key=lambda r: (r["time_s"], r["cell"]))
    encoded = json.dumps([r["_raw"] for r in selected], sort_keys=True, separators=(",", ":")).encode()
    return selected, hashlib.sha256(encoded).hexdigest()


def prefix_check(left, right, end=540):
    lrows, lhash = _prefix(left["_rows"], end)
    rrows, rhash = _prefix(right["_rows"], end)
    result = {"left_job": left["job_id"], "right_job": right["job_id"], "end_exclusive_s": end,
              "left_rows": len(lrows), "right_rows": len(rrows),
              "identical": lhash == rhash, "left_sha256": lhash, "right_sha256": rhash}
    if lhash != rhash:
        for a, b in zip(lrows, rrows):
            different = [key for key in sorted(set(a["_raw"]) | set(b["_raw"]))
                         if a["_raw"].get(key) != b["_raw"].get(key)]
            if different:
                result["first_difference"] = {"left_time_s": a["time_s"], "right_time_s": b["time_s"],
                                              "left_cell": a["cell"], "right_cell": b["cell"],
                                              "fields": different[:15]}
                break
    return result


def factorial_analysis(runs):
    grouped, checks, interactions, warnings = defaultdict(dict), [], [], []
    for run in runs:
        job = run["job"]
        if job.get("phase") != "test":
            continue
        key = job.get("trace_seed")
        arm = job.get("arm")
        if arm is None or key is None:
            warnings.append(f"Test job {run['job_id']} has no explicit arm/trace_seed; excluded from factorial")
            continue
        if arm in grouped[key]:
            raise ValueError(f"Ambiguous repeated test arm {arm} for trace {key}; use distinct experiment roots")
        grouped[key][arm] = run
    for trace, arms in sorted(grouped.items(), key=lambda item: str(item[0])):
        if "none" in arms:
            for arm, run in arms.items():
                if arm != "none":
                    checks.append({"trace_seed": trace, "kind": "common-pre180",
                                   **prefix_check(arms["none"], run, 180)})
        for variant in ("local", "guarded"):
            names = ("none", f"lb_{variant}", f"mo_{variant}", f"both_{variant}")
            for a, b in ((names[0], names[2]), (names[1], names[3])):
                if a in arms and b in arms:
                    checks.append({"trace_seed": trace, "kind": "same-controls-pre540",
                                   **prefix_check(arms[a], arms[b], 540)})
            if not all(name in arms for name in names):
                warnings.append(f"Trace {trace}, {variant}: factorial incomplete, missing {[n for n in names if n not in arms]}")
                continue
            for start, end in [(540, 720), (720, 900), (540, 900)]:
                summaries = [aggregate(arms[name]["_rows"], start, end) for name in names]
                for metric in INTERACTION_METRICS:
                    base, lb, mo, both = [m[metric] for m in summaries]
                    valid = all(value is not None for value in (base, lb, mo, both))
                    interactions.append({"trace_seed": trace, "variant": variant, "start_s": start, "end_s": end,
                        "metric": metric, "none": base, "lb": lb, "mo": mo, "both": both,
                        "interaction_both_minus_lb_minus_mo_plus_none": both - lb - mo + base if valid else None,
                        "both_minus_lb": both - lb if both is not None and lb is not None else None,
                        "both_minus_mo": both - mo if both is not None and mo is not None else None,
                        "all_denominators_supported": valid})
    return checks, interactions, warnings


def write_csv(path, rows):
    rows = list(rows)
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def build_report(root):
    root = Path(root)
    runs, skipped, excluded, errors = [], [], [], []
    for path in sorted((root / "results").glob("**/execution.json")):
        try:
            execution, job = execution_metadata(path.parent)
            if execution.get("status") != "complete":
                skipped.append({"run_directory": str(path.parent), "status": execution.get("status")})
                continue
            if job.get("phase") == "smoke":
                excluded.append({"run_directory": str(path.parent), "job_id": job["job_id"],
                                 "phase": "smoke", "status": "complete",
                                 "reason": "Explicit smoke-test phase; outside the 900-second campaign analysis"})
                continue
            rows = read_cells(path.parent / "cell_kpi.csv")
            runs.append({"job_id": job["job_id"], "job": job, "source_directory": str(path.parent.resolve()),
                         "windows": [aggregate(rows, start, end) for start, end in WINDOWS], "_rows": rows})
        except (OSError, ValueError, TypeError, KeyError) as exc:
            errors.append({"run_directory": str(path.parent), "error": str(exc)})
    checks, interactions, warnings = factorial_analysis(runs)
    output = root / "analysis"
    output.mkdir(parents=True, exist_ok=True)
    payload = {"schema": "drl-xapp-completed-kpi-report-v1", "method": METHOD,
               "completed_valid_runs": len(runs), "skipped_incomplete_runs": skipped,
               "excluded_runs": excluded,
               "invalid_complete_runs": errors, "warnings": warnings, "prefix_checks": checks,
               "factorial_interactions": interactions,
               "runs": [{k: v for k, v in run.items() if k != "_rows"} for run in runs]}
    (output / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    flat = []
    for run in runs:
        for window in run["windows"]:
            flat.append({"job_id": run["job_id"], **{k: run["job"].get(k) for k in
                ("phase", "arm", "trace_seed", "learning_seed", "mode", "reward")},
                **{k: v for k, v in window.items() if k != "cells"}})
    write_csv(output / "windows.csv", flat)
    write_csv(output / "factorial.csv", interactions)
    lines = [f"DRL xApp pilot: {len(runs)} completed valid runs", f"Incomplete skipped: {len(skipped)}; complete but invalid: {len(errors)}",
             f"Explicit smoke tests excluded: {len(excluded)}",
             f"Prefix checks: {sum(c['identical'] for c in checks)}/{len(checks)} identical",
             "", "The following test results are measured outcomes, not policy-selection scores.",
             "PP percentages are NA below 10 matching HO events. Jain=NA for no PRB load.",
             "Interactions are paired descriptive pilot estimates, not proof of significance or convergence.", "",
             "job | window | Jain | PP/attempt % | PP/success % | RLF | Goodput Mbps | delay ms"]
    fmt = lambda value: "NA" if value is None else f"{value:.4f}"
    for run in runs:
        if run["job"].get("phase") != "test":
            continue
        for w in run["windows"]:
            if w["start_s"] < 540:
                continue
            lines.append(f"{run['job_id']} | {w['start_s']:g}-{w['end_s']:g} | " + " | ".join(fmt(w[k]) for k in
                ("prb_jain_of_interval_means", "pp_per_attempt_pct", "pp_per_success_pct", "rlf", "goodput_mbps", "delay_ms")))
    if warnings or errors:
        lines += ["", "Report caveats:"] + warnings + [f"{e['run_directory']}: {e['error']}" for e in errors]
    lines += ["", "Metric definitions:"] + [f"{key}: {value}" for key, value in METHOD.items()]
    (output / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Experiment root containing results/<job_id>/execution.json")
    args = parser.parse_args()
    report = build_report(args.root)
    print(json.dumps({"completed_valid_runs": report["completed_valid_runs"],
                      "incomplete_skipped": len(report["skipped_incomplete_runs"]),
                      "excluded_smoke_runs": len(report["excluded_runs"]),
                      "invalid_complete_runs": len(report["invalid_complete_runs"]),
                      "output": str(args.root / "analysis")}, indent=2))
    if report["invalid_complete_runs"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
