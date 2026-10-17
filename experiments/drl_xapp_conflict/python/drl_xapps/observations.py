"""Causal aggregation of flushed, completed 1-second ns-3 cell KPI rows."""
import csv
import math
from pathlib import Path
import numpy as np
from .actions import CELLS

CELL_NAMES = {"gNB_5G": "5g", "gNB_4G_1": "4g1", "gNB_4G_2": "4g2"}
REQUIRED = ("time_s", "cell", "num_ues", "delivered_megabits_interval",
            "generated_load_mbps", "prb_used_reg", "prb_capacity_reg",
            "ho_attempts_interval", "ho_successes_interval", "ho_failures_interval", "pingpongs_interval", "rlfs_interval",
            "delay_sum_ms_interval", "delay_samples_interval")
STATE_SCHEMA = "causal-30-90-cell-ho-failure-v2"


def safe_ratio(numerator, denominator, default=0.0):
    return float(numerator / denominator) if denominator > 0 else float(default)


def load_completed(path, time_s, history_s=90):
    rows = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing KPI fields: {sorted(missing)}")
        for raw in reader:
            try:
                timestamp = float(raw["time_s"])
                if not max(0.0, time_s - history_s) <= timestamp < time_s:
                    continue
                cell = CELL_NAMES.get(raw["cell"], raw["cell"])
                if cell not in CELLS:
                    continue
                row = {k: float(raw[k]) for k in REQUIRED if k != "cell"}
                if not all(math.isfinite(v) for v in row.values()):
                    raise ValueError("Non-finite required cell KPI")
                row["cell"] = cell
                rows.append(row)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Incomplete or invalid completed KPI row: {raw}") from exc
    if not rows:
        raise ValueError(f"No completed KPI observations before t={time_s}")
    # A partial three-cell timestamp must not become an apparently unloaded cell.
    groups = {}
    for row in rows:
        key = row["time_s"]
        names = groups.setdefault(key, set())
        if row["cell"] in names:
            raise ValueError(f"Duplicate cell timestamp: {key}, {row['cell']}")
        names.add(row["cell"])
    if any(names != set(CELLS) for names in groups.values()):
        raise ValueError("A completed timestamp is missing one or more cell rows")
    return rows


def aggregate(rows, time_s, width_s):
    selected = [r for r in rows if max(0.0, time_s - width_s) <= r["time_s"] < time_s]
    # Each row covers the reported 1-second interval; do not extrapolate warm-up gaps.
    timestamps = sorted({r["time_s"] for r in selected})
    if not timestamps:
        raise ValueError("Empty aggregation window")
    if len(timestamps) > 1 and not np.allclose(np.diff(timestamps), 1.0):
        raise ValueError("This experiment requires contiguous 1-second KPI reports")
    duration = float(len(timestamps))
    total = lambda field: sum(r[field] for r in selected)
    cells = []
    for name in CELLS:
        group = [r for r in selected if r["cell"] == name]
        subtotal = lambda field: sum(r[field] for r in group)
        cells.append({"name": name,
            "prb": safe_ratio(subtotal("prb_used_reg"), subtotal("prb_capacity_reg")),
            "ues": subtotal("num_ues") / duration,
            "goodput_mbps": subtotal("delivered_megabits_interval") / duration})
    prb = np.asarray([c["prb"] for c in cells])
    jain = safe_ratio(prb.sum() ** 2, len(prb) * (prb ** 2).sum(), default=0)
    attempts, successes, failures, pp, rlf = (total(f) for f in ("ho_attempts_interval",
        "ho_successes_interval", "ho_failures_interval", "pingpongs_interval", "rlfs_interval"))
    ue_seconds = total("num_ues")
    generated = total("generated_load_mbps")
    delivered = total("delivered_megabits_interval")
    # Beta(1,9) weak prior avoids a perfect score with zero/sparse HO attempts.
    posterior = (pp + 1.0) / (attempts + 10.0)
    return {"duration_s": duration, "first_time_s": timestamps[0], "last_time_s": timestamps[-1],
        "cells": cells, "jain": jain, "active_load": bool(prb.sum() > 1e-8),
        "goodput_mbps": delivered / duration, "generated_mbps": generated / duration,
        "delivery_ratio": safe_ratio(delivered, generated, default=1),
        "has_traffic": generated > 0, "mean_ues": ue_seconds / duration,
        "ho_attempts": attempts, "ho_successes": successes, "ho_failures": failures,
        "ho_failure_ratio": safe_ratio(failures, attempts), "pingpongs": pp, "rlf": rlf,
        "pp_per_ho": safe_ratio(pp, attempts) if attempts > 0 else None,
        "pp_per_success": safe_ratio(pp, successes) if successes > 0 else None,
        "pp_posterior": posterior, "ho_confidence": attempts / (attempts + 10.0),
        "ho_success_support": min(successes, attempts) / (attempts + 10.0),
        "rlf_per_1000_ue_s": safe_ratio(1000 * rlf, ue_seconds),
        "delay_ms": safe_ratio(total("delay_sum_ms_interval"), total("delay_samples_interval"))}


def features(metrics):
    result = []
    for c in metrics["cells"]:
        result.extend((c["prb"], c["ues"] / 100, c["goodput_mbps"] / 50))
    result.extend((metrics["jain"], metrics["ho_attempts"] / 100,
        metrics["pingpongs"] / 30, metrics["pp_posterior"], metrics["ho_confidence"],
        metrics["rlf"] / 10, metrics["rlf_per_1000_ue_s"] / 5,
        metrics["delivery_ratio"], metrics["delay_ms"] / 100,
        metrics["goodput_mbps"] / 100, metrics["generated_mbps"] / 100,
        metrics["mean_ues"] / 300, float(metrics["active_load"]), float(metrics["has_traffic"]),
        metrics["ho_failure_ratio"]))
    return result


def observe(payload, reward_window_s=None):
    now = float(payload["time_s"])
    rows = load_completed(payload["cell_kpi_path"], now, max(90, reward_window_s or 0))
    short, long = aggregate(rows, now, 30), aggregate(rows, now, 90)
    p = payload["parameters"]
    controls = [p["cio_" + c] / 6 for c in CELLS] + [p["ttt_ms"] / 1024, p["hys_db"] / 6]
    state = np.asarray(features(short) + features(long) + controls + [now / 900], dtype=np.float32)
    metrics = {"short": short, "long": long}
    if reward_window_s is not None:
        metrics["outcome"] = aggregate(rows, now, reward_window_s)
    return np.clip(state, -10, 10), metrics
