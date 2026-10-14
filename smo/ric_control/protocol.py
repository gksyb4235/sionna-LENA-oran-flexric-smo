"""Validate a reviewed offline plan against a concrete running ns-3 instance."""

import math
import re

CELLS = ("5g", "4g1", "4g2")
KINDS = {"txp": 1, "ret": 2, "cio": 3, "ttt_ms": 4, "hys_db": 5}
REAL_TTT = {0, 40, 64, 80, 100, 128, 160, 256, 320, 480, 512, 640, 1024, 1280, 2560, 5120}


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{32}", value):
        raise ValueError("Invalid deployment/run identifier")
    return value


def key_for(kind, cell):
    return kind if kind in ("ttt_ms", "hys_db") else f"{kind}_{cell}"


def finite(value):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("Non-finite or non-numeric control value")
    return float(value)


def check_domain(kind, value, cell, ideal):
    value = finite(value)
    bounds = {
        "txp": (30, 43),
        "ret": (3, 12),
        "cio": ((5, 9) if cell == "5g" else (-2, 2)),
        "ttt_ms": (128, 720),
        "hys_db": (0.5, 5),
    }
    lo, hi = bounds[kind]
    if not lo <= value <= hi:
        raise ValueError(f"{kind}_{cell}={value} is outside the GNN control domain")
    if kind == "ret" and value not in (3, 5, 7, 9, 12):
        raise ValueError("RET needs an existing 3/5/7/9/12 degree cache")
    if kind == "ttt_ms" and (value != int(value) or not ideal and value not in REAL_TTT):
        raise ValueError(
            "TTT cannot be represented by this ns-3 RRC configuration; no silent rounding is allowed"
        )
    if kind == "hys_db" and value * 2 != round(value * 2):
        raise ValueError("HYS uses 0.5 dB steps")
    return value


def encode(receipt, live):
    """Return one atomic simulator batch carried inside one E2SM-RC request."""
    run = identifier(live["run_id"])
    command = identifier(receipt["deployment_id"])
    if live.get("finished") or not live.get("process_alive", True):
        raise ValueError("ns-3 run is not active")
    if live.get("replay_run_id") != receipt["run_id"]:
        raise ValueError("ns-3 must be explicitly bound to the evaluated replay run")
    if live.get("lease") and live["lease"] != command:
        raise ValueError("Another deployment owns the simulation control lease")
    if receipt["target_ric"] not in ("near_rt", "non_rt"):
        raise ValueError(
            "Live atomic deployment requires one Near-RT or Non-RT route; split is available only in the offline outbox"
        )
    candidate = receipt["candidate"]
    start = finite(candidate["interval_start_s"])
    end = finite(candidate["interval_end_s"])
    if not finite(live["sim_time_s"]) + 1 < start < end < finite(live["stop_time_s"]):
        raise ValueError(
            "Prediction interval has started or exceeds ns-3 duration. Re-evaluate or start a matching replay; times are never shifted automatically"
        )
    current = receipt["current_parameters"]
    changes = candidate["changes"]
    expected_keys = {key_for(k, c) for k in KINDS for c in CELLS}
    if set(current) != expected_keys or not changes or not set(changes) <= expected_keys:
        raise ValueError("Expected the complete eleven-control baseline and known changed controls")
    cell_rows = {row["label"]: row for row in live["cells"]}
    if set(cell_rows) != set(CELLS):
        raise ValueError("Live topology must contain the three modeled cells")
    records = []
    for label in CELLS:
        row = cell_rows[label]
        if not row.get("e2_enabled"):
            raise ValueError(f"E2 is disabled for {label}")
        for kind, code in KINDS.items():
            key = key_for(kind, label)
            expected = finite(current[key])
            actual = finite(row["parameters"][kind])
            if abs(actual - expected) > 1e-3:
                raise ValueError(f"Baseline mismatch: {key} prediction={expected}, ns3={actual}")
            value = check_domain(
                kind, changes.get(key, expected), label, row.get("ideal_rrc", False)
            )
            records.append((int(row["cell_id"]), code, value, expected))
    if not any(abs(v - e) > 1e-3 for _, _, v, e in records):
        raise ValueError("No controls change")
    payload = " ".join(
        ["AGORA1", run, command, f"{start:.9g}", f"{end:.9g}", str(len(records))]
        + [f"{value:.12g}" for record in records for value in record]
    )
    return {
        "payload": payload,
        "e2_node_id": cell_rows["5g"]["cell_id"],
        "start_s": start,
        "end_s": end,
        "clock": "ns3_simulation_seconds",
        "run_id": run,
        "command_id": command,
    }
