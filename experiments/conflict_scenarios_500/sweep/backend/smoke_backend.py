"""Explicit single-process 20-second integration smoke, separate from the 900-second sweep."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

HERE = Path(__file__).resolve().parent
CELLS = {"5g": "gNB_5G", "4g1": "gNB_4G_1", "4g2": "gNB_4G_2"}
INITIAL = {"txp_5g": 43, "txp_4g1": 43, "txp_4g2": 43,
           "ret_5g": 9, "ret_4g1": 9, "ret_4g2": 9,
           "cio_5g": 6, "cio_4g1": 0, "cio_4g2": -1, "ttt_ms": 512, "hys_db": 1}
EVENTS = [
    (2, {"txp_5g": 33, "txp_4g1": 37}),
    (4, {"ret_5g": 12, "ret_4g1": 3, "ret_4g2": 5}),
    (6, {"cio_4g2": 2, "cio_5g": 9, "cio_4g1": -2}),
    (8, {"ttt_ms": 720, "hys_db": 5}),
    (10, {"ttt_ms": 340, "hys_db": 2.5}),
    (12, {"txp_5g": 43, "txp_4g1": 43, "txp_4g2": 30}),
    (14, {"ret_5g": 9, "ret_4g1": 9, "ret_4g2": 9}),
    (16, {"ttt_ms": 128, "hys_db": 0.5}),
]
PROTECTED = ["scratch/khu-real-dual-nr-sionna-final.cc",
             "src/spectrum/model/sionna-lookup-channel-model.cc",
             "src/spectrum/model/sionna-lookup-channel-model.h",
             "contrib/nr/model/nr-ue-rrc.cc", "contrib/nr/model/nr-gnb-rrc.cc"]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(repo: Path, output: Path) -> dict:
    repo = repo.resolve(strict=True)
    output = output.absolute()
    if output.exists():
        raise FileExistsError(f"Smoke output already exists: {output}")
    output.mkdir(parents=True, exist_ok=False)
    schedule = output / "schedule.csv"
    with schedule.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["time_s", "parameter", "value"])
        for timestamp, updates in EVENTS:
            writer.writerows((timestamp, key, value) for key, value in sorted(updates.items()))
    binary = repo / "build/scratch/agora-conflict-sweep/ns3.48-khu-agora-conflict-sweep-default"
    cache = repo / "scenarios/khu-real/ret_caches"
    argv = [str(binary), "--simTime=20", "--N_Ues=300", "--numerology=0",
            "--centralFrequency18=1.8e9", "--bandwidth18=10e6",
            "--centralFrequency35=3.5e9", "--bandwidth35=10e6",
            "--sionnaUpdatePeriod=1s", "--kpiReportInterval=1", "--udpBaseIntervalMs=200",
            "--trafficMultiplierEnabled=true", "--RngSeed=1", "--RngRun=1",
            "--ns3::NrHelper::UseIdealRrc=true", "--ns3::NrHelper::E2ModeNr=false",
            "--sumoTrace=" + str(repo / "scenarios/khu-real/ue_position/ue_positions_seed1419.csv"),
            "--gnbPositions=" + str(repo / "scenarios/khu-real/gnbs-ret.csv"),
            "--resultsRoot=" + str(output / "data"), "--runTag=smoke",
            "--parameterSchedule=" + str(schedule), "--parameterScheduleRetCacheRoot=" + str(cache),
            "--sionnaCacheFile18=" + str(cache / "tilt_9deg/sionna_rt_cache_1p8ghz_2x2_1x1_lzf.h5"),
            "--sionnaCacheFile35=" + str(cache / "tilt_9deg/sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5"),
            "--cellRetBearingDeg=gNB_5G:20,gNB_4G_1:0,gNB_4G_2:10",
            "--handoverTtt=512", "--handoverHysteresis=1"]
    for flag, family in [("cellTxPowerDbm", "txp"), ("cellRetTiltDeg", "ret"), ("cellCioDb", "cio")]:
        argv.append("--" + flag + "=" + ",".join(f"{label}:{INITIAL[f'{family}_{cell}']}"
                                                    for cell, label in CELLS.items()))
    for flag, key in [("cellTttMs", "ttt_ms"), ("cellHysteresisDb", "hys_db")]:
        argv.append("--" + flag + "=" + ",".join(f"{label}:{INITIAL[key]}" for label in CELLS.values()))
    environment = {k: v for k, v in os.environ.items() if not k.startswith(("AGORA_", "RIC_", "DRL_"))}
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONPATH=str(repo / ".venv/lib/python3.12/site-packages"), PYTHONDONTWRITEBYTECODE="1")
    environment.pop("PYTHONHOME", None)
    before = {f: digest(repo / f) for f in PROTECTED}
    cpu = min(os.sched_getaffinity(0))
    started = time.monotonic()
    (output / "command.json").write_text(json.dumps(argv, indent=2) + "\n")
    with (output / "launcher.log").open("w") as log:
        process = subprocess.Popen(argv, cwd=output, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                   preexec_fn=lambda: os.sched_setaffinity(0, {cpu}))
        print(f"Started one 20-second smoke pid={process.pid}, cpu={cpu}, output={output}", flush=True)
        while process.poll() is None:
            time.sleep(10)
            kpi = output / "data/smoke/cell_kpi.csv"
            latest = None
            if kpi.exists():
                lines = kpi.read_text().splitlines()
                if len(lines) > 1:
                    latest = lines[-1].split(",")[0]
            print(f"wall={time.monotonic()-started:.1f}s latest_kpi_time={latest}", flush=True)
    elapsed = time.monotonic() - started
    issues = []
    if process.returncode != 0:
        issues.append(f"process_exit_{process.returncode}")
    receipts_path = output / "data/smoke/parameter_changes.csv"
    receipts = list(csv.DictReader(receipts_path.open())) if receipts_path.exists() else []
    expected_rows = []
    state = dict(INITIAL)
    for timestamp, updates in EVENTS:
        for key, value in sorted(updates.items()):
            expected_rows.append({"scheduled_time_s": timestamp, "applied_time_s": timestamp,
                                  "parameter": key, "before": state[key], "after": value, "status": "applied"})
        state.update(updates)
    if len(receipts) != len(expected_rows):
        issues.append("receipt_count_mismatch")
    for i, (actual, expected) in enumerate(zip(receipts, expected_rows)):
        for key, value in expected.items():
            valid = actual[key] == value if isinstance(value, str) else abs(float(actual[key]) - value) < 1e-6
            if not valid:
                issues.append(f"receipt_{i}_{key}_mismatch")
    kpi_path = output / "data/smoke/cell_kpi.csv"
    kpis = list(csv.DictReader(kpi_path.open())) if kpi_path.exists() else []
    reverse = {label: cell for cell, label in CELLS.items()}
    for row in kpis:
        cell = reverse[row["cell"]]
        timestamp = float(row["time_s"])
        expected = dict(INITIAL)
        for when, updates in EVENTS:
            if timestamp >= when:
                expected.update(updates)
        for column, key in [("tx_power_dbm", f"txp_{cell}"), ("ret_tilt_deg", f"ret_{cell}"),
                            ("cio_bias_db", f"cio_{cell}"), ("ttt_ms", "ttt_ms"), ("hysteresis_db", "hys_db")]:
            if abs(float(row[column]) - expected[key]) > 1e-6:
                issues.append(f"kpi_{timestamp}_{cell}_{column}_mismatch")
    if len(kpis) != 60 or not kpis or float(kpis[-1]["time_s"]) != 19.5:
        issues.append("kpi_coverage_mismatch")
    after = {f: digest(repo / f) for f in PROTECTED}
    if after != before:
        issues.append("protected_source_changed")
    report = {"passed": not issues, "issues": issues, "returncode": process.returncode,
              "duration_s": 20, "wall_seconds": elapsed, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "output": str(output), "binary_sha256": digest(binary), "events": len(EVENTS),
              "receipt_rows": len(receipts), "kpi_rows": len(kpis), "protected_sources_unchanged": before == after,
              "protected_sha256": after, "receipts": receipts,
              "scope": "short_backend_functionality_not_900s_stability_or_conflict_efficacy"}
    (output / "smoke_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.repo, args.output)
    (HERE / "smoke_report.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("receipts", "protected_sha256")}, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
