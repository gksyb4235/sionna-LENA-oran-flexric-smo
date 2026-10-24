"""Local Linux ns-3 two-switch batch runner; Python standard library only.

preflight/status/verify never start ns-3. Only the explicit run subcommand starts
jobs. Existing output roots are never reused, resumed, deleted, or overwritten.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
from typing import Any


SCHEMA = "ns3-two-switch-batch-v1"
CELLS = (("5g", "gNB_5G", 1), ("4g1", "gNB_4G_1", 2), ("4g2", "gNB_4G_2", 3))
PARAMETERS = tuple(f"{family}_{cell}" for family in ("txp", "ret", "cio")
                   for cell, _, _ in CELLS) + ("ttt_ms", "hys_db")
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.+-]{0,127}\Z")
RECEIPT_FIELDS = ("scheduled_time_s", "applied_time_s", "parameter", "before", "after", "status")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def finite(value: Any, label: str) -> float:
    if isinstance(value, (bool, str)):
        raise ValueError(f"{label} must be a finite JSON number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite JSON number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{label} must be a positive integer")
    return value


def child_path(root: Path, relative: str, label: str) -> Path:
    if not isinstance(relative, str) or not relative or any(c.isspace() for c in relative):
        raise ValueError(f"{label} must be a nonempty relative path without whitespace")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts or "\\" in relative:
        raise ValueError(f"{label} must stay inside backend_root")
    if path.parts and path.parts[0] == "experiments" and (root / "experiment/khu/studies").is_dir():
        path = Path("experiment/khu/studies", *path.parts[1:])
    resolved = (root / path).resolve()
    if resolved == root or not resolved.is_relative_to(root):
        raise ValueError(f"{label} resolves outside backend_root")
    return resolved


def validate_manifest(manifest: dict) -> dict:
    if manifest.get("schema") != SCHEMA:
        raise ValueError(f"manifest schema must be {SCHEMA}")
    if not SAFE_ID.fullmatch(str(manifest.get("experiment_id", ""))):
        raise ValueError("invalid experiment_id")
    backend = Path(manifest["backend_root"]).resolve()
    if not Path(manifest["backend_root"]).is_absolute() or any(c.isspace() for c in str(backend)):
        raise ValueError("backend_root must be absolute and contain no whitespace")
    for field in ("binary", "ret_cache_root", "output_root"):
        child_path(backend, manifest[field], field)
    output = child_path(backend, manifest["output_root"], "output_root")
    cache_root = child_path(backend, manifest["ret_cache_root"], "ret_cache_root")
    if output == cache_root or output.is_relative_to(cache_root) or cache_root.is_relative_to(output):
        raise ValueError("output_root and ret_cache_root must be separate trees")
    if finite(manifest.get("duration_s"), "duration_s") != 900:
        raise ValueError("this experiment requires duration_s=900")
    if finite(manifest.get("kpi_interval_s"), "kpi_interval_s") != 1:
        raise ValueError("this experiment requires kpi_interval_s=1")
    for field in ("n_ues", "rng_seed", "rng_run"):
        positive_integer(manifest.get(field), field)
    baseline = manifest.get("baseline_parameters", {})
    if set(baseline) != set(PARAMETERS):
        raise ValueError("baseline_parameters must contain exactly the eleven physical parameters")
    for parameter, value in baseline.items():
        number = finite(value, parameter)
        if parameter.startswith("txp_") and not 0 <= number <= 60:
            raise ValueError(f"unsupported TxP domain: {parameter}")
        if parameter.startswith("ret_") and not 0 <= number <= 90:
            raise ValueError(f"unsupported RET domain: {parameter}")
    if baseline["ttt_ms"] < 0 or baseline["hys_db"] < 0:
        raise ValueError("TTT and hysteresis must be nonnegative")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("jobs must be a nonempty list")
    seen = set()
    for job in jobs:
        job_id = job.get("job_id", "")
        if not isinstance(job_id, str) or not SAFE_ID.fullmatch(job_id) or job_id in seen or job_id == "_runner":
            raise ValueError(f"invalid or duplicate job_id: {job_id!r}")
        seen.add(job_id)
        seed = job.get("trace_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("trace_seed must be a nonnegative integer")
        child_path(backend, job["trace_path"], "trace_path")
        actions = job.get("actions")
        if not isinstance(actions, list) or len(actions) != 2:
            raise ValueError(f"{job_id}: exactly two actions are required")
        state = dict(baseline)
        families = []
        for action, expected_time in zip(actions, (180, 540)):
            if finite(action.get("time_s"), "action time_s") != expected_time:
                raise ValueError(f"{job_id}: action times must be 180 then 540")
            parameter = action.get("parameter", "")
            if parameter not in PARAMETERS or not parameter.startswith(("txp_", "ret_")):
                raise ValueError(f"{job_id}: scheduled parameter must be a cell TxP or RET")
            family = parameter.split("_")[0]
            families.append(family)
            value = finite(action.get("value"), "action value")
            before = float(state[parameter])
            if family == "txp" and not 0 <= value < before:
                raise ValueError(f"{job_id}: TxP action must strictly decrease within [0,60]")
            if family == "ret" and not before < value <= 90:
                raise ValueError(f"{job_id}: RET action must strictly increase within [0,90]")
            state[parameter] = value
        if set(families) != {"txp", "ret"}:
            raise ValueError(f"{job_id}: one TxP decrease and one RET increase are required")
        if job.get("order") != "_then_".join(families):
            raise ValueError(f"{job_id}: order does not match its actions")
    return manifest


def expected_parameters(manifest: dict, job: dict, timestamp: float) -> dict:
    state = dict(manifest["baseline_parameters"])
    for action in job["actions"]:
        if timestamp >= float(action["time_s"]):
            state[action["parameter"]] = float(action["value"])
    return state


def cache_for_angle(cache_root: Path, angle: float, cell: str) -> Path:
    if not float(angle).is_integer():
        raise ValueError("RET cache directory requires an exact integer angle")
    band = "3p5ghz" if cell == "5g" else "1p8ghz"
    return cache_root / f"tilt_{int(angle)}deg" / f"sionna_rt_cache_{band}_2x2_1x1_lzf.h5"


def initial_caches(manifest: dict) -> tuple[Path, Path]:
    backend = Path(manifest["backend_root"]).resolve()
    cache_root = child_path(backend, manifest["ret_cache_root"], "ret_cache_root")
    parameters = manifest["baseline_parameters"]
    low1, low2 = (float(parameters[f"ret_{cell}"]) for cell in ("4g1", "4g2"))
    if not low1.is_integer() or not low2.is_integer():
        raise ValueError("initial low-band cache angles must be exact integers")
    low = (cache_for_angle(cache_root, low1, "4g1") if low1 == low2 else
           cache_root / "merged_percell" / f"merged_1p8ghz_4g1_{int(low1)}deg_4g2_{int(low2)}deg.h5")
    return low, cache_for_angle(cache_root, parameters["ret_5g"], "5g")


def parameter_cli(parameters: dict) -> list[str]:
    def values(family: str) -> str:
        return ",".join(f"{label}:{parameters[f'{family}_{cell}']:g}" for cell, label, _ in CELLS)
    ttt, hys = parameters["ttt_ms"], parameters["hys_db"]
    return ["--cellTxPowerDbm=" + values("txp"), "--cellCioDb=" + values("cio"),
            "--cellRetTiltDeg=" + values("ret"),
            "--cellRetBearingDeg=gNB_5G:20,gNB_4G_1:0,gNB_4G_2:10",
            f"--handoverTtt={ttt:g}", f"--handoverHysteresis={hys:g}",
            "--cellTttMs=" + ",".join(f"{label}:{ttt:g}" for _, label, _ in CELLS),
            "--cellHysteresisDb=" + ",".join(f"{label}:{hys:g}" for _, label, _ in CELLS)]


def build_argv(manifest: dict, job: dict, schedule: Path) -> list[str]:
    backend = Path(manifest["backend_root"]).resolve()
    low_cache, mid_cache = initial_caches(manifest)
    cache_root = child_path(backend, manifest["ret_cache_root"], "ret_cache_root")
    return [str(child_path(backend, manifest["binary"], "binary")),
            "--centralFrequency18=1.8e9", "--bandwidth18=10e6", "--centralFrequency35=3.5e9",
            "--bandwidth35=10e6", "--numerology=0", "--sionnaCacheFile18=" + str(low_cache),
            "--sionnaCacheFile35=" + str(mid_cache), "--sionnaUpdatePeriod=1s",
            "--udpBaseIntervalMs=200", "--trafficMultiplierEnabled=true", "--simTime=900",
            "--kpiReportInterval=1.0", "--N_Ues=" + str(manifest["n_ues"]),
            "--sumoTrace=" + str(child_path(backend, job["trace_path"], "trace_path")),
            "--gnbPositions=" + str(child_path(backend, manifest.get("gnb_positions", "scenarios/khu-real/gnbs-ret.csv"), "gnb_positions")),
            "--resultsRoot=" + str(child_path(backend, manifest["output_root"], "output_root")), "--runTag=" + job["job_id"],
            "--RngSeed=" + str(manifest["rng_seed"]), "--RngRun=" + str(manifest["rng_run"]),
            "--ns3::NrHelper::E2ModeNr=false",
            "--parameterSchedule=" + str(schedule),
            "--parameterScheduleRetCacheRoot=" + str(cache_root),
            *parameter_cli(manifest["baseline_parameters"])]


def clean_environment(source: dict[str, str] | None = None, *, python_site_packages: Path | None = None) -> dict[str, str]:
    environment = dict(os.environ if source is None else source)
    for name in list(environment):
        if name.startswith(("AGORA_", "RIC_")):
            del environment[name]
    environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                       PYTHONDONTWRITEBYTECODE="1")
    if python_site_packages is not None:
        environment["PYTHONPATH"] = str(python_site_packages)
        environment.pop("PYTHONHOME", None)
    return environment


def memory_report() -> dict:
    path = Path("/proc/meminfo")
    if not path.is_file():
        return {"available": False, "reason": "/proc/meminfo unavailable"}
    fields = {}
    for line in path.read_text(encoding="ascii").splitlines():
        name, value = line.split(":", 1)
        fields[name] = int(value.strip().split()[0]) * 1024
    return {"available": True, "total_bytes": fields["MemTotal"],
            "available_bytes": fields.get("MemAvailable", fields["MemFree"]),
            "swap_total_bytes": fields.get("SwapTotal", 0),
            "swap_free_bytes": fields.get("SwapFree", 0)}


def process_memory(pid: int) -> dict:
    path = Path(f"/proc/{pid}/status")
    try:
        fields = {}
        for line in path.read_text(encoding="ascii").splitlines():
            name, _, value = line.partition(":")
            if name in ("VmRSS", "VmHWM"):
                fields[name] = int(value.strip().split()[0]) * 1024
        return {"rss_bytes": fields.get("VmRSS"), "peak_rss_bytes": fields.get("VmHWM")}
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return {"rss_bytes": None, "peak_rss_bytes": None}


def memory_launch_gate(memory: dict, min_free_gib: float, estimated_job_gib: float) -> dict:
    required = (min_free_gib + estimated_job_gib) * 1024 ** 3
    if not memory.get("available"):
        reason = "waiting_for_memory_measurement"
    elif memory["available_bytes"] < required:
        reason = "waiting_for_available_ram"
    else:
        reason = None
    return {"can_launch": reason is None, "reason": reason,
            "required_available_bytes": int(required),
            "projected_reserve_note": "MemAvailable must exceed reserve plus one estimated new job; this estimate is not an OOM guarantee."}


def read_manifest(path: Path) -> dict:
    return validate_manifest(json.loads(path.read_text(encoding="utf-8")))


def preflight(manifest: dict, manifest_path: Path | None = None, workers: int = 18,
              estimated_job_gib: float = 1.75) -> dict:
    validate_manifest(manifest)
    positive_integer(workers, "workers")
    if finite(estimated_job_gib, "estimated_job_gib") <= 0:
        raise ValueError("estimated_job_gib must be positive")
    backend = Path(manifest["backend_root"]).resolve()
    if not backend.is_dir():
        raise ValueError("backend_root does not exist")
    python_site = child_path(backend, manifest.get("python_site_packages", ".venv/lib/python3.12/site-packages"),
                             "python_site_packages")
    if not python_site.is_dir():
        raise FileNotFoundError(f"Embedded-Python site-packages directory is absent: {python_site}")
    output = child_path(backend, manifest["output_root"], "output_root")
    if output.exists():
        raise FileExistsError(f"Refusing existing output_root; preserve it and choose a new experiment: {output}")
    paths: dict[Path, set[str]] = {}

    def add(path: Path, role: str) -> None:
        resolved = path.resolve()
        if not resolved.is_relative_to(backend) or any(c.isspace() for c in str(resolved)):
            raise ValueError(f"input path escapes backend or contains whitespace: {path}")
        if resolved == output or resolved.is_relative_to(output):
            raise ValueError("input files must not reside under the new output root")
        if not resolved.is_file() or resolved.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty {role}: {resolved}")
        paths.setdefault(resolved, set()).add(role)

    binary = child_path(backend, manifest["binary"], "binary")
    add(binary, "binary")
    if os.name == "posix" and not os.access(binary, os.X_OK):
        raise ValueError("binary is not executable")
    sources = manifest.get("source_files", ["scratch/khu-real-dual-nr-sionna-final.cc"])
    if not isinstance(sources, list) or not sources:
        raise ValueError("source_files must be a nonempty list")
    for source in sources:
        add(child_path(backend, source, "source_files"), "source")
    add(child_path(backend, manifest.get("gnb_positions", "scenarios/khu-real/gnbs-ret.csv"), "gnb_positions"), "gnb_positions")
    low, mid = initial_caches(manifest)
    add(low, "initial_cache_18")
    add(mid, "initial_cache_35")
    cache_root = child_path(backend, manifest["ret_cache_root"], "ret_cache_root")
    plans = []
    for job in manifest["jobs"]:
        add(child_path(backend, job["trace_path"], "trace_path"), "trace:" + job["job_id"])
        actions = []
        state = dict(manifest["baseline_parameters"])
        for action in job["actions"]:
            item = {**action, "before": state[action["parameter"]]}
            state[action["parameter"]] = action["value"]
            item["parameters_after"] = dict(state)
            if action["parameter"].startswith("ret_"):
                cache = cache_for_angle(cache_root, action["value"], action["parameter"][4:])
                add(cache, "scheduled_ret_cache:" + job["job_id"])
                item["cache_path"] = str(cache.resolve())
            actions.append(item)
        schedule = output / "_runner" / job["job_id"] / "parameter_schedule.csv"
        plans.append({"job_id": job["job_id"], "trace_seed": job["trace_seed"],
                      "order": job["order"], "schedule": str(schedule),
                      "actions": actions, "argv": build_argv(manifest, job, schedule)})
    fingerprints = []
    expected = manifest.get("expected_sha256", {})
    if not isinstance(expected, dict):
        raise ValueError("expected_sha256 must map relative input paths to SHA256")
    for path, roles in sorted(paths.items(), key=lambda item: str(item[0])):
        relative = path.relative_to(backend).as_posix()
        value = sha256(path)
        if relative in expected and expected[relative] != value:
            raise ValueError(f"Frozen input SHA256 mismatch: {relative}")
        fingerprints.append({"path": str(path), "relative_path": relative,
                             "bytes": path.stat().st_size, "sha256": value, "roles": sorted(roles)})
    if set(expected) - {row["relative_path"] for row in fingerprints}:
        raise ValueError("expected_sha256 includes paths outside the recorded input set")
    memory = memory_report()
    estimated = min(workers, len(plans)) * estimated_job_gib * 1024 ** 3
    return {"schema": SCHEMA, "checked_at_utc": utc_now(), "preflight_passed": True,
            "simulation_launched": False, "experiment_id": manifest["experiment_id"],
            "manifest_sha256": sha256(manifest_path) if manifest_path else None,
            "manifest_content_sha256": canonical_hash(manifest),
            "runner_sha256": sha256(Path(__file__)), "output_root": str(output),
            "configured_workers": workers, "job_count": len(plans), "jobs": plans,
            "input_fingerprints": fingerprints, "memory": memory,
            "embedded_python_site_packages": str(python_site),
            "estimated_peak_memory_gib_per_job": estimated_job_gib,
            "estimated_peak_memory_bytes": int(estimated),
            "estimated_memory_note": "Planning estimate per concurrent job, not a measured bound. Insufficient RAM does not silently alter the configured worker limit.",
            "memory_warning": bool(memory.get("available") and estimated > memory["available_bytes"]),
            "environment_policy": "Remove AGORA_/RIC_ and PYTHONHOME; replace PYTHONPATH with verified backend site-packages; E2=false; OMP/MKL/OPENBLAS threads=1.",
            "interface_policy": "No RIC/LLM/model process; omit guiSrc/influxSrc to retain audited disabled defaults (ns-3 rejects empty string CLI values); E2ModeNr=false.",
            "cache_validation": "Exact-angle file existence and SHA256 only; HDF5 angle metadata and grid coverage require the independent cache audit."}


def last_cell_time(directory: Path) -> float | None:
    path = directory / "cell_kpi.csv"
    try:
        with path.open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            size = handle.tell()
            handle.seek(max(0, size - 32768))
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
        for line in reversed(lines):
            try:
                value = float(line.split(",", 1)[0])
                if math.isfinite(value):
                    return value
            except ValueError:
                continue
    except FileNotFoundError:
        pass
    return None


def verify_job(manifest: dict, job: dict, directory: Path) -> dict:
    """Validate applied receipts and every cell's full cumulative configuration."""
    issues = []
    receipt = directory / "parameter_changes.csv"
    records = []
    try:
        with receipt.open(encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != RECEIPT_FIELDS:
                issues.append("unexpected_parameter_changes_header")
            records = list(reader)
        if len(records) != 2:
            issues.append(f"expected_two_receipts_got_{len(records)}")
        state = dict(manifest["baseline_parameters"])
        for index, action in enumerate(job["actions"]):
            if index >= len(records):
                break
            row = records[index]
            expected = {"scheduled_time_s": action["time_s"], "applied_time_s": action["time_s"],
                        "before": state[action["parameter"]], "after": action["value"]}
            if row.get("parameter") != action["parameter"] or row.get("status") != "applied":
                issues.append(f"receipt_{index}_parameter_or_status_mismatch")
            for field, value in expected.items():
                observed = float(row[field])
                if not math.isfinite(observed) or abs(observed - float(value)) > 1e-6:
                    issues.append(f"receipt_{index}_{field}_mismatch")
            state[action["parameter"]] = action["value"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        issues.append("receipt_read_error:" + str(error))

    cell_path = directory / "cell_kpi.csv"
    observed_grid: set[tuple[int, int]] = set()
    row_count, last_time = 0, None
    try:
        with cell_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                row_count += 1
                timestamp, cell_id = float(row["time_s"]), int(row["cell_id"])
                if not math.isfinite(timestamp):
                    raise ValueError("nonfinite timestamp")
                slot = round(timestamp - 0.5)
                if cell_id not in (1, 2, 3) or not 0 <= slot < 900 or abs(timestamp - (slot + .5)) > 1e-6:
                    raise ValueError(f"unexpected cell/time at row {row_count}")
                key = (cell_id, slot)
                if key in observed_grid:
                    raise ValueError(f"duplicate cell/time at row {row_count}")
                observed_grid.add(key)
                last_time = timestamp if last_time is None else max(timestamp, last_time)
                cell = CELLS[cell_id - 1][0]
                expected = expected_parameters(manifest, job, timestamp)
                for column, parameter in (("tx_power_dbm", f"txp_{cell}"), ("ret_tilt_deg", f"ret_{cell}"),
                                          ("cio_bias_db", f"cio_{cell}"), ("ttt_ms", "ttt_ms"),
                                          ("hysteresis_db", "hys_db")):
                    value = float(row[column])
                    if not math.isfinite(value) or abs(value - float(expected[parameter])) > 1e-6:
                        raise ValueError(f"row {row_count} {parameter}: observed {value}, expected {expected[parameter]}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        issues.append("cell_csv_error:" + str(error))
    complete = len(observed_grid) == 2700 and row_count == 2700
    if not complete:
        issues.append(f"incomplete_900s_cell_grid:{len(observed_grid)}/2700")
    ue_path = directory / "ue_kpi.csv"
    if not ue_path.is_file() or ue_path.stat().st_size == 0:
        issues.append("missing_or_empty_ue_kpi_csv")
    artifacts = {path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
                 for path in (cell_path, ue_path, receipt) if path.is_file()}
    return {"job_id": job["job_id"], "verified_at_utc": utc_now(), "passed": not issues,
            "issues": issues, "cell_rows": row_count, "last_cell_time_s": last_time,
            "complete_900s_cell_grid": complete, "receipt_count": len(records),
            "parameters_after_all_actions": expected_parameters(manifest, job, 900),
            "artifacts": artifacts}


def stop_owned_process(process: subprocess.Popen, grace_s: float = 10) -> None:
    """Only stop the process group that this invocation created with setsid."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=grace_s)


def launch_job(manifest: dict, job: dict, root: Path, provenance: dict,
               popen_factory=subprocess.Popen) -> dict:
    metadata = root / "_runner" / job["job_id"]
    metadata.mkdir(parents=True, exist_ok=False)
    output = root / job["job_id"]
    if output.exists():
        raise FileExistsError(f"Job output unexpectedly exists: {output}")
    schedule = metadata / "parameter_schedule.csv"
    with schedule.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "parameter", "value"))
        writer.writerows((action["time_s"], action["parameter"], action["value"]) for action in job["actions"])
    argv = build_argv(manifest, job, schedule)
    record = {"job_id": job["job_id"], "trace_seed": job["trace_seed"], "order": job["order"],
              "status": "launching", "started_at_utc": utc_now(), "linux_pid": None,
              "argv": argv, "cwd": str(Path(manifest["backend_root"]).resolve()),
              "schedule_sha256": sha256(schedule), "manifest_content_sha256": canonical_hash(manifest),
              "provenance_path": str(root / "preflight.json"),
              "input_fingerprints_content_sha256": canonical_hash(provenance["input_fingerprints"]),
              "stdout_log": str(metadata / "stdout.log"), "stderr_log": str(metadata / "stderr.log"),
              "output_directory": str(output), "timed_out": False}
    atomic_json(metadata / "execution.json", record)
    stdout = (metadata / "stdout.log").open("xb", buffering=0)
    try:
        stderr = (metadata / "stderr.log").open("xb", buffering=0)
    except BaseException:
        stdout.close()
        raise
    try:
        python_site = child_path(Path(manifest["backend_root"]).resolve(),
                                  manifest.get("python_site_packages", ".venv/lib/python3.12/site-packages"),
                                  "python_site_packages")
        process = popen_factory(argv, cwd=manifest["backend_root"],
                                env=clean_environment(python_site_packages=python_site),
                                stdout=stdout, stderr=stderr, start_new_session=True)
    except BaseException as error:
        stdout.close()
        stderr.close()
        record.update(status="launch_failed", error=str(error), finished_at_utc=utc_now())
        atomic_json(metadata / "execution.json", record)
        raise
    record.update(status="running", linux_pid=process.pid)
    try:
        atomic_json(metadata / "execution.json", record)
    except BaseException:
        stop_owned_process(process)
        stdout.close()
        stderr.close()
        raise
    return {"process": process, "stdout": stdout, "stderr": stderr, "record": record,
            "metadata": metadata, "output": output, "job": job,
            "started_monotonic": time.monotonic(), "peak_rss_bytes": 0}


def finish_job(manifest: dict, running: dict, interrupted: bool = False) -> dict:
    process, record = running["process"], running["record"]
    running["stdout"].close()
    running["stderr"].close()
    record.update(returncode=process.poll(), finished_at_utc=utc_now(),
                  wall_seconds=time.monotonic() - running["started_monotonic"],
                  peak_rss_bytes_observed=running["peak_rss_bytes"], interrupted=interrupted)
    verification = verify_job(manifest, running["job"], running["output"])
    record["verification"] = verification
    record["status"] = ("interrupted" if interrupted else "timed_out" if record["timed_out"] else
                        "complete" if record["returncode"] == 0 and verification["passed"] else "failed")
    atomic_json(running["metadata"] / "execution.json", record)
    return record


def run_batch(manifest: dict, manifest_path: Path, *, workers: int = 18, timeout_s: float = 172800,
              min_free_gib: float = 3, estimated_job_gib: float = 1.75,
              startup_stagger_s: float = 10, poll_s: float = 1) -> dict:
    if sys.platform != "linux":
        raise RuntimeError("run is supported only on Linux; preflight and mocked unit tests do not launch ns-3")
    for name, value in (("timeout", timeout_s), ("min_free_gib", min_free_gib),
                        ("estimated_job_gib", estimated_job_gib),
                        ("startup_stagger_s", startup_stagger_s), ("poll_s", poll_s)):
        value = finite(value, name)
        if value < 0 or (name in ("timeout", "poll_s", "estimated_job_gib") and value == 0):
            raise ValueError(f"invalid {name}")
    # Every path, action and input digest is validated before the first process.
    provenance = preflight(manifest, manifest_path, workers, estimated_job_gib)
    root = Path(provenance["output_root"])
    root.mkdir(parents=True, exist_ok=False)
    (root / "_runner").mkdir()
    atomic_json(root / "manifest.json", manifest)
    atomic_json(root / "preflight.json", provenance)
    pending = list(manifest["jobs"])
    running: dict[str, dict] = {}
    completed = []
    next_launch = time.monotonic()
    started_utc, started = utc_now(), time.monotonic()
    state = {"experiment_id": manifest["experiment_id"], "runner_pid": os.getpid(),
             "manifest_content_sha256": canonical_hash(manifest),
             "started_at_utc": started_utc, "configured_workers": workers,
             "timeout_seconds_per_job": timeout_s, "minimum_available_memory_gib": min_free_gib,
             "estimated_new_job_memory_gib": estimated_job_gib,
             "memory_admission_required_available_gib": min_free_gib + estimated_job_gib,
             "startup_stagger_s": startup_stagger_s, "status": "running"}

    def save_state(reason: str | None = None) -> None:
        state.update(updated_at_utc=utc_now(), wall_seconds=time.monotonic() - started,
                     pending_job_ids=[job["job_id"] for job in pending],
                     running_jobs=[{**item["record"], "wall_seconds": time.monotonic() - item["started_monotonic"],
                                    "last_cell_time_s": last_cell_time(item["output"]),
                                    **process_memory(item["process"].pid)} for item in running.values()],
                     completed_jobs=completed, memory=memory_report(), launch_wait_reason=reason)
        atomic_json(root / "batch_status.json", state)

    save_state()
    prior_handlers = {number: signal.getsignal(number) for number in (signal.SIGTERM, signal.SIGINT)}
    stop_signal = None

    def request_stop(signum, frame):
        # A flag avoids interrupting Popen between child creation and ownership
        # bookkeeping. The main loop then stops only its tracked process groups.
        nonlocal stop_signal
        stop_signal = signum

    for number in prior_handlers:
        signal.signal(number, request_stop)
    try:
        while pending or running:
            if stop_signal is not None:
                raise KeyboardInterrupt(f"Signal {stop_signal} requested batch stop")
            for job_id, item in list(running.items()):
                memory = process_memory(item["process"].pid)
                item["peak_rss_bytes"] = max(item["peak_rss_bytes"], memory.get("peak_rss_bytes") or 0)
                if item["process"].poll() is None and time.monotonic() - item["started_monotonic"] >= timeout_s:
                    item["record"]["timed_out"] = True
                    stop_owned_process(item["process"])
                if item["process"].poll() is not None:
                    completed.append(finish_job(manifest, item))
                    del running[job_id]
            wait_reason = None
            if pending and len(running) < workers:
                memory = memory_report()
                gate = memory_launch_gate(memory, min_free_gib, estimated_job_gib)
                if not gate["can_launch"]:
                    wait_reason = gate["reason"]
                elif time.monotonic() < next_launch:
                    wait_reason = "startup_stagger"
                else:
                    job = pending.pop(0)
                    try:
                        item = launch_job(manifest, job, root, provenance)
                        running[job["job_id"]] = item
                    except Exception as error:
                        completed.append({"job_id": job["job_id"], "status": "launch_failed",
                                          "error": str(error), "finished_at_utc": utc_now()})
                    next_launch = time.monotonic() + startup_stagger_s
            save_state(wait_reason)
            if pending or running:
                time.sleep(poll_s)
    except BaseException as error:
        for item in running.values():
            stop_owned_process(item["process"])
        for item in running.values():
            completed.append(finish_job(manifest, item, interrupted=True))
        running.clear()
        state.update(status="interrupted", error=f"{type(error).__name__}: {error}", finished_at_utc=utc_now())
        save_state()
        raise
    finally:
        for number, handler in prior_handlers.items():
            signal.signal(number, handler)
    state.update(status="complete" if all(row["status"] == "complete" for row in completed) else "failed",
                 finished_at_utc=utc_now())
    save_state()
    return state


def status(manifest: dict) -> dict:
    root = child_path(Path(manifest["backend_root"]).resolve(), manifest["output_root"], "output_root")
    path = root / "batch_status.json"
    if not path.is_file():
        return {"experiment_id": manifest["experiment_id"], "status": "not_started", "output_root": str(root)}
    result = json.loads(path.read_text(encoding="utf-8"))
    result["queried_at_utc"] = utc_now()
    result["memory_now"] = memory_report()
    result["runner_pid_exists_now"] = Path(f"/proc/{result['runner_pid']}").exists()
    for row in result.get("running_jobs", []):
        row["last_cell_time_s"] = last_cell_time(root / row["job_id"])
        row["process_pid_exists_now"] = Path(f"/proc/{row['linux_pid']}").exists()
        row.update(process_memory(row["linux_pid"]))
    return result


def verify(manifest: dict) -> dict:
    root = child_path(Path(manifest["backend_root"]).resolve(), manifest["output_root"], "output_root")
    jobs = [verify_job(manifest, job, root / job["job_id"]) for job in manifest["jobs"]]
    batch_issues = []
    try:
        launched_manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        if canonical_hash(launched_manifest) != canonical_hash(manifest):
            batch_issues.append("requested_manifest_differs_from_launch_manifest")
    except (OSError, ValueError) as error:
        batch_issues.append("launch_manifest_unavailable:" + str(error))
    for row in jobs:
        path = root / "_runner" / row["job_id"] / "execution.json"
        try:
            execution = json.loads(path.read_text(encoding="utf-8"))
            row["process_returncode"] = execution.get("returncode")
            row["execution_status"] = execution.get("status")
            if execution.get("returncode") != 0 or execution.get("status") != "complete":
                row["issues"].append("execution_not_successfully_completed")
        except (OSError, ValueError) as error:
            row["issues"].append("execution_record_unavailable:" + str(error))
        row["passed"] = not row["issues"]
    return {"experiment_id": manifest["experiment_id"], "verified_at_utc": utc_now(),
            "all_passed": not batch_issues and all(row["passed"] for row in jobs),
            "batch_issues": batch_issues, "jobs": jobs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "run", "status", "verify"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=18)
    parser.add_argument("--timeout", type=float, default=172800)
    parser.add_argument("--min-free-gib", type=float, default=3)
    parser.add_argument("--estimated-job-gib", type=float, default=1.75)
    parser.add_argument("--startup-stagger-s", type=float, default=10)
    args = parser.parse_args()
    try:
        manifest = read_manifest(args.manifest)
        if args.command == "preflight":
            result = preflight(manifest, args.manifest, args.workers, args.estimated_job_gib)
        elif args.command == "run":
            result = run_batch(manifest, args.manifest, workers=args.workers, timeout_s=args.timeout,
                               min_free_gib=args.min_free_gib, estimated_job_gib=args.estimated_job_gib,
                               startup_stagger_s=args.startup_stagger_s)
        elif args.command == "status":
            result = status(manifest)
        else:
            result = verify(manifest)
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        if (args.command == "run" and result["status"] != "complete") or (args.command == "verify" and not result["all_passed"]):
            raise SystemExit(1)
    except (OSError, ValueError, RuntimeError) as error:
        print(json.dumps({"error": str(error), "command": args.command}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
