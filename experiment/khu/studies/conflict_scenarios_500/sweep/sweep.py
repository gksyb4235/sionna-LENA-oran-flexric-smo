"""Portable Linux/WSL fixed-parameter ns-3 sweep. Only `run` launches simulations.

Python 3.10+ standard library; ns-3 embeds its own configured Python dependencies.
Every attempt is immutable. Successful jobs are verified and skipped on resume.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
from typing import Any
from resource_limits import cpu_quota, memory_available

SCHEMA = "agora-conflict-sweep-v1"
CELLS = (("5g", "gNB_5G", 1), ("4g1", "gNB_4G_1", 2), ("4g2", "gNB_4G_2", 3))
PARAMETERS = tuple(f"{family}_{cell}" for family in ("txp", "ret", "cio")
                   for cell, _, _ in CELLS) + ("ttt_ms", "hys_db")
DOMAINS = {**{f"txp_{c}": (30, 33, 37, 40, 43) for c, _, _ in CELLS},
           **{f"ret_{c}": (3, 5, 7, 9, 12) for c, _, _ in CELLS},
           "cio_5g": (5, 6, 7, 8, 9), "cio_4g1": (-2, -1, 0, 1, 2),
           "cio_4g2": (-2, -1, 0, 1, 2), "ttt_ms": (128, 256, 340, 512, 720),
           "hys_db": tuple(i / 2 for i in range(1, 11))}
FAMILIES = {"ES": {"txp"}, "TM": {"txp"}, "CV": {"ret"}, "LB": {"cio"}, "MO": {"ttt", "hys"}}
SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}\Z")
RECEIPT_FIELDS = ("scheduled_time_s", "applied_time_s", "parameter", "before", "after", "status")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode("utf-8")).hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label}: expected a finite JSON number")
    return float(value)


def contained(root: Path, value: str, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute() or ".." in Path(value).parts or "\\" in value:
        raise ValueError(f"{label}: expected a relative path inside ns3_root")
    # Resolve recorded pre-layout configs without rewriting frozen contracts.
    # CPU-v1 Docker images intentionally keep their original internal layout.
    path = Path(value)
    canonical_studies = root / "experiment/khu/studies"
    if path.parts and path.parts[0] == "experiments" and canonical_studies.is_dir():
        path = Path("experiment/khu/studies", *path.parts[1:])
    result = (root / path).resolve()
    if result == root or not result.is_relative_to(root):
        raise ValueError(f"{label}: resolves outside ns3_root")
    return result


def validate_manifest(manifest: dict) -> dict:
    if manifest.get("schema") != SCHEMA or not SAFE_ID.fullmatch(manifest.get("experiment_id", "")):
        raise ValueError("Unsupported manifest schema or experiment_id")
    if manifest.get("duration_s") != 900 or manifest.get("kpi_interval_s") != 1:
        raise ValueError("This sweep requires 900 seconds and one-second KPI reports")
    for field in ("n_ues", "rng_seed", "rng_run"):
        value = manifest.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{field}: positive integer required")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Manifest has no jobs")
    seen = set()
    for job in jobs:
        ident = job.get("job_id", "")
        if not SAFE_ID.fullmatch(ident) or ident in seen:
            raise ValueError(f"Invalid or duplicated job_id {ident!r}")
        seen.add(ident)
        seed = job.get("trace_seed")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError(f"{ident}: invalid trace_seed")
        if job.get("initial_app") not in FAMILIES:
            raise ValueError(f"{ident}: invalid initial_app")
        state = dict(job["initial_parameters"])
        if set(state) != set(PARAMETERS):
            raise ValueError(f"{ident}: exactly eleven initial parameters required")
        for p, v in state.items():
            if finite(v, p) not in DOMAINS[p]:
                raise ValueError(f"{ident}: {p}={v} outside frozen parameter domain")
        events = job.get("events")
        if not isinstance(events, list) or [x.get("time_s") for x in events] not in ([180, 540], [180, 360, 540, 720]):
            raise ValueError(f"{ident}: expected two- or four-switch event schedule")
        for event in events:
            app, updates = event.get("app"), event.get("updates")
            if app not in FAMILIES or not isinstance(updates, dict) or not updates:
                raise ValueError(f"{ident}: invalid app or empty event")
            for p, v in updates.items():
                if p not in DOMAINS or p.split("_")[0] not in FAMILIES[app]:
                    raise ValueError(f"{ident}: parameter outside {app} control family")
                v = finite(v, p)
                if v not in DOMAINS[p] or v == state[p]:
                    raise ValueError(f"{ident}: invalid domain or no-op for {p}")
                if app == "ES" and v >= state[p] or app == "TM" and v <= state[p]:
                    raise ValueError(f"{ident}: wrong TxP direction")
                state[p] = v
    return manifest


def validate_config(config: dict) -> dict:
    root = Path(config["ns3_root"]).expanduser().resolve()
    if not Path(config["ns3_root"]).expanduser().is_absolute():
        raise ValueError("ns3_root must be absolute")
    for field in ("binary", "output_root", "ret_cache_root", "gnb_positions", "python_site_packages"):
        contained(root, config[field], field)
    python = config.get("python_executable", ".venv/bin/python")
    if not isinstance(python, str) or Path(python).is_absolute() or ".." in Path(python).parts or "\\" in python:
        raise ValueError("python_executable must be a relative path (venv symlink to system Python is allowed)")
    output = contained(root, config["output_root"], "output_root")
    for field in ("binary", "ret_cache_root", "gnb_positions", "python_site_packages"):
        target = contained(root, config[field], field)
        if target == output or target.is_relative_to(output) or output.is_relative_to(target):
            raise ValueError(f"output_root and input {field} overlap")
    sources = config.get("source_files")
    if not isinstance(sources, list) or not sources:
        raise ValueError("source_files must list the installed backend and modified library sources")
    for source in sources:
        contained(root, source, "source")
    for key in ("memory_reserve_gib", "estimated_job_gib", "disk_reserve_gib",
                "estimated_output_gib_per_job", "startup_stagger_s", "timeout_s"):
        value = finite(config[key], key)
        if value < 0 or key in ("estimated_job_gib", "estimated_output_gib_per_job", "timeout_s") and value == 0:
            raise ValueError(f"Invalid {key}")
    return config


def selected_jobs(manifest: dict, index: int, count: int, max_jobs: int | None = None,
                  tier: str = "all", pilot_ids: list[str] | None = None) -> list[dict]:
    if count < 1 or not 0 <= index < count or max_jobs is not None and max_jobs < 1:
        raise ValueError("Invalid shard index/count or max-jobs")
    if pilot_ids is not None and (len(set(pilot_ids)) != len(pilot_ids) or
                                  set(pilot_ids) - {j["job_id"] for j in manifest["jobs"]}):
        raise ValueError("Pilot IDs contain duplicates or unknown jobs")
    ordered = sorted(manifest["jobs"], key=lambda j: (j.get("screening_priority", 10**9), j["job_id"]))
    selected = [job for i, job in enumerate(ordered) if i % count == index
                and (tier == "all" or job.get("screening_tier") == tier)
                and (pilot_ids is None or job["job_id"] in pilot_ids)]
    return selected if max_jobs is None else selected[:max_jobs]


def jobs_in_range(manifest: dict, jobs: list[dict], start: int, end: int) -> list[dict]:
    """Select inclusive numeric ID suffixes, never CSV row positions or trace seeds."""
    known = {int(j["job_id"].rsplit("_", 1)[1]) for j in manifest["jobs"]}
    if start < 1 or end < start or end > max(known) or any(i not in known for i in range(start, end + 1)):
        raise ValueError(f"Scenario range must contain existing IDs {min(known)}..{max(known)}, inclusive")
    return sorted([j for j in jobs if start <= int(j["job_id"].rsplit("_", 1)[1]) <= end],
                  key=lambda j: int(j["job_id"].rsplit("_", 1)[1]))


def cpu_slots(cores_per_job: int, workers: int) -> list[list[int]]:
    """Allocate disjoint slots from the process's allowed logical CPU IDs."""
    if cores_per_job < 1 or workers < 1:
        raise ValueError("cores-per-job and workers must be positive")
    if not hasattr(os, "sched_getaffinity") or not shutil.which("taskset"):
        raise RuntimeError("CPU pinning requires Linux/WSL sched_getaffinity and taskset")
    available = sorted(os.sched_getaffinity(0))
    required = cores_per_job * workers
    quota = cpu_quota()
    if quota is not None and required > quota:
        raise ValueError(f"Requested {required} CPUs exceeds cgroup CPU quota {quota:g}")
    if required > len(available):
        raise ValueError(f"Requested {required} logical CPUs ({cores_per_job} x {workers}), only {len(available)} allowed")
    return [available[i * cores_per_job:(i + 1) * cores_per_job] for i in range(workers)]


def schedule_rows(job: dict) -> list[tuple]:
    return [(event["time_s"], p, event["updates"][p]) for event in job["events"]
            for p in PARAMETERS if p in event["updates"]]


def expected_state(job: dict, timestamp: float) -> dict:
    state = dict(job["initial_parameters"])
    for event in job["events"]:
        if timestamp >= event["time_s"]:
            state.update(event["updates"])
    return state


def cache_file(root: Path, angle: float, cell: str) -> Path:
    band = "3p5ghz" if cell == "5g" else "1p8ghz"
    return root / f"tilt_{int(angle)}deg" / f"sionna_rt_cache_{band}_2x2_1x1_lzf.h5"


def initial_caches(cache_root: Path, job: dict) -> tuple[Path, Path]:
    p = job["initial_parameters"]
    a, b = int(p["ret_4g1"]), int(p["ret_4g2"])
    low = cache_file(cache_root, a, "4g1") if a == b else cache_root / "merged_percell" / f"merged_1p8ghz_4g1_{a}deg_4g2_{b}deg.h5"
    return low, cache_file(cache_root, p["ret_5g"], "5g")


def argv_for(manifest: dict, config: dict, job: dict, attempt: Path) -> list[str]:
    root = Path(config["ns3_root"]).expanduser().resolve()
    cache_root = contained(root, config["ret_cache_root"], "ret_cache_root")
    low, mid = initial_caches(cache_root, job)
    p = job["initial_parameters"]
    def triples(family):
        return ",".join(f"{label}:{p[f'{family}_{cell}']:g}" for cell, label, _ in CELLS)
    def shared(value):
        return ",".join(f"{label}:{value:g}" for _, label, _ in CELLS)
    return [str(contained(root, config["binary"], "binary")),
            "--centralFrequency18=1.8e9", "--bandwidth18=10e6", "--centralFrequency35=3.5e9",
            "--bandwidth35=10e6", "--numerology=0", f"--sionnaCacheFile18={low}",
            f"--sionnaCacheFile35={mid}", "--sionnaUpdatePeriod=1s",
            "--udpBaseIntervalMs=200", "--trafficMultiplierEnabled=true", "--simTime=900",
            "--kpiReportInterval=1.0", f"--N_Ues={manifest['n_ues']}",
            "--sumoTrace=" + str(contained(root, job["trace_path"], "trace_path")),
            "--gnbPositions=" + str(contained(root, config["gnb_positions"], "gnb_positions")),
            f"--resultsRoot={attempt / 'data'}", f"--runTag={job['job_id']}",
            f"--RngSeed={manifest['rng_seed']}", f"--RngRun={manifest['rng_run']}",
            "--ns3::NrHelper::E2ModeNr=false", "--ns3::NrHelper::UseIdealRrc=true",
            f"--parameterSchedule={attempt / 'parameter_schedule.csv'}",
            f"--parameterScheduleRetCacheRoot={cache_root}",
            "--cellTxPowerDbm=" + triples("txp"), "--cellCioDb=" + triples("cio"),
            "--cellRetTiltDeg=" + triples("ret"),
            "--cellRetBearingDeg=gNB_5G:20,gNB_4G_1:0,gNB_4G_2:10",
            f"--handoverTtt={p['ttt_ms']:g}", f"--handoverHysteresis={p['hys_db']:g}",
            "--cellTttMs=" + shared(p["ttt_ms"]), "--cellHysteresisDb=" + shared(p["hys_db"])]


def clean_environment(config: dict) -> dict:
    result = {k: v for k, v in os.environ.items() if not k.startswith(("AGORA_", "DRL_", "RIC_"))}
    root = Path(config["ns3_root"]).expanduser().resolve()
    result.pop("PYTHONHOME", None)
    result.update(PYTHONPATH=str(contained(root, config["python_site_packages"], "python_site_packages")),
                  OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                  NUMEXPR_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1", PYTHONDONTWRITEBYTECODE="1")
    return result


def existing_parent(path: Path) -> Path:
    while not path.exists():
        path = path.parent
    return path


def preflight(manifest: dict, config: dict, jobs: list[dict], workers: int) -> dict:
    validate_manifest(manifest)
    validate_config(config)
    root = Path(config["ns3_root"]).expanduser().resolve()
    output = contained(root, config["output_root"], "output_root")
    cache_root = contained(root, config["ret_cache_root"], "ret_cache_root")
    paths: dict[Path, set[str]] = {}
    def add(path: Path, role: str):
        path = path.resolve()
        if not path.is_relative_to(root) or path.is_relative_to(output):
            raise ValueError(f"Input escapes root or lies in output: {path}")
        if not path.is_file() or path.stat().st_size == 0:
            raise FileNotFoundError(f"Missing or empty {role}: {path}")
        paths.setdefault(path, set()).add(role)
    binary = contained(root, config["binary"], "binary")
    add(binary, "binary")
    if sys.platform == "linux" and not os.access(binary, os.X_OK):
        raise ValueError("ns-3 binary is not executable")
    for source in config["source_files"]:
        add(contained(root, source, "source"), "source")
    add(contained(root, config["gnb_positions"], "gnb_positions"), "gnb_positions")
    site = contained(root, config["python_site_packages"], "python_site_packages")
    if not site.is_dir():
        raise FileNotFoundError(f"Embedded Python dependencies missing: {site}")
    for job in jobs:
        add(contained(root, job["trace_path"], "trace_path"), "trace")
        for path in initial_caches(cache_root, job):
            add(path, "initial_cache")
    # Backend eagerly validates the entire exact-angle cache registry.
    for angle in (3, 5, 7, 9, 12):
        for cell in ("5g", "4g1"):
            add(cache_file(cache_root, angle, cell), "ret_cache")
    expected = config.get("expected_sha256", {})
    fingerprints = []
    for path, roles in sorted(paths.items(), key=lambda pair: str(pair[0])):
        relative = path.relative_to(root).as_posix()
        digest = sha256(path)
        if relative in expected and expected[relative] != digest:
            raise ValueError(f"SHA256 mismatch: {relative}")
        fingerprints.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size,
                             "roles": sorted(roles)})
    unknown_expected = set(expected) - {f["path"] for f in fingerprints}
    if unknown_expected:
        raise ValueError("expected_sha256 includes inputs outside selected shard")
    abi = {"checked": False, "reason": "Linux ldd unavailable"}
    if sys.platform == "linux" and shutil.which("ldd"):
        result = subprocess.run(["ldd", str(binary)], capture_output=True, text=True, timeout=30)
        abi = {"checked": True, "returncode": result.returncode,
               "output": (result.stdout + result.stderr).strip()}
        if result.returncode != 0 or "not found" in abi["output"]:
            raise RuntimeError("Binary ABI check failed: " + abi["output"])
        libraries = set()
        for line in abi["output"].splitlines():
            match = re.search(r"(?:=>\s+)?(/\S+)\s+\(", line)
            if match:
                library = Path(match.group(1)).resolve()
                if library.is_file():
                    libraries.add(library)
        abi["linked_library_count"] = len(libraries)
        for library in sorted(libraries):
            name = library.relative_to(root).as_posix() if library.is_relative_to(root) else "@system_library:" + str(library)
            if not any(row["path"] == name for row in fingerprints):
                fingerprints.append({"path": name, "sha256": sha256(library),
                                     "bytes": library.stat().st_size, "roles": ["linked_library"]})
    cache_audit = {"checked": False, "reason": "Linux/WSL required for configured embedded Python environment"}
    if sys.platform == "linux":
        audit_script = Path(__file__).resolve().parent / "backend" / "cache_tools.py"
        python = root / config.get("python_executable", ".venv/bin/python")
        if not audit_script.is_file() or not python.is_file():
            raise FileNotFoundError("Backend cache audit helper or configured venv Python is missing")
        with tempfile.TemporaryDirectory(prefix="agora-cache-audit-") as temp:
            target = Path(temp) / "audit.json"
            result = subprocess.run([str(python), str(audit_script), "--cache-root", str(cache_root),
                                     "--gnb-positions", str(contained(root, config["gnb_positions"], "gnb_positions")),
                                     "--output", str(target)], capture_output=True, text=True, timeout=120,
                                    env=clean_environment(config))
            if result.returncode != 0:
                raise RuntimeError("RET cache metadata validation failed: " + (result.stdout + result.stderr)[-5000:])
            cache_audit = read_json(target)
    free = shutil.disk_usage(existing_parent(output)).free
    estimated = (config["disk_reserve_gib"] + len(jobs) * config["estimated_output_gib_per_job"]) * 1024 ** 3
    if free < estimated:
        raise RuntimeError(f"Insufficient free disk: {free / 1024**3:.1f} GiB; planning requirement {estimated / 1024**3:.1f} GiB")
    available = memory_available()
    return {"preflight_passed": True, "simulation_launched": False, "at_utc": utc_now(),
            "hostname": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version, "manifest_hash": canonical_hash(manifest),
            "runner_sha256": sha256(Path(__file__)), "input_fingerprints": fingerprints,
            "abi": abi, "selected_jobs": len(jobs), "workers": workers,
            "available_memory_bytes": available, "free_disk_bytes": free,
            "memory_estimate_bytes": min(workers, len(jobs)) * config["estimated_job_gib"] * 1024 ** 3,
            "resource_estimates_are_not_guarantees": True,
            "cache_validation": cache_audit,
            "mode": "No actual xApp, RIC or LLM; predetermined absolute parameter schedule; Ideal RRC."}


def last_cell_time(directory: Path) -> float | None:
    try:
        with (directory / "cell_kpi.csv").open("rb") as handle:
            handle.seek(0, os.SEEK_END)
            handle.seek(max(0, handle.tell() - 32768))
            for line in reversed(handle.read().decode("utf-8", errors="replace").splitlines()):
                try:
                    value = float(line.split(",", 1)[0])
                    if math.isfinite(value):
                        return value
                except ValueError:
                    continue
    except OSError:
        pass
    return None


def verify_job(job: dict, directory: Path, hashes: bool = True) -> dict:
    issues, records = [], []
    receipt = directory / "parameter_changes.csv"
    try:
        with receipt.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != RECEIPT_FIELDS:
                raise ValueError("Unexpected receipt columns")
            records = list(reader)
        expected = schedule_rows(job)
        if len(records) != len(expected):
            raise ValueError(f"Expected {len(expected)} receipts, got {len(records)}")
        # The backend stores a grouped event in std::map, so fields may be
        # emitted lexically rather than in input CSV order. Order inside the
        # same atomic timestamp is immaterial; duplicates remain an error.
        indexed = {}
        previous_time = -math.inf
        for row in records:
            when = float(row["scheduled_time_s"])
            if not math.isfinite(when) or when < previous_time:
                raise ValueError("Nonfinite or decreasing receipt timestamp")
            previous_time = when
            key = (when, row["parameter"])
            if key in indexed:
                raise ValueError(f"Duplicate receipt key {key}")
            indexed[key] = row
        if set(indexed) != {(float(when), p) for when, p, _ in expected}:
            raise ValueError("Receipt timestamp/parameter keys differ from scheduled updates")
        state = dict(job["initial_parameters"])
        for i, (when, parameter, value) in enumerate(expected):
            row = indexed[(float(when), parameter)]
            if row["parameter"] != parameter or row["status"] != "applied":
                raise ValueError(f"Receipt {i} wrong parameter/status")
            for key, wanted in (("scheduled_time_s", when), ("applied_time_s", when),
                                ("before", state[parameter]), ("after", value)):
                observed = float(row[key])
                if not math.isfinite(observed) or abs(observed - wanted) > 1e-6:
                    raise ValueError(f"Receipt {i} {key}: {observed} != {wanted}")
            state[parameter] = value
    except (OSError, ValueError, KeyError, TypeError) as error:
        issues.append("receipt_error:" + str(error))
    cell_path = directory / "cell_kpi.csv"
    grid, row_count, last = set(), 0, None
    try:
        with cell_path.open(encoding="utf-8-sig", newline="") as handle:
            for row in csv.DictReader(handle):
                row_count += 1
                timestamp, cell_id = float(row["time_s"]), int(row["cell_id"])
                if not math.isfinite(timestamp):
                    raise ValueError("Nonfinite timestamp")
                slot = round(timestamp - 0.5)
                key = (cell_id, slot)
                if cell_id not in (1, 2, 3) or not 0 <= slot < 900 or abs(timestamp - slot - .5) > 1e-6 or key in grid:
                    raise ValueError(f"Unexpected or duplicate cell/time at row {row_count}")
                grid.add(key)
                last = timestamp if last is None else max(last, timestamp)
                cell = CELLS[cell_id - 1][0]
                expected = expected_state(job, timestamp)
                for column, p in (("tx_power_dbm", f"txp_{cell}"), ("ret_tilt_deg", f"ret_{cell}"),
                                  ("cio_bias_db", f"cio_{cell}"), ("ttt_ms", "ttt_ms"), ("hysteresis_db", "hys_db")):
                    value = float(row[column])
                    if not math.isfinite(value) or abs(value - expected[p]) > 1e-6:
                        raise ValueError(f"Row {row_count} {p}={value}, expected {expected[p]}")
    except (OSError, ValueError, KeyError, TypeError) as error:
        issues.append("cell_csv_error:" + str(error))
    if len(grid) != 2700 or row_count != 2700:
        issues.append(f"incomplete_cell_grid:{len(grid)}/2700")
    ue_path = directory / "ue_kpi.csv"
    if not ue_path.is_file() or ue_path.stat().st_size == 0:
        issues.append("missing_or_empty_ue_csv")
    artifacts = {}
    for path in (receipt, cell_path, ue_path):
        if path.is_file():
            artifacts[path.name] = {"bytes": path.stat().st_size}
            if hashes:
                artifacts[path.name]["sha256"] = sha256(path)
    return {"job_id": job["job_id"], "passed": not issues, "issues": issues,
            "at_utc": utc_now(), "cell_rows": row_count, "last_cell_time_s": last,
            "receipt_count": len(records), "artifacts": artifacts}


@contextlib.contextmanager
def exclusive_lock(path: Path):
    if sys.platform != "linux":
        raise RuntimeError("Execution locks require Linux/WSL")
    import fcntl
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+", encoding="utf-8") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another runner owns this output/shard: {path}") from error
        handle.seek(0)
        handle.truncate()
        json.dump({"pid": os.getpid(), "host": socket.gethostname(), "at_utc": utc_now()}, handle)
        handle.flush()
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def attempt_paths(root: Path, job: dict) -> list[Path]:
    job_root = root / "jobs" / job["job_id"]
    if job_root.is_symlink() or not job_root.resolve().is_relative_to(root.resolve()):
        raise ValueError("Job output path escapes result root or is a symlink")
    paths = sorted(job_root.glob("attempt[0-9][0-9][0-9][0-9]"))
    if any(path.is_symlink() or not path.resolve().is_relative_to(root.resolve()) for path in paths):
        raise ValueError("Attempt output path escapes result root or is a symlink")
    return paths


def completed_attempt(root: Path, job: dict, manifest_hash: str, verify: bool) -> Path | None:
    for path in reversed(attempt_paths(root, job)):
        execution = path / "execution.json"
        if execution.is_file():
            record = read_json(execution)
            if record.get("status") == "complete":
                if record.get("manifest_hash") != manifest_hash or record.get("job_hash") != canonical_hash(job):
                    raise ValueError(f"Complete result belongs to a different manifest: {path}")
                if verify and not verify_job(job, path / "data" / job["job_id"])["passed"]:
                    raise ValueError(f"Previously completed result no longer verifies: {path}")
                return path
    return None


def stop_owned(process: subprocess.Popen) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def launch(manifest: dict, config: dict, job: dict, root: Path, provenance: dict,
           cpu_ids: list[int] | None = None) -> dict:
    job_root = root / "jobs" / job["job_id"]
    attempts = attempt_paths(root, job)
    job_root.mkdir(parents=True, exist_ok=True)
    number = max([int(p.name[7:]) for p in attempts], default=0) + 1
    if number > 9999:
        raise ValueError("Attempt limit exceeded")
    attempt = job_root / f"attempt{number:04d}"
    attempt.mkdir(exist_ok=False)
    (attempt / "data").mkdir()
    with (attempt / "parameter_schedule.csv").open("x", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("time_s", "parameter", "value"))
        writer.writerows(schedule_rows(job))
    atomic_json(attempt / "job.json", job)
    argv = argv_for(manifest, config, job, attempt)
    environment = clean_environment(config)
    if cpu_ids is not None:
        allowed = set(os.sched_getaffinity(0))
        if not cpu_ids or len(set(cpu_ids)) != len(cpu_ids) or not set(cpu_ids) <= allowed:
            raise ValueError("CPU slot is empty, duplicated or outside current affinity")
        taskset = shutil.which("taskset")
        if taskset is None:
            raise RuntimeError("taskset is required for per-job CPU pinning")
        argv = [taskset, "--cpu-list", ",".join(map(str, cpu_ids)), *argv]
        for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
            environment[key] = str(len(cpu_ids))
    record = {"job_id": job["job_id"], "job_hash": canonical_hash(job), "manifest_hash": canonical_hash(manifest),
              "status": "launching", "started_at_utc": utc_now(), "hostname": socket.gethostname(),
              "argv": argv, "cpu_ids": cpu_ids, "cwd": str(attempt), "provenance_hash": canonical_hash(provenance),
              "schedule_sha256": sha256(attempt / "parameter_schedule.csv"), "attempt": number}
    atomic_json(attempt / "execution.json", record)
    stdout = (attempt / "stdout.log").open("xb", buffering=0)
    stderr = (attempt / "stderr.log").open("xb", buffering=0)
    try:
        process = subprocess.Popen(argv, cwd=attempt, env=environment,
                                   stdout=stdout, stderr=stderr, start_new_session=True)
        record.update(status="running", pid=process.pid)
        atomic_json(attempt / "execution.json", record)
    except BaseException as error:
        if "process" in locals():
            stop_owned(process)
        stdout.close()
        stderr.close()
        record.update(status="launch_failed", error=str(error), finished_at_utc=utc_now())
        atomic_json(attempt / "execution.json", record)
        raise
    return {"process": process, "stdout": stdout, "stderr": stderr, "record": record,
            "attempt": attempt, "job": job, "start": time.monotonic()}


def finish(item: dict, reason: str | None = None) -> dict:
    item["stdout"].close()
    item["stderr"].close()
    result = verify_job(item["job"], item["attempt"] / "data" / item["job"]["job_id"])
    record = item["record"]
    code = item["process"].poll()
    record.update(status=reason or ("complete" if code == 0 and result["passed"] else "failed"),
                  returncode=code, wall_seconds=time.monotonic() - item["start"],
                  finished_at_utc=utc_now(), verification=result)
    atomic_json(item["attempt"] / "execution.json", record)
    return record


def output_root(config: dict) -> Path:
    return contained(Path(config["ns3_root"]).expanduser().resolve(), config["output_root"], "output_root")


def run_batch(manifest: dict, config: dict, jobs: list[dict], *, workers: int, shard_index: int,
              shard_count: int, resume: bool, retry_failed: bool, cores_per_job: int | None = None) -> dict:
    if sys.platform != "linux":
        raise RuntimeError("run is supported only on Linux/WSL")
    slots = cpu_slots(cores_per_job, workers) if cores_per_job is not None else None
    provenance = preflight(manifest, config, jobs, workers)
    provenance.update(cores_per_job=cores_per_job, cpu_slots=slots)
    root = output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    with exclusive_lock(root / "locks" / f"shard-{shard_index}-of-{shard_count}.lock"):
        # Frozen root configuration prevents overlapping shard topologies and changed manifests.
        with exclusive_lock(root / "locks" / "metadata.lock"):
            contract = {"manifest_hash": canonical_hash(manifest), "shard_count": shard_count,
                        "config_hash": canonical_hash(config)}
            contract_path = root / "execution_contract.json"
            if contract_path.exists():
                if read_json(contract_path) != contract:
                    raise ValueError("Output root belongs to a different manifest/config/shard-count; choose a new output_root")
                if not resume:
                    raise FileExistsError("Output already initialized; use --resume")
            else:
                atomic_json(contract_path, contract)
                atomic_json(root / "manifest.json", manifest)
                atomic_json(root / "config.json", config)
            # All shards and retries must use the same bytes for shared inputs.
            # New trace fingerprints may be added as previously unseen shards run.
            fingerprints_path = root / "frozen_inputs.json"
            frozen = read_json(fingerprints_path) if fingerprints_path.is_file() else {}
            for item in provenance["input_fingerprints"]:
                name, digest = item["path"], item["sha256"]
                if name in frozen and frozen[name] != digest:
                    raise ValueError(f"Input changed since an earlier launch: {name}; choose a new output_root")
                frozen[name] = digest
            atomic_json(fingerprints_path, frozen)
        provenance_dir = root / "provenance"
        provenance_dir.mkdir(exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        atomic_json(provenance_dir / f"shard{shard_index}-{stamp}.json", provenance)
        pending, skipped = [], []
        for job in jobs:
            completed = completed_attempt(root, job, canonical_hash(manifest), verify=True)
            if completed:
                skipped.append(job["job_id"])
            elif attempt_paths(root, job) and not retry_failed:
                raise ValueError(f"{job['job_id']} has unsuccessful attempt(s); add --retry-failed to preserve them and create a new attempt")
            else:
                # A crashed runner can leave an ns-3 process alive. Never start a duplicate.
                for old in attempt_paths(root, job):
                    path = old / "execution.json"
                    if path.is_file():
                        record = read_json(path)
                        if record.get("status") == "running" and record.get("hostname") == socket.gethostname() and Path(f"/proc/{record.get('pid', 0)}").exists():
                            raise RuntimeError(f"Prior process may still be alive for {job['job_id']}; inspect PID {record['pid']} before retry")
                pending.append(job)
        running, finished = {}, []
        stopped = False
        previous = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
        def stop_handler(signum, frame):
            nonlocal stopped
            stopped = True
        for sig in previous:
            signal.signal(sig, stop_handler)
        started, next_launch = time.monotonic(), time.monotonic()
        state = {"status": "running", "started_at_utc": utc_now(), "pid": os.getpid(),
                 "hostname": socket.gethostname(), "shard_index": shard_index, "shard_count": shard_count,
                 "workers": workers, "cores_per_job": cores_per_job, "cpu_slots": slots, "skipped_complete": skipped}
        def save(reason=None):
            state.update(updated_at_utc=utc_now(), wall_seconds=time.monotonic() - started,
                         pending=[j["job_id"] for j in pending], completed=finished, wait_reason=reason,
                         running=[{"job_id": ident, "pid": item["process"].pid, "cpu_ids": item["record"].get("cpu_ids"),
                                   "wall_seconds": time.monotonic() - item["start"],
                                   "last_cell_time_s": last_cell_time(item["attempt"] / "data" / ident)}
                                  for ident, item in running.items()])
            atomic_json(root / f"status-shard-{shard_index}.json", state)
        try:
            while pending or running:
                if stopped:
                    raise KeyboardInterrupt("Stop requested")
                for ident, item in list(running.items()):
                    reason = None
                    if item["process"].poll() is None and time.monotonic() - item["start"] > config["timeout_s"]:
                        stop_owned(item["process"])
                        reason = "timed_out"
                    if item["process"].poll() is not None:
                        record = finish(item, reason)
                        finished.append(record)
                        print(f"[{record['status']}] {ident} exit={record['returncode']} wall={record['wall_seconds']:.1f}s", file=sys.stderr, flush=True)
                        del running[ident]
                reason = None
                if pending and len(running) < workers:
                    available = memory_available()
                    free = shutil.disk_usage(root).free
                    if free < (config["disk_reserve_gib"] + config["estimated_output_gib_per_job"]) * 1024 ** 3:
                        raise RuntimeError("Free disk below launch reserve")
                    if available is None or available < (config["memory_reserve_gib"] + config["estimated_job_gib"]) * 1024 ** 3:
                        reason = "waiting_for_available_memory"
                        if not running:
                            raise RuntimeError("Insufficient available RAM for one new job plus reserve; adjust workers/config after measuring")
                    elif time.monotonic() < next_launch:
                        reason = "startup_stagger"
                    else:
                        job = pending.pop(0)
                        if slots is None:
                            item = launch(manifest, config, job, root, provenance)
                        else:
                            occupied = {tuple(i["record"]["cpu_ids"]) for i in running.values()}
                            slot = next(s for s in slots if tuple(s) not in occupied)
                            item = launch(manifest, config, job, root, provenance, cpu_ids=slot)
                        running[job["job_id"]] = item
                        print(f"[start] {job['job_id']} pid={item['process'].pid} cpu={item['record'].get('cpu_ids')} remaining={len(pending)}", file=sys.stderr, flush=True)
                        next_launch = time.monotonic() + config["startup_stagger_s"]
                save(reason)
                if pending or running:
                    time.sleep(1)
        except BaseException as error:
            for item in running.values():
                stop_owned(item["process"])
            for item in running.values():
                finished.append(finish(item, "interrupted"))
            running.clear()
            state.update(status="interrupted", error=f"{type(error).__name__}: {error}")
            save()
            raise
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
        state.update(status="complete" if all(r["status"] == "complete" for r in finished) else "failed",
                     finished_at_utc=utc_now())
        save()
        return state


def status(manifest: dict, config: dict, jobs: list[dict], verify: bool = False) -> dict:
    root, rows = output_root(config), []
    manifest_hash = canonical_hash(manifest)
    for job in jobs:
        attempts = attempt_paths(root, job)
        completed = completed_attempt(root, job, manifest_hash, verify=verify)
        path = completed or (attempts[-1] if attempts else None)
        row = {"job_id": job["job_id"], "trace_seed": job["trace_seed"], "status": "not_started",
               "attempts": len(attempts)}
        if path and (path / "execution.json").is_file():
            record = read_json(path / "execution.json")
            row.update(status=record["status"], returncode=record.get("returncode"), cpu_ids=record.get("cpu_ids"),
                       wall_seconds=record.get("wall_seconds"), output=str(path / "data" / job["job_id"]),
                       last_cell_time_s=last_cell_time(path / "data" / job["job_id"]))
            if record.get("status") == "running":
                row.update(pid=record.get("pid"), hostname=record.get("hostname"))
                if record.get("hostname") == socket.gethostname() and sys.platform == "linux":
                    row["process_alive"] = Path(f"/proc/{record['pid']}").exists()
            if verify:
                row["verification"] = verify_job(job, path / "data" / job["job_id"])
                row["verified_complete"] = record.get("status") == "complete" and record.get("returncode") == 0 and row["verification"]["passed"]
        rows.append(row)
    counts = {label: sum(r["status"] == label for r in rows) for label in sorted({r["status"] for r in rows})}
    result = {"experiment_id": manifest["experiment_id"], "at_utc": utc_now(), "output_root": str(root),
            "manifest_hash": manifest_hash,
            "selected_jobs": len(rows), "counts": counts, "jobs": rows,
            "selection_empty": not rows,
            "all_verified_complete": all(r.get("verified_complete", False) for r in rows) if verify else None}
    if verify:
        frozen_path = root / "frozen_inputs.json"
        frozen = read_json(frozen_path) if frozen_path.is_file() else {}
        roles = {}
        for path in sorted((root / "provenance").glob("*.json")):
            provenance = read_json(path)
            if provenance.get("manifest_hash") != manifest_hash:
                raise ValueError(f"Stored provenance manifest mismatch: {path}")
            for item in provenance.get("input_fingerprints", []):
                roles.setdefault(item["path"], set()).update(item["roles"])
        shared_roles = {"source", "trace", "gnb_positions", "initial_cache", "ret_cache"}
        shared = {name: digest for name, digest in frozen.items() if roles.get(name, set()) & shared_roles}
        result.update(frozen_inputs=frozen, shared_input_sha256=shared,
                      input_provenance_available=bool(shared),
                      shared_input_scope="source/cache/trace/gnb input bytes; locally compiled binaries and linked libraries are recorded but excluded from cross-PC equality")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("plan", "preflight", "run", "status", "collect"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-jobs", type=int)
    parser.add_argument("--start-index", type=int, help="Inclusive conflict scenario ID, starting at 1")
    parser.add_argument("--end-index", type=int, help="Inclusive conflict scenario ID, up to 500")
    parser.add_argument("--cores-per-job", type=int, help="Pin each concurrent job to this many logical CPUs")
    parser.add_argument("--tier", choices=("all", "A", "B", "C", "D"), default="all")
    parser.add_argument("--pilot20", action="store_true", help="Select the reviewed twenty-case pilot list")
    parser.add_argument("--pilot-ids", type=Path, help="Explicit JSON list or {job_ids:[...]} for pilot selection")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--report", type=Path, help="Save JSON report without overwriting an existing file")
    args = parser.parse_args()
    try:
        if args.workers < 1:
            raise ValueError("workers must be positive")
        manifest = validate_manifest(read_json(args.manifest))
        config = validate_config(read_json(args.config))
        pilot_ids = None
        if args.pilot20 or args.pilot_ids:
            pilot_path = args.pilot_ids or Path(__file__).resolve().parent / "data" / "pilot_ids.json"
            pilot_ids = read_json(pilot_path)
            if isinstance(pilot_ids, dict):
                pilot_ids = pilot_ids.get("job_ids", pilot_ids.get("scenario_ids"))
            if not isinstance(pilot_ids, list) or not all(isinstance(x, str) for x in pilot_ids):
                raise ValueError("Pilot list must contain job IDs")
            if args.pilot20 and len(pilot_ids) != 20:
                raise ValueError("--pilot20 requires exactly twenty IDs")
        ranged = args.start_index is not None or args.end_index is not None
        if ranged and (args.start_index is None or args.end_index is None or args.shard_index != 0 or args.shard_count != 1):
            raise ValueError("Supply both range bounds; do not combine ranges with modulo sharding")
        jobs = selected_jobs(manifest, args.shard_index, args.shard_count, None if ranged else args.max_jobs, args.tier, pilot_ids)
        if ranged:
            jobs = jobs_in_range(manifest, jobs, args.start_index, args.end_index)
            if args.max_jobs is not None:
                if args.max_jobs < 1:
                    raise ValueError("max-jobs must be positive")
                jobs = jobs[:args.max_jobs]
        slots = cpu_slots(args.cores_per_job, args.workers) if args.cores_per_job is not None and args.command in ("plan", "preflight", "run") else None
        if args.command == "plan":
            root = output_root(config)
            result = {"simulation_launched": False, "manifest_hash": canonical_hash(manifest),
                      "selected_jobs": len(jobs), "shard_index": args.shard_index, "shard_count": args.shard_count,
                      "cores_per_job": args.cores_per_job, "cpu_slots": slots,
                      "jobs": [{"job_id": j["job_id"], "trace_seed": j["trace_seed"],
                                "schedule_rows": schedule_rows(j),
                                "argv_example": argv_for(manifest, config, j, root / "jobs" / j["job_id"] / "attempt0001")}
                               for j in jobs]}
        elif args.command == "preflight":
            result = preflight(manifest, config, jobs, args.workers)
            result.update(cores_per_job=args.cores_per_job, cpu_slots=slots)
        elif args.command == "run":
            result = run_batch(manifest, config, jobs, workers=args.workers, shard_index=args.shard_index,
                               shard_count=args.shard_count, resume=args.resume, retry_failed=args.retry_failed,
                               cores_per_job=args.cores_per_job)
        else:
            result = status(manifest, config, jobs, verify=args.command == "collect")
        if args.report:
            with args.report.open("x", encoding="utf-8") as handle:
                json.dump(result, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
            print(json.dumps({"report": str(args.report), "selected_jobs": len(jobs),
                              "status": result.get("status"), "counts": result.get("counts")}, ensure_ascii=False))
        else:
            print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
        if args.command == "run" and result["status"] != "complete" or args.command == "collect" and not result["all_verified_complete"]:
            raise SystemExit(1)
    except (OSError, ValueError, RuntimeError, KeyError) as error:
        print(json.dumps({"error": str(error), "command": args.command}, ensure_ascii=False), file=sys.stderr)
        raise SystemExit(2) from error


if __name__ == "__main__":
    main()
