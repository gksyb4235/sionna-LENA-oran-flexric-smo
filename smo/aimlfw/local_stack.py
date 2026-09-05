"""Local multi-process wiring and health checks for the complete SMO path.

Run with ``python -m smo.aimlfw.local_stack start``.  The runner preserves
the component boundaries from the design: services communicate over their
FastAPI contracts, while GNN MCP remains embedded in KPI Advisor and the
Retraining Controller is a separate polling worker.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pymongo import MongoClient

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
AGENTIC_AI_ROOT = REPOSITORY_ROOT / "smo" / "agentic_ai"
DEEPAGENTS_SOURCE_ROOT = AGENTIC_AI_ROOT / "deepagents" / "libs" / "deepagents"
KPI_ADVISOR_ROOT = AGENTIC_AI_ROOT / "agents" / "kpi-advisor"
DASHBOARD_ROOT = AGENTIC_AI_ROOT / "dashboard"
A1_ROOT = REPOSITORY_ROOT / "A1_Mediator_standalone" / "A1_Mediator"
STATE_ROOT = REPOSITORY_ROOT / "smo" / ".state" / "local-stack"
PID_FILE = STATE_ROOT / "processes.json"
LOG_ROOT = STATE_ROOT / "logs"


@dataclass(frozen=True)
class ServiceSpec:
    name: str
    command: tuple[str, ...]
    health_url: str | None
    required_http_status: int = 200


def _uvicorn(module: str, app_dir: Path, port: int) -> tuple[str, ...]:
    return (
        sys.executable,
        "-m",
        "uvicorn",
        module,
        "--app-dir",
        str(app_dir),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    )


def service_specs() -> tuple[ServiceSpec, ...]:
    """Return the authoritative local process and port configuration."""
    return (
        ServiceSpec(
            "a1-mediator",
            _uvicorn("app.main:app", A1_ROOT, 9000),
            "http://127.0.0.1:9000/a1-p/healthcheck",
        ),
        ServiceSpec(
            "data-extractor",
            (sys.executable, "-m", "smo.aimlfw.data_extractor.server"),
            "http://127.0.0.1:8101/health",
        ),
        ServiceSpec(
            "feature-store",
            (sys.executable, "-m", "smo.aimlfw.feature_store.server"),
            "http://127.0.0.1:8102/health",
        ),
        ServiceSpec(
            "training-manager",
            (sys.executable, "-m", "smo.aimlfw.training_manager.server"),
            "http://127.0.0.1:8103/health",
        ),
        ServiceSpec(
            "model-registry",
            (sys.executable, "-m", "smo.aimlfw.model_registry.server"),
            "http://127.0.0.1:8104/health",
        ),
        ServiceSpec(
            "inference-service",
            (sys.executable, "-m", "smo.aimlfw.inference_service.server"),
            "http://127.0.0.1:8105/status",
        ),
        ServiceSpec(
            "kpi-advisor",
            _uvicorn("server:app", KPI_ADVISOR_ROOT, 8110),
            "http://127.0.0.1:8110/health",
        ),
        ServiceSpec(
            "retraining-controller",
            (sys.executable, "-m", "smo.aimlfw.retraining_controller.worker"),
            None,
        ),
        ServiceSpec(
            "dashboard",
            _uvicorn("server:app", DASHBOARD_ROOT, 8000),
            "http://127.0.0.1:8000/health",
        ),
    )


def stack_environment() -> dict[str, str]:
    """Inject shared storage and service addresses without merging boundaries."""
    env = dict(os.environ)
    python_path = [
        str(REPOSITORY_ROOT),
        str(KPI_ADVISOR_ROOT),
        str(AGENTIC_AI_ROOT),
        str(DEEPAGENTS_SOURCE_ROOT),
    ]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(python_path),
            "AGENTIC_AI_ROOT": str(AGENTIC_AI_ROOT),
            "DATA_EXTRACTOR_PORT": "8101",
            "FEATURE_STORE_PORT": "8102",
            "TRAINING_MANAGER_PORT": "8103",
            "MODEL_REGISTRY_PORT": "8104",
            "INFERENCE_SERVICE_PORT": "8105",
            "INFERENCE_SERVICE_BASE_URL": "http://127.0.0.1:8105",
            "TRAINING_MANAGER_BASE_URL": "http://127.0.0.1:8103",
            "KPI_ADVISOR_AGENT_URL": "http://127.0.0.1:8110",
            "A1_MEDIATOR_BASE_URL": "http://127.0.0.1:9000",
            "KNOWLEDGE_MONGODB_URI": env.get(
                "KNOWLEDGE_MONGODB_URI",
                env.get("MODEL_REGISTRY_MONGODB_URI", "mongodb://127.0.0.1:27017"),
            ),
        }
    )
    return env


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except (OSError, ValueError):
        return False
    return True


def _saved_processes() -> dict[str, int]:
    if not PID_FILE.is_file():
        return {}
    try:
        payload = json.loads(PID_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return {
        str(name): int(pid)
        for name, pid in payload.items()
        if isinstance(name, str) and isinstance(pid, int)
    }


def check_local_stack() -> dict[str, Any]:
    """Check MongoDB plus every network boundary; no trained GNN is accepted as degraded."""
    report: dict[str, Any] = {"components": {}}
    pids = _saved_processes()
    mongo_uri = stack_environment()["KNOWLEDGE_MONGODB_URI"]
    try:
        MongoClient(mongo_uri, serverSelectionTimeoutMS=1500).admin.command("ping")
        report["components"]["mongodb"] = {"status": "ready"}
    except Exception as exc:  # noqa: BLE001 - health report retains dependency reason.
        report["components"]["mongodb"] = {"status": "not_ready", "reason": str(exc)}

    with httpx.Client(timeout=3.0) as client:
        for spec in service_specs():
            if spec.health_url is None:
                pid = pids.get(spec.name)
                report["components"][spec.name] = {
                    "status": "ready" if pid is not None and _pid_alive(pid) else "not_ready",
                    "pid": pid,
                }
                continue
            try:
                response = client.get(spec.health_url)
                body = response.json()
                status = "ready" if response.status_code == spec.required_http_status else "not_ready"
                if spec.name in {"inference-service", "kpi-advisor"}:
                    model_status = body.get("model_load_status", body.get("status"))
                    if model_status in {"not_loaded", "not_ready"}:
                        status = "not_ready"
                report["components"][spec.name] = {
                    "status": status,
                    "http_status": response.status_code,
                    "details": body,
                }
            except Exception as exc:  # noqa: BLE001
                report["components"][spec.name] = {"status": "not_ready", "reason": str(exc)}

    inference = report["components"].get("inference-service", {})
    report["components"]["gnn-mcp"] = {
        "status": inference.get("status", "not_ready"),
        "mode": "embedded_in_kpi_advisor",
        "inference_service": "http://127.0.0.1:8105",
    }
    statuses = [component["status"] for component in report["components"].values()]
    report["status"] = "ready" if statuses and all(status == "ready" for status in statuses) else "degraded"
    return report


def start_local_stack() -> int:
    """Start every local process and keep supervising until SIGINT/SIGTERM."""
    existing = {name: pid for name, pid in _saved_processes().items() if _pid_alive(pid)}
    if existing:
        raise RuntimeError(f"local stack already has live processes: {existing}")
    STATE_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    env = stack_environment()
    processes: dict[str, subprocess.Popen[bytes]] = {}
    log_handles: list[Any] = []
    stopping = False

    def request_stop(_signum: int, _frame: Any) -> None:
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        for spec in service_specs():
            log_handle = (LOG_ROOT / f"{spec.name}.log").open("ab")
            log_handles.append(log_handle)
            processes[spec.name] = subprocess.Popen(
                spec.command,
                cwd=REPOSITORY_ROOT,
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
            )
        PID_FILE.write_text(
            json.dumps({name: process.pid for name, process in processes.items()}, indent=2),
            encoding="utf-8",
        )
        time.sleep(1.0)
        print(json.dumps(check_local_stack(), ensure_ascii=False, indent=2), flush=True)
        while not stopping:
            failed = {
                name: process.returncode
                for name, process in processes.items()
                if process.poll() is not None
            }
            if failed:
                print(json.dumps({"status": "process_failed", "processes": failed}), file=sys.stderr)
                return 1
            time.sleep(0.5)
        return 0
    finally:
        for process in processes.values():
            if process.poll() is None:
                process.terminate()
        for process in processes.values():
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
        for handle in log_handles:
            handle.close()
        PID_FILE.unlink(missing_ok=True)


def stop_local_stack() -> None:
    """Gracefully stop only PIDs recorded by this runner."""
    for pid in _saved_processes().values():
        if _pid_alive(pid):
            os.kill(pid, signal.SIGTERM)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("start", "check", "stop"))
    args = parser.parse_args()
    if args.command == "start":
        raise SystemExit(start_local_stack())
    if args.command == "stop":
        stop_local_stack()
        return
    print(json.dumps(check_local_stack(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
