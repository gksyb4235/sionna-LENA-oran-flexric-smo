"""Handoff adapter from Decomposition Agent to Planning Agent."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

JAECHAN_ROOT = Path(
    os.getenv("AGENTIC_AI_ROOT", str(Path(__file__).resolve().parents[3]))
).resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
PLANNING_ROOT = JAECHAN_ROOT / "agents" / "planning"
_PLANNING_VENV_PYTHON = PLANNING_ROOT / ".venv" / "bin" / "python"
PLANNING_PYTHON = _PLANNING_VENV_PYTHON if os.access(_PLANNING_VENV_PYTHON, os.X_OK) else Path(sys.executable)
PLANNING_AGENT = PLANNING_ROOT / "agent.py"
HANDOFF_DIR = PROJECT_ROOT / ".handoff"
DEFAULT_TIMEOUT_SECONDS = 120


def _ensure_path(path: Path, *, root: Path = JAECHAN_ROOT) -> Path:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"Path must stay under {root}; got {resolved}")
    return resolved


def _load_env_file() -> None:
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _agent_pythonpath(agent_root: Path, python_path: Path) -> str:
    venv_root = python_path.resolve().parents[1]
    candidates = sorted((venv_root / "lib").glob("python*/site-packages"))
    parts = [str(candidates[0])] if candidates else []
    parts.extend([str(agent_root.resolve()), str(JAECHAN_ROOT)])
    return os.pathsep.join(parts)


def _subprocess_env(agent_root: Path = PLANNING_ROOT, python_path: Path = PLANNING_PYTHON) -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _timeout_seconds() -> int:
    raw_value = os.getenv("PLANNING_AGENT_TIMEOUT_SEC", str(DEFAULT_TIMEOUT_SECONDS))
    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return max(1, value)


def validate_planning_report(payload: dict[str, Any]) -> None:
    """Validate the minimal Planning Agent response contract."""

    required = {"intent", "subtask_plans", "overall_status"}
    optional = {"run_id"}
    keys = set(payload.keys())
    if not required.issubset(keys) or not keys.issubset(required | optional):
        raise ValueError("Planning report must contain intent, subtask_plans, overall_status, and optional run_id.")
    if "run_id" in payload and not isinstance(payload["run_id"], str):
        raise TypeError("Planning report 'run_id' must be a string when provided.")
    if not isinstance(payload["intent"], str):
        raise TypeError("Planning report 'intent' must be a string.")
    if not isinstance(payload["subtask_plans"], list):
        raise TypeError("Planning report 'subtask_plans' must be a list.")
    valid_statuses = {"running", "completed", "mock_completed", "partial", "blocked", "pending_approval", "rejected"}
    if payload["overall_status"] not in valid_statuses:
        raise ValueError("Planning report overall_status is invalid.")

def _planning_invoke_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    return url if url.endswith("/invoke") else f"{url}/invoke"


def _invoke_planning_http(decomposition: dict[str, Any], base_url: str) -> dict[str, Any]:
    body = json.dumps(decomposition, ensure_ascii=False).encode("utf-8")
    request = Request(
        _planning_invoke_url(base_url),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=_timeout_seconds()) as response:  # noqa: S310 - URL is operator-configured.
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Planning agent HTTP request failed: {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Planning agent HTTP request failed: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Planning agent HTTP response must be a JSON object.")
    validate_planning_report(payload)
    return payload


def _invoke_planning_cli(decomposition: dict[str, Any]) -> dict[str, Any]:
    _ensure_path(PROJECT_ROOT)
    _ensure_path(PLANNING_ROOT)
    if not PLANNING_PYTHON.exists() or not PLANNING_AGENT.exists():
        raise RuntimeError("Planning agent local runtime is not available. Set PLANNING_AGENT_URL or install planning/.venv.")

    handoff_dir = _ensure_path(HANDOFF_DIR)
    handoff_dir.mkdir(parents=True, exist_ok=True)
    input_path = _ensure_path(handoff_dir / f"decomposition-{uuid.uuid4().hex}.json")
    input_path.write_text(json.dumps(decomposition, ensure_ascii=False), encoding="utf-8")
    try:
        completed = subprocess.run(
            [str(PLANNING_PYTHON), str(PLANNING_AGENT), "--json", "--input", str(input_path)],
            cwd=str(PLANNING_ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=_timeout_seconds(),
            env=_subprocess_env(),
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "no stderr"
        raise RuntimeError(f"Planning agent CLI failed: {stderr}") from exc
    finally:
        input_path.unlink(missing_ok=True)

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        print(completed.stdout, file=sys.stderr)
        raise RuntimeError("Planning agent CLI did not return valid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Planning agent CLI response must be a JSON object.")
    validate_planning_report(payload)
    return payload


def validate_representative_core_report(payload: dict[str, Any]) -> None:
    """Validate the minimal Representative Core Agent response contract."""

    required = {
        "intent",
        "command_proposal",
        "kubectl_preview",
        "risk_level",
        "approval_required",
        "approval",
        "status",
        "result",
        "errors",
        "source",
    }
    if set(payload.keys()) != required:
        raise ValueError("Representative Core report must contain exactly the expected output fields.")
    if not isinstance(payload["intent"], str):
        raise TypeError("Representative Core report 'intent' must be a string.")
    if not isinstance(payload["kubectl_preview"], list):
        raise TypeError("Representative Core report 'kubectl_preview' must be a list.")
    if payload["risk_level"] not in {"low", "medium", "high"}:
        raise ValueError("Representative Core report risk_level is invalid.")
    if not isinstance(payload["approval_required"], bool):
        raise TypeError("Representative Core report 'approval_required' must be a boolean.")
    if payload["status"] not in {"pending_approval", "completed", "blocked", "rejected"}:
        raise ValueError("Representative Core report status is invalid.")


def _planning_approve_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/invoke"):
        url = url[: -len("/invoke")]
    return url if url.endswith("/approve") else f"{url}/approve"


def _validate_planning_approval_response(payload: dict[str, Any]) -> None:
    if "subtask_plans" in payload and "overall_status" in payload:
        validate_planning_report(payload)
    else:
        validate_representative_core_report(payload)


def _approve_planning_http(
    report: dict[str, Any],
    *,
    approved: bool,
    approved_by: str | None,
    reason: str | None,
    base_url: str,
    planning_run_id: str | None = None,
) -> dict[str, Any]:
    request_payload: dict[str, Any] = {
        "report": report,
        "approved": approved,
        "approved_by": approved_by,
        "reason": reason,
    }
    if planning_run_id:
        request_payload["run_id"] = planning_run_id
    body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        _planning_approve_url(base_url),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=_timeout_seconds()) as response:  # noqa: S310 - URL is operator-configured.
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Planning approval HTTP request failed: {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Planning approval HTTP request failed: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Planning approval HTTP response must be a JSON object.")
    _validate_planning_approval_response(payload)
    return payload

def _approve_planning_cli(
    report: dict[str, Any],
    *,
    approved: bool,
    approved_by: str | None,
    reason: str | None,
    planning_run_id: str | None = None,
) -> dict[str, Any]:
    _ensure_path(PROJECT_ROOT)
    _ensure_path(PLANNING_ROOT)
    if not PLANNING_PYTHON.exists() or not PLANNING_AGENT.exists():
        raise RuntimeError("Planning agent local runtime is not available. Set PLANNING_AGENT_URL or install planning/.venv.")

    command = [str(PLANNING_PYTHON), str(PLANNING_AGENT), "--json", "--approve-core" if approved else "--reject-core"]
    if approved_by:
        command.extend(["--approved-by", approved_by])
    if reason:
        command.extend(["--reason", reason])
    request_payload: dict[str, Any] = {"report": report}
    if planning_run_id:
        request_payload["run_id"] = planning_run_id

    try:
        completed = subprocess.run(
            command,
            cwd=str(PLANNING_ROOT),
            input=json.dumps(request_payload, ensure_ascii=False),
            check=True,
            capture_output=True,
            text=True,
            timeout=_timeout_seconds(),
            env=_subprocess_env(),
        )
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.strip() if exc.stderr else "no stderr"
        raise RuntimeError(f"Planning approval CLI failed: {stderr}") from exc

    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        print(completed.stdout, file=sys.stderr)
        raise RuntimeError("Planning approval CLI did not return valid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Planning approval CLI response must be a JSON object.")
    _validate_planning_approval_response(payload)
    return payload

def approve_representative_core_report(
    report: dict[str, Any],
    *,
    approved: bool,
    approved_by: str | None = None,
    reason: str | None = None,
    planning_run_id: str | None = None,
) -> dict[str, Any]:
    """Approve or reject a Representative Core pending report through the Planning Agent."""

    _load_env_file()
    planning_url = os.getenv("PLANNING_AGENT_URL", "").strip()
    if planning_url:
        return _approve_planning_http(
            report,
            approved=approved,
            approved_by=approved_by,
            reason=reason,
            base_url=planning_url,
            planning_run_id=planning_run_id,
        )
    return _approve_planning_cli(
        report,
        approved=approved,
        approved_by=approved_by,
        reason=reason,
        planning_run_id=planning_run_id,
    )

def invoke_planning_agent(decomposition: dict[str, Any]) -> dict[str, Any]:
    """Send Decomposition Agent output to the Planning Agent."""

    _load_env_file()
    planning_url = os.getenv("PLANNING_AGENT_URL", "").strip()
    if planning_url:
        return _invoke_planning_http(decomposition, planning_url)
    return _invoke_planning_cli(decomposition)
