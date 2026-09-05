"""Agent Ops Dashboard server."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

JAECHAN_ROOT = Path(
    os.getenv("AGENTIC_AI_ROOT", str(Path(__file__).resolve().parents[1]))
).resolve()
PLANNING_ROOT = JAECHAN_ROOT / "agents" / "planning"
DECOMPOSITION_ROOT = JAECHAN_ROOT / "agents" / "decomposition"
_DECOMPOSITION_VENV_PYTHON = DECOMPOSITION_ROOT / ".venv" / "bin" / "python"
DECOMPOSITION_PYTHON = (
    _DECOMPOSITION_VENV_PYTHON if os.access(_DECOMPOSITION_VENV_PYTHON, os.X_OK) else Path(sys.executable)
)
DECOMPOSITION_AGENT = DECOMPOSITION_ROOT / "agent.py"
PLANNING_STATE_ROOT = PLANNING_ROOT / ".state" / "planning-runs"
DASHBOARD_ROOT = Path(__file__).resolve().parent
STATIC_ROOT = DASHBOARD_ROOT / "static"
INVOKE_DEBUG_LOG = DASHBOARD_ROOT / "invoke-debug.log"

if str(PLANNING_ROOT) not in sys.path:
    sys.path.insert(0, str(PLANNING_ROOT))
if str(JAECHAN_ROOT) not in sys.path:
    sys.path.append(str(JAECHAN_ROOT))

from agent_ops.telemetry import list_event_run_ids, read_events, safe_run_id  # noqa: E402

app = FastAPI(title="Agent Ops Dashboard", version="0.1.0")
app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")


@app.get("/health")
def readiness() -> dict[str, str]:
    return {"status": "ready", "component": "dashboard"}


class InvokeIntentRequest(BaseModel):
    intent: str


class ApprovalRequest(BaseModel):
    report: dict[str, Any] | None = None
    approved: bool
    approved_by: str = "dashboard"
    reason: str | None = None


def _ensure_under_root(path: Path, root: Path = JAECHAN_ROOT) -> Path:
    resolved = path.resolve()
    if resolved != root and root not in resolved.parents:
        raise RuntimeError(f"Path must stay under {root}; got {resolved}")
    return resolved


def _state_path(run_id: str) -> Path:
    safe = safe_run_id(run_id)
    root = _ensure_under_root(PLANNING_STATE_ROOT)
    return _ensure_under_root(root / f"{safe}.json", root)


def _load_state(run_id: str) -> dict[str, Any] | None:
    path = _state_path(run_id)
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, dict) else None


def _parse_agent_json(stdout: str) -> dict[str, Any]:
    stripped = stdout.strip()
    if not stripped:
        raise RuntimeError("decomposition-agent returned no JSON output")
    try:
        payload = json.loads(stripped)
        if isinstance(payload, dict):
            return payload
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    index = 0
    while index < len(stripped):
        if stripped[index] != "{":
            index += 1
            continue
        try:
            payload, end = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(payload, dict) and ("run_id" in payload or "overall_status" in payload):
            return payload
        index += max(end, 1)
    raise RuntimeError("decomposition-agent output did not contain a planning report JSON object")


def _short_error(text: str, limit: int = 1800) -> str:
    clean = "\n".join(line for line in text.strip().splitlines() if line.strip())
    if not clean:
        return "unknown error"
    return clean[-limit:]


class InvokeFailure(RuntimeError):
    def __init__(self, *, call_id: str, stage: str, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.call_id = call_id
        self.stage = stage
        self.message = message
        self.status_code = status_code


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _tail_text(text: str | None, limit: int = 1800) -> str:
    if not text:
        return ""
    clean = "\n".join(line for line in text.splitlines() if line.strip())
    return clean[-limit:]


def _safe_env_summary(env: dict[str, str]) -> dict[str, Any]:
    return {
        "PYTHONPATH_present": "PYTHONPATH" in env,
        "PYTHONNOUSERSITE": env.get("PYTHONNOUSERSITE"),
        "OPENAI_API_KEY_present": bool(env.get("OPENAI_API_KEY")),
        "PATH_present": "PATH" in env,
    }


def _append_invoke_debug(call_id: str, event: str, **fields: Any) -> None:
    try:
        log_path = _ensure_under_root(INVOKE_DEBUG_LOG, DASHBOARD_ROOT)
        payload = {
            "timestamp": _now_iso(),
            "call_id": call_id,
            "event": event,
            **fields,
        }
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return


def _invoke_error_detail(exc: InvokeFailure) -> dict[str, Any]:
    return {
        "message": exc.message,
        "call_id": exc.call_id,
        "stage": exc.stage,
        "debug_log": str(INVOKE_DEBUG_LOG),
    }


def _preflight_code() -> str:
    return '''
import json
import sys

result = {"sys_path_head": sys.path[:8]}
import schemas
result["schemas_file"] = getattr(schemas, "__file__", None)
import agent
agent.prefer_venv_deepagents_package()
import deepagents
result["deepagents_file"] = getattr(deepagents, "__file__", None)
result["deepagents_path"] = list(getattr(deepagents, "__path__", []))
result["has_create_deep_agent"] = hasattr(deepagents, "create_deep_agent")
from deepagents import create_deep_agent
result["import_create_deep_agent"] = bool(create_deep_agent)
print(json.dumps(result, ensure_ascii=False))
'''


def _run_import_preflight(
    *,
    call_id: str,
    python_path: Path,
    cwd: Path,
    env: dict[str, str],
    client_host: str | None,
    request_path: str,
) -> None:
    argv = [str(python_path), "-c", _preflight_code()]
    _append_invoke_debug(
        call_id,
        "preflight_started",
        client_host=client_host,
        request_path=request_path,
        server_pid=os.getpid(),
        cwd=str(cwd),
        argv=[str(python_path), "-c", "<import preflight>"],
        env=_safe_env_summary(env),
    )
    completed = subprocess.run(
        argv,
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    _append_invoke_debug(
        call_id,
        "preflight_completed",
        returncode=completed.returncode,
        stdout_tail=_tail_text(completed.stdout),
        stderr_tail=_tail_text(completed.stderr),
    )
    if completed.returncode != 0:
        raise InvokeFailure(
            call_id=call_id,
            stage="preflight",
            message=f"decomposition import preflight failed: {_short_error(completed.stderr or completed.stdout)}",
        )


def _agent_pythonpath(agent_root: Path, python_path: Path) -> str:
    venv_root = python_path.resolve().parents[1]
    candidates = sorted((venv_root / "lib").glob("python*/site-packages"))
    parts = [str(candidates[0])] if candidates else []
    parts.extend([str(agent_root.resolve()), str(JAECHAN_ROOT)])
    return os.pathsep.join(parts)


def _run_decomposition_intent(
    intent: str,
    *,
    call_id: str | None = None,
    client_host: str | None = None,
    request_path: str = "/api/invoke",
) -> dict[str, Any]:
    call_id = call_id or f"invoke-{uuid.uuid4().hex}"
    # Keep the venv executable path itself. Resolving .venv/bin/python follows the
    # symlink to the uv-managed base interpreter and drops the venv site-packages.
    _ensure_under_root(DECOMPOSITION_PYTHON.parent)
    python_path = DECOMPOSITION_PYTHON
    agent_path = _ensure_under_root(DECOMPOSITION_AGENT)
    cwd = _ensure_under_root(DECOMPOSITION_ROOT)
    if not python_path.exists():
        raise InvokeFailure(call_id=call_id, stage="setup", message=f"Missing decomposition Python runtime: {python_path}")
    if not agent_path.exists():
        raise InvokeFailure(call_id=call_id, stage="setup", message=f"Missing decomposition agent entrypoint: {agent_path}")

    timeout = int(os.getenv("DASHBOARD_INVOKE_TIMEOUT_SECONDS", "600"))
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"

    _run_import_preflight(
        call_id=call_id,
        python_path=python_path,
        cwd=cwd,
        env=env,
        client_host=client_host,
        request_path=request_path,
    )

    argv = [str(python_path), str(agent_path), "--json", intent]
    _append_invoke_debug(
        call_id,
        "decomposition_started",
        client_host=client_host,
        request_path=request_path,
        server_pid=os.getpid(),
        cwd=str(cwd),
        argv=[str(python_path), str(agent_path), "--json", "<intent>"],
        env=_safe_env_summary(env),
    )
    try:
        completed = subprocess.run(
            argv,
            cwd=str(cwd),
            env=env,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _append_invoke_debug(call_id, "decomposition_timeout", timeout=timeout)
        raise InvokeFailure(
            call_id=call_id,
            stage="decomposition",
            message=f"decomposition-agent timed out after {timeout} seconds",
            status_code=504,
        ) from exc

    _append_invoke_debug(
        call_id,
        "decomposition_completed",
        returncode=completed.returncode,
        stdout_tail=_tail_text(completed.stdout),
        stderr_tail=_tail_text(completed.stderr),
    )
    if completed.returncode != 0:
        raise InvokeFailure(
            call_id=call_id,
            stage="decomposition",
            message=f"decomposition-agent failed: {_short_error(completed.stderr or completed.stdout)}",
        )
    try:
        return _parse_agent_json(completed.stdout)
    except Exception as exc:  # noqa: BLE001 - preserve call_id/stage for UI and debug log.
        _append_invoke_debug(call_id, "parse_failed", error=str(exc), stdout_tail=_tail_text(completed.stdout))
        raise InvokeFailure(call_id=call_id, stage="parse", message=str(exc)) from exc


def _state_files(limit: int = 50) -> list[Path]:
    root = _ensure_under_root(PLANNING_STATE_ROOT)
    if not root.exists():
        return []
    return sorted(root.glob("*.json"), key=lambda path: path.stat().st_mtime, reverse=True)[:limit]


def _current_agent(state: dict[str, Any] | None, events: list[dict[str, Any]]) -> str | None:
    for event in reversed(events):
        target = event.get("target_agent")
        agent = event.get("agent")
        event_type = event.get("event_type")
        if event_type in {"subagent_request", "pending_approval", "planning_handoff"} and isinstance(target, str):
            return target
        if isinstance(agent, str):
            return agent
    if state and isinstance(state.get("pending_approval"), dict):
        pending = state["pending_approval"]
        agent = pending.get("representative_agent_id")
        return str(agent) if agent else "representative-core-agent"
    return None


def _run_summary(run_id: str, state: dict[str, Any] | None = None) -> dict[str, Any]:
    events = read_events(run_id)
    state = state if state is not None else _load_state(run_id)
    state_mtime = _state_path(run_id).stat().st_mtime if _state_path(run_id).exists() else 0
    started = events[0]["timestamp"] if events else None
    updated = events[-1]["timestamp"] if events else (time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(state_mtime)) if state_mtime else None)
    return {
        "run_id": run_id,
        "intent": state.get("intent") if state else None,
        "overall_status": state.get("overall_status") if state else "events_only",
        "current_subtask_index": state.get("current_subtask_index") if state else None,
        "subtask_count": len(state.get("subtasks", [])) if state else None,
        "pending_approval": bool(state and state.get("pending_approval")),
        "current_agent": _current_agent(state, events),
        "started_at": started,
        "updated_at": updated,
        "event_count": len(events),
    }


def _known_run_ids(limit: int = 50) -> list[str]:
    seen: list[str] = []
    for path in _state_files(limit=limit):
        seen.append(path.stem)
    for run_id in list_event_run_ids(limit=limit):
        if run_id not in seen:
            seen.append(run_id)
    return seen[:limit]


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    html = (STATIC_ROOT / "index.html").read_text(encoding="utf-8")
    for asset in ("styles.css", "app.js"):
        version = int((STATIC_ROOT / asset).stat().st_mtime)
        html = html.replace(f"/static/{asset}", f"/static/{asset}?v={version}")
    return html


@app.get("/api/health")
def api_health() -> dict[str, str]:
    return {"status": "ok", "service": "agent-ops-dashboard"}


@app.post("/api/invoke")
def invoke_intent(payload: InvokeIntentRequest, request: Request) -> dict[str, Any]:
    intent = payload.intent.strip()
    if not intent:
        raise HTTPException(status_code=400, detail="intent must be a non-empty string")
    call_id = f"invoke-{uuid.uuid4().hex}"
    client_host = request.client.host if request.client else None
    try:
        result = _run_decomposition_intent(
            intent,
            call_id=call_id,
            client_host=client_host,
            request_path=str(request.url.path),
        )
    except InvokeFailure as exc:
        raise HTTPException(status_code=exc.status_code, detail=_invoke_error_detail(exc)) from exc
    except Exception as exc:  # noqa: BLE001 - dashboard should surface agent startup/runtime failures.
        failure = InvokeFailure(call_id=call_id, stage="dashboard", message=str(exc))
        _append_invoke_debug(call_id, "unexpected_failure", error=str(exc), client_host=client_host)
        raise HTTPException(status_code=400, detail=_invoke_error_detail(failure)) from exc

    run_id = result.get("run_id")
    run = None
    if isinstance(run_id, str) and run_id.strip():
        try:
            run = get_run(run_id)
        except HTTPException:
            run = None
    return {"result": result, "run": run}


@app.get("/api/runs")
def list_runs(limit: int = 25) -> dict[str, Any]:
    bounded_limit = min(max(limit, 1), 100)
    runs = [_run_summary(run_id) for run_id in _known_run_ids(limit=bounded_limit)]
    return {"runs": runs}


@app.get("/api/runs/{run_id}")
def get_run(run_id: str) -> dict[str, Any]:
    safe = safe_run_id(run_id)
    state = _load_state(safe)
    events = read_events(safe)
    if state is None and not events:
        raise HTTPException(status_code=404, detail="run_id not found")
    return {"summary": _run_summary(safe, state), "state": state, "events": events}


@app.get("/api/runs/{run_id}/events")
def get_events(run_id: str) -> dict[str, Any]:
    safe = safe_run_id(run_id)
    return {"run_id": safe, "events": read_events(safe)}


@app.get("/api/runs/{run_id}/stream")
async def stream_events(run_id: str) -> StreamingResponse:
    safe = safe_run_id(run_id)

    async def generator():
        sent = 0
        while True:
            events = read_events(safe)
            for event in events[sent:]:
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            sent = len(events)
            await asyncio.sleep(1)

    return StreamingResponse(generator(), media_type="text/event-stream")


@app.post("/api/runs/{run_id}/approve")
def approve_run(run_id: str, payload: ApprovalRequest) -> dict[str, Any]:
    safe = safe_run_id(run_id)
    state = _load_state(safe)
    if state is None:
        raise HTTPException(status_code=404, detail="run_id not found")
    report = payload.report
    if report is None:
        pending = state.get("pending_approval")
        if not isinstance(pending, dict) or not isinstance(pending.get("report"), dict):
            raise HTTPException(status_code=400, detail="run has no pending approval report")
        report = pending["report"]
    try:
        from agent import approve_core_report_from_payload

        result = approve_core_report_from_payload(
            {"run_id": safe, "report": report},
            approved=payload.approved,
            approved_by=payload.approved_by,
            reason=payload.reason,
        )
    except Exception as exc:  # noqa: BLE001 - dashboard should surface approval/runtime failures.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"result": result, "run": get_run(safe)}
