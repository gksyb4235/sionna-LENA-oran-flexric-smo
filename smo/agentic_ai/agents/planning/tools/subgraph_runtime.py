"""Runtime adapters for Planning Agent sub-graphs."""

from __future__ import annotations

import json
import math
import operator
import os
import subprocess
import sys
from pathlib import Path
from typing import Annotated, Any, TypedDict
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from langgraph.graph import END, START, StateGraph

JAECHAN_ROOT = Path(
    os.getenv("AGENTIC_AI_ROOT", str(Path(__file__).resolve().parents[3]))
).resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(JAECHAN_ROOT) not in sys.path:
    sys.path.append(str(JAECHAN_ROOT))

from agent_ops.telemetry import default_run_id, log_event, monotonic_ms, new_call_id  # noqa: E402
from tools.knowledge_registry import (  # noqa: E402
    resolve_agent_communication_contract,
    validate_contract_payload,
    validate_contract_schema,
)
PROBE_ROOT = JAECHAN_ROOT / "agents" / "probe"
_PROBE_VENV_PYTHON = PROBE_ROOT / ".venv" / "bin" / "python"
PROBE_PYTHON = _PROBE_VENV_PYTHON if os.access(_PROBE_VENV_PYTHON, os.X_OK) else Path(sys.executable)
PROBE_AGENT = PROBE_ROOT / "agent.py"
MONITORING_ROOT = JAECHAN_ROOT / "agents" / "monitoring"
_MONITORING_VENV_PYTHON = MONITORING_ROOT / ".venv" / "bin" / "python"
MONITORING_PYTHON = (
    _MONITORING_VENV_PYTHON if os.access(_MONITORING_VENV_PYTHON, os.X_OK) else Path(sys.executable)
)
MONITORING_AGENT = MONITORING_ROOT / "agent.py"
REPRESENTATIVE_CORE_ROOT = JAECHAN_ROOT / "agents" / "representative-core"
_REPRESENTATIVE_CORE_VENV_PYTHON = REPRESENTATIVE_CORE_ROOT / ".venv" / "bin" / "python"
REPRESENTATIVE_CORE_PYTHON = (
    _REPRESENTATIVE_CORE_VENV_PYTHON
    if os.access(_REPRESENTATIVE_CORE_VENV_PYTHON, os.X_OK)
    else Path(sys.executable)
)
REPRESENTATIVE_CORE_AGENT = REPRESENTATIVE_CORE_ROOT / "agent.py"
DEFAULT_TIMEOUT_SECONDS = 240
KPI_ADVISOR_AGENT_ID = "kpi-advisor-agent"
KPI_ADVISOR_TIMEOUT_SECONDS = 60
PUBLIC_RUNTIME_AGENT_NODE_FIELDS = {
    "id",
    "agent_id",
    "type",
    "role",
    "name",
    "label",
    "source",
    "description",
    "capabilities",
    "contract_ref",
}


def _node_contract_schema(node: dict[str, Any], key: str) -> dict[str, Any]:
    contract = node.get("_runtime_communication_contract")
    if not isinstance(contract, dict) or contract.get("schema_dialect") != "json-schema-subset-v1":
        raise ValueError(f"Agent {node.get('id')!r} has no resolved MongoDB communication contract dialect.")
    schema = contract.get(key) if isinstance(contract, dict) else None
    if not isinstance(schema, dict):
        raise ValueError(f"Agent {node.get('id')!r} has no MongoDB {key}.")
    return schema


def _validate_response_contract(
    payload: dict[str, Any],
    response_schema: dict[str, Any] | None,
    fallback,
) -> None:
    if response_schema is None:
        fallback(payload)
        return
    validate_contract_payload(payload, response_schema, path="$.subagent_response")


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


def _timeout_seconds(env_name: str = "PROBE_AGENT_TIMEOUT_SEC") -> int:
    raw_value = os.getenv(env_name) or os.getenv("PROBE_AGENT_TIMEOUT_SEC") or str(DEFAULT_TIMEOUT_SECONDS)
    try:
        value = int(raw_value)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS
    return max(1, value)




def _emit_event(run_id: str | None, agent: str, event_type: str, **kwargs: Any) -> None:
    try:
        log_event(run_id=run_id or default_run_id("planning-call"), agent=agent, event_type=event_type, **kwargs)
    except Exception as exc:  # noqa: BLE001 - telemetry must not break runtime adapters.
        print(f"subgraph-runtime telemetry warning: {exc}", file=sys.stderr)


def _agent_pythonpath(agent_root: Path, python_path: Path) -> str:
    venv_root = python_path.resolve().parents[1]
    candidates = sorted((venv_root / "lib").glob("python*/site-packages"))
    parts = [str(candidates[0])] if candidates else []
    parts.extend([str(agent_root.resolve()), str(JAECHAN_ROOT)])
    return os.pathsep.join(parts)


def _subprocess_env(
    run_id: str | None,
    subtask_id: str | None,
    call_id: str | None,
    *,
    agent_root: Path,
    python_path: Path,
) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONNOUSERSITE"] = "1"
    if run_id:
        env["AGENT_RUN_ID"] = run_id
    if subtask_id:
        env["AGENT_SUBTASK_ID"] = subtask_id
    if call_id:
        env["AGENT_CALL_ID"] = call_id
    return env


def _with_correlation(payload: dict[str, Any], run_id: str | None, subtask_id: str | None, call_id: str | None) -> dict[str, Any]:
    if run_id:
        payload["run_id"] = run_id
    if subtask_id:
        payload["subtask_id"] = subtask_id
    if call_id:
        payload["call_id"] = call_id
    return payload

def mock_execute_subgraph(
    *,
    subtask_id: str,
    subtask: str,
    langgraph_spec: dict[str, Any],
    representative_agent: dict[str, Any],
) -> dict[str, Any]:
    """Simulate representative-agent execution until an Execution Agent exists."""

    return {
        "status": "mock_completed",
        "runtime": "mock_subgraph_runtime",
        "subtask_id": subtask_id,
        "subtask": subtask,
        "representative_agent_id": representative_agent["id"],
        "representative_agent_name": representative_agent["name"],
        "message": "Mock execution completed. Replace this adapter when the Execution Agent is available.",
        "compiled": bool(langgraph_spec.get("compiled", False)),
    }


def _representative_core_invoke_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    return url if url.endswith("/invoke") else f"{url}/invoke"


def _validate_representative_core_report(payload: dict[str, Any]) -> None:
    required = (
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
    )
    if tuple(payload.keys()) != required:
        raise ValueError("Representative Core report has an unexpected contract.")
    if payload["status"] not in {"pending_approval", "completed", "blocked", "rejected"}:
        raise ValueError("Representative Core report status is invalid.")
    if payload["risk_level"] not in {"low", "medium", "high"}:
        raise ValueError("Representative Core report risk_level is invalid.")
    if not isinstance(payload["kubectl_preview"], list):
        raise TypeError("Representative Core report kubectl_preview must be a list.")
    if not isinstance(payload["errors"], list):
        raise TypeError("Representative Core report errors must be a list.")


def _approval_matches(payload: dict[str, Any], *, state: str, approved: bool) -> bool:
    approval = payload.get("approval")
    return (
        payload.get("approval_required") is True
        and isinstance(approval, dict)
        and approval.get("required") is True
        and approval.get("state") == state
        and approval.get("approved") is approved
    )


def _claims_execution(payload: dict[str, Any]) -> bool:
    result = payload.get("result")
    return isinstance(result, dict) and (result.get("executed") is True or result.get("success") is True)


def _validate_core_invoke_transition(payload: dict[str, Any]) -> None:
    """An initial Core call may only propose an approval-gated mutation or fail closed."""

    status = payload.get("status")
    if status == "pending_approval":
        if not _approval_matches(payload, state="pending", approved=False):
            raise ValueError("Initial Representative Core proposal has an invalid pending approval state.")
        if not isinstance(payload.get("command_proposal"), dict) or not payload.get("kubectl_preview"):
            raise ValueError("Initial Representative Core proposal must include a command and kubectl preview.")
        result = payload.get("result")
        if not isinstance(result, dict) or result.get("executed") is not False or _claims_execution(payload):
            raise ValueError("Initial Representative Core proposal must contain a not-executed result.")
        return
    if status == "blocked":
        approval = payload.get("approval")
        result = payload.get("result")
        if (
            isinstance(approval, dict)
            and approval.get("state") == "blocked"
            and approval.get("approved") is False
            and isinstance(result, dict)
            and result.get("executed") is False
            and not _claims_execution(payload)
        ):
            return
        raise ValueError("Blocked initial Representative Core response must prove it did not execute.")
    raise ValueError("Initial Representative Core call must not complete or reject a mutation before approval.")


def _validate_core_approval_transition(payload: dict[str, Any], *, approved: bool) -> None:
    """Bind the Core response state to the user's stored approval decision."""

    status = payload.get("status")
    if not approved:
        if status != "rejected" or not _approval_matches(payload, state="rejected", approved=False):
            raise ValueError("Rejected Representative Core approval must return the rejected state.")
        if _claims_execution(payload):
            raise ValueError("Rejected Representative Core approval cannot claim execution.")
        return
    if status == "completed":
        result = payload.get("result")
        if not _approval_matches(payload, state="approved", approved=True):
            raise ValueError("Completed Representative Core approval has an invalid approved state.")
        if not isinstance(result, dict) or result.get("executed") is not True or result.get("success") is not True:
            raise ValueError("Completed Representative Core approval must contain a successful executed result.")
        return
    if status == "blocked" and _approval_matches(payload, state="blocked", approved=False):
        result = payload.get("result")
        if isinstance(result, dict) and result.get("executed") is False and not _claims_execution(payload):
            return
    if status == "blocked" and _approval_matches(payload, state="approved", approved=True):
        result = payload.get("result")
        if isinstance(result, dict) and result.get("executed") is True and result.get("success") is False:
            return
    raise ValueError("Approved Representative Core request returned an invalid state transition.")


def _invoke_representative_core_http(intent: str, base_url: str, *, run_id: str | None = None, subtask_id: str | None = None, call_id: str | None = None, response_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    body = json.dumps(_with_correlation({"intent": intent}, run_id, subtask_id, call_id), ensure_ascii=False).encode("utf-8")
    request = Request(
        _representative_core_invoke_url(base_url),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=_timeout_seconds("REPRESENTATIVE_CORE_AGENT_TIMEOUT_SEC")) as response:  # noqa: S310 - operator configured URL.
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Representative Core Agent HTTP request failed: {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Representative Core Agent HTTP request failed: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Representative Core Agent HTTP response must be a JSON object.")
    _validate_response_contract(payload, response_schema, _validate_representative_core_report)
    return payload


def _invoke_representative_core_cli(intent: str, *, run_id: str | None = None, subtask_id: str | None = None, call_id: str | None = None, response_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    _ensure_path(REPRESENTATIVE_CORE_ROOT)
    if not REPRESENTATIVE_CORE_PYTHON.exists() or not REPRESENTATIVE_CORE_AGENT.exists():
        raise RuntimeError(
            "Representative Core Agent local runtime is not available. Set CORE_REPRESENTATIVE_AGENT_URL or install representative-core/.venv."
        )
    completed = subprocess.run(
        [str(REPRESENTATIVE_CORE_PYTHON), str(REPRESENTATIVE_CORE_AGENT), "--json", intent],
        cwd=str(REPRESENTATIVE_CORE_ROOT),
        check=True,
        capture_output=True,
        text=True,
        timeout=_timeout_seconds("REPRESENTATIVE_CORE_AGENT_TIMEOUT_SEC"),
        env=_subprocess_env(run_id, subtask_id, call_id, agent_root=REPRESENTATIVE_CORE_ROOT, python_path=REPRESENTATIVE_CORE_PYTHON),
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Representative Core Agent CLI did not return valid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Representative Core Agent CLI response must be a JSON object.")
    _validate_response_contract(payload, response_schema, _validate_representative_core_report)
    return payload


def invoke_representative_core_agent(intent: str, *, run_id: str | None = None, subtask_id: str | None = None, call_id: str | None = None, response_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    _load_env_file()
    core_url = os.getenv("CORE_REPRESENTATIVE_AGENT_URL", "").strip()
    if core_url:
        return _invoke_representative_core_http(intent, core_url, run_id=run_id, subtask_id=subtask_id, call_id=call_id, response_schema=response_schema)
    return _invoke_representative_core_cli(intent, run_id=run_id, subtask_id=subtask_id, call_id=call_id, response_schema=response_schema)


def _representative_core_approve_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    if url.endswith("/invoke"):
        url = url[: -len("/invoke")]
    return url if url.endswith("/approve") else f"{url}/approve"


def _approval_request_payload(
    report: dict[str, Any],
    *,
    approved: bool,
    approved_by: str | None,
    reason: str | None,
    run_id: str | None,
    subtask_id: str | None,
    call_id: str | None,
) -> dict[str, Any]:
    return _with_correlation(
        {"report": report, "approved": approved, "approved_by": approved_by, "reason": reason},
        run_id,
        subtask_id,
        call_id,
    )


def _approve_representative_core_http(
    report: dict[str, Any],
    *,
    approved: bool,
    approved_by: str | None = None,
    reason: str | None = None,
    base_url: str,
    run_id: str | None = None,
    subtask_id: str | None = None,
    call_id: str | None = None,
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = json.dumps(
        _approval_request_payload(
            report,
            approved=approved,
            approved_by=approved_by,
            reason=reason,
            run_id=run_id,
            subtask_id=subtask_id,
            call_id=call_id,
        ),
        ensure_ascii=False,
    ).encode("utf-8")
    request = Request(
        _representative_core_approve_url(base_url),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=_timeout_seconds("REPRESENTATIVE_CORE_AGENT_TIMEOUT_SEC")) as response:  # noqa: S310
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Representative Core Agent approve request failed: {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Representative Core Agent approve request failed: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Representative Core Agent approve response must be a JSON object.")
    _validate_response_contract(payload, response_schema, _validate_representative_core_report)
    return payload


def _approve_representative_core_cli(
    report: dict[str, Any],
    *,
    approved: bool,
    approved_by: str | None = None,
    reason: str | None = None,
    run_id: str | None = None,
    subtask_id: str | None = None,
    call_id: str | None = None,
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _ensure_path(REPRESENTATIVE_CORE_ROOT)
    if not REPRESENTATIVE_CORE_PYTHON.exists() or not REPRESENTATIVE_CORE_AGENT.exists():
        raise RuntimeError(
            "Representative Core Agent local runtime is not available. Set CORE_REPRESENTATIVE_AGENT_URL or install representative-core/.venv."
        )
    command = [str(REPRESENTATIVE_CORE_PYTHON), str(REPRESENTATIVE_CORE_AGENT), "--json"]
    command.append("--approve" if approved else "--reject")
    if approved_by:
        command.extend(["--approved-by", approved_by])
    if reason:
        command.extend(["--reason", reason])
    completed = subprocess.run(
        command,
        cwd=str(REPRESENTATIVE_CORE_ROOT),
        input=json.dumps(report, ensure_ascii=False),
        check=True,
        capture_output=True,
        text=True,
        timeout=_timeout_seconds("REPRESENTATIVE_CORE_AGENT_TIMEOUT_SEC"),
        env=_subprocess_env(run_id, subtask_id, call_id, agent_root=REPRESENTATIVE_CORE_ROOT, python_path=REPRESENTATIVE_CORE_PYTHON),
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Representative Core Agent approve CLI did not return valid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Representative Core Agent approve CLI response must be a JSON object.")
    _validate_response_contract(payload, response_schema, _validate_representative_core_report)
    return payload


def approve_representative_core_report(
    report: dict[str, Any],
    *,
    approved: bool,
    approved_by: str | None = None,
    reason: str | None = None,
    run_id: str | None = None,
    subtask_id: str | None = None,
    call_id: str | None = None,
    contract_ref: dict[str, Any] | None = None,
) -> dict[str, Any]:
    _load_env_file()
    call_id = call_id or new_call_id()
    if not isinstance(contract_ref, dict):
        raise ValueError("Representative Core Agent approval requires a MongoDB contract_ref.")
    contract = resolve_agent_communication_contract("representative-core-agent", contract_ref)
    if contract.get("schema_dialect") != "json-schema-subset-v1":
        raise ValueError("Representative Core Agent has no supported MongoDB communication contract for approval.")
    approval_request_schema = contract.get("approval_request_schema")
    approval_response_schema = contract.get("approval_response_schema")
    if not isinstance(approval_request_schema, dict) or not isinstance(approval_response_schema, dict):
        raise ValueError("Representative Core Agent MongoDB contract has no approval request/response schema.")
    validate_contract_schema(approval_request_schema, path="representative-core-agent.approval_request_schema")
    validate_contract_schema(approval_response_schema, path="representative-core-agent.approval_response_schema")
    validate_contract_payload(
        _approval_request_payload(
            report,
            approved=approved,
            approved_by=approved_by,
            reason=reason,
            run_id=run_id,
            subtask_id=subtask_id,
            call_id=call_id,
        ),
        approval_request_schema,
        path="$.representative_core_approval_request",
    )
    started = monotonic_ms()
    _emit_event(
        run_id,
        "planning-agent",
        "approval_forwarded",
        status="approved" if approved else "rejected",
        subtask_id=subtask_id,
        target_agent="representative-core-agent",
        request={"approved": approved, "approved_by": approved_by, "reason": reason},
        command_preview=report.get("kubectl_preview"),
        risk_level=report.get("risk_level"),
    )
    core_url = os.getenv("CORE_REPRESENTATIVE_AGENT_URL", "").strip()
    try:
        if core_url:
            payload = _approve_representative_core_http(
                report,
                approved=approved,
                approved_by=approved_by,
                reason=reason,
                base_url=core_url,
                run_id=run_id,
                subtask_id=subtask_id,
                call_id=call_id,
                response_schema=approval_response_schema,
            )
        else:
            payload = _approve_representative_core_cli(
                report,
                approved=approved,
                approved_by=approved_by,
                reason=reason,
                run_id=run_id,
                subtask_id=subtask_id,
                call_id=call_id,
                response_schema=approval_response_schema,
            )
        _validate_core_approval_transition(payload, approved=approved)
    except Exception as exc:
        _emit_event(
            run_id,
            "planning-agent",
            "approval_response",
            status="blocked",
            subtask_id=subtask_id,
            target_agent="representative-core-agent",
            duration_ms=monotonic_ms() - started,
            error=str(exc),
        )
        raise
    _emit_event(
        run_id,
        "planning-agent",
        "approval_response",
        status=str(payload.get("status")),
        subtask_id=subtask_id,
        target_agent="representative-core-agent",
        response=payload,
        command_preview=payload.get("kubectl_preview"),
        risk_level=payload.get("risk_level"),
        duration_ms=monotonic_ms() - started,
    )
    return payload


def _probe_invoke_url(base_url: str) -> str:
    url = base_url.rstrip("/")
    return url if url.endswith("/invoke") else f"{url}/invoke"


def _validate_monitoring_report(payload: dict[str, Any]) -> None:
    required = ("intent", "command", "evaluation_request", "evaluation_result", "scope", "metric", "window_seconds", "status", "results", "errors", "source")
    if tuple(payload.keys()) != required:
        raise ValueError("Monitoring report has an unexpected contract.")
    if payload["status"] not in {"completed", "partial", "blocked"}:
        raise ValueError("Monitoring report status is invalid.")
    if not isinstance(payload["evaluation_request"], dict) or not isinstance(payload["evaluation_result"], dict):
        raise TypeError("Monitoring report evaluation fields must be objects.")
    if not isinstance(payload["results"], list):
        raise TypeError("Monitoring report results must be a list.")
    if not isinstance(payload["errors"], list):
        raise TypeError("Monitoring report errors must be a list.")


def _invoke_probe_http(intent: str, base_url: str, *, run_id: str | None = None, subtask_id: str | None = None, call_id: str | None = None, evaluation_request: dict[str, Any] | None = None, response_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    request_payload = {"intent": intent}
    if isinstance(evaluation_request, dict):
        request_payload["evaluation_request"] = evaluation_request
    body = json.dumps(_with_correlation(request_payload, run_id, subtask_id, call_id), ensure_ascii=False).encode("utf-8")
    request = Request(
        _probe_invoke_url(base_url),
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=_timeout_seconds()) as response:  # noqa: S310 - operator configured URL.
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Probe agent HTTP request failed: {exc.code} {detail}") from exc
    except URLError as exc:
        raise RuntimeError(f"Probe agent HTTP request failed: {exc.reason}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Probe agent HTTP response must be a JSON object.")
    _validate_response_contract(payload, response_schema, _validate_monitoring_report)
    return payload


def _invoke_probe_cli(intent: str, *, run_id: str | None = None, subtask_id: str | None = None, call_id: str | None = None, evaluation_request: dict[str, Any] | None = None, response_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    _ensure_path(PROBE_ROOT)
    if not PROBE_PYTHON.exists() or not PROBE_AGENT.exists():
        raise RuntimeError("Probe agent local runtime is not available. Set PROBE_AGENT_URL or install probe/.venv.")
    command = [str(PROBE_PYTHON), str(PROBE_AGENT), "--json"]
    if isinstance(evaluation_request, dict):
        command.extend(["--evaluation-request-json", json.dumps(evaluation_request, ensure_ascii=False)])
    command.append(intent)
    completed = subprocess.run(
        command,
        cwd=str(PROBE_ROOT),
        check=True,
        capture_output=True,
        text=True,
        timeout=_timeout_seconds(),
        env=_subprocess_env(run_id, subtask_id, call_id, agent_root=PROBE_ROOT, python_path=PROBE_PYTHON),
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Probe agent CLI did not return valid JSON.") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("Probe agent CLI response must be a JSON object.")
    _validate_response_contract(payload, response_schema, _validate_monitoring_report)
    return payload


def invoke_probe_agent(intent: str, *, run_id: str | None = None, subtask_id: str | None = None, call_id: str | None = None, evaluation_request: dict[str, Any] | None = None, response_schema: dict[str, Any] | None = None) -> dict[str, Any]:
    _load_env_file()
    monitoring_url = os.getenv("PROBE_AGENT_URL", "").strip()
    if monitoring_url:
        return _invoke_probe_http(intent, monitoring_url, run_id=run_id, subtask_id=subtask_id, call_id=call_id, evaluation_request=evaluation_request, response_schema=response_schema)
    return _invoke_probe_cli(intent, run_id=run_id, subtask_id=subtask_id, call_id=call_id, evaluation_request=evaluation_request, response_schema=response_schema)


def invoke_kpi_advisor_agent(
    intent: str,
    evaluation_request: dict[str, Any],
    *,
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke KPI Advisor through its existing HTTP runtime."""
    base_url = os.getenv("KPI_ADVISOR_AGENT_URL", "http://127.0.0.1:8110").strip().rstrip("/")
    payload = {
        "intent": intent,
        "baseline_parameter_set": evaluation_request.get("baseline_parameter_set"),
        "time_step": evaluation_request.get("time_step"),
    }
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    request = Request(
        f"{base_url}/invoke",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=KPI_ADVISOR_TIMEOUT_SECONDS) as response:  # noqa: S310 - configured service URL.
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"KPI Advisor HTTP request failed: {exc.code} {detail}") from exc
    except (TimeoutError, URLError) as exc:
        raise RuntimeError(f"KPI Advisor HTTP request failed: {exc}") from exc
    if not isinstance(result, dict):
        raise RuntimeError("KPI Advisor HTTP response must be a JSON object")
    _validate_response_contract(result, response_schema, lambda value: None)
    return result


def _validate_monitoring_agent_report(payload: dict[str, Any]) -> None:
    required = (
        "run_id", "subtask_id", "status", "expected", "execution",
        "probe_report", "feedback", "planner_response", "errors", "source",
    )
    if tuple(payload.keys()) != required:
        raise ValueError("Monitoring Agent report has an unexpected contract.")
    if payload.get("status") not in {
        "healthy", "observed", "replan_required", "observation_unavailable",
        "awaiting_execution", "stopped",
    }:
        raise ValueError("Monitoring Agent report status is invalid.")


def invoke_monitoring_agent(
    planning_report: dict[str, Any],
    *,
    subtask_id: str,
    notify_planner: bool,
    notification_delivery: str,
    run_id: str | None = None,
    call_id: str | None = None,
    response_schema: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run Monitoring locally; it delegates all live data collection to Probe."""

    _load_env_file()
    _ensure_path(MONITORING_ROOT)
    if not MONITORING_PYTHON.exists() or not MONITORING_AGENT.exists():
        raise RuntimeError("Monitoring Agent local runtime is not available.")
    payload = {
        "planning_report": planning_report,
        "subtask_id": subtask_id,
        "notify_planner": notify_planner,
        "notification_delivery": notification_delivery,
    }
    expected = planning_report.get("subtask_plans", [])[-1].get("langgraph_spec", {}).get("evaluation_request", {})
    schedule = expected.get("schedule") if isinstance(expected, dict) else None
    duration = schedule.get("duration_seconds", 0) if isinstance(schedule, dict) else 0
    duration_seconds = int(duration) if isinstance(duration, int | float) and not isinstance(duration, bool) else 0
    command = [str(MONITORING_PYTHON), str(MONITORING_AGENT), "--json"]
    if not notify_planner:
        command.append("--no-notify")
    completed = subprocess.run(
        command,
        cwd=str(MONITORING_ROOT),
        input=json.dumps(payload, ensure_ascii=False),
        check=True,
        capture_output=True,
        text=True,
        timeout=max(_timeout_seconds("MONITORING_AGENT_TIMEOUT_SEC"), duration_seconds + 120),
        env=_subprocess_env(run_id, subtask_id, call_id, agent_root=MONITORING_ROOT, python_path=MONITORING_PYTHON),
    )
    try:
        report = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Monitoring Agent CLI did not return valid JSON.") from exc
    if not isinstance(report, dict):
        raise RuntimeError("Monitoring Agent CLI response must be a JSON object.")
    _validate_response_contract(report, response_schema, _validate_monitoring_agent_report)
    return report



def _agent_display_name(node: dict[str, Any]) -> str:
    return str(node.get("name") or node.get("id") or "agent")


def _is_completion_node(node: dict[str, Any]) -> bool:
    return node.get("role") == "completion_checker" or node.get("name") == "planning-completion-check"


def _graph_layout(
    langgraph_spec: dict[str, Any],
) -> tuple[
    list[dict[str, Any]],
    dict[str, dict[str, Any]],
    dict[str, list[dict[str, Any]]],
    dict[str, list[dict[str, Any]]],
]:
    raw_nodes = [node for node in langgraph_spec.get("nodes", []) if isinstance(node, dict)]
    node_ids = [str(node.get("id") or "") for node in raw_nodes]
    if any(not node_id for node_id in node_ids) or len(node_ids) != len(set(node_ids)):
        raise ValueError("duplicate_or_missing_node_id")
    nodes = dict(zip(node_ids, raw_nodes, strict=True))
    entrypoint = str(langgraph_spec.get("entrypoint") or "")
    if entrypoint not in nodes:
        raise ValueError("entrypoint_not_found")

    outgoing = {node_id: [] for node_id in nodes}
    incoming = {node_id: [] for node_id in nodes}
    for edge in langgraph_spec.get("edges", []):
        if not isinstance(edge, dict):
            raise ValueError("invalid_edge")
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source not in nodes:
            raise ValueError("edge_source_not_found")
        if target not in nodes:
            raise ValueError("edge_target_not_found")
        outgoing[source].append(edge)
        incoming[target].append(edge)

    terminal_nodes = {str(item) for item in langgraph_spec.get("terminal_nodes", [])}
    order: list[dict[str, Any]] = []
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError("cycle_not_supported")
        if node_id in visited:
            return
        visiting.add(node_id)
        order.append(nodes[node_id])
        if node_id not in terminal_nodes:
            for edge in outgoing[node_id]:
                visit(str(edge["target"]))
        visiting.remove(node_id)
        visited.add(node_id)

    visit(entrypoint)
    return order, nodes, outgoing, incoming


def _nearest_preceding_runtime_agents(
    node_id: str,
    nodes: dict[str, dict[str, Any]],
    incoming: dict[str, list[dict[str, Any]]],
) -> set[str]:
    pending = [str(edge["source"]) for edge in incoming[node_id]]
    visited: set[str] = set()
    found: set[str] = set()
    while pending:
        candidate = pending.pop()
        if candidate in visited:
            continue
        visited.add(candidate)
        node = nodes[candidate]
        if node.get("type") == "agent" and not _is_completion_node(node):
            found.add(candidate)
            continue
        pending.extend(str(edge["source"]) for edge in incoming[candidate])
    return found


def _selected_tools_after(
    node_id: str,
    nodes: dict[str, dict[str, Any]],
    outgoing: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    pending = [str(edge["target"]) for edge in outgoing[node_id]]
    visited: set[str] = set()
    tools: list[dict[str, Any]] = []
    while pending:
        candidate = pending.pop(0)
        if candidate in visited:
            continue
        visited.add(candidate)
        node = nodes[candidate]
        if node.get("type") != "tool":
            continue
        tools.append(node)
        pending.extend(str(edge["target"]) for edge in outgoing[candidate])
    return tools


def validate_subgraph_execution_topology(langgraph_spec: dict[str, Any]) -> list[dict[str, Any]]:
    """Validate a safe LangGraph DAG, including the sole conditional Core gate."""

    order, nodes, _outgoing, incoming = _graph_layout(langgraph_spec)
    if not order or order[0].get("type") != "agent" or _is_completion_node(order[0]):
        raise ValueError("entrypoint_must_be_an_agent")
    runtime_adapter_ids = {
        "probe-agent",
        "monitoring-agent",
        "representative-core-agent",
        KPI_ADVISOR_AGENT_ID,
    }
    runtime_agent_names = {
        "probe agent",
        "monitoring agent",
        "representative core agent",
        "kpi advisor agent",
    }
    terminal_node_ids = {str(node_id) for node_id in langgraph_spec.get("terminal_nodes", [])}
    for node in langgraph_spec.get("nodes", []):
        if not isinstance(node, dict):
            continue
        node_id = str(node.get("id") or "")
        node_name = str(node.get("name") or "").strip().lower()
        completion = _is_completion_node(node)
        if node_id in runtime_adapter_ids and node.get("type") != "agent":
            raise ValueError("runtime_agent_id_requires_agent_node_type")
        if "communication_contract" in node:
            raise ValueError("embedded_communication_contract_not_supported")
        has_contract_ref = isinstance(node.get("contract_ref"), dict)
        if not completion and node_id in runtime_adapter_ids and set(node).difference(PUBLIC_RUNTIME_AGENT_NODE_FIELDS):
            raise ValueError("runtime_agent_node_contains_unsupported_fields")
        if completion and (node_id in runtime_adapter_ids or has_contract_ref):
            raise ValueError("registered_agent_cannot_be_completion_checker")
        if node_id in terminal_node_ids and node_id in runtime_adapter_ids and (node.get("type") != "agent" or completion):
            raise ValueError("registered_agent_cannot_be_terminal_node")
        if not completion and has_contract_ref and node_id not in runtime_adapter_ids:
            raise ValueError("registered_agent_id_must_be_canonical")
        if not completion and node.get("type") != "agent" and node_name in runtime_agent_names:
            raise ValueError("runtime_agent_name_requires_agent_node_type")
        if not completion and node.get("type") == "tool" and node.get("source") == "planning_agent":
            raise ValueError("planning_agent_source_is_reserved_for_completion")
    runtime_agents = [
        (index, str(node.get("id") or ""))
        for index, node in enumerate(order)
        if node.get("type") == "agent"
        and node.get("role") != "completion_checker"
        and node.get("name") != "planning-completion-check"
    ]
    declared_runtime_agent_ids = [
        str(node.get("id") or "")
        for node in langgraph_spec.get("nodes", [])
        if isinstance(node, dict)
        and node.get("type") == "agent"
        and node.get("role") != "completion_checker"
        and node.get("name") != "planning-completion-check"
    ]
    if not runtime_agents:
        raise ValueError("no_executable_agent_node")
    if len(declared_runtime_agent_ids) != len(set(declared_runtime_agent_ids)):
        raise ValueError("duplicate_runtime_agent_node")
    if set(declared_runtime_agent_ids) != {node_id for _, node_id in runtime_agents}:
        raise ValueError("unreachable_runtime_agent_node")
    core_positions = [index for index, node_id in runtime_agents if node_id == "representative-core-agent"]
    monitoring_positions = [index for index, node_id in runtime_agents if node_id == "monitoring-agent"]
    probe_positions = [index for index, node_id in runtime_agents if node_id == "probe-agent"]

    if monitoring_positions and probe_positions:
        raise ValueError("monitoring_agent_must_delegate_probe_internally")
    if core_positions and probe_positions:
        if min(probe_positions) < min(core_positions):
            raise ValueError("legacy_probe_conditional_recovery_not_supported")
        raise ValueError("legacy_core_probe_post_action_path_not_supported")
    if monitoring_positions and core_positions:
        if len(monitoring_positions) != 1 or len(core_positions) != 1:
            raise ValueError("conditional_core_requires_one_monitoring_gate_and_one_core_action")
        monitoring_index = monitoring_positions[0]
        core_index = core_positions[0]
        preceding_agents = [node_id for index, node_id in runtime_agents if index < core_index]
        if monitoring_index >= core_index or not preceding_agents or preceding_agents[-1] != "monitoring-agent":
            raise ValueError("conditional_core_must_be_immediately_gated_by_monitoring_agent")
        gate_edges = [
            edge for edge in langgraph_spec.get("edges", [])
            if isinstance(edge, dict)
            and edge.get("target") == "representative-core-agent"
        ]
        if len(gate_edges) != 1 or gate_edges[0].get("condition") != "only_if_mismatch":
            raise ValueError("monitoring_core_edge_requires_only_if_mismatch")
    nodes = {str(node.get("id")): node for node in langgraph_spec.get("nodes", []) if isinstance(node, dict)}
    conditional_markers = ("mismatch", "only_if", "conditional", "if-mismatch", "if mismatch")
    for edge in langgraph_spec.get("edges", []):
        if not isinstance(edge, dict) or str(edge.get("target") or "") != "representative-core-agent":
            continue
        condition = str(edge.get("condition") or "").lower()
        if any(marker in condition for marker in conditional_markers):
            source = str(edge.get("source") or "")
            source_node = nodes.get(source, {})
            previous_agents = _nearest_preceding_runtime_agents(source, nodes, incoming)
            if source != "monitoring-agent" and not (source_node.get("type") == "tool" and previous_agents == {"monitoring-agent"}):
                raise ValueError("conditional_core_edge_must_originate_from_monitoring_gate")
    return order


def _agent_intent(original_intent: str, subtask: str, node: dict[str, Any], selected_tools: list[dict[str, Any]], previous_results: list[dict[str, Any]]) -> str:
    context = {
        "node_id": node.get("id"),
        "node_name": node.get("name"),
        "selected_tools": selected_tools,
        "previous_node_results": previous_results,
    }
    return f"Original intent: {original_intent}\nSubtask: {subtask}\nPlanning context: {json.dumps(context, ensure_ascii=False)}"


_THRESHOLD_COMPARATORS = {
    "==": lambda actual, expected: actual == expected,
    "!=": lambda actual, expected: actual != expected,
    ">": lambda actual, expected: actual > expected,
    ">=": lambda actual, expected: actual >= expected,
    "<": lambda actual, expected: actual < expected,
    "<=": lambda actual, expected: actual <= expected,
}


def _threshold_evidence_maps(evaluation: Any, expected_request: Any) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]] | None:
    if not isinstance(evaluation, dict) or evaluation.get("type") != "thresholds":
        return None
    if not isinstance(expected_request, dict) or expected_request.get("type") != "thresholds":
        return None
    planned_items = expected_request.get("checks")
    evidence_items = evaluation.get("checks")
    if not isinstance(planned_items, list) or not planned_items or not isinstance(evidence_items, list):
        return None
    planned = {
        check.get("check_id"): check
        for check in planned_items
        if isinstance(check, dict) and isinstance(check.get("check_id"), str) and check.get("check_id")
    }
    evidence = {
        check.get("check_id"): check
        for check in evidence_items
        if isinstance(check, dict) and isinstance(check.get("check_id"), str) and check.get("check_id")
    }
    if len(planned) != len(planned_items) or len(evidence) != len(evidence_items) or set(planned) != set(evidence):
        return None
    return planned, evidence


def _bound_threshold_comparison(check: dict[str, Any], planned: dict[str, Any], evaluation: dict[str, Any]) -> bool | None:
    if check.get("valid") is not True or check.get("observation") != planned.get("observation"):
        return None
    planned_expected = planned.get("expected")
    if not isinstance(planned_expected, dict) or check.get("operator") != planned_expected.get("operator"):
        return None
    actual = check.get("actual")
    expected = check.get("expected")
    operator = check.get("operator")
    if isinstance(actual, bool) or not isinstance(actual, int | float):
        return None
    if isinstance(expected, bool) or not isinstance(expected, int | float):
        return None
    if not math.isfinite(actual) or not math.isfinite(expected) or not isinstance(operator, str):
        return None
    literal_expected = planned_expected.get("value")
    if isinstance(literal_expected, bool) or (literal_expected is not None and expected != literal_expected):
        return None
    source_check_id = planned_expected.get("value_from_check_id")
    if isinstance(source_check_id, str):
        query_values = evaluation.get("query_values")
        source_value = query_values.get(source_check_id) if isinstance(query_values, dict) else None
        if isinstance(source_value, bool) or not isinstance(source_value, int | float):
            return None
        if not math.isfinite(source_value) or expected != source_value:
            return None
    comparator = _THRESHOLD_COMPARATORS.get(operator)
    return comparator(actual, expected) if comparator is not None else None


def _optional_check_is_skipped(check: dict[str, Any], planned: dict[str, Any]) -> bool:
    planned_expected = planned.get("expected")
    return (
        planned.get("optional") is True
        and check.get("observation") == planned.get("observation")
        and isinstance(planned_expected, dict)
        and check.get("operator") == planned_expected.get("operator")
        and check.get("skipped") is True
        and check.get("valid") is False
        and check.get("ok") is None
    )


def has_confirmed_threshold_mismatch(evaluation: Any, expected_request: Any) -> bool:
    maps = _threshold_evidence_maps(evaluation, expected_request)
    if maps is None or evaluation.get("status") != "failed":
        return False
    planned, evidence = maps
    mismatch = False
    for check_id, planned_check in planned.items():
        check = evidence[check_id]
        if _optional_check_is_skipped(check, planned_check):
            continue
        comparison = _bound_threshold_comparison(check, planned_check, evaluation)
        if comparison is None or check.get("ok") is not comparison:
            return False
        mismatch = mismatch or comparison is False
    return mismatch


def _has_confirmed_threshold_satisfaction(evaluation: Any, expected_request: Any) -> bool:
    maps = _threshold_evidence_maps(evaluation, expected_request)
    if maps is None or evaluation.get("status") != "passed":
        return False
    planned, evidence = maps
    verified = False
    for check_id, planned_check in planned.items():
        check = evidence[check_id]
        if _optional_check_is_skipped(check, planned_check):
            continue
        if check.get("ok") is not True or _bound_threshold_comparison(check, planned_check, evaluation) is not True:
            return False
        verified = True
    return verified


def monitoring_gate_outcome(report: Any) -> str:
    if not isinstance(report, dict) or not isinstance(report.get("expected"), dict):
        return "invalid"
    status = report.get("status")
    expected = report["expected"]
    probe = report.get("probe_report")
    evaluation = probe.get("evaluation_result") if isinstance(probe, dict) else None
    probe_matches = (
        isinstance(probe, dict)
        and probe.get("status") == "completed"
        and probe.get("evaluation_request") == expected
        and isinstance(evaluation, dict)
    )
    if status == "healthy":
        return "satisfied" if probe_matches and _has_confirmed_threshold_satisfaction(evaluation, expected) and report.get("feedback") is None else "invalid"
    if status != "replan_required":
        return "unavailable"
    feedback = report.get("feedback")
    if not isinstance(feedback, dict):
        return "invalid"
    if feedback.get("source") != "monitoring-agent" or feedback.get("status") != "replan_required":
        return "invalid"
    if feedback.get("expected") != expected:
        return "invalid"
    observed = feedback.get("observed")
    if not isinstance(observed, dict) or observed.get("status") != "failed":
        return "invalid"
    if feedback.get("reason") == "expected_network_state_not_observed":
        return "mismatch" if probe_matches and observed == evaluation and has_confirmed_threshold_mismatch(evaluation, expected) else "invalid"
    return "invalid"


def _adapter_completion(result: dict[str, Any]) -> str:
    status = result.get("status")
    runtime = result.get("runtime")
    if runtime == KPI_ADVISOR_AGENT_ID:
        return "completed" if status == "completed" else "advisor_unavailable"
    if status in {"pending_approval", "rejected", "blocked"}:
        return str(status)
    if runtime == "representative-core-agent":
        report = result.get("representative_core_report")
        if isinstance(report, dict):
            if report.get("status") in {"pending_approval", "rejected", "blocked"}:
                return str(report.get("status"))
            execution = report.get("result")
            if report.get("status") == "completed" and isinstance(execution, dict) and execution.get("success") is True:
                return "completed"
        return "failed"
    if runtime == "probe-agent":
        report = result.get("monitoring_report")
        if isinstance(report, dict):
            evaluation = report.get("evaluation_result")
            if isinstance(evaluation, dict) and evaluation.get("status") != "passed":
                return "failed"
            return "completed" if report.get("status") == "completed" else "failed"
        return "failed"
    if runtime == "monitoring-agent":
        report = result.get("monitoring_agent_report")
        if isinstance(report, dict):
            outcome = monitoring_gate_outcome(report)
            if outcome == "satisfied":
                return "completed"
            if outcome == "mismatch":
                return "failed"
            if report.get("status") == "stopped":
                return "rejected"
            return "blocked"
        return "failed"
    return "completed" if status == "completed" else "failed"


def _blocked_graph_result(
    *,
    subtask_id: str,
    subtask: str,
    representative_agent: dict[str, Any],
    langgraph_spec: dict[str, Any],
    reason: str,
    node_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "status": "blocked",
        "runtime": "subgraph-runner",
        "subtask_id": subtask_id,
        "subtask": subtask,
        "representative_agent_id": representative_agent["id"],
        "representative_agent_name": representative_agent["name"],
        "message": reason,
        "compiled": bool(langgraph_spec.get("compiled", False)),
        "node_results": node_results or [],
        "executed_node_ids": [str(item.get("node_id")) for item in node_results or [] if isinstance(item, dict)],
        "errors": [{"code": "subgraph_execution_blocked", "message": reason}],
    }


def _invoke_probe_node(
    *,
    original_intent: str,
    subtask_id: str,
    subtask: str,
    langgraph_spec: dict[str, Any],
    node: dict[str, Any],
    selected_tools: list[dict[str, Any]],
    previous_results: list[dict[str, Any]],
    run_id: str | None,
    call_id: str,
) -> dict[str, Any]:
    monitoring_intent = _agent_intent(original_intent, subtask, node, selected_tools, previous_results)
    started = monotonic_ms()
    evaluation_request = langgraph_spec.get("evaluation_request") if isinstance(langgraph_spec.get("evaluation_request"), dict) else {"type": "report_values", "completion_rule": "data_returned"}
    _emit_event(run_id, "planning-agent", "graph_node_started", status="running", subtask_id=subtask_id, target_agent="probe-agent", request={"node_id": node.get("id"), "selected_tools": selected_tools})
    _emit_event(run_id, "planning-agent", "subagent_request", status="running", subtask_id=subtask_id, target_agent="probe-agent", transport="http" if os.getenv("PROBE_AGENT_URL", "").strip() else "cli", request={"call_id": call_id, "intent": monitoring_intent, "evaluation_request": evaluation_request, "selected_tools": selected_tools})
    try:
        validate_contract_payload(
            _with_correlation(
                {"intent": monitoring_intent, "evaluation_request": evaluation_request},
                run_id,
                subtask_id,
                call_id,
            ),
            _node_contract_schema(node, "request_schema"),
            path="$.probe_request",
        )
        report = invoke_probe_agent(
            monitoring_intent,
            run_id=run_id,
            subtask_id=subtask_id,
            call_id=call_id,
            evaluation_request=evaluation_request,
            response_schema=_node_contract_schema(node, "response_schema"),
        )
        if report.get("evaluation_request") != evaluation_request:
            raise ValueError("Probe Agent response evaluation_request does not match the Planning request.")
    except Exception as exc:  # noqa: BLE001
        _emit_event(run_id, "planning-agent", "graph_node_blocked", status="blocked", subtask_id=subtask_id, target_agent="probe-agent", duration_ms=monotonic_ms() - started, error=str(exc))
        return {"status": "blocked", "runtime": "probe-agent", "subtask_id": subtask_id, "subtask": subtask, "representative_agent_id": str(node.get("id")), "representative_agent_name": _agent_display_name(node), "message": f"Probe Agent execution failed: {exc}", "compiled": bool(langgraph_spec.get("compiled", False)), "monitoring_report": None}
    result = {"status": report["status"], "runtime": "probe-agent", "subtask_id": subtask_id, "subtask": subtask, "representative_agent_id": str(node.get("id")), "representative_agent_name": _agent_display_name(node), "message": "Probe Agent execution completed.", "compiled": bool(langgraph_spec.get("compiled", False)), "monitoring_report": report}
    _emit_event(run_id, "planning-agent", "subagent_response", status=str(report.get("status")), subtask_id=subtask_id, target_agent="probe-agent", response=report, duration_ms=monotonic_ms() - started)
    _emit_event(run_id, "planning-agent", "graph_node_completed", status=_adapter_completion(result), subtask_id=subtask_id, target_agent="probe-agent", response=result, duration_ms=monotonic_ms() - started)
    return result


def _invoke_kpi_advisor_node(
    *,
    original_intent: str,
    subtask_id: str,
    subtask: str,
    langgraph_spec: dict[str, Any],
    node: dict[str, Any],
    run_id: str | None,
) -> dict[str, Any]:
    """Run the advisor and localize timeout/error as ``advisor_unavailable``."""
    started = monotonic_ms()
    evaluation_request = langgraph_spec.get("evaluation_request")
    _emit_event(
        run_id,
        "planning-agent",
        "graph_node_started",
        status="running",
        subtask_id=subtask_id,
        target_agent=KPI_ADVISOR_AGENT_ID,
        request={"node_id": node.get("id"), "evaluation_request": evaluation_request},
    )
    try:
        if not isinstance(evaluation_request, dict):
            raise ValueError("KPI Advisor requires a structured evaluation_request")
        request_payload = {
            "intent": original_intent,
            "baseline_parameter_set": evaluation_request.get("baseline_parameter_set"),
            "time_step": evaluation_request.get("time_step"),
        }
        validate_contract_payload(request_payload, _node_contract_schema(node, "request_schema"), path="$.kpi_advisor")
        report = invoke_kpi_advisor_agent(
            original_intent,
            evaluation_request,
            response_schema=_node_contract_schema(node, "response_schema"),
        )
    except Exception as exc:  # noqa: BLE001 - this node is explicitly failure-tolerant.
        result = {
            "status": "advisor_unavailable",
            "runtime": KPI_ADVISOR_AGENT_ID,
            "subtask_id": subtask_id,
            "subtask": subtask,
            "message": f"KPI Advisor is unavailable: {exc}",
            "error_code": "advisor_unavailable",
        }
        _emit_event(
            run_id,
            "planning-agent",
            "graph_node_completed",
            status="advisor_unavailable",
            subtask_id=subtask_id,
            target_agent=KPI_ADVISOR_AGENT_ID,
            response=result,
            duration_ms=monotonic_ms() - started,
        )
        return result
    result = {
        "status": "completed",
        "runtime": KPI_ADVISOR_AGENT_ID,
        "subtask_id": subtask_id,
        "subtask": subtask,
        "message": "KPI Advisor execution completed.",
        "kpi_advisor_report": report,
    }
    _emit_event(
        run_id,
        "planning-agent",
        "graph_node_completed",
        status="completed",
        subtask_id=subtask_id,
        target_agent=KPI_ADVISOR_AGENT_ID,
        response=result,
        duration_ms=monotonic_ms() - started,
    )
    return result


def _monitoring_planning_report(
    *,
    planning_report: dict[str, Any] | None,
    original_intent: str,
    subtask_id: str,
    subtask: str,
    langgraph_spec: dict[str, Any],
    node: dict[str, Any],
    previous_results: list[dict[str, Any]],
    run_id: str | None,
) -> dict[str, Any]:
    base = planning_report if isinstance(planning_report, dict) else {}
    plans = [
        plan for plan in base.get("subtask_plans", [])
        if isinstance(plan, dict) and plan.get("subtask_id") != subtask_id
    ]
    public_node = {key: value for key, value in node.items() if not key.startswith("_runtime_")}
    plans.append({
        "subtask_id": subtask_id,
        "subtask": subtask,
        "langgraph_spec": langgraph_spec,
        "representative_agent": public_node,
        "attempts": [],
        "status": "completed",
        "result": {
            "status": "completed",
            "runtime": "subgraph-runner",
            "node_results": previous_results,
        },
        "knowledge_context_used": bool(langgraph_spec.get("knowledge_context_used", False)),
    })
    return {
        "run_id": str(base.get("run_id") or run_id or default_run_id("planning-monitor")),
        "intent": str(base.get("intent") or original_intent),
        "subtask_plans": plans,
        "overall_status": "completed",
    }


def _invoke_monitoring_node(
    *,
    original_intent: str,
    subtask_id: str,
    subtask: str,
    langgraph_spec: dict[str, Any],
    node: dict[str, Any],
    previous_results: list[dict[str, Any]],
    planning_report: dict[str, Any] | None,
    run_id: str | None,
    call_id: str,
) -> dict[str, Any]:
    started = monotonic_ms()
    has_inline_core = any(
        isinstance(candidate, dict) and candidate.get("id") == "representative-core-agent"
        for candidate in langgraph_spec.get("nodes", [])
    )
    notify_planner = not has_inline_core
    notification_delivery = "response"
    snapshot = _monitoring_planning_report(
        planning_report=planning_report,
        original_intent=original_intent,
        subtask_id=subtask_id,
        subtask=subtask,
        langgraph_spec=langgraph_spec,
        node=node,
        previous_results=previous_results,
        run_id=run_id,
    )
    _emit_event(run_id, "planning-agent", "graph_node_started", status="running", subtask_id=subtask_id, target_agent="monitoring-agent", request={"node_id": node.get("id")})
    _emit_event(run_id, "planning-agent", "subagent_request", status="running", subtask_id=subtask_id, target_agent="monitoring-agent", transport="cli", request={"call_id": call_id, "planning_report": snapshot})
    try:
        validate_contract_payload(
            {
                "planning_report": snapshot,
                "subtask_id": subtask_id,
                "notify_planner": notify_planner,
                "notification_delivery": notification_delivery,
            },
            _node_contract_schema(node, "request_schema"),
            path="$.monitoring_request",
        )
        report = invoke_monitoring_agent(
            snapshot,
            subtask_id=subtask_id,
            notify_planner=notify_planner,
            notification_delivery=notification_delivery,
            run_id=run_id,
            call_id=call_id,
            response_schema=_node_contract_schema(node, "response_schema"),
        )
        if report.get("run_id") != snapshot["run_id"]:
            raise ValueError("Monitoring Agent response run_id does not match the Planning request.")
        if report.get("subtask_id") != subtask_id:
            raise ValueError("Monitoring Agent response subtask_id does not match the Planning request.")
        expected = langgraph_spec.get("evaluation_request")
        if report.get("expected") != expected:
            raise ValueError("Monitoring Agent response expected state does not match the Planning request.")
    except Exception as exc:  # noqa: BLE001
        _emit_event(run_id, "planning-agent", "graph_node_blocked", status="blocked", subtask_id=subtask_id, target_agent="monitoring-agent", duration_ms=monotonic_ms() - started, error=str(exc))
        return {
            "status": "blocked",
            "runtime": "monitoring-agent",
            "subtask_id": subtask_id,
            "subtask": subtask,
            "representative_agent_id": str(node.get("id")),
            "representative_agent_name": _agent_display_name(node),
            "message": f"Monitoring Agent execution failed: {exc}",
            "compiled": bool(langgraph_spec.get("compiled", False)),
            "monitoring_agent_report": None,
        }
    result = {
        "status": report["status"],
        "runtime": "monitoring-agent",
        "subtask_id": subtask_id,
        "subtask": subtask,
        "representative_agent_id": str(node.get("id")),
        "representative_agent_name": _agent_display_name(node),
        "message": "Monitoring Agent verified the expected network outcome.",
        "compiled": bool(langgraph_spec.get("compiled", False)),
        "monitoring_agent_report": report,
    }
    completion = _adapter_completion(result)
    _emit_event(run_id, "planning-agent", "subagent_response", status=str(report.get("status")), subtask_id=subtask_id, target_agent="monitoring-agent", response=report, duration_ms=monotonic_ms() - started)
    _emit_event(run_id, "planning-agent", "graph_node_completed", status=completion, subtask_id=subtask_id, target_agent="monitoring-agent", response=result, duration_ms=monotonic_ms() - started)
    return result


def _invoke_core_node(
    *,
    original_intent: str,
    subtask_id: str,
    subtask: str,
    langgraph_spec: dict[str, Any],
    node: dict[str, Any],
    selected_tools: list[dict[str, Any]],
    previous_results: list[dict[str, Any]],
    run_id: str | None,
    call_id: str,
) -> dict[str, Any]:
    core_intent = _agent_intent(original_intent, subtask, node, selected_tools, previous_results)
    started = monotonic_ms()
    _emit_event(run_id, "planning-agent", "graph_node_started", status="running", subtask_id=subtask_id, target_agent="representative-core-agent", request={"node_id": node.get("id"), "selected_tools": selected_tools})
    _emit_event(run_id, "planning-agent", "subagent_request", status="running", subtask_id=subtask_id, target_agent="representative-core-agent", transport="http" if os.getenv("CORE_REPRESENTATIVE_AGENT_URL", "").strip() else "cli", request={"call_id": call_id, "intent": core_intent, "selected_tools": selected_tools})
    try:
        validate_contract_payload(
            _with_correlation({"intent": core_intent}, run_id, subtask_id, call_id),
            _node_contract_schema(node, "request_schema"),
            path="$.representative_core_request",
        )
        report = invoke_representative_core_agent(
            core_intent,
            run_id=run_id,
            subtask_id=subtask_id,
            call_id=call_id,
            response_schema=_node_contract_schema(node, "response_schema"),
        )
        _validate_core_invoke_transition(report)
    except Exception as exc:  # noqa: BLE001
        _emit_event(run_id, "planning-agent", "graph_node_blocked", status="blocked", subtask_id=subtask_id, target_agent="representative-core-agent", duration_ms=monotonic_ms() - started, error=str(exc))
        return {"status": "blocked", "runtime": "representative-core-agent", "subtask_id": subtask_id, "subtask": subtask, "representative_agent_id": str(node.get("id")), "representative_agent_name": _agent_display_name(node), "message": f"Representative Core Agent execution failed: {exc}", "compiled": bool(langgraph_spec.get("compiled", False)), "representative_core_report": None}
    result = {"status": report["status"], "runtime": "representative-core-agent", "subtask_id": subtask_id, "subtask": subtask, "representative_agent_id": str(node.get("id")), "representative_agent_name": _agent_display_name(node), "message": "Representative Core Agent returned a kubectl approval request.", "compiled": bool(langgraph_spec.get("compiled", False)), "representative_core_report": report}
    _emit_event(run_id, "planning-agent", "subagent_response", status=str(report.get("status")), subtask_id=subtask_id, target_agent="representative-core-agent", response=report, command_preview=report.get("kubectl_preview"), risk_level=report.get("risk_level"), duration_ms=monotonic_ms() - started)
    event_type = "graph_node_pending_approval" if report.get("status") == "pending_approval" else "graph_node_completed"
    _emit_event(run_id, "planning-agent", event_type, status=_adapter_completion(result), subtask_id=subtask_id, target_agent="representative-core-agent", response=result, command_preview=report.get("kubectl_preview"), risk_level=report.get("risk_level"), duration_ms=monotonic_ms() - started)
    return result


def _result_with_graph(result: dict[str, Any], node_results: list[dict[str, Any]], langgraph_spec: dict[str, Any]) -> dict[str, Any]:
    return {
        **result,
        "compiled": True,
        "node_results": node_results,
        "executed_node_ids": [str(item.get("node_id")) for item in node_results if isinstance(item, dict)],
    }


class _SubgraphState(TypedDict):
    node_results: Annotated[list[dict[str, Any]], operator.add]


def _next_targets(
    node_id: str,
    nodes: dict[str, dict[str, Any]],
    outgoing: dict[str, list[dict[str, Any]]],
    terminal_nodes: set[str],
) -> list[str]:
    if node_id in terminal_nodes:
        return []
    targets: list[str] = []
    for edge in outgoing[node_id]:
        target = str(edge["target"])
        routed = END if _is_completion_node(nodes[target]) else target
        if routed not in targets:
            targets.append(routed)
    return targets


def _route_value(targets: list[str]) -> str | list[str]:
    if not targets:
        return END
    return targets[0] if len(targets) == 1 else targets


def _last_node_result(state: _SubgraphState, node_id: str) -> dict[str, Any] | None:
    return next(
        (item for item in reversed(state.get("node_results", [])) if item.get("node_id") == node_id),
        None,
    )


def _compile_runtime_graph(
    *,
    original_intent: str,
    subtask_id: str,
    subtask: str,
    langgraph_spec: dict[str, Any],
    runtime_nodes: dict[str, dict[str, Any]],
    run_id: str | None,
    call_id: str | None,
    planning_report: dict[str, Any] | None,
) -> tuple[Any, bool]:
    order, nodes, outgoing, incoming = _graph_layout(langgraph_spec)
    terminal_nodes = {str(item) for item in langgraph_spec.get("terminal_nodes", [])}
    conditional_core = (
        "monitoring-agent" in nodes
        and "representative-core-agent" in nodes
        and _nearest_preceding_runtime_agents("representative-core-agent", nodes, incoming) == {"monitoring-agent"}
    )
    builder = StateGraph(_SubgraphState)

    for node in order:
        node_id = str(node.get("id") or "")
        if _is_completion_node(node):
            continue
        node_type = str(node.get("type") or "")
        if node_type == "tool":
            builder.add_node(node_id, lambda _state: {})
            continue
        if node_type != "agent":
            raise ValueError(f"unsupported_node_type:{node_type or 'unknown'}")
        runtime_node = runtime_nodes[node_id]
        selected_tools = _selected_tools_after(node_id, nodes, outgoing)

        def run_agent(state: _SubgraphState, *, current_node: dict[str, Any] = runtime_node, tools: list[dict[str, Any]] = selected_tools) -> dict[str, Any]:
            current_id = str(current_node["id"])
            previous_results = list(state.get("node_results", []))
            call = call_id or new_call_id()
            try:
                resolved_node = {
                    **current_node,
                    "_runtime_communication_contract": resolve_agent_communication_contract(current_id, current_node.get("contract_ref")),
                }
                for schema_name in ("request_schema", "response_schema"):
                    validate_contract_schema(_node_contract_schema(resolved_node, schema_name), path=f"{current_id}.{schema_name}")
            except ValueError as exc:
                result = _blocked_graph_result(subtask_id=subtask_id, subtask=subtask, representative_agent=current_node, langgraph_spec=langgraph_spec, reason=f"agent_communication_contract_unavailable:{exc}", node_results=previous_results)
                return {"node_results": [{"node_id": current_id, "node_type": "agent", "runtime": result.get("runtime"), "status": "blocked", "result": result}]}
            if current_id == "probe-agent":
                result = _invoke_probe_node(original_intent=original_intent, subtask_id=subtask_id, subtask=subtask, langgraph_spec=langgraph_spec, node=resolved_node, selected_tools=tools, previous_results=previous_results, run_id=run_id, call_id=call)
            elif current_id == "monitoring-agent":
                result = _invoke_monitoring_node(original_intent=original_intent, subtask_id=subtask_id, subtask=subtask, langgraph_spec=langgraph_spec, node=resolved_node, previous_results=previous_results, planning_report=planning_report, run_id=run_id, call_id=call)
            elif current_id == "representative-core-agent":
                result = _invoke_core_node(original_intent=original_intent, subtask_id=subtask_id, subtask=subtask, langgraph_spec=langgraph_spec, node=resolved_node, selected_tools=tools, previous_results=previous_results, run_id=run_id, call_id=call)
            elif current_id == KPI_ADVISOR_AGENT_ID:
                result = _invoke_kpi_advisor_node(
                    original_intent=original_intent,
                    subtask_id=subtask_id,
                    subtask=subtask,
                    langgraph_spec=langgraph_spec,
                    node=resolved_node,
                    run_id=run_id,
                )
            else:
                result = _blocked_graph_result(subtask_id=subtask_id, subtask=subtask, representative_agent=current_node, langgraph_spec=langgraph_spec, reason=f"no_runtime_adapter:{current_id}", node_results=previous_results)
            return {"node_results": [{"node_id": current_id, "node_type": "agent", "runtime": result.get("runtime"), "status": _adapter_completion(result), "result": result}]}

        builder.add_node(node_id, run_agent)

    entrypoint = str(langgraph_spec["entrypoint"])
    if _is_completion_node(nodes[entrypoint]):
        raise ValueError("entrypoint_must_be_an_agent")
    builder.add_edge(START, entrypoint)

    for node in order:
        node_id = str(node.get("id") or "")
        if _is_completion_node(node):
            continue
        targets = _next_targets(node_id, nodes, outgoing, terminal_nodes)
        if node.get("type") == "tool":
            if targets:
                for target in targets:
                    builder.add_edge(node_id, target)
            else:
                builder.add_edge(node_id, END)
            continue

        def route_after_agent(state: _SubgraphState, *, current_id: str = node_id, next_targets: list[str] = targets) -> str | list[str]:
            node_result = _last_node_result(state, current_id)
            if not isinstance(node_result, dict):
                return END
            result = node_result.get("result")
            if current_id == "monitoring-agent" and conditional_core:
                report = result.get("monitoring_agent_report") if isinstance(result, dict) else None
                if monitoring_gate_outcome(report) != "mismatch":
                    return END
                return _route_value([target for target in next_targets if target != END])
            if node_result.get("status") not in {"completed", "advisor_unavailable"}:
                return END
            return _route_value(next_targets)

        builder.add_conditional_edges(node_id, route_after_agent)

    return builder.compile(), conditional_core


def execute_subgraph(
    *,
    original_intent: str,
    subtask_id: str,
    subtask: str,
    langgraph_spec: dict[str, Any],
    representative_agent: dict[str, Any],
    run_id: str | None = None,
    call_id: str | None = None,
    planning_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Compile and execute a registered subgraph with LangGraph."""

    try:
        order = validate_subgraph_execution_topology(langgraph_spec)
    except ValueError as exc:
        return _blocked_graph_result(subtask_id=subtask_id, subtask=subtask, representative_agent=representative_agent, langgraph_spec=langgraph_spec, reason=str(exc))
    runtime_nodes: dict[str, dict[str, Any]] = {}
    for node in order:
        if node.get("type") != "agent" or _is_completion_node(node):
            continue
        try:
            node_id = str(node.get("id") or "")
            reference = node.get("contract_ref")
            if not isinstance(reference, dict):
                raise ValueError(f"Agent {node_id!r} has no MongoDB contract_ref.")
            if set(reference) != {"agent_id", "version", "hash"} or reference.get("agent_id") != node_id:
                raise ValueError(f"Agent {node_id!r} has an invalid MongoDB contract_ref.")
            runtime_nodes[node_id] = node
        except ValueError as exc:
            return _blocked_graph_result(
                subtask_id=subtask_id,
                subtask=subtask,
                representative_agent=representative_agent,
                langgraph_spec=langgraph_spec,
                reason=f"agent_communication_contract_unavailable:{exc}",
            )
    try:
        graph, conditional_core = _compile_runtime_graph(
            original_intent=original_intent,
            subtask_id=subtask_id,
            subtask=subtask,
            langgraph_spec=langgraph_spec,
            runtime_nodes=runtime_nodes,
            run_id=run_id,
            call_id=call_id,
            planning_report=planning_report,
        )
        state = graph.invoke({"node_results": []}, {"recursion_limit": max(25, len(order) * 4)})
    except Exception as exc:  # noqa: BLE001 - convert LangGraph compile/runtime errors to the public blocked contract.
        return _blocked_graph_result(subtask_id=subtask_id, subtask=subtask, representative_agent=representative_agent, langgraph_spec=langgraph_spec, reason=f"langgraph_execution_failed:{exc}")

    node_results = [item for item in state.get("node_results", []) if isinstance(item, dict)]
    if not node_results:
        return _blocked_graph_result(subtask_id=subtask_id, subtask=subtask, representative_agent=representative_agent, langgraph_spec=langgraph_spec, reason="no_executable_agent_node", node_results=node_results)

    executed_ids = {str(item.get("node_id")) for item in node_results}
    if conditional_core and "monitoring-agent" in executed_ids and "representative-core-agent" not in executed_ids:
        monitoring_result = next((item.get("result") for item in reversed(node_results) if item.get("node_id") == "monitoring-agent"), None)
        report = monitoring_result.get("monitoring_agent_report") if isinstance(monitoring_result, dict) else None
        if monitoring_gate_outcome(report) == "satisfied":
            _emit_event(run_id, "planning-agent", "graph_node_skipped", status="completed", subtask_id=subtask_id, target_agent="representative-core-agent", response={"reason": "expected_network_state_already_observed", "gate_agent": "monitoring-agent"})
            return _result_with_graph({**monitoring_result, "status": "completed", "message": "Expected network state is already satisfied; conditional Core remediation was skipped."}, node_results, langgraph_spec)

    selected = (
        next((item for item in reversed(node_results) if item.get("status") == "pending_approval"), None)
        or next((item for item in reversed(node_results) if item.get("status") != "completed"), None)
        or node_results[-1]
    )
    result = selected.get("result") if isinstance(selected.get("result"), dict) else {}
    status = str(selected.get("status") or "failed")
    return _result_with_graph({**result, "status": status}, node_results, langgraph_spec)
