"""CLI and runtime for expected-outcome monitoring."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from schemas import MonitoringAgentReport, MonitoringRequest, validate_monitoring_report, validate_monitoring_request

JAECHAN_ROOT = Path(
    os.getenv("AGENTIC_AI_ROOT", str(Path(__file__).resolve().parents[2]))
).resolve()
PROJECT_ROOT = Path(__file__).resolve().parent
PROBE_ROOT = JAECHAN_ROOT / "agents" / "probe"
PLANNING_ROOT = JAECHAN_ROOT / "agents" / "planning"
_PROBE_VENV_PYTHON = PROBE_ROOT / ".venv" / "bin" / "python"
PROBE_PYTHON = _PROBE_VENV_PYTHON if os.access(_PROBE_VENV_PYTHON, os.X_OK) else Path(sys.executable)
PROBE_AGENT = PROBE_ROOT / "agent.py"
_PLANNING_VENV_PYTHON = PLANNING_ROOT / ".venv" / "bin" / "python"
PLANNING_PYTHON = _PLANNING_VENV_PYTHON if os.access(_PLANNING_VENV_PYTHON, os.X_OK) else Path(sys.executable)
PLANNING_AGENT = PLANNING_ROOT / "agent.py"

if str(JAECHAN_ROOT) not in sys.path:
    sys.path.append(str(JAECHAN_ROOT))

try:
    from agent_ops.telemetry import default_run_id, log_event  # noqa: E402
except ModuleNotFoundError:  # Local package tests can run outside the jaechan monorepo.
    def default_run_id(prefix: str) -> str:
        return f"{prefix}-{uuid.uuid4().hex}"

    def log_event(**_: Any) -> None:
        return None


class AgentConnectionError(RuntimeError):
    pass


def _load_env_path(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip('"').strip("'")


def load_env_file() -> None:
    _load_env_path(PROJECT_ROOT / ".env")
    _load_env_path(PLANNING_ROOT / ".env")


def _timeout(name: str, default: int = 240) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _emit_event(run_id: str, event_type: str, **kwargs: Any) -> None:
    try:
        log_event(run_id=run_id or default_run_id("monitoring"), agent="monitoring-agent", event_type=event_type, **kwargs)
    except Exception as exc:  # noqa: BLE001 - telemetry must not break monitoring.
        print(f"monitoring-agent telemetry warning: {exc}", file=sys.stderr)


def _post_json(url: str, payload: dict[str, Any], timeout: int, *, bearer_token: str | None = None) -> dict[str, Any]:
    headers = {"Content-Type": "application/json"}
    if bearer_token:
        headers["Authorization"] = f"Bearer {bearer_token}"
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310 - operator configured URL.
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        reason = exc.reason if isinstance(exc, URLError) else str(exc)
        raise AgentConnectionError(f"Cannot reach {url}: {reason}") from exc
    if not isinstance(result, dict):
        raise RuntimeError(f"{url} did not return a JSON object.")
    return result


def _invoke_url(base_url: str, endpoint: str) -> str:
    url = base_url.rstrip("/")
    return url if url.endswith(endpoint) else f"{url}{endpoint}"


def _subprocess_env(run_id: str, subtask_id: str) -> dict[str, str]:
    env = dict(os.environ)
    env["AGENT_RUN_ID"] = run_id
    env["AGENT_SUBTASK_ID"] = subtask_id
    env["AGENT_CALL_ID"] = f"monitoring-{uuid.uuid4().hex}"
    return env


def _validate_probe_report(payload: dict[str, Any]) -> None:
    required = (
        "intent",
        "command",
        "evaluation_request",
        "evaluation_result",
        "scope",
        "metric",
        "window_seconds",
        "status",
        "results",
        "errors",
        "source",
    )
    if tuple(payload.keys()) != required or payload.get("status") not in {"completed", "partial", "blocked"}:
        raise ValueError("Probe Agent returned an invalid monitoring report.")
    if not isinstance(payload.get("evaluation_result"), dict):
        raise TypeError("Probe Agent evaluation_result must be an object.")


def invoke_probe_agent(
    intent: str,
    evaluation_request: dict[str, Any],
    *,
    run_id: str,
    subtask_id: str,
) -> dict[str, Any]:
    base_url = os.getenv("PROBE_AGENT_URL", "").strip()
    payload = {"intent": intent, "evaluation_request": evaluation_request}
    if base_url:
        try:
            report = _post_json(_invoke_url(base_url, "/invoke"), payload, _timeout("PROBE_AGENT_TIMEOUT_SEC"))
        except AgentConnectionError:
            report = None
    else:
        report = None
    if report is None:
        if not PROBE_PYTHON.exists() or not PROBE_AGENT.exists():
            raise RuntimeError("Probe runtime is unavailable. Set PROBE_AGENT_URL or install probe/.venv.")
        command = [
            str(PROBE_PYTHON),
            str(PROBE_AGENT),
            "--json",
            "--evaluation-request-json",
            json.dumps(evaluation_request, ensure_ascii=False),
            intent,
        ]
        completed = subprocess.run(
            command,
            cwd=str(PROBE_ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=_timeout("PROBE_AGENT_TIMEOUT_SEC"),
            env=_subprocess_env(run_id, subtask_id),
        )
        try:
            report = json.loads(completed.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Probe Agent CLI did not return valid JSON.") from exc
    if not isinstance(report, dict):
        raise RuntimeError("Probe Agent response must be a JSON object.")
    _validate_probe_report(report)
    return report


def notify_planning_agent(feedback: dict[str, Any]) -> dict[str, Any]:
    base_url = os.getenv("PLANNING_AGENT_URL", "").strip()
    if base_url:
        token = os.getenv("MONITORING_AGENT_TOKEN", "").strip()
        if token:
            try:
                return _post_json(
                    _invoke_url(base_url, "/feedback"),
                    feedback,
                    _timeout("PLANNING_AGENT_TIMEOUT_SEC"),
                    bearer_token=token,
                )
            except AgentConnectionError:
                pass
    if not PLANNING_PYTHON.exists() or not PLANNING_AGENT.exists():
        suffix = " and MONITORING_AGENT_TOKEN" if base_url else ""
        raise RuntimeError(f"Planning runtime is unavailable. Set PLANNING_AGENT_URL{suffix} or install planning/.venv.")
    completed = subprocess.run(
        [str(PLANNING_PYTHON), str(PLANNING_AGENT), "--json", "--monitoring-feedback"],
        cwd=str(PLANNING_ROOT),
        input=json.dumps(feedback, ensure_ascii=False),
        check=True,
        capture_output=True,
        text=True,
        timeout=_timeout("PLANNING_AGENT_TIMEOUT_SEC"),
        env=_subprocess_env(str(feedback["run_id"]), str(feedback["subtask_id"])),
    )
    try:
        response = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Planning Agent CLI did not return valid JSON.") from exc
    if not isinstance(response, dict):
        raise RuntimeError("Planning Agent response must be a JSON object.")
    return response


def select_plan(request: MonitoringRequest) -> tuple[dict[str, Any], int]:
    plans = request.planning_report["subtask_plans"]
    if request.subtask_id:
        for index, plan in enumerate(plans):
            if plan.get("subtask_id") == request.subtask_id:
                return plan, index
        raise ValueError(f"subtask_id not found: {request.subtask_id}")
    return plans[-1], len(plans) - 1


def _latest_core_report(plans: list[dict[str, Any]], through_index: int) -> dict[str, Any] | None:
    def find(result: Any) -> dict[str, Any] | None:
        if not isinstance(result, dict):
            return None
        report = result.get("representative_core_report")
        if isinstance(report, dict):
            return report
        node_results = result.get("node_results")
        if isinstance(node_results, list):
            for node_result in reversed(node_results):
                found = find(node_result.get("result") if isinstance(node_result, dict) else None)
                if found:
                    return found
        return None

    for plan in reversed(plans[: through_index + 1]):
        report = find(plan.get("result"))
        if report:
            return report
    return None


def execution_summary(planning_report: dict[str, Any], plan: dict[str, Any], index: int) -> dict[str, Any]:
    result = plan.get("result") if isinstance(plan.get("result"), dict) else {}
    core_report = _latest_core_report(planning_report["subtask_plans"], index)
    action_result = core_report.get("result") if isinstance(core_report, dict) and isinstance(core_report.get("result"), dict) else {}
    return {
        "plan_status": plan.get("status"),
        "runtime_status": result.get("status"),
        "representative_agent": plan.get("representative_agent"),
        "action_status": core_report.get("status") if isinstance(core_report, dict) else None,
        "action_success": action_result.get("success") if isinstance(action_result.get("success"), bool) else None,
        "previous_action": core_report.get("command_proposal") if isinstance(core_report, dict) else None,
    }


def plan_fingerprint(plan: dict[str, Any]) -> str:
    snapshot = {
        "subtask_id": plan.get("subtask_id"),
        "langgraph_spec": plan.get("langgraph_spec", {}),
    }
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def probe_intent(planning_report: dict[str, Any], plan: dict[str, Any], expected: dict[str, Any], execution: dict[str, Any]) -> str:
    return (
        f"Original intent: {planning_report['intent']}\n"
        f"Subtask to verify: {plan.get('subtask', '')}\n"
        f"Expected outcome from Planning Agent: {json.dumps(expected, ensure_ascii=False)}\n"
        f"Previous action: {json.dumps(execution.get('previous_action'), ensure_ascii=False)}\n"
        "Use the existing Probe query catalog to collect current RAN/Core evidence for this exact expectation."
    )


def feedback_event(
    planning_report: dict[str, Any],
    plan: dict[str, Any],
    expected: dict[str, Any],
    execution: dict[str, Any],
    observed: dict[str, Any],
    *,
    reason: str,
    probe_errors: list[dict[str, Any]],
) -> dict[str, Any]:
    attempt = len(plan.get("attempts", [])) if isinstance(plan.get("attempts"), list) else 0
    identity = json.dumps(
        [planning_report["run_id"], plan["subtask_id"], attempt, expected, reason],
        sort_keys=True,
        ensure_ascii=False,
    ).encode("utf-8")
    return {
        "event_id": f"monitoring-{hashlib.sha256(identity).hexdigest()[:24]}",
        "source": "monitoring-agent",
        "run_id": planning_report["run_id"],
        "subtask_id": plan["subtask_id"],
        "plan_fingerprint": plan_fingerprint(plan),
        "status": "replan_required",
        "reason": reason,
        "expected": expected,
        "observed": observed,
        "previous_action": execution.get("previous_action"),
        "probe_errors": probe_errors,
    }


def _build_report(
    *,
    run_id: str,
    subtask_id: str,
    status: str,
    expected: dict[str, Any],
    execution: dict[str, Any],
    probe_report: dict[str, Any] | None = None,
    feedback: dict[str, Any] | None = None,
    planner_response: dict[str, Any] | None = None,
    errors: list[dict[str, Any]] | None = None,
) -> MonitoringAgentReport:
    report = MonitoringAgentReport(
        run_id=run_id,
        subtask_id=subtask_id,
        status=status,  # type: ignore[arg-type]
        expected=expected,
        execution=execution,
        probe_report=probe_report,
        feedback=feedback,
        planner_response=planner_response,
        errors=errors or [],
        source={"probe": "probe-agent", "planning_feedback": "planning-agent"},
    )
    validate_monitoring_report(report.to_dict())
    return report


def _monitoring_schedule(expected: dict[str, Any]) -> tuple[int, int | None]:
    schedule = expected.get("schedule")
    if not isinstance(schedule, dict):
        raise ValueError("Planning Agent must provide a schedule object.")
    duration = schedule.get("duration_seconds")
    if not isinstance(duration, int | float) or isinstance(duration, bool) or duration < 0:
        raise ValueError("Planning Agent schedule must provide a non-negative duration_seconds.")
    duration_seconds = int(duration)
    if duration_seconds == 0:
        return 0, None
    interval = schedule.get("interval_seconds")
    if not isinstance(interval, int | float) or isinstance(interval, bool) or interval <= 0:
        raise ValueError("Planning Agent schedule must provide a positive interval_seconds for timed monitoring.")
    return duration_seconds, int(interval)


def run_monitoring(payload: dict[str, Any]) -> MonitoringAgentReport:
    load_env_file()
    request = validate_monitoring_request(payload)
    plan, index = select_plan(request)
    run_id = request.planning_report["run_id"]
    subtask_id = plan.get("subtask_id")
    if not isinstance(subtask_id, str) or not subtask_id:
        raise ValueError("Selected subtask plan has no subtask_id.")
    spec = plan.get("langgraph_spec") if isinstance(plan.get("langgraph_spec"), dict) else {}
    expected = spec.get("evaluation_request") if isinstance(spec.get("evaluation_request"), dict) else {"type": "report_values", "completion_rule": "data_returned"}
    execution = execution_summary(request.planning_report, plan, index)
    _emit_event(run_id, "run_started", status="running", subtask_id=subtask_id, request={"expected": expected})

    if plan.get("status") == "pending_approval" or execution.get("action_status") == "pending_approval":
        return _build_report(run_id=run_id, subtask_id=subtask_id, status="awaiting_execution", expected=expected, execution=execution)
    if plan.get("status") == "rejected" or execution.get("action_status") == "rejected":
        return _build_report(run_id=run_id, subtask_id=subtask_id, status="stopped", expected=expected, execution=execution)

    errors: list[dict[str, Any]] = []
    probe_report: dict[str, Any] | None = None
    probe_reports: list[dict[str, Any]] = []
    feedback = None
    status = "running"
    try:
        duration, interval = _monitoring_schedule(expected)
    except ValueError as exc:
        errors.append({"stage": "planning_contract", "error": str(exc)})
        status = "observation_unavailable"
        duration = 0
        interval = None
    if status == "observation_unavailable":
        observation_intent = ""
    else:
        observation_intent = probe_intent(request.planning_report, plan, expected, execution)
    started = time.monotonic()
    while status != "observation_unavailable":
        try:
            probe_report = invoke_probe_agent(
                observation_intent,
                expected,
                run_id=run_id,
                subtask_id=subtask_id,
            )
        except Exception as exc:  # noqa: BLE001 - observation failure is not outcome mismatch.
            errors.append({"stage": "probe", "error": str(exc)})
            status = "observation_unavailable"
            break
        probe_reports.append(probe_report)
        sample_number = len(probe_reports)
        _emit_event(run_id, "monitoring_sample", status=str(probe_report.get("status")), subtask_id=subtask_id, response={"sample_number": sample_number, "evaluation_result": probe_report.get("evaluation_result")})
        if probe_report["evaluation_request"] != expected:
            status = "observation_unavailable"
            errors.append({"stage": "probe_contract", "error": "Probe evaluation_request does not match Planning expectation."})
            break
        evaluation_status = probe_report["evaluation_result"].get("status")
        if probe_report["status"] != "completed" or evaluation_status not in {"passed", "failed"}:
            status = "observation_unavailable"
            break
        if evaluation_status == "failed":
            status = "replan_required"
            break
        elapsed = time.monotonic() - started
        if elapsed >= duration:
            status = "healthy" if expected.get("type") == "thresholds" else "observed"
            break
        time.sleep(min(interval or duration, max(0.0, duration - elapsed)))

    if probe_reports:
        probe_report = dict(probe_reports[-1])
        if duration:
            probe_report["monitoring_duration_seconds"] = duration
            probe_report["monitoring_samples"] = [
                {
                    "sample_number": sample_number,
                    "collection_status": report.get("status"),
                    "evaluation_status": report.get("evaluation_result", {}).get("status"),
                }
                for sample_number, report in enumerate(probe_reports, start=1)
            ]
    if status == "replan_required" and probe_report is not None:
        feedback = feedback_event(
            request.planning_report,
            plan,
            expected,
            execution,
            probe_report["evaluation_result"],
            reason="expected_network_state_not_observed",
            probe_errors=probe_report.get("errors", []),
        )

    planner_response = None
    if feedback and request.notify_planner and request.notification_delivery == "direct":
        try:
            planner_response = notify_planning_agent(feedback)
        except Exception as exc:  # noqa: BLE001 - preserve mismatch evidence for retry.
            errors.append({"stage": "planning_feedback", "error": str(exc)})
    _emit_event(run_id, "run_finished", status=status, subtask_id=subtask_id, response={"feedback": feedback}, error=errors or None)
    return _build_report(
        run_id=run_id,
        subtask_id=subtask_id,
        status=status,
        expected=expected,
        execution=execution,
        probe_report=probe_report,
        feedback=feedback,
        planner_response=planner_response,
        errors=errors,
    )


def read_json_input(path: str | None) -> dict[str, Any]:
    text = Path(path).read_text(encoding="utf-8") if path else sys.stdin.read()
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise TypeError("Monitoring input must be a JSON object.")
    return payload


def format_agent_section(payload: dict[str, Any]) -> str:
    return f"------monitoring-agent------\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n----end----"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify a Planning Agent result with live Probe evidence.")
    parser.add_argument("--input", help="Planning/monitoring request JSON. Reads stdin when omitted.")
    parser.add_argument("--json", action="store_true", help="Print only JSON.")
    parser.add_argument("--no-notify", action="store_true", help="Do not send mismatch feedback to Planning Agent.")
    args = parser.parse_args(argv)
    try:
        payload = read_json_input(args.input)
        if args.no_notify:
            payload["notify_planner"] = False
        report = run_monitoring(payload).to_dict()
    except Exception as exc:  # noqa: BLE001 - CLI should surface validation/runtime failures.
        print(f"monitoring-agent error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else format_agent_section(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
