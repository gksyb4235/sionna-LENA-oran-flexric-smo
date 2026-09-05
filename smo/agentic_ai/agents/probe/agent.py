"""CLI and core runtime for the Probe Agent."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from schemas import MonitoringCommand, MonitoringReport, validate_monitoring_command, validate_monitoring_report
from tools.grafana_mcp_tools import (
    CORE_FREE5GC_NFS,
    _extract_instant_values,
    build_nf_ready_pod_promql,
    build_nf_replica_promql,
    expand_influxql_macros,
    extract_influx_series,
    extract_influx_values,
    list_free5gc_monitoring_targets,
    load_grafana_config,
    monitor_target_cpu_average,
    monitor_target_replica_health,
    monitoring_source,
    query_influxql,
    query_prometheus,
)

JAECHAN_ROOT = Path(
    os.getenv("AGENTIC_AI_ROOT", str(Path(__file__).resolve().parents[2]))
).resolve()
PROJECT_ROOT = Path(__file__).resolve().parent
SKILL_PATH = PROJECT_ROOT / "skills" / "probe-tools" / "SKILL.md"
DECOMPOSITION_ENV = JAECHAN_ROOT / "agents" / "decomposition" / ".env"
if str(JAECHAN_ROOT) not in sys.path:
    sys.path.append(str(JAECHAN_ROOT))

from agent_ops.llm_models import model_for  # noqa: E402
from agent_ops.telemetry import default_run_id, log_event  # noqa: E402
NF_ASSIGNMENT_PATTERN = re.compile(r"\b(?:nf|NF)\s*[:=]\s*([A-Za-z0-9_.:-]+)")
KNOWN_NF_PATTERN = re.compile(r"\b(AMF|SMF|UPF|NRF|AUSF|UDM|UDR|PCF|NSSF)s?\b", re.IGNORECASE)
QUOTED_PATTERN = re.compile(r"['\"]([A-Za-z0-9_.:-]+)['\"]")
UPPER_TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Z0-9_.:-]{1,}\b")
IGNORED_UPPER_TOKENS = {"CPU", "NF", "NFS", "RAM", "MEM", "GPU"}
REPLICA_COUNT_PATTERN = re.compile(r"(?i)(?:\b(\d+)\s*(?:amfs?|replicas?|pods?)\b|\b(?:replicas?|replica count|amfs?)\D{0,24}(\d+)\b)")
REPLICA_HEALTH_TERMS = (
    "replica",
    "replicas",
    "amfs",
    "deployment",
    "pod",
    "pods",
    "ready",
    "available",
    "healthy",
    "health",
    "operating properly",
    "going well",
    "well",
)
CPU_TERMS = ("cpu", "usage", "utilization", "cpu usage")
COMPARATORS = {
    "==": lambda actual, expected: actual == expected,
    "!=": lambda actual, expected: actual != expected,
    ">=": lambda actual, expected: actual >= expected,
    ">": lambda actual, expected: actual > expected,
    "<=": lambda actual, expected: actual <= expected,
    "<": lambda actual, expected: actual < expected,
}
MAX_QUERY_RETRIES = 5
LEARNED_CATALOG_START = "<!-- learned-query-catalog:start -->"
LEARNED_CATALOG_END = "<!-- learned-query-catalog:end -->"
INFLUXQL_SELECT_PATTERN = re.compile(r"(?is)^SELECT\s+(.+?)\s+(FROM\s+.+)$")
PROMETHEUS_METRIC_PATTERN = re.compile(r"^[A-Za-z_:][A-Za-z0-9_:]*$")

def prefer_venv_deepagents_package() -> None:
    """Ensure the installed deepagents package wins over the local source checkout."""

    import sysconfig

    site_paths: list[str] = []
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    venv_site = Path(sys.prefix) / "lib" / version / "site-packages"
    if venv_site.exists():
        site_paths.append(str(venv_site))
    purelib = sysconfig.get_paths().get("purelib")
    if purelib and purelib not in site_paths:
        site_paths.append(purelib)

    for site_path in reversed(site_paths):
        if site_path in sys.path:
            sys.path.remove(site_path)
        sys.path.insert(0, site_path)

    loaded = sys.modules.get("deepagents")
    if loaded is None:
        return
    module_file = getattr(loaded, "__file__", None)
    module_paths = [str(item) for item in getattr(loaded, "__path__", [])]
    locations = [str(module_file or ""), *module_paths]
    if any("/.venv/" in item for item in locations):
        return
    for name in [name for name in sys.modules if name == "deepagents" or name.startswith("deepagents.")]:
        sys.modules.pop(name, None)




def _emit_event(event_type: str, **kwargs: Any) -> None:
    try:
        log_event(
            run_id=default_run_id("monitoring"),
            agent="probe-agent",
            event_type=event_type,
            subtask_id=os.getenv("AGENT_SUBTASK_ID") or None,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must not break monitoring.
        print(f"probe-agent telemetry warning: {exc}", file=sys.stderr)

def ensure_project_scope() -> None:
    project_root = PROJECT_ROOT.resolve()
    if project_root != JAECHAN_ROOT and JAECHAN_ROOT not in project_root.parents:
        raise RuntimeError(f"Project root must stay under {JAECHAN_ROOT}; got {project_root}")


def _load_env_path(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_env_file() -> None:
    _load_env_path(PROJECT_ROOT / ".env")
    if not os.getenv("OPENAI_API_KEY"):
        os.environ.pop("OPENAI_API_KEY", None)
        _load_env_path(DECOMPOSITION_ENV)


def load_agent_instructions() -> str:
    path = PROJECT_ROOT / "agent.md"
    if not path.exists():
        raise FileNotFoundError(f"Missing monitoring instructions: {path}")
    return path.read_text(encoding="utf-8")


def _content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                text = block.get("text") or block.get("content")
                parts.append(text if isinstance(text, str) else json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    first_object: dict[str, Any] | None = None
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
        if isinstance(payload, dict):
            if first_object is None:
                first_object = payload
            if {"operation", "metric", "scope"}.issubset(payload.keys()):
                return payload
        index += max(end, 1)
    if first_object is not None:
        return first_object
    raise ValueError("Monitoring LLM response did not contain a JSON object.")


def create_probe_agent():
    """Create the DeepAgents-backed monitoring command harness."""

    prefer_venv_deepagents_package()
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    ensure_project_scope()
    load_env_file()
    model = model_for("probe-agent")
    backend = FilesystemBackend(root_dir=PROJECT_ROOT, virtual_mode=True)
    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt=load_agent_instructions(),
        skills=["/skills"],
        backend=backend,
    )


def infer_nf_from_intent(intent: str) -> str:
    """Infer one NF identifier from a monitoring intent for deterministic fallbacks."""

    for pattern in (NF_ASSIGNMENT_PATTERN, QUOTED_PATTERN):
        match = pattern.search(intent)
        if match:
            return match.group(1)
    known_match = KNOWN_NF_PATTERN.search(intent)
    if known_match:
        return known_match.group(1).upper()
    for token in UPPER_TOKEN_PATTERN.findall(intent):
        if token.upper() not in IGNORED_UPPER_TOKENS:
            return token
    raise ValueError("Could not infer NF identifier from intent. Pass --nf explicitly or ask for each NF.")


def build_command_prompt(
    intent: str,
    nf: str | None = None,
    *,
    evaluation_request: dict[str, Any] | None = None,
    retry_context: dict[str, Any] | None = None,
    attempt: int = 1,
) -> str:
    explicit_hint = nf.strip() if isinstance(nf, str) and nf.strip() else ""
    retry_text = (
        "\nPrevious failed attempt:\n"
        f"{json.dumps(retry_context, ensure_ascii=False, indent=2)}\n"
        "Create a corrected command. Do not repeat an unchanged query unless the error is clearly transient.\n"
        if retry_context
        else ""
    )
    return f"""Convert the monitoring request into one JSON monitoring command.

User intent:
{intent}

Explicit NF override:
{explicit_hint or '(none)'}

Planning Agent evaluation request:
{json.dumps(evaluation_request or {}, ensure_ascii=False, indent=2)}

Attempt:
{attempt} of {MAX_QUERY_RETRIES + 1}
{retry_text}
The runtime can execute read-only PromQL and InfluxQL through Grafana. Use the existing skill catalog first. For a monitoring request not yet in the catalog, create the smallest read-only query plan that can answer it.

Command fields:
- operation: free-form operation name. Use `prometheus_query_plan` for PromQL and `influxql_query_plan` for InfluxQL.
- metric: free-form metric or health dimension, for example cpu_usage, replica_health, memory_usage, restart_count, latency.
- scope: free-form scope label, for example single_nf, explicit_nfs, core_free5gc_nfs, pod, deployment, service.
- nfs: explicit NF names when relevant.
- window_seconds: integer seconds, default 60.
- expected_replicas: integer expected replica count when stated, otherwise null.
- queries: list of query specs. PromQL uses tool=query_prometheus and query_type=instant. InfluxQL uses tool=query_influxql and query_type=influxql. Each query also has label, target, query, purpose, and optional unit.
- evaluation: copy the Planning Agent evaluation request without changing its contract.

Selection rules:
- Choose the metric that answers the current subtask. A request asking whether replicas/pods/deployments are going well, ready, available, healthy, or operating properly is replica/deployment health, not CPU.
- Use CPU only when CPU usage/utilization/average is explicitly requested.
- Use InfluxQL for RAN KPM data from measurement `kpm`; keep Grafana macros `$timeFilter` and `$__interval` in the command.
- For future metrics, prefer a read-only query plan over inventing a new fixed operation.
- If an explicit NF override is present, use that NF as the target.
- For every threshold check, translate its observation into the smallest read-only query and copy its opaque `check_id` unchanged into the matching query spec. The query `label` is display-only.
- Preserve an evaluation check's expected reducer. Without one, multi-sample lower bounds use min and upper bounds use max.

Return exactly one JSON object with exactly these keys:
- operation
- metric
- scope
- nfs
- window_seconds
- expected_replicas
- queries
- evaluation
"""


def _expected_replicas_from_intent(intent: str) -> int | None:
    for match in REPLICA_COUNT_PATTERN.finditer(intent):
        for group in match.groups():
            if group:
                try:
                    return int(group)
                except ValueError:
                    continue
    return None


def _intent_requests_replica_health(intent: str) -> bool:
    lower = intent.lower()
    has_replica_signal = any(term in lower for term in REPLICA_HEALTH_TERMS)
    has_cpu_signal = any(term in lower for term in CPU_TERMS)
    return has_replica_signal and not (has_cpu_signal and "replica" not in lower and "deployment" not in lower)


def _operation_for_intent(intent: str) -> tuple[str, str, int | None]:
    if _intent_requests_replica_health(intent):
        return "prometheus_query_plan", "replica_health", _expected_replicas_from_intent(intent)
    return "cpu_average", "cpu_usage", None


def _replica_health_plan(nfs: list[str], expected_replicas: int | None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    queries: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    for nf in nfs:
        target = nf.upper()
        nf_value = nf.lower()
        metrics = [
            (
                "desired replicas",
                "desired deployment replica count",
                build_nf_replica_promql(nf_value, "kube_deployment_spec_replicas"),
            ),
            (
                "available replicas",
                "available deployment replica count",
                build_nf_replica_promql(nf_value, "kube_deployment_status_replicas_available"),
            ),
            (
                "ready replicas",
                "ready deployment replica count",
                build_nf_replica_promql(nf_value, "kube_deployment_status_replicas_ready"),
            ),
            ("ready pods", "ready pod count", build_nf_ready_pod_promql(nf_value)),
        ]
        query_ids: list[str] = []
        for label_suffix, observation_suffix, query in metrics:
            check_id = f"probe-check-{len(queries) + 1}"
            query_ids.append(check_id)
            queries.append(
                {
                    "tool": "query_prometheus",
                    "check_id": check_id,
                    "label": f"{target} {label_suffix}",
                    "target": target,
                    "query": query,
                    "purpose": f"{target} {observation_suffix}",
                    "query_type": "instant",
                    "unit": "replicas",
                },
            )
        expected: dict[str, Any]
        if expected_replicas is None:
            expected = {"operator": ">=", "value_from_check_id": query_ids[0]}
        else:
            expected = {"operator": "==", "value": expected_replicas}
            checks.append(
                {
                    "check_id": query_ids[0],
                    "observation": f"{target} desired deployment replica count",
                    "expected": {"operator": "==", "value": expected_replicas},
                    "optional": True,
                }
            )
        checks.extend(
            [
                {
                    "check_id": query_ids[1],
                    "observation": f"{target} available deployment replica count",
                    "expected": dict(expected),
                    "optional": False,
                },
                {
                    "check_id": query_ids[2],
                    "observation": f"{target} ready deployment replica count",
                    "expected": dict(expected),
                    "optional": True,
                },
                {
                    "check_id": query_ids[3],
                    "observation": f"{target} ready pod count",
                    "expected": dict(expected),
                    "optional": True,
                },
            ]
        )
    return queries, {"type": "thresholds", "checks": checks}


def _command_with_evaluation_request(command: MonitoringCommand, evaluation_request: dict[str, Any] | None) -> MonitoringCommand:
    if not isinstance(evaluation_request, dict):
        return command
    payload = command.to_dict()
    payload["evaluation"] = evaluation_request
    return validate_monitoring_command(payload)


def _heuristic_monitoring_command(intent: str, nf: str | None = None) -> MonitoringCommand:
    operation, metric, expected_replicas = _operation_for_intent(intent)
    if nf and nf.strip():
        scope = "single_nf"
        nfs = [nf.strip().upper()]
    else:
        lower = intent.lower()
        if operation == "cpu_average" and "pod" in lower and ("all" in lower or "every" in lower):
            scope = "all_free5gc_pods"
            nfs = []
        elif "each" in lower or "every nf" in lower or "all nf" in lower:
            scope = "core_free5gc_nfs"
            nfs = [nf_name.upper() for nf_name in CORE_FREE5GC_NFS]
        else:
            scope = "single_nf"
            nfs = [infer_nf_from_intent(intent).upper()]

    if operation == "prometheus_query_plan" and metric == "replica_health":
        plan_nfs = nfs or [nf_name.upper() for nf_name in CORE_FREE5GC_NFS]
        queries, evaluation = _replica_health_plan(plan_nfs, expected_replicas)
        return validate_monitoring_command(
            {
                "operation": "prometheus_query_plan",
                "metric": "replica_health",
                "scope": scope,
                "nfs": plan_nfs,
                "window_seconds": 60,
                "expected_replicas": expected_replicas,
                "queries": queries,
                "evaluation": evaluation,
            }
        )

    return validate_monitoring_command(
        {
            "operation": operation,
            "metric": metric,
            "scope": scope,
            "nfs": nfs,
            "window_seconds": 60,
            "expected_replicas": expected_replicas,
            "queries": [],
            "evaluation": {},
        }
    )


def build_monitoring_command(
    intent: str,
    *,
    nf: str | None = None,
    evaluation_request: dict[str, Any] | None = None,
    retry_context: dict[str, Any] | None = None,
    attempt: int = 1,
) -> MonitoringCommand:
    ensure_project_scope()
    load_env_file()
    if nf and nf.strip() and not isinstance(evaluation_request, dict):
        command = _heuristic_monitoring_command(intent, nf=nf)
        return _command_with_evaluation_request(command, evaluation_request)
    if os.getenv("MONITORING_USE_DETERMINISTIC_TEST_COMMAND") == "1" and not isinstance(evaluation_request, dict):
        command = _heuristic_monitoring_command(intent, nf=nf)
        return _command_with_evaluation_request(command, evaluation_request)

    agent = create_probe_agent()
    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": build_command_prompt(
                        intent,
                        nf=nf,
                        evaluation_request=evaluation_request,
                        retry_context=retry_context,
                        attempt=attempt,
                    ),
                }
            ]
        }
    )
    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not messages:
        raise RuntimeError("Monitoring LLM returned no messages.")
    payload = extract_json_object(_content_to_text(getattr(messages[-1], "content", messages[-1])))
    command = validate_monitoring_command(payload)
    if not isinstance(evaluation_request, dict) and _intent_requests_replica_health(intent) and command.metric == "cpu_usage":
        command = _heuristic_monitoring_command(intent, nf=nf)
    return _command_with_evaluation_request(command, evaluation_request)


def _targets_for_command(command: MonitoringCommand) -> list[dict[str, str]]:
    if command.scope in {"single_nf", "explicit_nfs"}:
        return [
            {"name": nf.upper(), "query_value": nf.lower(), "target_type": "core_nf", "label": "container"}
            for nf in command.nfs
        ]
    if command.scope in {"core_free5gc_nfs", "all_free5gc_pods", "active_free5gc_containers"}:
        return list_free5gc_monitoring_targets(command.scope)
    raise ValueError(f"Unsupported monitoring scope for legacy execution: {command.scope}")


def _resolve_expected_value(expected: dict[str, Any], values_by_check_id: dict[str, float]) -> float | None:
    if "value" in expected:
        value = expected["value"]
        return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None
    value_from_check_id = expected.get("value_from_check_id")
    if isinstance(value_from_check_id, str):
        return values_by_check_id.get(value_from_check_id)
    return None


def _evaluation_request(command: MonitoringCommand) -> dict[str, Any]:
    if command.evaluation:
        return command.evaluation
    return {"type": "report_values", "completion_rule": "data_returned"}


def _threshold_actual(item: dict[str, Any] | None, operator: str, reducer: Any) -> tuple[float | None, str | None, str | None]:
    if not isinstance(item, dict):
        return None, None, "missing query result"
    raw_values = item.get("values")
    values = [float(value) for value in raw_values if isinstance(value, (int, float)) and not isinstance(value, bool)] if isinstance(raw_values, list) else []
    if not values and isinstance(item.get("first_value"), (int, float)) and not isinstance(item.get("first_value"), bool):
        values = [float(item["first_value"])]
    if not values:
        return None, None, "missing actual value"
    if len(values) == 1:
        return values[0], "single", None
    selected = reducer if isinstance(reducer, str) else None
    if selected is None:
        selected = "min" if operator in {">", ">="} else "max" if operator in {"<", "<="} else None
    reducers = {
        "min": min,
        "max": max,
        "mean": lambda items: sum(items) / len(items),
        "sum": sum,
        "first": lambda items: items[0],
        "last": lambda items: items[-1],
    }
    reduce_values = reducers.get(selected)
    if reduce_values is None:
        return None, selected, "multiple samples require a supported reducer"
    return float(reduce_values(values)), selected, None


def _evaluate_thresholds(command: MonitoringCommand, query_results: list[dict[str, Any]]) -> dict[str, Any]:
    evaluation = _evaluation_request(command)
    if evaluation.get("type") != "thresholds":
        values_present = any(item.get("sample_count", 0) > 0 for item in query_results)
        return {
            "type": evaluation.get("type", "report_values"),
            "status": "passed" if values_present else "failed",
            "values_present": values_present,
            "checks": [],
        }

    results_by_check_id = {
        item["check_id"]: item
        for item in query_results
        if isinstance(item.get("check_id"), str)
    }
    values_by_check_id = {
        item["check_id"]: item["first_value"]
        for item in query_results
        if item.get("sample_count") == 1
        and isinstance(item.get("check_id"), str)
        and isinstance(item.get("first_value"), (int, float))
    }
    check_results: list[dict[str, Any]] = []
    for raw_check in evaluation.get("checks", []):
        if not isinstance(raw_check, dict):
            continue
        check_id = raw_check.get("check_id")
        observation = str(raw_check.get("observation") or "check")
        expected_request = raw_check.get("expected") if isinstance(raw_check.get("expected"), dict) else {}
        operator = str(expected_request.get("operator", ""))
        optional = bool(raw_check.get("optional", False))
        item = results_by_check_id.get(check_id) if isinstance(check_id, str) else None
        actual, reducer, actual_reason = _threshold_actual(item, operator, expected_request.get("reducer"))
        expected = _resolve_expected_value(expected_request, values_by_check_id)
        if actual is None or expected is None:
            check_results.append(
                {
                    "check_id": check_id,
                    "observation": observation,
                    "ok": None,
                    "valid": False,
                    "skipped": optional,
                    "actual": actual,
                    "expected": expected,
                    "operator": operator,
                    "reducer": reducer,
                    "reason": actual_reason or "missing expected value",
                }
            )
            continue
        comparator = COMPARATORS.get(operator)
        if comparator is None:
            check_results.append(
                {
                    "check_id": check_id,
                    "observation": observation,
                    "ok": None,
                    "valid": False,
                    "skipped": optional,
                    "actual": actual,
                    "expected": expected,
                    "operator": operator,
                    "reducer": reducer,
                    "reason": "unsupported operator",
                }
            )
            continue
        check_results.append(
            {
                "check_id": check_id,
                "observation": observation,
                "ok": bool(comparator(actual, expected)),
                "valid": True,
                "actual": actual,
                "expected": expected,
                "operator": operator,
                "reducer": reducer,
            }
        )

    mandatory_results = [item for item in check_results if not item.get("skipped")]
    if not mandatory_results:
        health_status = "unknown"
        status = "unknown"
    elif any(item.get("valid") is False for item in mandatory_results):
        health_status = "unknown"
        status = "unknown"
    elif all(item.get("ok") for item in mandatory_results):
        health_status = "healthy"
        status = "passed"
    else:
        health_status = "unhealthy"
        status = "failed"
    return {
        "type": "thresholds",
        "status": status,
        "health_status": health_status,
        "healthy": health_status == "healthy",
        "checks": check_results,
        "query_values": values_by_check_id,
    }


def execute_query_plan(intent: str, command: MonitoringCommand) -> MonitoringReport:
    cfg = load_grafana_config()
    query_time = time.time()
    query_results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for spec in command.queries:
        tool = spec.get("tool", "query_prometheus")
        check_id = spec.get("check_id")
        label = spec.get("label", "query")
        target = spec.get("target", "")
        query = spec.get("query", "")
        try:
            if tool == "query_influxql":
                payload = query_influxql(query, config=cfg, window_seconds=command.window_seconds)
                values = extract_influx_values(payload)
                series = extract_influx_series(payload)
                executed_query = expand_influxql_macros(query, command.window_seconds)
            else:
                payload = query_prometheus(query, config=cfg, timestamp=query_time)
                values = _extract_instant_values(payload)
                series = []
                executed_query = query
        except Exception as exc:  # noqa: BLE001 - keep partial query-plan reports.
            errors.append({"check_id": check_id, "label": label, "target": target, "tool": tool, "query": query, "error": str(exc)})
            continue

        entry = {
            "check_id": check_id,
            "label": label,
            "target": target,
            "tool": tool,
            "metric": command.metric,
            "purpose": spec.get("purpose"),
            "unit": spec.get("unit"),
            "query": query,
            "executed_query": executed_query,
            "values": values,
            "first_value": values[0] if values else None,
            "sample_count": len(values),
            "time_range": {"end_unix": query_time, "start_unix": query_time - command.window_seconds},
        }
        if series:
            entry["series"] = series
        query_results.append(entry)
        if not values:
            errors.append(
                {
                    "check_id": check_id,
                    "label": label,
                    "target": target,
                    "tool": tool,
                    "query": query,
                    "error": "No samples returned.",
                }
            )

    evaluation_request = _evaluation_request(command)
    evaluation_result = _evaluate_thresholds(command, query_results)
    result = {
        "target": ",".join(command.nfs) if command.nfs else command.scope,
        "target_type": command.scope,
        "metric": command.metric,
        "operation": command.operation,
        "window_seconds": command.window_seconds,
        "expected_replicas": command.expected_replicas,
        "query_results": query_results,
        "evaluation_request": evaluation_request,
        "evaluation_result": evaluation_result,
        "evaluation": evaluation_result,
    }
    results = [result] if query_results else []
    has_samples = any(item.get("sample_count", 0) > 0 for item in query_results)
    skipped_check_ids = {
        item.get("check_id")
        for item in evaluation_result.get("checks", [])
        if isinstance(item, dict) and item.get("skipped") is True
    }
    blocking_errors = [error for error in errors if error.get("check_id") not in skipped_check_ids]
    evaluation_known = evaluation_result.get("status") in {"passed", "failed"}
    if has_samples and evaluation_known and not blocking_errors:
        status = "completed"
    elif has_samples:
        status = "partial"
    else:
        status = "blocked"

    report = MonitoringReport(
        intent=intent,
        command=command.to_dict(),
        evaluation_request=evaluation_request,
        evaluation_result=evaluation_result,
        scope=command.scope,
        metric=command.metric,
        window_seconds=command.window_seconds,
        status=status,  # type: ignore[arg-type]
        results=results,
        errors=errors,
        source=monitoring_source(),
    )
    validate_monitoring_report(report.to_dict())
    return report


def _evaluate_legacy_results(evaluation_request: dict[str, Any], results: list[dict[str, Any]], status: str) -> dict[str, Any]:
    if evaluation_request.get("type") == "thresholds":
        raise ValueError("Threshold evaluation requires a query plan with matching check_id values.")
    values_present = status in {"completed", "partial"} and bool(results)
    return {
        "type": evaluation_request.get("type", "report_values"),
        "status": "passed" if values_present else "failed",
        "values_present": values_present,
        "checks": [],
    }


def execute_monitoring_command(intent: str, command: MonitoringCommand) -> MonitoringReport:
    if command.queries:
        return execute_query_plan(intent, command)

    targets = _targets_for_command(command)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    for target in targets:
        try:
            if command.operation == "replica_health":
                result = monitor_target_replica_health(target, expected_replicas=command.expected_replicas)
            else:
                result = monitor_target_cpu_average(target)
        except Exception as exc:  # noqa: BLE001 - keep monitoring partial when one target fails.
            errors.append(
                {
                    "target": target.get("name"),
                    "target_type": target.get("target_type"),
                    "error": str(exc),
                }
            )
            continue
        results.append(result)

    if results and not errors:
        status = "completed"
    elif results and errors:
        status = "partial"
    else:
        status = "blocked"

    evaluation_request = _evaluation_request(command)
    evaluation_result = _evaluate_legacy_results(evaluation_request, results, status)

    report = MonitoringReport(
        intent=intent,
        command=command.to_dict(),
        evaluation_request=evaluation_request,
        evaluation_result=evaluation_result,
        scope=command.scope,
        metric=command.metric,
        window_seconds=command.window_seconds,
        status=status,  # type: ignore[arg-type]
        results=results,
        errors=errors,
        source=monitoring_source(),
    )
    validate_monitoring_report(report.to_dict())
    return report


def _catalog_entry(spec: dict[str, Any]) -> tuple[str, str, str | None, str | None]:
    tool = str(spec.get("tool", "query_prometheus"))
    query = str(spec.get("query", "")).strip()
    if tool == "query_influxql":
        match = INFLUXQL_SELECT_PATTERN.match(query)
        if match:
            return tool, f"SELECT <fields> {match.group(2).strip()}", "fields", match.group(1).strip()
    if tool == "query_prometheus" and PROMETHEUS_METRIC_PATTERN.fullmatch(query):
        return tool, "<metric>", "metric", query
    if tool == "query_prometheus":
        for nf in CORE_FREE5GC_NFS:
            nf_pattern = re.compile(rf"(?i)(?<![A-Za-z0-9]){re.escape(nf)}(?![A-Za-z0-9])")
            if nf_pattern.search(query):
                return tool, nf_pattern.sub("<nf>", query), "nf", nf
    return tool, query, None, None


def _add_catalog_entry(content: str, spec: dict[str, Any]) -> str:
    tool, pattern, variable_name, variable = _catalog_entry(spec)
    pattern_line = f"- `{tool}`: `{pattern}`"
    variable_line = f"  - {variable_name}: `{variable}`" if variable_name and variable else None
    start = content.find(pattern_line)

    if start >= 0:
        block_candidates = [
            position
            for position in (
                content.find("\n- `", start + len(pattern_line)),
                content.find("\n## ", start + len(pattern_line)),
                content.find(LEARNED_CATALOG_END, start + len(pattern_line)),
            )
            if position >= 0
        ]
        end = min(block_candidates) if block_candidates else len(content)
        if not variable_line or variable_line in content[start:end]:
            return content
        return content[:end].rstrip() + f"\n{variable_line}\n\n" + content[end:].lstrip("\n")

    entry = pattern_line + (f"\n{variable_line}" if variable_line else "")
    marker = content.find(LEARNED_CATALOG_END)
    if marker < 0:
        return (
            content.rstrip()
            + "\n\n## Learned Query Patterns\n\n"
            + LEARNED_CATALOG_START
            + f"\n{entry}\n"
            + LEARNED_CATALOG_END
            + "\n"
        )
    return content[:marker].rstrip() + f"\n\n{entry}\n" + content[marker:]


def _learn_successful_queries(command: MonitoringCommand, report: MonitoringReport) -> bool:
    successful_queries = {
        (query_result.get("tool"), query_result.get("query"))
        for result in report.results
        for query_result in result.get("query_results", [])
        if isinstance(query_result, dict) and query_result.get("sample_count", 0) > 0
    }
    specs = [spec for spec in command.queries if (spec.get("tool"), spec.get("query")) in successful_queries]
    if not specs:
        return False

    skill_path = SKILL_PATH.resolve()
    if PROJECT_ROOT.resolve() not in skill_path.parents:
        raise RuntimeError(f"Skill path must stay under {PROJECT_ROOT}; got {skill_path}")
    content = skill_path.read_text(encoding="utf-8")
    updated = content
    for spec in specs:
        updated = _add_catalog_entry(updated, spec)
    if updated == content:
        return False

    # ponytail: last writer wins; add file locking if concurrent skill learning is enabled.
    temporary_path = skill_path.with_name(f".{skill_path.name}.{os.getpid()}.tmp")
    temporary_path.write_text(updated, encoding="utf-8")
    temporary_path.replace(skill_path)
    return True


def run_monitoring(intent: str, *, nf: str | None = None, evaluation_request: dict[str, Any] | None = None) -> MonitoringReport:
    _emit_event("run_started", status="running", request={"intent": intent, "nf": nf, "evaluation_request": evaluation_request})
    try:
        retry_context: dict[str, Any] | None = None
        for attempt in range(1, MAX_QUERY_RETRIES + 2):
            try:
                command = build_monitoring_command(
                    intent,
                    nf=nf,
                    evaluation_request=evaluation_request,
                    retry_context=retry_context,
                    attempt=attempt,
                )
            except (TypeError, ValueError) as exc:
                if attempt > MAX_QUERY_RETRIES:
                    raise
                retry_context = {"attempt": attempt, "status": "invalid_command", "errors": [{"error": str(exc)}]}
                continue
            _emit_event(
                "command_built",
                status="completed",
                request={"attempt": attempt, "max_retries": MAX_QUERY_RETRIES},
                response=command.to_dict(),
            )
            report = execute_monitoring_command(intent, command)
            if (report.status == "completed" and report.evaluation_result.get("status") != "unknown") or not command.queries:
                break
            retry_context = {
                "attempt": attempt,
                "command": command.to_dict(),
                "status": report.status,
                "errors": report.errors,
                "evaluation_result": report.evaluation_result,
            }
        if report.status == "completed" and command.queries:
            try:
                if _learn_successful_queries(command, report):
                    _emit_event("skill_updated", status="completed", response={"path": str(SKILL_PATH)})
            except Exception as exc:  # noqa: BLE001 - skill learning must not discard a valid report.
                _emit_event("skill_update_failed", status="partial", error=str(exc))
    except Exception as exc:
        _emit_event("run_finished", status="blocked", error=str(exc))
        raise
    _emit_event("run_finished", status=report.status, response=report.to_dict())
    return report


def format_agent_section(agent_name: str, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"------{agent_name}------\n{body}\n----end----"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Monitor free5gc targets through Grafana/Prometheus.")
    parser.add_argument("intent", type=str, help="Natural language monitoring intent.")
    parser.add_argument("--nf", type=str, help="Explicit NF identifier. Overrides inference from intent.")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Print only JSON without CLI section markers.")
    parser.add_argument("--evaluation-request-json", type=str, help="Planning Agent evaluation request JSON.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        evaluation_request = json.loads(args.evaluation_request_json) if args.evaluation_request_json else None
        if evaluation_request is not None and not isinstance(evaluation_request, dict):
            raise ValueError("--evaluation-request-json must decode to an object.")
        report = run_monitoring(args.intent, nf=args.nf, evaluation_request=evaluation_request).to_dict()
    except Exception as exc:  # noqa: BLE001 - CLI should surface operational failures.
        print(f"probe-agent error: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_agent_section("probe-agent", report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
