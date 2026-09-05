"""Schemas for the Probe Agent."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Literal

Status = Literal["completed", "partial", "blocked"]

LEGACY_OPERATIONS = {"cpu_average", "replica_health"}
LEGACY_SCOPES = {"single_nf", "explicit_nfs", "core_free5gc_nfs", "all_free5gc_pods", "active_free5gc_containers"}
READ_ONLY_QUERY_TOOLS = {"query_prometheus", "query_influxql"}
COMMAND_KEYS = ("operation", "metric", "scope", "nfs", "window_seconds", "expected_replicas", "queries", "evaluation")
REPORT_KEYS = ("intent", "command", "evaluation_request", "evaluation_result", "scope", "metric", "window_seconds", "status", "results", "errors", "source")
MAX_QUERY_LENGTH = 5000
INFLUXQL_WRITE_PATTERN = re.compile(r"(?i)\b(INTO|DROP|DELETE|CREATE|ALTER|GRANT|REVOKE)\b")


@dataclass(frozen=True)
class MonitoringCommand:
    operation: str
    metric: str
    scope: str
    nfs: list[str]
    window_seconds: int
    expected_replicas: int | None = None
    queries: list[dict[str, Any]] = field(default_factory=list)
    evaluation: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "metric": self.metric,
            "scope": self.scope,
            "nfs": self.nfs,
            "window_seconds": self.window_seconds,
            "expected_replicas": self.expected_replicas,
            "queries": self.queries,
            "evaluation": self.evaluation,
        }


@dataclass(frozen=True)
class MonitoringReport:
    intent: str
    command: dict[str, Any]
    evaluation_request: dict[str, Any]
    evaluation_result: dict[str, Any]
    scope: str
    metric: str
    window_seconds: int
    status: Status
    results: list[dict[str, Any]]
    errors: list[dict[str, Any]]
    source: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "command": self.command,
            "evaluation_request": self.evaluation_request,
            "evaluation_result": self.evaluation_result,
            "scope": self.scope,
            "metric": self.metric,
            "window_seconds": self.window_seconds,
            "status": self.status,
            "results": self.results,
            "errors": self.errors,
            "source": self.source,
        }


def _normalize_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string.")
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field_name} must not be empty.")
    return normalized


def _normalize_string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if not isinstance(value, list):
        raise TypeError("nfs must be a list of strings.")
    normalized: list[str] = []
    for item in value:
        if not isinstance(item, str):
            raise TypeError("Every nfs item must be a string.")
        text = item.strip()
        if text:
            normalized.append(text)
    return normalized


def _normalize_expected_replicas(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise TypeError("expected_replicas must be an integer or null.")
    try:
        expected = int(value)
    except (TypeError, ValueError):
        raise TypeError("expected_replicas must be an integer or null.") from None
    if expected < 0 or expected > 1000:
        raise ValueError("expected_replicas must be between 0 and 1000.")
    return expected


def _normalize_query_specs(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise TypeError("queries must be a list of query objects.")

    normalized: list[dict[str, Any]] = []
    check_ids: set[str] = set()
    for index, item in enumerate(value, start=1):
        if not isinstance(item, dict):
            raise TypeError("Every query spec must be an object.")
        tool = item.get("tool", "query_prometheus")
        if not isinstance(tool, str) or not tool.strip():
            raise TypeError("query tool must be a non-empty string.")
        tool = tool.strip()
        if tool not in READ_ONLY_QUERY_TOOLS:
            raise ValueError("Monitoring query plans may only call read-only Grafana query tools.")

        query = item.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query spec requires a non-empty query string.")
        query = query.strip()
        if len(query) > MAX_QUERY_LENGTH:
            raise ValueError("Monitoring query is too long.")
        if any(ord(ch) < 32 and ch not in "\t\n\r" for ch in query):
            raise ValueError("Monitoring query contains unsupported control characters.")
        if tool == "query_influxql":
            if not re.match(r"(?is)^SELECT\b", query) or ";" in query or INFLUXQL_WRITE_PATTERN.search(query):
                raise ValueError("InfluxQL monitoring queries must be one read-only SELECT statement.")

        label = item.get("label") or item.get("purpose") or f"query_{index}"
        target = item.get("target") or ""
        purpose = item.get("purpose") or label
        expected_query_type = "influxql" if tool == "query_influxql" else "instant"
        query_type = item.get("query_type", expected_query_type)
        if not isinstance(label, str) or not label.strip():
            raise TypeError("query label must be a non-empty string.")
        if not isinstance(target, str):
            raise TypeError("query target must be a string when provided.")
        if not isinstance(purpose, str) or not purpose.strip():
            raise TypeError("query purpose must be a non-empty string.")
        if not isinstance(query_type, str) or not query_type.strip():
            raise TypeError("query_type must be a non-empty string.")
        if query_type.strip() != expected_query_type:
            raise ValueError(f"{tool} requires query_type={expected_query_type}.")

        spec: dict[str, Any] = {
            "tool": tool,
            "label": label.strip(),
            "target": target.strip(),
            "query": query,
            "purpose": purpose.strip(),
            "query_type": expected_query_type,
        }
        check_id = item.get("check_id")
        if check_id is not None:
            if not isinstance(check_id, str) or not check_id.strip():
                raise TypeError("query check_id must be a non-empty string when provided.")
            check_id = check_id.strip()
            if check_id in check_ids:
                raise ValueError(f"Duplicate query check_id: {check_id}")
            check_ids.add(check_id)
            spec["check_id"] = check_id
        for optional_key in ("unit", "description"):
            optional_value = item.get(optional_key)
            if optional_value is not None:
                if not isinstance(optional_value, str):
                    raise TypeError(f"{optional_key} must be a string when provided.")
                spec[optional_key] = optional_value.strip()
        normalized.append(spec)
    return normalized


def _normalize_evaluation(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("evaluation must be an object.")
    evaluation = dict(value)
    if not evaluation:
        return evaluation
    evaluation_type = evaluation.get("type")
    if evaluation_type not in {"thresholds", "report_values"}:
        raise ValueError("evaluation type must be thresholds or report_values.")
    if evaluation_type == "report_values":
        if evaluation.get("completion_rule") != "data_returned":
            raise ValueError("report_values evaluation requires completion_rule=data_returned.")
        return evaluation
    checks = evaluation.get("checks")
    if not isinstance(checks, list) or not checks:
        raise ValueError("threshold evaluation requires at least one check.")
    check_ids: set[str] = set()
    normalized_checks: list[dict[str, Any]] = []
    for raw_check in checks:
        if not isinstance(raw_check, dict):
            raise TypeError("Every threshold check must be an object.")
        legacy_fields = {"query_label", "value_from_query", "operator", "value", "value_from", "value_from_check_id", "reducer"}.intersection(raw_check)
        if legacy_fields:
            raise ValueError(f"threshold check fields must use the nested expected contract; unsupported fields: {sorted(legacy_fields)}")
        check_id = raw_check.get("check_id")
        observation = raw_check.get("observation")
        expected = raw_check.get("expected")
        if not isinstance(check_id, str) or not check_id.strip():
            raise ValueError("Every threshold check requires a non-empty check_id.")
        if not isinstance(observation, str) or not observation.strip():
            raise ValueError("Every threshold check requires a non-empty observation.")
        if not isinstance(expected, dict):
            raise ValueError("Every threshold check requires an expected object.")
        if not isinstance(expected.get("operator"), str) or not expected["operator"].strip():
            raise ValueError("Every threshold check expected object requires an operator.")
        operator = expected["operator"].strip()
        if operator not in {"==", "!=", ">", ">=", "<", "<="}:
            raise ValueError(f"Unsupported threshold operator: {operator}")
        if "value" not in expected and "value_from_check_id" not in expected:
            raise ValueError("Every threshold check expected object requires a value or value reference.")
        if "value" in expected and (not isinstance(expected["value"], (int, float)) or isinstance(expected["value"], bool)):
            raise TypeError("threshold expected value must be numeric.")
        if "value_from_check_id" in expected and (not isinstance(expected["value_from_check_id"], str) or not expected["value_from_check_id"].strip()):
            raise TypeError("threshold value_from_check_id must be a non-empty string.")
        check_id = check_id.strip()
        if check_id in check_ids:
            raise ValueError(f"Duplicate threshold check_id: {check_id}")
        check_ids.add(check_id)
        check = dict(raw_check)
        check["check_id"] = check_id
        check["observation"] = observation.strip()
        check["expected"] = {**expected, "operator": operator}
        normalized_checks.append(check)
    evaluation["checks"] = normalized_checks
    return evaluation


def validate_monitoring_command(data: dict[str, Any]) -> MonitoringCommand:
    if not isinstance(data, dict):
        raise TypeError("Monitoring command must be a JSON object.")

    operation = _normalize_string(data.get("operation"), "operation")
    metric = _normalize_string(data.get("metric"), "metric")
    scope = _normalize_string(data.get("scope"), "scope")
    nfs = _normalize_string_list(data.get("nfs", []))
    window_seconds = data.get("window_seconds", 60)
    expected_replicas = _normalize_expected_replicas(data.get("expected_replicas"))
    queries = _normalize_query_specs(data.get("queries", data.get("query_plan", [])))
    evaluation = _normalize_evaluation(data.get("evaluation", {}))

    if evaluation.get("type") == "thresholds":
        query_check_ids = {spec.get("check_id") for spec in queries if isinstance(spec.get("check_id"), str)}
        missing = [check["check_id"] for check in evaluation["checks"] if check["check_id"] not in query_check_ids]
        if missing:
            raise ValueError(f"threshold checks have no matching query check_id: {missing}")
        unknown_references = [
            check["expected"]["value_from_check_id"]
            for check in evaluation["checks"]
            if isinstance(check["expected"].get("value_from_check_id"), str)
            and check["expected"]["value_from_check_id"] not in query_check_ids
        ]
        if unknown_references:
            raise ValueError(f"threshold checks reference unknown query check_id: {unknown_references}")

    if not isinstance(window_seconds, int) or isinstance(window_seconds, bool):
        raise TypeError("window_seconds must be an integer.")
    if window_seconds < 1 or window_seconds > 3600:
        raise ValueError("window_seconds must be between 1 and 3600.")

    if not queries:
        if operation not in LEGACY_OPERATIONS:
            raise ValueError("Custom monitoring operations require a query plan in queries.")
        if operation == "cpu_average" and metric != "cpu_usage":
            raise ValueError("operation=cpu_average requires metric=cpu_usage when no query plan is provided.")
        if operation == "replica_health" and metric != "replica_status":
            raise ValueError("operation=replica_health requires metric=replica_status when no query plan is provided.")
        if scope not in LEGACY_SCOPES:
            raise ValueError("Unsupported legacy monitoring scope.")
        if scope in {"single_nf", "explicit_nfs"} and not nfs:
            raise ValueError("single_nf and explicit_nfs scopes require at least one NF.")
        if operation == "replica_health" and scope not in {"single_nf", "explicit_nfs", "core_free5gc_nfs"}:
            raise ValueError("replica_health supports only NF scopes when no query plan is provided.")

    if scope == "single_nf" and len(nfs) > 1:
        nfs = nfs[:1]

    return MonitoringCommand(
        operation=operation,
        metric=metric,
        scope=scope,
        nfs=nfs,
        window_seconds=window_seconds,
        expected_replicas=expected_replicas,
        queries=queries,
        evaluation=evaluation,
    )


def validate_monitoring_report(data: dict[str, Any]) -> None:
    if tuple(data.keys()) != REPORT_KEYS:
        raise ValueError("Monitoring report contains unexpected fields or ordering.")
    if not isinstance(data["intent"], str):
        raise TypeError("intent must be a string.")
    validate_monitoring_command(data["command"])
    if not isinstance(data["evaluation_request"], dict):
        raise TypeError("evaluation_request must be an object.")
    if not isinstance(data["evaluation_result"], dict):
        raise TypeError("evaluation_result must be an object.")
    if data["evaluation_result"].get("status") not in {"passed", "failed", "unknown"}:
        raise ValueError("evaluation_result status must be passed, failed, or unknown.")
    if not isinstance(data["scope"], str) or not data["scope"].strip():
        raise TypeError("scope in report must be a non-empty string.")
    if not isinstance(data["metric"], str) or not data["metric"].strip():
        raise TypeError("metric in report must be a non-empty string.")
    if not isinstance(data["window_seconds"], int) or data["window_seconds"] <= 0:
        raise ValueError("window_seconds must be a positive integer.")
    if data["status"] not in {"completed", "partial", "blocked"}:
        raise ValueError("Unsupported monitoring status.")
    if not isinstance(data["results"], list):
        raise TypeError("results must be a list.")
    if not isinstance(data["errors"], list):
        raise TypeError("errors must be a list.")
    if not isinstance(data["source"], dict):
        raise TypeError("source must be an object.")
    if data["status"] == "completed" and not data["results"]:
        raise ValueError("completed monitoring reports must include at least one result.")
