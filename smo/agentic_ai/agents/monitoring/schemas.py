"""Contracts for the Monitoring Agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Status = Literal[
    "healthy",
    "observed",
    "replan_required",
    "observation_unavailable",
    "awaiting_execution",
    "stopped",
]
STATUSES = set(Status.__args__)  # type: ignore[attr-defined]
NotificationDelivery = Literal["direct", "response"]
NOTIFICATION_DELIVERIES = set(NotificationDelivery.__args__)  # type: ignore[attr-defined]
REPORT_KEYS = (
    "run_id",
    "subtask_id",
    "status",
    "expected",
    "execution",
    "probe_report",
    "feedback",
    "planner_response",
    "errors",
    "source",
)


@dataclass(frozen=True)
class MonitoringRequest:
    planning_report: dict[str, Any]
    subtask_id: str | None
    notify_planner: bool
    notification_delivery: NotificationDelivery


@dataclass(frozen=True)
class MonitoringAgentReport:
    run_id: str
    subtask_id: str
    status: Status
    expected: dict[str, Any]
    execution: dict[str, Any]
    probe_report: dict[str, Any] | None
    feedback: dict[str, Any] | None
    planner_response: dict[str, Any] | None
    errors: list[dict[str, Any]]
    source: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "subtask_id": self.subtask_id,
            "status": self.status,
            "expected": self.expected,
            "execution": self.execution,
            "probe_report": self.probe_report,
            "feedback": self.feedback,
            "planner_response": self.planner_response,
            "errors": self.errors,
            "source": self.source,
        }


def validate_monitoring_request(payload: Any) -> MonitoringRequest:
    if not isinstance(payload, dict):
        raise TypeError("Monitoring input must be a JSON object.")
    planning_report = payload.get("planning_report")
    if not isinstance(planning_report, dict):
        raise TypeError("planning_report must be an object.")
    run_id = planning_report.get("run_id")
    if not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("planning_report.run_id must be a non-empty string.")
    if not isinstance(planning_report.get("intent"), str):
        raise TypeError("planning_report.intent must be a string.")
    plans = planning_report.get("subtask_plans")
    if not isinstance(plans, list) or not plans:
        raise ValueError("planning_report.subtask_plans must be a non-empty list.")
    if not all(isinstance(plan, dict) for plan in plans):
        raise TypeError("Every subtask plan must be an object.")
    subtask_id = payload.get("subtask_id")
    if subtask_id is not None and (not isinstance(subtask_id, str) or not subtask_id.strip()):
        raise TypeError("subtask_id must be a non-empty string when provided.")
    notify_planner = payload.get("notify_planner", True)
    if not isinstance(notify_planner, bool):
        raise TypeError("notify_planner must be a boolean.")
    notification_delivery = payload.get("notification_delivery", "direct")
    if not isinstance(notification_delivery, str):
        raise TypeError("notification_delivery must be a string.")
    if notification_delivery not in NOTIFICATION_DELIVERIES:
        raise ValueError("notification_delivery must be direct or response.")
    return MonitoringRequest(
        planning_report,
        subtask_id.strip() if isinstance(subtask_id, str) else None,
        notify_planner,
        notification_delivery,  # type: ignore[arg-type]
    )


def validate_monitoring_report(payload: dict[str, Any]) -> None:
    if tuple(payload.keys()) != REPORT_KEYS:
        raise ValueError("Monitoring Agent report contains unexpected fields or ordering.")
    if not isinstance(payload["run_id"], str) or not payload["run_id"]:
        raise TypeError("run_id must be a non-empty string.")
    if not isinstance(payload["subtask_id"], str) or not payload["subtask_id"]:
        raise TypeError("subtask_id must be a non-empty string.")
    if payload["status"] not in STATUSES:
        raise ValueError("Unsupported Monitoring Agent status.")
    if not isinstance(payload["expected"], dict) or not isinstance(payload["execution"], dict):
        raise TypeError("expected and execution must be objects.")
    for key in ("probe_report", "feedback", "planner_response"):
        if payload[key] is not None and not isinstance(payload[key], dict):
            raise TypeError(f"{key} must be an object or null.")
    if not isinstance(payload["errors"], list) or not isinstance(payload["source"], dict):
        raise TypeError("errors must be a list and source must be an object.")
