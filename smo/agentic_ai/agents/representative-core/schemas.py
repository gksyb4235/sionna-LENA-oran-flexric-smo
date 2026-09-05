"""Schemas for the Representative Core Agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

Operation = Literal[
    "get",
    "describe",
    "logs",
    "label",
    "annotate",
    "patch",
    "delete",
    "pod_reset",
    "rollout_restart",
    "scale",
    "exec",
    "attach",
    "cp",
    "apply",
    "replace",
    "edit",
    "node_operation",
    "namespace_operation",
]
Resource = Literal["pod", "deployment", "statefulset", "service", "namespace", "node", "manifest"]
TargetScope = Literal["single_nf", "explicit_nfs", "core_free5gc_nfs", "raw_name"]
RiskLevel = Literal["low", "medium", "high"]
Status = Literal["pending_approval", "completed", "blocked", "rejected"]

COMMAND_PROPOSAL_KEYS = ("operation", "resource", "namespace", "target_scope", "nfs", "parameters", "rationale")
REPORT_KEYS = (
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
OPERATIONS = set(Operation.__args__)  # type: ignore[attr-defined]
RESOURCES = set(Resource.__args__)  # type: ignore[attr-defined]
TARGET_SCOPES = set(TargetScope.__args__)  # type: ignore[attr-defined]
RISK_LEVELS = set(RiskLevel.__args__)  # type: ignore[attr-defined]
STATUSES = set(Status.__args__)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class CoreCommandProposal:
    operation: Operation
    resource: Resource
    namespace: str
    target_scope: TargetScope
    nfs: list[str]
    parameters: dict[str, Any]
    rationale: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "operation": self.operation,
            "resource": self.resource,
            "namespace": self.namespace,
            "target_scope": self.target_scope,
            "nfs": self.nfs,
            "parameters": self.parameters,
            "rationale": self.rationale,
        }


@dataclass(frozen=True)
class CoreAgentReport:
    intent: str
    command_proposal: dict[str, Any]
    kubectl_preview: list[str]
    risk_level: RiskLevel
    approval_required: bool
    approval: dict[str, Any]
    status: Status
    result: dict[str, Any]
    errors: list[dict[str, Any]]
    source: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "intent": self.intent,
            "command_proposal": self.command_proposal,
            "kubectl_preview": self.kubectl_preview,
            "risk_level": self.risk_level,
            "approval_required": self.approval_required,
            "approval": self.approval,
            "status": self.status,
            "result": self.result,
            "errors": self.errors,
            "source": self.source,
        }


def _normalize_string(value: Any, *, field: str, default: str = "") -> str:
    if value is None:
        return default
    if not isinstance(value, str):
        raise TypeError(f"{field} must be a string.")
    return value.strip() or default


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


def _normalize_parameters(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise TypeError("parameters must be a JSON object.")
    return dict(value)


def validate_core_command_proposal(data: dict[str, Any]) -> CoreCommandProposal:
    if not isinstance(data, dict):
        raise TypeError("Core command proposal must be a JSON object.")

    operation = _normalize_string(data.get("operation"), field="operation").replace("-", "_").lower()
    resource = _normalize_string(data.get("resource", "deployment"), field="resource", default="deployment").replace("-", "_").lower()
    namespace = _normalize_string(data.get("namespace", "free5gc-v4"), field="namespace", default="free5gc-v4")
    target_scope = _normalize_string(data.get("target_scope", "single_nf"), field="target_scope", default="single_nf").replace("-", "_").lower()
    nfs = _normalize_string_list(data.get("nfs", []))
    parameters = _normalize_parameters(data.get("parameters", {}))
    rationale = _normalize_string(data.get("rationale", ""), field="rationale")

    if operation not in OPERATIONS:
        raise ValueError(f"Unsupported operation: {operation}")
    if resource not in RESOURCES:
        raise ValueError(f"Unsupported resource: {resource}")
    if target_scope not in TARGET_SCOPES:
        raise ValueError(f"Unsupported target_scope: {target_scope}")
    if target_scope in {"single_nf", "explicit_nfs"} and not nfs and resource not in {"namespace", "node", "manifest"}:
        raise ValueError("single_nf and explicit_nfs target scopes require at least one NF.")
    if target_scope == "single_nf" and len(nfs) > 1:
        nfs = nfs[:1]

    return CoreCommandProposal(
        operation=operation,  # type: ignore[arg-type]
        resource=resource,  # type: ignore[arg-type]
        namespace=namespace,
        target_scope=target_scope,  # type: ignore[arg-type]
        nfs=nfs,
        parameters=parameters,
        rationale=rationale,
    )


def validate_core_agent_report(data: dict[str, Any]) -> None:
    if not isinstance(data, dict):
        raise TypeError("Representative Core report must be a JSON object.")
    if tuple(data.keys()) != REPORT_KEYS:
        raise ValueError("Representative Core report contains unexpected fields or ordering.")
    if not isinstance(data["intent"], str):
        raise TypeError("intent must be a string.")
    validate_core_command_proposal(data["command_proposal"])
    if not isinstance(data["kubectl_preview"], list) or not all(isinstance(item, str) for item in data["kubectl_preview"]):
        raise TypeError("kubectl_preview must be a list of strings.")
    if data["risk_level"] not in RISK_LEVELS:
        raise ValueError("Invalid risk_level.")
    if not isinstance(data["approval_required"], bool):
        raise TypeError("approval_required must be a boolean.")
    if not isinstance(data["approval"], dict):
        raise TypeError("approval must be an object.")
    if data["status"] not in STATUSES:
        raise ValueError("Invalid status.")
    if not isinstance(data["result"], dict):
        raise TypeError("result must be an object.")
    if not isinstance(data["errors"], list):
        raise TypeError("errors must be a list.")
    if not isinstance(data["source"], dict):
        raise TypeError("source must be an object.")
