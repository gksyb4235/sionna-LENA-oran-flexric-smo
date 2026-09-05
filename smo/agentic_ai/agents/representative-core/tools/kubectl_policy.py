"""kubectl proposal validation, risk classification, and preview generation."""

from __future__ import annotations

import json
import os
import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from schemas import CoreCommandProposal

JAECHAN_ROOT = Path(
    os.getenv("AGENTIC_AI_ROOT", str(Path(__file__).resolve().parents[3]))
).resolve()
DEFAULT_NAMESPACE = "free5gc-v4"
CORE_FREE5GC_NFS = ("amf", "smf", "upf", "nrf", "ausf", "udm", "udr", "pcf", "nssf")
CORE_NF_SET = set(CORE_FREE5GC_NFS)
SUSPICIOUS_PATTERN = re.compile(r"[;&|><`$\n\r]")
SAFE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.:/=-]+$")
SAFE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.\-/]+$")
READ_ONLY_OPERATIONS = {"get", "describe", "logs"}
MEDIUM_RISK_OPERATIONS = {"label", "annotate", "patch"}
HIGH_RISK_OPERATIONS = {
    "delete",
    "pod_reset",
    "rollout_restart",
    "exec",
    "attach",
    "cp",
    "apply",
    "replace",
    "edit",
    "node_operation",
    "namespace_operation",
}


@dataclass(frozen=True)
class PolicyEvaluation:
    risk_level: str
    kubectl_argvs: list[list[str]]
    errors: list[dict[str, str]]

    @property
    def kubectl_preview(self) -> list[str]:
        return [shlex.join(argv) for argv in self.kubectl_argvs]


def configured_namespace() -> str:
    return os.getenv("FREE5GC_CORE_NAMESPACE", DEFAULT_NAMESPACE).strip() or DEFAULT_NAMESPACE


def policy_source() -> dict[str, Any]:
    return {
        "type": "kubectl_policy",
        "namespace": configured_namespace(),
        "core_nf_label": "nf",
        "core_nfs": list(CORE_FREE5GC_NFS),
        "approval_model": "planning_agent_user_approval_required",
    }


def _contains_suspicious_value(value: Any) -> bool:
    if isinstance(value, str):
        return bool(SUSPICIOUS_PATTERN.search(value))
    if isinstance(value, list):
        return any(_contains_suspicious_value(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_suspicious_value(key) or _contains_suspicious_value(item) for key, item in value.items())
    return False


def _safe_name(value: str, *, field: str) -> str:
    text = value.strip()
    if not text or not SAFE_NAME_PATTERN.match(text) or SUSPICIOUS_PATTERN.search(text):
        raise ValueError(f"Unsafe {field}: {value}")
    return text


def _safe_key(value: str, *, field: str) -> str:
    text = value.strip()
    if not text or not SAFE_KEY_PATTERN.match(text) or SUSPICIOUS_PATTERN.search(text):
        raise ValueError(f"Unsafe {field}: {value}")
    return text


def _safe_value(value: Any, *, field: str) -> str:
    if value is None:
        raise ValueError(f"Missing {field}.")
    text = str(value).strip()
    if not text or SUSPICIOUS_PATTERN.search(text):
        raise ValueError(f"Unsafe {field}: {text}")
    return text


def normalize_nf(value: str) -> str:
    nf = value.strip().lower()
    if nf not in CORE_NF_SET:
        raise ValueError(f"Unsupported free5gc Core NF target: {value}")
    return nf


def target_nfs(proposal: CoreCommandProposal) -> list[str]:
    if proposal.target_scope == "core_free5gc_nfs":
        return list(CORE_FREE5GC_NFS)
    if proposal.resource in {"namespace", "node", "manifest"}:
        return []
    return [normalize_nf(nf) for nf in proposal.nfs]


def deployment_name(nf: str) -> str:
    return f"free5gc-free5gc-{nf}"


def _resource_ref(resource: str, nf: str) -> str:
    if resource == "deployment":
        return f"deployment/{deployment_name(nf)}"
    if resource == "statefulset":
        return f"statefulset/free5gc-{nf}"
    if resource == "service":
        return f"service/free5gc-free5gc-{nf}"
    if resource == "pod":
        return "pods"
    return resource


def _selector_args(resource: str, nf: str) -> list[str]:
    if resource == "pod":
        return ["pods", "-l", f"nf={nf}"]
    return [_resource_ref(resource, nf)]


def _namespace_args(namespace: str) -> list[str]:
    return ["-n", _safe_name(namespace, field="namespace")]


def risk_for(proposal: CoreCommandProposal) -> str:
    if proposal.operation in READ_ONLY_OPERATIONS:
        return "low"
    if proposal.operation == "scale":
        replicas = proposal.parameters.get("replicas")
        try:
            return "high" if int(replicas) == 0 else "medium"
        except (TypeError, ValueError):
            return "medium"
    if proposal.operation in MEDIUM_RISK_OPERATIONS:
        return "medium"
    if proposal.operation in HIGH_RISK_OPERATIONS:
        return "high"
    return "high"


def _manifest_path(value: Any) -> str:
    raw = _safe_value(value, field="manifest_path")
    path = Path(raw)
    resolved = (JAECHAN_ROOT / raw).resolve() if not path.is_absolute() else path.resolve()
    if resolved != JAECHAN_ROOT and JAECHAN_ROOT not in resolved.parents:
        raise ValueError(f"manifest_path must stay under {JAECHAN_ROOT}")
    return str(resolved)


def _command_for_nf(proposal: CoreCommandProposal, nf: str) -> list[str]:
    ns = proposal.namespace or configured_namespace()
    operation = proposal.operation
    resource = proposal.resource
    params = proposal.parameters
    namespace = _namespace_args(ns)

    if operation == "get":
        return ["kubectl", "get", *_selector_args(resource, nf), *namespace]
    if operation == "describe":
        return ["kubectl", "describe", *_selector_args(resource, nf), *namespace]
    if operation == "logs":
        tail = int(params.get("tail", 100)) if str(params.get("tail", "100")).isdigit() else 100
        return ["kubectl", "logs", *namespace, _resource_ref("deployment", nf), f"--tail={tail}"]
    if operation == "label":
        key = _safe_key(str(params.get("key", "")), field="label key")
        value = _safe_value(params.get("value"), field="label value")
        return ["kubectl", "label", _resource_ref(resource, nf), *namespace, f"{key}={value}", "--overwrite"]
    if operation == "annotate":
        key = _safe_key(str(params.get("key", "")), field="annotation key")
        value = _safe_value(params.get("value"), field="annotation value")
        return ["kubectl", "annotate", _resource_ref(resource, nf), *namespace, f"{key}={value}", "--overwrite"]
    if operation == "patch":
        patch_type = _safe_name(str(params.get("patch_type", "merge")), field="patch_type")
        patch_body = params.get("patch")
        if not isinstance(patch_body, dict):
            raise ValueError("patch operation requires parameters.patch as an object.")
        return ["kubectl", "patch", _resource_ref(resource, nf), *namespace, "--type", patch_type, "-p", json.dumps(patch_body, separators=(",", ":"))]
    if operation == "delete":
        if resource == "pod":
            return ["kubectl", "delete", "pods", *namespace, "-l", f"nf={nf}"]
        return ["kubectl", "delete", _resource_ref(resource, nf), *namespace]
    if operation == "pod_reset":
        return ["kubectl", "delete", "pods", *namespace, "-l", f"nf={nf}"]
    if operation == "rollout_restart":
        return ["kubectl", "rollout", "restart", f"deployment/{deployment_name(nf)}", *namespace]
    if operation == "scale":
        raw_replicas = params.get("replicas")
        try:
            if isinstance(raw_replicas, bool):
                raise ValueError
            replicas = int(raw_replicas)
        except (TypeError, ValueError):
            raise ValueError("scale operation requires non-negative integer parameters.replicas.") from None
        if replicas < 0:
            raise ValueError("scale operation requires non-negative integer parameters.replicas.")
        return ["kubectl", "scale", f"deployment/{deployment_name(nf)}", *namespace, f"--replicas={replicas}"]
    if operation == "exec":
        command = params.get("command")
        if not isinstance(command, list) or not command or not all(isinstance(item, str) and item.strip() for item in command):
            raise ValueError("exec operation requires parameters.command as a non-empty list of strings.")
        if _contains_suspicious_value(command):
            raise ValueError("exec command contains suspicious shell characters.")
        return ["kubectl", "exec", *namespace, f"deployment/{deployment_name(nf)}", "--", *command]
    if operation == "attach":
        return ["kubectl", "attach", *namespace, f"deployment/{deployment_name(nf)}"]
    if operation == "edit":
        return ["kubectl", "edit", _resource_ref(resource, nf), *namespace]
    raise ValueError(f"Operation {operation} is not valid for NF targets.")


def _non_nf_command(proposal: CoreCommandProposal) -> list[str]:
    operation = proposal.operation
    params = proposal.parameters
    ns = proposal.namespace or configured_namespace()
    if proposal.resource == "namespace" or operation == "namespace_operation":
        action = _safe_name(str(params.get("action", operation)), field="namespace action")
        name = _safe_name(str(params.get("name", ns)), field="namespace name")
        if action in {"delete", "remove"}:
            return ["kubectl", "delete", "namespace", name]
        if action in {"label", "annotate"}:
            key = _safe_key(str(params.get("key", "")), field="namespace metadata key")
            value = _safe_value(params.get("value"), field="namespace metadata value")
            return ["kubectl", action, "namespace", name, f"{key}={value}", "--overwrite"]
        return ["kubectl", action, "namespace", name]
    if proposal.resource == "node" or operation == "node_operation":
        action = _safe_name(str(params.get("action", operation)), field="node action")
        name = _safe_name(str(params.get("name", "")), field="node name")
        return ["kubectl", action, "node", name]
    if proposal.resource == "manifest" or operation in {"apply", "replace"}:
        manifest = _manifest_path(params.get("manifest_path"))
        return ["kubectl", operation, "-n", _safe_name(ns, field="namespace"), "-f", manifest]
    if operation == "cp":
        source = _safe_value(params.get("source"), field="copy source")
        destination = _safe_value(params.get("destination"), field="copy destination")
        return ["kubectl", "cp", "-n", _safe_name(ns, field="namespace"), source, destination]
    raise ValueError("Command target is unclear.")


def evaluate_policy(proposal: CoreCommandProposal) -> PolicyEvaluation:
    errors: list[dict[str, str]] = []
    argvs: list[list[str]] = []
    risk = risk_for(proposal)

    try:
        if proposal.namespace and proposal.namespace != configured_namespace() and proposal.resource not in {"namespace", "node"}:
            raise ValueError(f"Only namespace {configured_namespace()} is supported for free5gc Core NF actions.")
        executable_payload = {
            "operation": proposal.operation,
            "resource": proposal.resource,
            "namespace": proposal.namespace,
            "target_scope": proposal.target_scope,
            "nfs": proposal.nfs,
            "parameters": proposal.parameters,
        }
        if _contains_suspicious_value(executable_payload):
            raise ValueError("Command proposal contains suspicious shell characters.")
        nfs = target_nfs(proposal)
        if nfs:
            for nf in nfs:
                argvs.append(_command_for_nf(proposal, nf))
        else:
            argvs.append(_non_nf_command(proposal))
    except Exception as exc:  # noqa: BLE001 - convert policy failures into report errors.
        errors.append({"error": str(exc)})

    return PolicyEvaluation(risk_level=risk, kubectl_argvs=argvs, errors=errors)
