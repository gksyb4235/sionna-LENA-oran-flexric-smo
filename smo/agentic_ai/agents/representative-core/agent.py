"""CLI and core runtime for the Representative Core Agent."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

from schemas import CoreAgentReport, CoreCommandProposal, validate_core_agent_report, validate_core_command_proposal
from tools.approval import approved_approval, pending_approval, rejected_approval
from tools.kubectl_executor import execute_kubectl_commands
from tools.kubectl_policy import CORE_FREE5GC_NFS, evaluate_policy, policy_source

JAECHAN_ROOT = Path(
    os.getenv("AGENTIC_AI_ROOT", str(Path(__file__).resolve().parents[2]))
).resolve()
PROJECT_ROOT = Path(__file__).resolve().parent
DECOMPOSITION_ENV = JAECHAN_ROOT / "agents" / "decomposition" / ".env"
if str(JAECHAN_ROOT) not in sys.path:
    sys.path.append(str(JAECHAN_ROOT))

from agent_ops.llm_models import model_for  # noqa: E402
from agent_ops.telemetry import default_run_id, log_event  # noqa: E402
NF_PATTERN = re.compile(r"\b(AMF|SMF|UPF|NRF|AUSF|UDM|UDR|PCF|NSSF)\b", re.IGNORECASE)
REPLICA_COUNT_PATTERN = re.compile(
    r"(?i)(?:\b(?:to|=)\s*(\d+)\s*replicas?\b|\b(\d+)\s*(?:replicas?|pods?|amfs?|smfs?|upfs?|nrfs?|ausfs?|udms?|udrs?|pcfs?|nssfs?)\b|\breplicas?\D{0,24}(\d+)\b)"
)

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
            run_id=default_run_id("representative-core"),
            agent="representative-core-agent",
            event_type=event_type,
            subtask_id=os.getenv("AGENT_SUBTASK_ID") or None,
            **kwargs,
        )
    except Exception as exc:  # noqa: BLE001 - telemetry must not break representative-core.
        print(f"representative-core-agent telemetry warning: {exc}", file=sys.stderr)

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
        raise FileNotFoundError(f"Missing representative core instructions: {path}")
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
            if {"operation", "resource", "target_scope"}.issubset(payload.keys()):
                return payload
        index += max(end, 1)
    if first_object is not None:
        return first_object
    raise ValueError("Representative Core LLM response did not contain a JSON object.")


def create_representative_core_agent():
    """Create the DeepAgents-backed kubectl command proposal harness."""

    prefer_venv_deepagents_package()
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    ensure_project_scope()
    load_env_file()
    model = model_for("representative-core-agent")
    backend = FilesystemBackend(root_dir=PROJECT_ROOT, virtual_mode=True)
    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt=load_agent_instructions(),
        skills=["/skills"],
        backend=backend,
    )


def build_command_prompt(intent: str) -> str:
    return f"""Convert the free5gc Core NF operation request into one safe JSON kubectl command proposal.

User intent:
{intent}

Supported free5gc Core NFs:
{", ".join(nf.upper() for nf in CORE_FREE5GC_NFS)}

Output exactly one JSON object with exactly these keys:
- operation: get, describe, logs, label, annotate, patch, delete, pod_reset, rollout_restart, scale, exec, attach, cp, apply, replace, edit, node_operation, namespace_operation
- resource: pod, deployment, statefulset, service, namespace, node, manifest
- namespace: default free5gc-v4 for Core NF actions
- target_scope: single_nf, explicit_nfs, core_free5gc_nfs, raw_name
- nfs: list of Core NF names when the target is one or more NFs
- parameters: object with operation-specific arguments
- rationale: short explanation for the proposed kubectl action

Rules:
- Do not output a shell command string.
- Do not execute anything.
- Destructive or high-risk requests are still represented as JSON proposals; approval is handled after validation.
- If a request names one NF, use target_scope=single_nf and nfs=[that NF].
- If a request says each/all Core NF, use target_scope=core_free5gc_nfs.
"""


def _infer_nfs(intent: str) -> list[str]:
    matches = [match.group(1).upper() for match in NF_PATTERN.finditer(intent)]
    seen: list[str] = []
    for item in matches:
        if item not in seen:
            seen.append(item)
    return seen


def _infer_replica_count(intent: str) -> int | None:
    for match in REPLICA_COUNT_PATTERN.finditer(intent):
        for group in match.groups():
            if not group:
                continue
            try:
                return int(group)
            except ValueError:
                continue
    return None


def _heuristic_command_proposal(intent: str) -> CoreCommandProposal:
    lower = intent.lower()
    nfs = _infer_nfs(intent)
    all_markers = ("each nf", "every nf", "all nf", "all core", "\uac01 nf", "\ubaa8\ub4e0 nf", "\uac01 core", "\ubaa8\ub4e0 core")
    target_scope = "core_free5gc_nfs" if any(marker in lower for marker in all_markers) else "single_nf"
    if target_scope == "single_nf" and not nfs:
        nfs = ["AMF"]

    operation = "get"
    resource = "pod"
    parameters: dict[str, Any] = {}
    if "rollout" in lower or "restart" in lower or "\uc7ac\uc2dc\uc791" in intent:
        operation = "rollout_restart"
        resource = "deployment"
    elif "reset" in lower or "\ub9ac\uc14b" in intent:
        operation = "pod_reset"
        resource = "pod"
    elif "delete" in lower or "\uc0ad\uc81c" in intent:
        operation = "delete"
        resource = "pod"
    elif "scale" in lower or "\uc2a4\ucf00\uc77c" in intent:
        operation = "scale"
        resource = "deployment"
        replicas = _infer_replica_count(intent)
        if replicas is None:
            replicas = 0 if re.search(r"replicas?\s*[:=]?\s*0|\b0\b", lower) else 1
        parameters["replicas"] = replicas
    elif "describe" in lower or "\uc0c1\uc138" in intent:
        operation = "describe"
        resource = "pod"
    elif "logs" in lower or "\ub85c\uadf8" in intent:
        operation = "logs"
        resource = "pod"
        parameters["tail"] = 100

    return validate_core_command_proposal(
        {
            "operation": operation,
            "resource": resource,
            "namespace": os.getenv("FREE5GC_CORE_NAMESPACE", "free5gc-v4"),
            "target_scope": target_scope,
            "nfs": nfs,
            "parameters": parameters,
            "rationale": "Deterministic test proposal inferred from the user intent.",
        }
    )


def build_command_proposal(intent: str) -> CoreCommandProposal:
    ensure_project_scope()
    load_env_file()
    if os.getenv("REPRESENTATIVE_CORE_USE_DETERMINISTIC_TEST_COMMAND") == "1":
        return _heuristic_command_proposal(intent)

    agent = create_representative_core_agent()
    result = agent.invoke({"messages": [{"role": "user", "content": build_command_prompt(intent)}]})
    messages = result.get("messages", []) if isinstance(result, dict) else []
    if not messages:
        raise RuntimeError("Representative Core LLM returned no messages.")
    payload = extract_json_object(_content_to_text(getattr(messages[-1], "content", messages[-1])))
    return validate_core_command_proposal(payload)


def build_pending_report(intent: str, proposal: CoreCommandProposal) -> CoreAgentReport:
    policy = evaluate_policy(proposal)
    errors = policy.errors
    if errors:
        status = "blocked"
        approval = {
            "required": True,
            "state": "blocked",
            "approved": False,
            "risk_level": policy.risk_level,
            "approved_by": None,
            "reason": "Command proposal could not be safely converted to kubectl argv.",
        }
        result = {"executed": False, "reason": "policy_validation_failed"}
    else:
        status = "pending_approval"
        approval = pending_approval(risk_level=policy.risk_level)
        result = {
            "executed": False,
            "reason": "waiting_for_user_approval",
            "command_count": len(policy.kubectl_argvs),
        }

    report = CoreAgentReport(
        intent=intent,
        command_proposal=proposal.to_dict(),
        kubectl_preview=policy.kubectl_preview,
        risk_level=policy.risk_level,  # type: ignore[arg-type]
        approval_required=True,
        approval=approval,
        status=status,  # type: ignore[arg-type]
        result=result,
        errors=errors,
        source=policy_source(),
    )
    validate_core_agent_report(report.to_dict())
    return report


def run_representative_core(intent: str) -> CoreAgentReport:
    _emit_event("run_started", status="running", request={"intent": intent})
    try:
        proposal = build_command_proposal(intent)
        report = build_pending_report(intent, proposal)
    except Exception as exc:
        _emit_event("run_finished", status="blocked", error=str(exc))
        raise
    _emit_event(
        "run_finished",
        status=report.status,
        response=report.to_dict(),
        command_preview=report.kubectl_preview,
        risk_level=report.risk_level,
    )
    return report


def approve_report(report_payload: dict[str, Any], *, approved: bool, approved_by: str | None = None, reason: str | None = None) -> CoreAgentReport:
    validate_core_agent_report(report_payload)
    proposal = validate_core_command_proposal(report_payload["command_proposal"])
    policy = evaluate_policy(proposal)
    if policy.errors:
        report = dict(report_payload)
        report["status"] = "blocked"
        report["approval"] = {
            "required": True,
            "state": "blocked",
            "approved": False,
            "risk_level": policy.risk_level,
            "approved_by": approved_by,
            "reason": "Command proposal could not be safely converted to kubectl argv.",
        }
        report["result"] = {"executed": False, "reason": "policy_validation_failed"}
        report["errors"] = policy.errors
        validate_core_agent_report(report)
        return CoreAgentReport(**report)

    report = dict(report_payload)
    report["kubectl_preview"] = policy.kubectl_preview
    report["risk_level"] = policy.risk_level
    report["source"] = policy_source()
    if not approved:
        report["status"] = "rejected"
        report["approval"] = rejected_approval(approved_by=approved_by, reason=reason, risk_level=policy.risk_level)
        report["result"] = {"executed": False, "reason": "user_rejected"}
        report["errors"] = []
        validate_core_agent_report(report)
        return CoreAgentReport(**report)

    result = execute_kubectl_commands(policy.kubectl_argvs)
    report["status"] = "completed" if result.get("success") else "blocked"
    report["approval"] = approved_approval(approved_by=approved_by, reason=reason, risk_level=policy.risk_level)
    report["result"] = result
    report["errors"] = [] if result.get("success") else [{"error": "One or more kubectl commands failed."}]
    validate_core_agent_report(report)
    return CoreAgentReport(**report)


def format_agent_section(agent_name: str, payload: dict[str, Any]) -> str:
    body = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"------{agent_name}------\n{body}\n----end----"


def _read_json_input(path: str | None) -> dict[str, Any]:
    raw = Path(path).read_text(encoding="utf-8") if path else sys.stdin.read()
    if not raw.strip():
        raise ValueError("No JSON input provided.")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise TypeError("Input must be a JSON object.")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create and approve kubectl proposals for free5gc Core NF actions.")
    parser.add_argument("intent", nargs="?", help="Natural language Core NF action intent.")
    parser.add_argument("--json", dest="json_output", action="store_true", help="Print only JSON without CLI section markers.")
    parser.add_argument("--input", dest="input_path", help="Read a pending Representative Core report from a JSON file or stdin.")
    parser.add_argument("--approve", action="store_true", help="Approve and execute a pending report from --input/stdin.")
    parser.add_argument("--reject", action="store_true", help="Reject a pending report from --input/stdin without executing.")
    parser.add_argument("--approved-by", dest="approved_by", help="User or system approving/rejecting the command.")
    parser.add_argument("--reason", dest="reason", help="Approval or rejection reason.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.approve or args.reject:
            payload = _read_json_input(args.input_path)
            report = approve_report(payload, approved=bool(args.approve and not args.reject), approved_by=args.approved_by, reason=args.reason).to_dict()
        else:
            if not args.intent:
                raise ValueError("intent is required unless --approve or --reject is used.")
            report = run_representative_core(args.intent).to_dict()
    except Exception as exc:  # noqa: BLE001 - CLI should surface operational failures.
        print(f"representative-core-agent error: {exc}", file=sys.stderr)
        return 1
    if args.json_output:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(format_agent_section("representative-core-agent", report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
