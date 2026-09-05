"""CLI entry point for the Decomposition Agent."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from schemas import DecompositionResult
from tools.decomposition_pattern_search import find_decomposition_pattern_context
from tools.planning_handoff import approve_representative_core_report, invoke_planning_agent

JAECHAN_ROOT = Path(
    os.getenv("AGENTIC_AI_ROOT", str(Path(__file__).resolve().parents[2]))
).resolve()
PROJECT_ROOT = Path(__file__).resolve().parent
if str(JAECHAN_ROOT) not in sys.path:
    sys.path.append(str(JAECHAN_ROOT))

from agent_ops.llm_models import model_for  # noqa: E402
from agent_ops.telemetry import default_run_id, log_event  # noqa: E402

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




def _emit_event(run_id: str, event_type: str, **kwargs: Any) -> None:
    try:
        log_event(run_id=run_id, agent="decomposition-agent", event_type=event_type, **kwargs)
    except Exception as exc:  # noqa: BLE001 - telemetry must not break decomposition.
        print(f"decomposition-agent telemetry warning: {exc}", file=sys.stderr)


def record_decomposition_events(
    decomposition_output: dict[str, Any],
    *,
    run_id: str | None = None,
    source: str = "cli",
) -> str:
    if not isinstance(run_id, str) or not run_id.strip():
        run_id = default_run_id("decomposition")
    _emit_event(
        run_id,
        "run_started",
        status="completed",
        request={"intent": decomposition_output.get("intent"), "source": source},
    )
    _emit_event(run_id, "decomposition_result", status="completed", response=decomposition_output)
    return run_id


def record_planning_handoff_event(run_id: str, decomposition_output: dict[str, Any], planning_output: dict[str, Any]) -> None:
    _emit_event(
        run_id,
        "planning_handoff",
        status=str(planning_output.get("overall_status", "unknown")),
        target_agent="planning-agent",
        request=decomposition_output,
        response={
            "run_id": planning_output.get("run_id"),
            "overall_status": planning_output.get("overall_status"),
            "subtask_plan_count": len(planning_output.get("subtask_plans", [])) if isinstance(planning_output.get("subtask_plans"), list) else None,
        },
    )

def ensure_project_scope() -> None:
    """Ensure this project is running from the allowed DPU Host workspace."""

    project_root = PROJECT_ROOT.resolve()
    if project_root != JAECHAN_ROOT and JAECHAN_ROOT not in project_root.parents:
        msg = f"Project root must stay under {JAECHAN_ROOT}; got {project_root}"
        raise RuntimeError(msg)


def load_env_file(path: Path | None = None) -> None:
    """Load a simple KEY=VALUE .env file without requiring python-dotenv."""

    env_path = path or PROJECT_ROOT / ".env"
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


def load_agent_instructions() -> str:
    """Read the agent behavior contract from agent.md."""

    instructions_path = PROJECT_ROOT / "agent.md"
    if not instructions_path.exists():
        raise FileNotFoundError(f"Missing agent instructions: {instructions_path}")
    return instructions_path.read_text(encoding="utf-8")


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
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(json.dumps(block, ensure_ascii=False))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def extract_json_object(text: str) -> dict[str, Any]:
    """Extract the public-contract JSON object from a model response."""

    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        stripped = "\n".join(lines).strip()

    decoder = json.JSONDecoder()
    required_keys = {"intent", "subtasks", "golden_goal_context_used"}
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
            if required_keys.issubset(payload.keys()):
                return payload
        index += max(end, 1)

    if first_object is not None:
        return first_object
    raise ValueError("Model response did not contain a JSON object.")


def build_user_prompt(intent: str, decomposition_patterns: list[dict[str, Any]], pattern_context_used: bool) -> str:
    """Build the model prompt for one decomposition request."""

    context_json = json.dumps(decomposition_patterns, ensure_ascii=False, indent=2)
    return f"""Decompose the following user intent into subtasks.

Original user intent:
{intent}

Decomposition pattern context used: {str(pattern_context_used).lower()}

Matched decomposition patterns:
{context_json}

Apply every required semantic and validation rule in each matched active pattern.
Preserve the target, desired condition, requested duration, monitoring responsibility, and mismatch notification behavior.
Do not create a direct remediation subtask when the pattern says to notify the Planning Agent.

Return exactly one JSON object with exactly these fields:
- intent
- subtasks
- golden_goal_context_used
"""


def build_repair_prompt(error: Exception) -> str:
    """Ask once for a corrected public decomposition contract."""

    return f"""The previous decomposition was rejected: {error}
Return the same three-field JSON contract again after fixing its JSON structure and field types.
Return JSON only.
"""


def create_decomposition_agent():
    """Create the DeepAgents-backed decomposition agent."""

    prefer_venv_deepagents_package()
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    ensure_project_scope()
    load_env_file()

    model = model_for("decomposition-agent")
    backend = FilesystemBackend(root_dir=PROJECT_ROOT, virtual_mode=True)

    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt=load_agent_instructions(),
        skills=["/skills"],
        backend=backend,
    )


def run_decomposition(intent: str) -> DecompositionResult:
    """Run one decomposition request and return the normalized result."""

    ensure_project_scope()
    load_env_file()

    pattern_context = find_decomposition_pattern_context(intent)
    agent = create_decomposition_agent()

    prompt = build_user_prompt(
        intent=intent,
        decomposition_patterns=pattern_context.documents,
        pattern_context_used=pattern_context.used,
    )
    conversation: list[dict[str, str]] = [{"role": "user", "content": prompt}]
    last_error: Exception | None = None
    for attempt in range(2):
        result = agent.invoke({"messages": conversation})
        messages = result.get("messages", []) if isinstance(result, dict) else []
        if not messages:
            raise RuntimeError("Agent returned no messages.")
        final_message = messages[-1]
        final_content = _content_to_text(getattr(final_message, "content", final_message))
        try:
            payload = extract_json_object(final_content)
            decomposition = DecompositionResult.from_model_payload(
                payload=payload,
                intent=intent,
                golden_goal_context_used=pattern_context.used,
            )
            return decomposition
        except (TypeError, ValueError) as exc:
            last_error = exc
            if attempt == 1:
                raise
            conversation.extend([
                {"role": "assistant", "content": final_content},
                {"role": "user", "content": build_repair_prompt(exc)},
            ])
    raise RuntimeError(f"Decomposition failed: {last_error}")


def run_decomposition_then_planning(intent: str) -> dict[str, Any]:
    """Run decomposition and hand the result to the Planning Agent."""

    decomposition = run_decomposition(intent).to_dict()
    run_id = default_run_id("planning")
    record_decomposition_events(decomposition, run_id=run_id, source="http")
    planning = invoke_planning_agent({**decomposition, "run_id": run_id})
    record_planning_handoff_event(str(planning.get("run_id") or run_id), decomposition, planning)
    return planning


def format_agent_section(agent_name: str, payload: dict[str, Any]) -> str:
    """Format one human-readable CLI result section."""

    body = json.dumps(payload, ensure_ascii=False, indent=2)
    return f"------{agent_name}------\n{body}\n----end----"


def extract_monitoring_reports(planning_output: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract Probe Agent reports embedded in Planning Agent subtask results."""

    reports: list[dict[str, Any]] = []
    subtask_plans = planning_output.get("subtask_plans")
    if not isinstance(subtask_plans, list):
        return reports
    for plan in subtask_plans:
        if not isinstance(plan, dict):
            continue
        result = plan.get("result")
        if not isinstance(result, dict):
            continue
        report = result.get("monitoring_report")
        if isinstance(report, dict):
            reports.append(report)
    return reports


def extract_representative_core_reports(planning_output: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract Representative Core Agent reports embedded in Planning Agent subtask results."""

    reports: list[dict[str, Any]] = []
    run_id = planning_output.get("run_id")
    subtask_plans = planning_output.get("subtask_plans")
    if not isinstance(subtask_plans, list):
        return reports
    for plan in subtask_plans:
        if not isinstance(plan, dict):
            continue
        result = plan.get("result")
        if not isinstance(result, dict):
            continue
        report = result.get("representative_core_report")
        if isinstance(report, dict):
            report_copy = dict(report)
            if isinstance(run_id, str) and run_id.strip():
                report_copy["_planning_run_id"] = run_id
            reports.append(report_copy)
    return reports

def representative_report_needs_approval(report: dict[str, Any]) -> bool:
    """Return whether a Representative Core report should trigger CLI approval."""

    return report.get("status") == "pending_approval" and bool(report.get("approval_required"))


def _approval_request_payload(report: dict[str, Any], index: int, total: int) -> dict[str, Any]:
    return {
        "request": f"{index}/{total}",
        "message": "kubectl has not been executed yet. Enter y to execute this command; Enter or N leaves the cluster unchanged.",
        "intent": report.get("intent"),
        "risk_level": report.get("risk_level"),
        "kubectl_preview": report.get("kubectl_preview", []),
        "approval_required": report.get("approval_required"),
        "approval": report.get("approval"),
    }


def prompt_for_representative_approvals(
    reports: list[dict[str, Any]],
    *,
    input_fn=input,
    output_fn=print,
) -> list[dict[str, Any]]:
    """Prompt for pending kubectl approvals and return approval/rejection reports."""

    pending_reports = [report for report in reports if representative_report_needs_approval(report)]
    approval_results: list[dict[str, Any]] = []
    seen_mutating_previews: set[tuple[str, ...]] = set()
    total = len(pending_reports)
    for index, report in enumerate(pending_reports, start=1):
        preview_signature = tuple(str(item) for item in report.get("kubectl_preview", []))
        mutating = report.get("risk_level") in {"medium", "high"}
        if mutating and preview_signature in seen_mutating_previews:
            output_fn("")
            output_fn(
                format_agent_section(
                    "approval-request-duplicate-skipped",
                    {
                        **_approval_request_payload(report, index, total),
                        "status": "skipped",
                        "reason": "duplicate_mutating_kubectl_preview",
                    },
                )
            )
            continue
        if mutating:
            seen_mutating_previews.add(preview_signature)

        output_fn("")
        output_fn(format_agent_section("approval-request", _approval_request_payload(report, index, total)))
        try:
            answer = input_fn("Approve kubectl execution? [y/N]: ")
        except EOFError:
            answer = ""
        approved = answer.strip().lower() in {"y", "yes"}
        reason = "Approved from decomposition CLI." if approved else "Rejected from decomposition CLI."
        planning_run_id = report.get("_planning_run_id")
        clean_report = {key: value for key, value in report.items() if not str(key).startswith("_")}
        approval_kwargs: dict[str, Any] = {
            "approved": approved,
            "approved_by": "decomposition-cli",
            "reason": reason,
        }
        if isinstance(planning_run_id, str) and planning_run_id.strip():
            approval_kwargs["planning_run_id"] = planning_run_id
        approval_result = approve_representative_core_report(clean_report, **approval_kwargs)
        approval_results.append(approval_result)
        if isinstance(approval_result, dict) and "subtask_plans" in approval_result:
            _print_planning_continuation(output_fn, approval_result)
        else:
            output_fn("")
            output_fn(format_agent_section("representative-core-agent-approval", approval_result))
    return approval_results

def approval_results_executed_successfully(results: list[dict[str, Any]]) -> bool:
    for result in results:
        execution = result.get("result")
        if result.get("status") == "completed" and isinstance(execution, dict) and execution.get("executed"):
            return True
    return False


def remaining_decomposition_after_planning(decomposition_output: dict[str, Any], planning_output: dict[str, Any]) -> dict[str, Any] | None:
    subtasks = decomposition_output.get("subtasks")
    subtask_plans = planning_output.get("subtask_plans")
    if not isinstance(subtasks, list) or not isinstance(subtask_plans, list):
        return None
    completed_count = len(subtask_plans)
    remaining = [item for item in subtasks[completed_count:] if isinstance(item, str) and item.strip()]
    if not remaining:
        return None
    return {
        "intent": str(decomposition_output.get("intent", "")),
        "subtasks": remaining,
        "golden_goal_context_used": bool(decomposition_output.get("golden_goal_context_used", False)),
    }


def append_planning_sections(output_sections: list[tuple[str, dict[str, Any]]], planning_output: dict[str, Any], *, planning_agent_name: str = "planning-agent") -> list[dict[str, Any]]:
    output_sections.append((planning_agent_name, planning_output))
    for monitoring_report in extract_monitoring_reports(planning_output):
        output_sections.append(("probe-agent", monitoring_report))
    representative_reports = extract_representative_core_reports(planning_output)
    for representative_report in representative_reports:
        display_report = {key: value for key, value in representative_report.items() if not str(key).startswith("_")}
        output_sections.append(("representative-core-agent", display_report))
    return representative_reports


def _print_planning_continuation(output_fn, planning_output: dict[str, Any]) -> None:
    continuation_sections: list[tuple[str, dict[str, Any]]] = []
    append_planning_sections(
        continuation_sections,
        planning_output,
        planning_agent_name="planning-agent-after-approval",
    )
    for agent_name, payload in continuation_sections:
        output_fn("")
        output_fn(format_agent_section(agent_name, payload))

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the Decomposition Agent and hand its output to the Planning Agent.",
    )
    parser.add_argument(
        "--decompose-only",
        action="store_true",
        help="Only print the decomposition contract JSON and do not invoke the Planning Agent.",
    )
    parser.add_argument(
        "--plan",
        action="store_true",
        help="Explicitly run the default decomposition-to-planning flow.",
    )
    parser.add_argument(
        "--json",
        dest="json_output",
        action="store_true",
        help="Print only the final JSON payload for scripts and agent-to-agent adapters.",
    )
    parser.add_argument(
        "intent",
        type=str,
        help="Natural language user intent to decompose and plan.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    representative_reports: list[dict[str, Any]] = []

    try:
        if args.decompose_only:
            decomposition_output = run_decomposition(args.intent).to_dict()
            record_decomposition_events(decomposition_output, source="cli")
            output_sections = [("decomposition-agent", decomposition_output)]
            final_payload = decomposition_output
        else:
            decomposition_output = run_decomposition(args.intent).to_dict()
            run_id = default_run_id("planning")
            record_decomposition_events(decomposition_output, run_id=run_id, source="cli")
            planning_output = invoke_planning_agent({**decomposition_output, "run_id": run_id})
            record_planning_handoff_event(str(planning_output.get("run_id") or run_id), decomposition_output, planning_output)
            output_sections = [("decomposition-agent", decomposition_output)]
            representative_reports = append_planning_sections(output_sections, planning_output)
            final_payload = planning_output
    except Exception as exc:  # noqa: BLE001 - CLI must surface dependency/API failures.
        print(f"decomposition-agent error: {exc}", file=sys.stderr)
        return 1

    if args.json_output:
        print(json.dumps(final_payload, ensure_ascii=False, indent=2))
    else:
        for index, (agent_name, payload) in enumerate(output_sections):
            if index:
                print()
            print(format_agent_section(agent_name, payload))
        if not args.decompose_only:
            try:
                approval_results = prompt_for_representative_approvals(representative_reports)
                if approval_results_executed_successfully(approval_results):
                    remaining_payload = remaining_decomposition_after_planning(decomposition_output, final_payload)
                    if remaining_payload:
                        continuation_output = invoke_planning_agent(remaining_payload)
                        continuation_sections: list[tuple[str, dict[str, Any]]] = []
                        append_planning_sections(
                            continuation_sections,
                            continuation_output,
                            planning_agent_name="planning-agent-after-approval",
                        )
                        for agent_name, payload in continuation_sections:
                            print()
                            print(format_agent_section(agent_name, payload))
            except Exception as exc:  # noqa: BLE001 - CLI must surface approval/execution failures.
                print(f"decomposition-agent approval error: {exc}", file=sys.stderr)
                return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
