"""Schema helpers for the Planning Agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SubtaskStatus = Literal[
    "pending",
    "running",
    "completed",
    "mock_completed",
    "failed",
    "blocked",
    "partial",
    "pending_approval",
    "rejected",
    "advisor_unavailable",
]
OverallStatus = Literal["running", "completed", "mock_completed", "partial", "blocked", "pending_approval", "rejected"]
SUBTASK_STATUSES = set(SubtaskStatus.__args__)  # type: ignore[attr-defined]
OVERALL_STATUSES = set(OverallStatus.__args__)  # type: ignore[attr-defined]


@dataclass(frozen=True)
class DecompositionInput:
    intent: str
    subtasks: list[str]
    golden_goal_context_used: bool


@dataclass(frozen=True)
class SubtaskPlan:
    subtask_id: str
    subtask: str
    langgraph_spec: dict[str, Any]
    representative_agent: dict[str, Any]
    attempts: list[dict[str, Any]]
    status: SubtaskStatus
    result: dict[str, Any]
    knowledge_context_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask_id": self.subtask_id,
            "subtask": self.subtask,
            "langgraph_spec": self.langgraph_spec,
            "representative_agent": self.representative_agent,
            "attempts": self.attempts,
            "status": self.status,
            "result": self.result,
            "knowledge_context_used": self.knowledge_context_used,
        }


@dataclass(frozen=True)
class PlanningReport:
    run_id: str
    intent: str
    subtask_plans: list[SubtaskPlan]
    overall_status: OverallStatus

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "intent": self.intent,
            "subtask_plans": [plan.to_dict() for plan in self.subtask_plans],
            "overall_status": self.overall_status,
        }


def validate_decomposition_input(data: dict[str, Any]) -> DecompositionInput:
    """Validate Decomposition Agent output for Planning Agent input."""

    if not isinstance(data, dict):
        raise TypeError("Decomposition input must be a JSON object.")
    intent = data.get("intent")
    subtasks = data.get("subtasks")
    golden_goal_context_used = data.get("golden_goal_context_used")

    if not isinstance(intent, str) or not intent.strip():
        raise ValueError("Decomposition input must contain a non-empty 'intent' string.")
    if not isinstance(subtasks, list):
        raise TypeError("Decomposition input must contain a 'subtasks' list.")
    normalized_subtasks: list[str] = []
    for item in subtasks:
        if not isinstance(item, str):
            raise TypeError("Every subtask must be a string.")
        task = item.strip()
        if task:
            normalized_subtasks.append(task)
    if not normalized_subtasks:
        raise ValueError("Decomposition input must contain at least one non-empty subtask.")
    if not isinstance(golden_goal_context_used, bool):
        raise TypeError("Decomposition input must contain boolean 'golden_goal_context_used'.")

    return DecompositionInput(
        intent=intent,
        subtasks=normalized_subtasks,
        golden_goal_context_used=golden_goal_context_used,
    )


def validate_langgraph_spec(spec: dict[str, Any]) -> None:
    required = {"nodes", "edges", "entrypoint", "terminal_nodes", "representative_agent"}
    missing = required.difference(spec)
    if missing:
        raise ValueError(f"LangGraph spec missing required keys: {sorted(missing)}")
    if not isinstance(spec["nodes"], list) or not spec["nodes"]:
        raise ValueError("LangGraph spec must contain at least one node.")
    if not isinstance(spec["edges"], list):
        raise TypeError("LangGraph spec 'edges' must be a list.")
    node_ids = {node.get("id") for node in spec["nodes"] if isinstance(node, dict)}
    if spec["entrypoint"] not in node_ids:
        raise ValueError("LangGraph spec entrypoint must reference a node id.")
    if spec["representative_agent"] not in node_ids:
        raise ValueError("LangGraph spec representative_agent must reference a node id.")
    if not all(node_id in node_ids for node_id in spec["terminal_nodes"]):
        raise ValueError("Every LangGraph spec terminal node must reference a node id.")


def validate_planning_report(data: dict[str, Any]) -> None:
    """Validate the public Planning Agent output contract."""

    if not isinstance(data, dict):
        raise TypeError("Planning report must be a JSON object.")
    required = {"intent", "subtask_plans", "overall_status"}
    optional = {"run_id"}
    keys = set(data.keys())
    if not required.issubset(keys) or not keys.issubset(required | optional):
        raise ValueError("Planning report must contain intent, subtask_plans, overall_status, and optional run_id.")
    if "run_id" in data and not isinstance(data["run_id"], str):
        raise TypeError("Planning report 'run_id' must be a string when provided.")
    if not isinstance(data["intent"], str):
        raise TypeError("Planning report 'intent' must be a string.")
    if not isinstance(data["subtask_plans"], list) or not data["subtask_plans"]:
        raise ValueError("Planning report must contain at least one subtask plan.")
    for plan in data["subtask_plans"]:
        if not isinstance(plan, dict):
            raise TypeError("Every subtask plan must be an object.")
        for key in ("subtask_id", "subtask", "langgraph_spec", "representative_agent", "attempts", "status", "result"):
            if key not in plan:
                raise ValueError(f"Subtask plan missing required key: {key}")
        if plan["status"] not in SUBTASK_STATUSES:
            raise ValueError("Invalid subtask plan status.")
        validate_langgraph_spec(plan["langgraph_spec"])
    if data["overall_status"] not in OVERALL_STATUSES:
        raise ValueError("Invalid planning report overall_status.")
