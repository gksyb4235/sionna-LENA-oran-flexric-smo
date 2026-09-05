"""Canonical registered agent/tool catalog for Planning Agent."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_DIR = PROJECT_ROOT / "registry"
ACTIVE_STATUSES = {"active", "available"}

PLANNING_AGENT_NODE: dict[str, Any] = {
    "id": "planning-agent",
    "type": "agent",
    "role": "planner",
    "name": "Planning Agent",
    "source": "local_runtime_registry",
    "description": "Registered planner node used only to report that no executable registered subgraph is available.",
    "capabilities": ["registered-subgraph-gating"],
}


def _load_documents(filename: str) -> list[dict[str, Any]]:
    path = REGISTRY_DIR / filename
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Registry file must contain a JSON list: {path}")
    documents: list[dict[str, Any]] = []
    for document in payload:
        if not isinstance(document, dict):
            raise TypeError(f"Registry document must be an object in {path}")
        documents.append(document)
    return documents


def _filter_status(documents: list[dict[str, Any]], *, include_planned: bool = False) -> list[dict[str, Any]]:
    if include_planned:
        return [dict(document) for document in documents]
    return [dict(document) for document in documents if str(document.get("status", "active")).lower() in ACTIVE_STATUSES]


def registered_agent_documents(*, include_planned: bool = False) -> list[dict[str, Any]]:
    return _filter_status(_load_documents("agents.json"), include_planned=include_planned)


def registered_tool_documents(*, include_planned: bool = False) -> list[dict[str, Any]]:
    return _filter_status(_load_documents("tools.json"), include_planned=include_planned)


def _prompt_document(document: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "id",
        "agent_id",
        "tool_id",
        "type",
        "role",
        "name",
        "domain",
        "description",
        "can_do",
        "cannot_do",
        "capabilities",
        "operations",
        "tools",
        "owner_agent",
        "tags",
        "keywords",
        "status",
    )
    return {key: document[key] for key in keys if key in document}


def registered_agents_for_prompt() -> list[dict[str, Any]]:
    return [_prompt_document(document) for document in registered_agent_documents()]


def registered_tools_for_prompt() -> list[dict[str, Any]]:
    return [_prompt_document(document) for document in registered_tool_documents()]


def known_runtime_representatives() -> dict[str, dict[str, Any]]:
    representatives: dict[str, dict[str, Any]] = {}
    for document in registered_agent_documents():
        agent_id = str(document.get("agent_id") or document.get("id") or "").strip()
        if not agent_id:
            continue
        representatives[agent_id] = {
            "id": agent_id,
            "type": "agent",
            "role": document.get("role", "representative_agent"),
            "name": document.get("name", agent_id),
            "source": "local_runtime_registry",
            "description": document.get("description", ""),
            "capabilities": document.get("capabilities", []),
        }
    return representatives


def known_runtime_tools() -> dict[str, dict[str, Any]]:
    tools: dict[str, dict[str, Any]] = {}
    for document in registered_tool_documents():
        tool_id = str(document.get("tool_id") or document.get("id") or "").strip()
        if not tool_id:
            continue
        tools[tool_id] = {
            "id": tool_id,
            "type": "tool",
            "name": document.get("name", tool_id),
            "source": "local_runtime_registry",
            "description": document.get("description", ""),
            "capabilities": document.get("capabilities", []),
        }
    return tools
