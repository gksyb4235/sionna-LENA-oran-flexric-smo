"""LLM-backed LangGraph-style sub-graph planner."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from schemas import validate_langgraph_spec
from tools.knowledge_registry import (
    CommunicationContractError,
    KnowledgeRegistryResult,
    PLANNER_MANIFEST_FIELDS,
    canonicalize_knowledge_agent_identities,
    communication_contract_ref,
    planner_manifest_is_valid,
    validate_subgraph_communication_contracts,
)
from tools.subgraph_runtime import validate_subgraph_execution_topology

JAECHAN_ROOT = Path(
    os.getenv("AGENTIC_AI_ROOT", str(Path(__file__).resolve().parents[3]))
).resolve()
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DECOMPOSITION_ENV = JAECHAN_ROOT / "agents" / "decomposition" / ".env"

if str(JAECHAN_ROOT) not in sys.path:
    sys.path.append(str(JAECHAN_ROOT))

from agent_ops.llm_models import model_for  # noqa: E402

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
        raise FileNotFoundError(f"Missing planning instructions: {path}")
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
            if isinstance(payload.get("langgraph_spec"), dict):
                return payload
        index += max(end, 1)
    if first_object is not None:
        return first_object
    raise ValueError("Planning model response did not contain a JSON object.")


def create_planning_llm_agent():
    from deepagents import create_deep_agent
    from deepagents.backends import FilesystemBackend

    load_env_file()
    model = model_for("planning-agent")
    backend = FilesystemBackend(root_dir=PROJECT_ROOT, virtual_mode=True)
    return create_deep_agent(
        model=model,
        tools=[],
        system_prompt=load_agent_instructions(),
        skills=["/skills"],
        backend=backend,
    )


def _compact_candidates(
    documents: list[dict[str, Any]],
    *,
    id_key: str,
    fields: tuple[str, ...],
    require_manifest: bool = False,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for document in documents:
        identifier = document.get(id_key) or document.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            continue
        manifest = document.get("planner_manifest")
        contract = document.get("communication_contract")
        authoritative_ref = (
            communication_contract_ref(identifier, contract) if isinstance(contract, dict) else None
        )
        if require_manifest and not planner_manifest_is_valid(document):
            continue
        candidate = {id_key: identifier, **{key: document[key] for key in fields if key in document}}
        if id_key == "agent_id":
            candidate["planner_manifest"] = {
                key: manifest[key] for key in PLANNER_MANIFEST_FIELDS if key in manifest
            }
            candidate["contract_ref"] = authoritative_ref
        candidates.append(candidate)
    return candidates


def build_llm_prompt(*, intent: str, golden_goal_context_used: bool, subtask_id: str, subtask: str, knowledge: KnowledgeRegistryResult, completed_actions: list[dict[str, Any]] | None = None) -> str:
    knowledge_payload = {
        "knowledge_db_available": knowledge.available,
        "knowledge_context_used": knowledge.used,
        "knowledge_error": knowledge.error,
        "query_keywords": knowledge.query_keywords,
        "candidate_agents": _compact_candidates(
            knowledge.agents,
            id_key="agent_id",
            fields=(
                "id", "type", "name", "_matched_keywords", "_match_strategy",
            ),
            require_manifest=True,
        ),
        "candidate_tools": _compact_candidates(
            knowledge.tools,
            id_key="tool_id",
            fields=(
                "name", "type", "domain", "description", "capabilities", "tags", "owner_agent",
                "operations", "_matched_keywords", "_match_strategy",
            ),
        ),
    }
    return f"""Create a LangGraph-style sub-graph spec for one decomposed subtask.

Original user intent:
{intent}

Golden goal context used by decomposition: {str(golden_goal_context_used).lower()}

Subtask id:
{subtask_id}

Subtask:
{subtask}

Knowledge DB candidate context:
{json.dumps(knowledge_payload, ensure_ascii=False, indent=2)}

Previously completed subtask actions:
{json.dumps(completed_actions or [], ensure_ascii=False, indent=2)}

Planning requirements:
- Use only Knowledge DB candidate agents/tools from the candidate context above.
- Do not use local runtime registry entries as planning candidates.
- Do not invent placeholder agents or tools.
- Candidate documents are compact projections. Fields omitted here remain authoritative in Knowledge DB.
- Candidate documents may include _matched_keywords and _match_strategy. Treat them as approximate relevance signals, not exact-match requirements.
- The Planning Agent owns, compiles, executes, and manages the graph. Agent nodes are workers invoked by Planning, not graph runtime owners.
- The representative agent is the selected worker Planning communicates with; it does not have to be the entrypoint or first worker node.
- The graph is declarative only. Do not claim it is compiled or actually executed.
- Use stable ASCII node ids. Prefer each candidate document's agent_id/tool_id/id exactly when creating node ids.
- Route a simple read, metric lookup, snapshot, or report to Probe Agent.
- Route expected-outcome verification, "maintain for a duration", post-action validation, or "recover only if mismatched" to Monitoring Agent. Monitoring Agent delegates live collection to Probe Agent, so do not add Probe Agent before or after it.
- Do not combine Representative Core Agent and Probe Agent in one graph. Put post-action expected-state verification in a separate Monitoring Agent subtask.
- Monitoring is selective, never an automatic post-hook for every plan.
- Use previously completed subtask actions as history. If a phrase like "after scaling" merely refers to an already completed action, do not route it as a new mutation. A confirmed post-action drift is a new recovery need and may reuse the previously successful action.
- For conditional recovery, Planning must create and manage the recovery graph with worker order monitoring-agent -> representative-core-agent, and the edge entering Core must use `only_if_mismatch`. An intermediate registered tool node is allowed. Monitoring Agent is the gate: a passed expectation ends the graph and a confirmed mismatch alone may continue to the approval-gated Core action.
- If the subtask asks Monitoring Agent only to notify Planning Agent, use one linear `monitoring-agent -> completion-check` graph. Monitoring returns healthy, unavailable, or mismatch feedback in its response; Planning decides whether to create a separate recovery graph. Do not model those outcomes as separate terminal branches and do not add Representative Core Agent.
- If the subtask already contains validated Monitoring feedback with confirmed mismatch evidence, route recovery directly to Representative Core Agent. Do not add another Monitoring Agent before Core; Planning separately resumes monitoring only for the time remaining before its original absolute deadline.
- Never use probe-agent -> representative-core-agent for conditional recovery.
- If the subtask requires a current Core NF change, including restoration after confirmed drift, route it to Representative Core Agent so the existing approval flow can ask the user.
- Include at least one representative agent node and one completion-check terminal node.
- Copy the selected agent's opaque `contract_ref` onto its graph node; never expand it into contract fields.
- Add a generic langgraph_spec.evaluation_request that expresses the requested observation or expectation. After selection, Planning resolves the full MongoDB contract and validates the request fail-closed.
- For an exact requested state, use equality checks for every signal required by that state; do not weaken equality to a lower bound.

Return exactly one JSON object with exactly this top-level key:
- langgraph_spec

langgraph_spec must contain version, subtask_id, subtask, nodes, edges, entrypoint, terminal_nodes, representative_agent, compiled=false, knowledge_context_used, planning_basis, and evaluation_request.
"""


def _slug(value: str) -> str:
    chars: list[str] = []
    for char in value.lower():
        if char.isalnum():
            chars.append(char)
        elif chars and chars[-1] != "-":
            chars.append("-")
    return "".join(chars).strip("-")[:64] or "node"


def _coerce_node_id(value: Any) -> str:
    if isinstance(value, str):
        return _slug(value) if value.strip() else ""
    if isinstance(value, dict):
        for key in ("id", "node_id", "agent_id", "tool_id", "name"):
            coerced = _coerce_node_id(value.get(key))
            if coerced:
                return coerced
    if isinstance(value, list) and value:
        return _coerce_node_id(value[0])
    return ""


def _normalize_node(node: dict[str, Any], fallback_id: str, fallback_type: str) -> dict[str, Any]:
    normalized = dict(node)
    node_type = normalized.get("type")
    name = normalized.get("name")
    node_id = _coerce_node_id(normalized.get("id") or normalized.get("agent_id") or normalized.get("tool_id") or name)
    normalized["id"] = node_id or fallback_id
    normalized["type"] = node_type if isinstance(node_type, str) and node_type.strip() else fallback_type
    normalized["name"] = name if isinstance(name, str) and name.strip() else normalized["id"]
    normalized.setdefault("source", "llm_inferred")
    normalized.setdefault("description", "")
    capabilities = normalized.get("capabilities")
    if isinstance(capabilities, str):
        normalized["capabilities"] = [capabilities]
    elif not isinstance(capabilities, list):
        normalized["capabilities"] = []
    return normalized


def _normalize_evaluation_request(raw: Any, _subtask: str, *, monitoring_selected: bool = False) -> dict[str, Any]:
    if not isinstance(raw, dict):
        if monitoring_selected:
            raise ValueError("Monitoring Agent selection requires an evaluation_request from Planning Agent.")
        return {"type": "report_values", "completion_rule": "data_returned"}
    return dict(raw)


def normalize_langgraph_spec(raw_spec: dict[str, Any], subtask_id: str, subtask: str, knowledge_used: bool) -> dict[str, Any]:
    nodes: list[dict[str, Any]] = []
    raw_nodes = raw_spec.get("nodes")
    if isinstance(raw_nodes, list):
        for index, node in enumerate(raw_nodes, start=1):
            if isinstance(node, dict):
                fallback_type = "agent" if index == 1 else "tool"
                nodes.append(_normalize_node(node, f"{fallback_type}-{subtask_id}-{index}", fallback_type))
    if not nodes:
        nodes.append({
            "id": f"agent-placeholder-{subtask_id}",
            "type": "agent",
            "name": "placeholder-representative-agent",
            "role": "representative",
            "source": "llm_placeholder",
            "description": "LLM-inferred representative agent used until Knowledge DB agents are registered.",
            "capabilities": ["subtask-planning"],
        })

    node_ids = {str(node["id"]) for node in nodes}
    entrypoint_candidate = _coerce_node_id(raw_spec.get("entrypoint"))
    entrypoint = entrypoint_candidate if entrypoint_candidate in node_ids else nodes[0]["id"]
    representative_candidate = _coerce_node_id(raw_spec.get("representative_agent"))
    representative = representative_candidate if representative_candidate in node_ids else entrypoint

    terminal_nodes = raw_spec.get("terminal_nodes")
    terminal_candidates = [_coerce_node_id(item) for item in terminal_nodes] if isinstance(terminal_nodes, list) else [_coerce_node_id(terminal_nodes)]
    terminals = [node_id for node_id in terminal_candidates if node_id in node_ids]
    terminal_aliases: dict[str, str] = {}
    if not terminals:
        completion_id = f"agent-completion-check-{subtask_id}"
        if completion_id not in node_ids:
            nodes.append({
                "id": completion_id,
                "type": "agent",
                "name": "planning-completion-check",
                "role": "completion_checker",
                "source": "planning_agent",
                "description": "Checks whether the representative agent completed the subtask.",
                "capabilities": ["completion-monitoring"],
            })
            node_ids.add(completion_id)
        terminals = [completion_id]
        terminal_aliases = {node_id: completion_id for node_id in terminal_candidates if node_id}
    for index, node in enumerate(nodes):
        if node.get("id") in terminals and node.get("type") == "terminal":
            nodes[index] = {
                **node,
                "type": "agent",
                "role": "completion_checker",
                "name": "planning-completion-check",
                "source": "planning_agent",
            }

    monitoring_only = "monitoring-agent" in node_ids and "representative-core-agent" not in node_ids
    if monitoring_only:
        monitoring_node = next(node for node in nodes if node.get("id") == "monitoring-agent")
        completion_node = next(
            (
                node
                for node in nodes
                if node.get("id") != "monitoring-agent"
                and (node.get("role") == "completion_checker" or node.get("name") == "planning-completion-check")
            ),
            None,
        )
        if completion_node is None:
            completion_node = {
                "id": f"agent-completion-check-{subtask_id}",
                "type": "agent",
                "name": "planning-completion-check",
                "role": "completion_checker",
                "source": "planning_agent",
                "description": "Checks whether the representative agent completed the subtask.",
                "capabilities": ["completion-monitoring"],
            }
        nodes = [monitoring_node, completion_node]
        node_ids = {str(node["id"]) for node in nodes}
        entrypoint = "monitoring-agent"
        representative = "monitoring-agent"
        terminals = [str(completion_node["id"])]

    edges: list[dict[str, str]] = []
    raw_edges = raw_spec.get("edges")
    if monitoring_only:
        edges.append({"source": "monitoring-agent", "target": terminals[0], "condition": "on_response"})
    elif isinstance(raw_edges, list):
        for edge in raw_edges:
            if not isinstance(edge, dict):
                continue
            source = _coerce_node_id(edge.get("source", edge.get("from")))
            target = _coerce_node_id(edge.get("target", edge.get("to")))
            target = terminal_aliases.get(target, target)
            if source in node_ids and target in node_ids:
                condition = edge.get("condition")
                if "monitoring-agent" in node_ids and target == "representative-core-agent":
                    condition = "only_if_mismatch"
                edges.append({"source": source, "target": target, "condition": condition if isinstance(condition, str) and condition else "handoff_or_tool_use"})
    if "monitoring-agent" in node_ids and "representative-core-agent" in node_ids:
        edges = [
            edge for edge in edges
            if not (edge["source"] == "monitoring-agent" and edge["target"] in terminals)
        ]
    if not edges:
        previous = entrypoint
        for node in nodes:
            node_id = node["id"]
            if node_id == previous:
                continue
            condition = "only_if_mismatch" if "monitoring-agent" in node_ids and node_id == "representative-core-agent" else "handoff_or_tool_use"
            edges.append({"source": previous, "target": node_id, "condition": condition})
            previous = node_id

    planning_basis = raw_spec.get("planning_basis")
    spec = {
        "version": "langgraph-style-v1",
        "subtask_id": subtask_id,
        "subtask": subtask,
        "nodes": nodes,
        "edges": edges,
        "entrypoint": entrypoint,
        "terminal_nodes": terminals,
        "representative_agent": representative,
        "compiled": False,
        "knowledge_context_used": bool(knowledge_used),
        "planning_basis": planning_basis if isinstance(planning_basis, str) and planning_basis.strip() else "llm_generated",
        "evaluation_request": _normalize_evaluation_request(
            raw_spec.get("evaluation_request"),
            subtask,
            monitoring_selected=any(node.get("id") == "monitoring-agent" for node in nodes),
        ),
    }
    validate_langgraph_spec(spec)
    return spec


def build_langgraph_spec_with_llm(*, intent: str, golden_goal_context_used: bool, subtask_id: str, subtask: str, knowledge: KnowledgeRegistryResult, completed_actions: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    agent = create_planning_llm_agent()
    prompt = build_llm_prompt(
        intent=intent,
        golden_goal_context_used=golden_goal_context_used,
        subtask_id=subtask_id,
        subtask=subtask,
        knowledge=knowledge,
        completed_actions=completed_actions,
    )
    messages: list[Any] = [{"role": "user", "content": prompt}]
    for attempt in range(2):
        result = agent.invoke({"messages": messages})
        response_messages = result.get("messages", []) if isinstance(result, dict) else []
        if not response_messages:
            raise RuntimeError("Planning LLM returned no messages.")
        final_content = _content_to_text(getattr(response_messages[-1], "content", response_messages[-1]))
        try:
            payload = extract_json_object(final_content)
            spec_payload = payload.get("langgraph_spec")
            if isinstance(spec_payload, dict):
                spec = normalize_langgraph_spec(spec_payload, subtask_id, subtask, knowledge.used)
                spec = canonicalize_knowledge_agent_identities(spec, knowledge)
                validate_subgraph_execution_topology(spec)
                validate_subgraph_communication_contracts(spec, knowledge)
                return spec
            error = "Planning LLM response must contain a langgraph_spec object."
        except CommunicationContractError as exc:
            if exc.code != "knowledge_agent_communication_contract_violation":
                raise
            error = str(exc)
        except ValueError as exc:
            error = str(exc)
        if attempt:
            raise ValueError(error)
        messages = [
            *response_messages,
            {
                "role": "user",
                "content": (
                    "Fix only the response contract error from your previous answer. "
                    "Return exactly one valid JSON object with exactly one top-level key named "
                    "`langgraph_spec`. Preserve the intended graph and candidate selection. "
                    "Do not add prose or Markdown fences.\n\n"
                    f"Parsing or validation error:\n{error}\n\nPrevious response:\n{final_content}"
                ),
            },
        ]
    raise AssertionError("unreachable")
