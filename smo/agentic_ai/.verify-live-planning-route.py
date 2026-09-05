from tools.knowledge_registry import find_knowledge_for_subtask
from tools.llm_subgraph_planner import build_langgraph_spec_with_llm
from agent import build_registered_subgraph


intent = "AMF의 pod 개수를 2개로 만들고 5분 동안 2개를 유지하는지 확인하고, 유지하지 못하면 2개로 수정해줘"
subtask = "5분 동안 AMF pod가 정확히 2개로 유지되는지 확인하고, 불일치할 때만 다시 2개로 조정"
knowledge = find_knowledge_for_subtask(subtask, intent=intent)
spec = build_langgraph_spec_with_llm(
    intent=intent,
    golden_goal_context_used=False,
    subtask_id="subtask-003",
    subtask=subtask,
    knowledge=knowledge,
    completed_actions=[{
        "operation": "scale",
        "resource": "deployment",
        "namespace": "free5gc-v4",
        "nfs": ["AMF"],
        "parameters": {"replicas": 2},
        "status": "completed",
    }],
)
registered, block_reason = build_registered_subgraph(
    intent=intent,
    subtask_id="subtask-003",
    subtask=subtask,
    knowledge=knowledge,
    llm_langgraph_spec=spec,
    completed_actions=[{"operation": "scale", "nfs": ["AMF"], "parameters": {"replicas": 2}, "status": "completed"}],
)
print({
    "candidate_agent_ids": [item.get("agent_id") or item.get("id") for item in knowledge.agents],
    "representative_agent": spec.get("representative_agent"),
    "node_ids": [item.get("id") for item in spec.get("nodes", [])],
    "edges": spec.get("edges"),
    "evaluation_request": spec.get("evaluation_request"),
    "registered_block_reason": block_reason,
    "registered_representative": registered.get("representative_agent"),
    "registered_node_ids": [item.get("id") for item in registered.get("nodes", [])],
    "registered_edges": registered.get("edges"),
    "registered_terminal_nodes": registered.get("terminal_nodes"),
})
