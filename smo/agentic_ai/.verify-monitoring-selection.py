from tools.knowledge_registry import find_knowledge_for_subtask


intent = "AMF pod를 2개로 만들고 5분 동안 2개를 유지하는지 확인하고, 유지하지 못하면 2개로 수정해줘"
result = find_knowledge_for_subtask(
    "5분 동안 AMF pod 2개 유지 여부를 확인하고 불일치할 때만 복구",
    intent=intent,
)
print({
    "available": result.available,
    "used": result.used,
    "agent_ids": [item.get("agent_id") or item.get("id") for item in result.agents],
    "match_strategies": [item.get("_match_strategy") for item in result.agents],
})
