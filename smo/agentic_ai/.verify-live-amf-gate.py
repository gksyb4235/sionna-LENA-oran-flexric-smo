from unittest.mock import patch

from tools import subgraph_runtime


expected = {
    "type": "thresholds",
    "duration_seconds": 0,
    "checks": [
        {"name": "AMF desired is 2", "query_label": "AMF_desired_replicas", "operator": "==", "value": 2},
        {"name": "AMF available is 2", "query_label": "AMF_available_replicas", "operator": "==", "value": 2},
        {"name": "AMF ready is 2", "query_label": "AMF_ready_replicas", "operator": "==", "value": 2},
        {"name": "AMF ready pods is 2", "query_label": "AMF_ready_pods", "operator": "==", "value": 2},
    ],
}
graph = {
    "nodes": [
        {"id": "monitoring-agent", "type": "agent", "name": "Monitoring Agent"},
        {"id": "representative-core-agent", "type": "agent", "name": "Representative Core Agent"},
        {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
    ],
    "edges": [
        {"source": "monitoring-agent", "target": "representative-core-agent", "condition": "only_if_mismatch"},
        {"source": "representative-core-agent", "target": "done"},
    ],
    "entrypoint": "monitoring-agent",
    "terminal_nodes": ["done"],
    "representative_agent": "monitoring-agent",
    "compiled": False,
    "knowledge_context_used": True,
    "evaluation_request": expected,
}
prior_core_report = {
    "status": "completed",
    "command_proposal": {
        "operation": "scale",
        "resource": "deployment",
        "namespace": "free5gc-v4",
        "nfs": ["AMF"],
        "parameters": {"replicas": 2},
    },
    "result": {"success": True},
}
planning_report = {
    "run_id": "planning-live-amf-gate-check",
    "intent": "AMF pod를 2개로 만들고 2개를 유지하는지 확인하며 불일치할 때만 2개로 복구",
    "subtask_plans": [{
        "subtask_id": "subtask-001",
        "subtask": "AMF를 2개로 설정",
        "langgraph_spec": {"evaluation_request": expected},
        "representative_agent": {"id": "representative-core-agent"},
        "attempts": [],
        "status": "completed",
        "result": {"status": "completed", "runtime": "representative-core-agent", "representative_core_report": prior_core_report},
        "knowledge_context_used": True,
    }],
    "overall_status": "running",
}

with patch.object(subgraph_runtime, "invoke_representative_core_agent", side_effect=AssertionError("Core must be skipped when AMF is already 2")) as core:
    result = subgraph_runtime.execute_subgraph(
        original_intent=planning_report["intent"],
        subtask_id="subtask-002",
        subtask="AMF pod가 2개인지 검증하고 불일치할 때만 2개로 복구",
        langgraph_spec=graph,
        representative_agent=graph["nodes"][0],
        run_id=planning_report["run_id"],
        planning_report=planning_report,
    )

monitoring = result.get("monitoring_agent_report") or {}
probe = monitoring.get("probe_report") or {}
checks = probe.get("evaluation_result", {}).get("checks", [])
print({
    "status": result.get("status"),
    "runtime": result.get("runtime"),
    "executed_node_ids": result.get("executed_node_ids"),
    "monitoring_status": monitoring.get("status"),
    "probe_collection_status": probe.get("status"),
    "probe_evaluation_status": probe.get("evaluation_result", {}).get("status"),
    "probe_errors": probe.get("errors"),
    "queries": [item.get("query") for result_item in probe.get("results", []) for item in result_item.get("query_results", [])],
    "checks": [
        {"name": item.get("name"), "actual": item.get("actual"), "expected": item.get("expected"), "ok": item.get("ok")}
        for item in checks
    ],
    "core_invoked": core.call_count > 0,
})
