from __future__ import annotations

import json
import os
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import agent
import server
from fastapi.testclient import TestClient
from schemas import validate_decomposition_input, validate_planning_report
from tools import llm_subgraph_planner, subgraph_runtime
from tools.knowledge_registry import (
    _annotate_and_filter,
    _find_active_catalog,
    _query_keywords,
    canonicalize_knowledge_agent_identities,
    communication_contract_ref,
    resolve_agent_communication_contract,
)
from tools.llm_subgraph_planner import build_llm_prompt, normalize_langgraph_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]

REPORT_VALUES_SCHEMA = {
    "type": "object",
    "required": ["type", "completion_rule"],
    "properties": {
        "type": {"const": "report_values"},
        "completion_rule": {"const": "data_returned"},
    },
    "additionalProperties": False,
}
THRESHOLDS_SCHEMA = {
    "type": "object",
    "required": ["type", "schedule", "checks"],
    "properties": {
        "type": {"const": "thresholds"},
        "schedule": {
            "type": "object",
            "required": ["duration_seconds"],
            "properties": {
                "duration_seconds": {"type": "integer", "minimum": 0},
                "interval_seconds": {"type": "integer", "exclusiveMinimum": 0},
            },
            "additionalProperties": False,
        },
        "checks": {
            "type": "array",
            "minItems": 1,
            "x-unique-by": "check_id",
            "items": {
                "type": "object",
                "required": ["check_id", "observation", "expected"],
                "properties": {
                    "check_id": {"type": "string", "minLength": 1},
                    "observation": {"type": "string", "minLength": 1},
                    "expected": {"type": "object"},
                },
                "additionalProperties": False,
            },
        },
    },
    "additionalProperties": False,
}


def communication_contract(evaluation_schema: dict[str, object]) -> dict[str, object]:
    return {
        "version": "test",
        "schema_dialect": "json-schema-subset-v1",
        "planning": {"evaluation_request_schema": evaluation_schema},
        "request_schema": {"type": "object"},
        "response_schema": {"type": "object"},
    }


def planner_manifest(
    agent_id: str,
    evaluation_schema: dict[str, object],
    version: str = "test",
) -> dict[str, object]:
    contract = communication_contract(evaluation_schema)
    return {
        "contract_version": version,
        "purpose": "Select the matching runtime agent.",
        "capabilities": ["test-capability"],
        "constraints": ["Use the authoritative contract after selection"],
        "planning_contract": {"supported_types": ["report_values", "thresholds"]},
        "planning_schema_path": "planning.evaluation_request_schema",
        "contract_ref": communication_contract_ref(agent_id, contract),
    }


RUNTIME_TEST_CONTRACT = {
    "version": "runtime-test",
    "schema_dialect": "json-schema-subset-v1",
    "request_schema": {"type": "object"},
    "response_schema": {"type": "object"},
    "approval_request_schema": {"type": "object"},
    "approval_response_schema": {"type": "object"},
}


def runtime_contract_ref(agent_id: str) -> dict[str, str]:
    return communication_contract_ref(agent_id, RUNTIME_TEST_CONTRACT)


def runtime_agent_node(agent_id: str, name: str) -> dict[str, object]:
    return {
        "id": agent_id,
        "type": "agent",
        "name": name,
        "contract_ref": runtime_contract_ref(agent_id),
    }


def runtime_resolved_node(agent_id: str, name: str) -> dict[str, object]:
    return {**runtime_agent_node(agent_id, name), "_runtime_communication_contract": dict(RUNTIME_TEST_CONTRACT)}


def failed_threshold_evaluation() -> dict[str, object]:
    return {
        "type": "thresholds",
        "status": "failed",
        "checks": [{
            "check_id": "opaque-check",
            "observation": "AMF pod count",
            "ok": False,
            "valid": True,
            "actual": 1,
            "expected": 2,
            "operator": "==",
        }],
        "query_values": {"opaque-check": 1},
    }


def expected_threshold_request() -> dict[str, object]:
    return {
        "type": "thresholds",
        "schedule": {"duration_seconds": 0},
        "checks": [{
            "check_id": "opaque-check",
            "observation": "AMF pod count",
            "expected": {"operator": "==", "value": 2},
        }],
    }


def passed_threshold_evaluation() -> dict[str, object]:
    return {
        "type": "thresholds",
        "status": "passed",
        "checks": [{
            "check_id": "opaque-check",
            "observation": "AMF pod count",
            "ok": True,
            "valid": True,
            "actual": 2,
            "expected": 2,
            "operator": "==",
        }],
        "query_values": {"opaque-check": 2},
    }


def sample_decomposition() -> dict[str, object]:
    return {
        "intent": "Launch a new mobile app",
        "subtasks": ["Define launch goals", "Prepare app store release"],
        "golden_goal_context_used": False,
    }


def fake_llm_spec(**kwargs: object) -> dict[str, object]:
    subtask_id = str(kwargs["subtask_id"])
    subtask = str(kwargs["subtask"])
    lower = subtask.lower()
    if any(word in lower for word in ("monitor", "cpu", "health")):
        representative_id = "probe-agent"
        tool_id = "grafana-prometheus-tool"
        representative_name = "Probe Agent"
        tool_name = "Grafana Prometheus Tool"
    elif any(word in lower for word in ("scale", "replica", "restart", "delete", "patch")):
        representative_id = "representative-core-agent"
        tool_id = "core-kubectl-command-tool"
        representative_name = "Representative Core Agent"
        tool_name = "Core kubectl Command Tool"
    else:
        representative_id = f"agent-placeholder-{subtask_id}"
        tool_id = None
        representative_name = "placeholder-representative-agent"
        tool_name = "placeholder-tool"
    completion_id = f"agent-completion-check-{subtask_id}"
    nodes = [
        {
            "id": representative_id,
            "type": "agent",
            "name": representative_name,
            "source": "fake_llm",
            "description": "fake LLM selected representative",
            "capabilities": ["fake-llm-selection"],
        },
        {
            "id": completion_id,
            "type": "agent",
            "name": "planning-completion-check",
            "source": "planning_agent",
            "description": "completion check",
            "capabilities": ["completion-monitoring"],
        },
    ]
    edges = [{"source": representative_id, "target": completion_id, "condition": "report_completion_status"}]
    if tool_id:
        nodes.insert(1, {"id": tool_id, "type": "tool", "name": tool_name, "source": "fake_llm", "description": "fake LLM selected tool", "capabilities": ["fake-tool-selection"]})
        edges = [
            {"source": representative_id, "target": tool_id, "condition": "tool_use"},
            {"source": tool_id, "target": completion_id, "condition": "report_completion_status"},
        ]
    return {
        "version": "langgraph-style-v1",
        "subtask_id": subtask_id,
        "subtask": subtask,
        "nodes": nodes,
        "edges": edges,
        "entrypoint": representative_id,
        "terminal_nodes": [completion_id],
        "representative_agent": representative_id,
        "compiled": False,
        "knowledge_context_used": False,
        "planning_basis": "fake llm selected registered representative",
        "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
    }

def knowledge_with_candidates() -> agent.KnowledgeRegistryResult:
    return agent.KnowledgeRegistryResult(
        used=True,
        available=True,
        error=None,
        query_keywords=["monitor", "cpu", "scale", "replica", "amf", "smf"],
        agents=[
            {
                "id": "probe-agent",
                "agent_id": "probe-agent",
                "_match_score": 0.99,
                "type": "agent",
                "name": "Probe Agent",
                "description": "Monitors free5gc NFs and reports metrics.",
                "capabilities": ["free5gc-monitoring", "cpu-average", "replica-health"],
                "planner_manifest": planner_manifest("probe-agent", {"oneOf": [REPORT_VALUES_SCHEMA, THRESHOLDS_SCHEMA]}),
                "communication_contract": communication_contract({"oneOf": [REPORT_VALUES_SCHEMA, THRESHOLDS_SCHEMA]}),
            },
            {
                "id": "representative-core-agent",
                "agent_id": "representative-core-agent",
                "type": "agent",
                "name": "Representative Core Agent",
                "description": "Creates approval-gated kubectl proposals for free5gc Core actions.",
                "capabilities": ["free5gc-core-actions", "scale-deployment", "approval-request"],
                "planner_manifest": planner_manifest("representative-core-agent", REPORT_VALUES_SCHEMA),
                "communication_contract": communication_contract(REPORT_VALUES_SCHEMA),
            },
            {
                "id": "monitoring-agent",
                "agent_id": "monitoring-agent",
                "type": "agent",
                "name": "Monitoring Agent",
                "description": "Verifies expected outcomes over time and gates conditional recovery.",
                "capabilities": ["expected-outcome-verification", "timed-monitoring", "conditional-remediation-gate"],
                "planner_manifest": planner_manifest("monitoring-agent", THRESHOLDS_SCHEMA),
                "communication_contract": communication_contract(THRESHOLDS_SCHEMA),
            },
        ],
        tools=[
            {
                "id": "grafana-prometheus-tool",
                "tool_id": "grafana-prometheus-tool",
                "_match_score": 0.98,
                "type": "tool",
                "name": "Grafana Prometheus Tool",
                "description": "Read-only Grafana/Prometheus query tool.",
                "capabilities": ["query_prometheus", "monitor_nf_cpu_average", "monitor_nf_replica_health"],
                "owner_agent": "probe-agent",
            },
            {
                "id": "core-kubectl-command-tool",
                "tool_id": "core-kubectl-command-tool",
                "type": "tool",
                "name": "Core kubectl Command Tool",
                "description": "Approval-gated kubectl command preview/execution tool.",
                "capabilities": ["kubectl-preview", "scale-deployment"],
                "owner_agent": "representative-core-agent",
            },
        ],
    )


class PlanningContractTests(unittest.TestCase):
    def setUp(self) -> None:
        settle = patch.dict(os.environ, {"PLANNING_REPRESENTATIVE_SETTLE_SECONDS": "0"})
        settle.start()
        self.addCleanup(settle.stop)
        resolver = patch.object(
            subgraph_runtime,
            "resolve_agent_communication_contract",
            side_effect=lambda _agent_id, _contract_ref: dict(RUNTIME_TEST_CONTRACT),
        )
        resolver.start()
        self.addCleanup(resolver.stop)

    def test_monitoring_feedback_http_requires_shared_token(self) -> None:
        client = TestClient(server.app)
        with patch.dict(os.environ, {"MONITORING_AGENT_TOKEN": "test-token"}):
            with patch.object(server, "load_env_file"):
                response = client.post("/feedback", json={})

        self.assertEqual(response.status_code, 401)

    def test_monitoring_feedback_adds_one_evidence_backed_recovery_subtask(self) -> None:
        graph = fake_llm_spec(subtask_id="subtask-001", subtask="Scale AMF to 2 replicas")
        expectation = {
            "type": "thresholds",
            "schedule": {"duration_seconds": 0},
            "checks": [{"check_id": "opaque-check", "observation": "AMF pod count", "expected": {"operator": "==", "value": 2}}],
        }
        graph["evaluation_request"] = expectation
        action = {"operation": "scale", "parameters": {"replicas": 2}}
        state = {
            "run_id": "planning-feedback-test",
            "intent": "Scale AMF to 2 replicas",
            "subtasks": ["Scale AMF to 2 replicas"],
            "golden_goal_context_used": False,
            "current_subtask_index": 1,
            "subtask_plans": [{
                "subtask_id": "subtask-001",
                "subtask": "Scale AMF to 2 replicas",
                "langgraph_spec": graph,
                "representative_agent": {"id": "representative-core-agent", "name": "Representative Core Agent"},
                "attempts": [{"attempt_number": 1}],
                "status": "completed",
                "result": {
                    "status": "completed",
                    "representative_core_report": {"status": "completed", "command_proposal": action},
                },
                "knowledge_context_used": True,
            }],
            "pending_approval": None,
            "overall_status": "completed",
        }
        observed = failed_threshold_evaluation()
        feedback = {
            "event_id": "monitoring-feedback-test",
            "source": "monitoring-agent",
            "run_id": state["run_id"],
            "subtask_id": "subtask-001",
            "plan_fingerprint": agent._plan_fingerprint(state["subtask_plans"][0]),
            "status": "replan_required",
            "reason": "expected_network_state_not_observed",
            "expected": expectation,
            "observed": observed,
            "previous_action": action,
            "probe_errors": [],
        }

        with patch.object(agent, "load_planning_state", return_value=state):
            with patch.object(agent, "save_planning_state") as save:
                with patch.object(agent, "run_planning_loop", side_effect=agent._state_to_report) as run:
                    report = agent.apply_monitoring_feedback(feedback).to_dict()

        self.assertEqual(state["subtask_plans"][0]["status"], "completed")
        self.assertNotIn("monitoring_replan_count", state)
        self.assertEqual(state["current_subtask_index"], 1)
        self.assertNotIn("Do not repeat", state["subtasks"][1])
        self.assertNotIn("different registered", state["subtasks"][1])
        self.assertNotIn("monitoring_failed_action_signatures", state)
        self.assertIn("approval-gated mutation agent", state["subtasks"][1])
        self.assertIn("do not merely re-observe", state["subtasks"][1])
        self.assertIn("Reusing a previously successful action is allowed", state["subtasks"][1])
        self.assertIn('"check_id":"opaque-check"', state["subtasks"][1])
        self.assertIn('"actual":1', state["subtasks"][1])
        self.assertEqual(report["overall_status"], "running")
        save.assert_called_once_with(state)
        run.assert_called_once_with(state)

        with patch.object(agent, "load_planning_state", return_value=state):
            with patch.object(agent, "run_planning_loop", side_effect=agent._state_to_report) as duplicate_run:
                duplicate = agent.apply_monitoring_feedback(feedback).to_dict()
        self.assertEqual(duplicate["run_id"], state["run_id"])
        duplicate_run.assert_called_once_with(state)

        recovery_plan = dict(state["subtask_plans"][0])
        recovery_plan.update({"subtask_id": "subtask-002", "subtask": state["subtasks"][1], "status": "completed"})
        state["subtask_plans"].append(recovery_plan)
        state["current_subtask_index"] = 2
        agent._set_state_status(state)
        self.assertEqual(state["overall_status"], "completed")

    def test_recovery_approval_resumes_only_remaining_monitoring_window(self) -> None:
        expectation = {
            "type": "thresholds",
            "schedule": {"duration_seconds": 300, "interval_seconds": 30},
            "checks": [{"check_id": "opaque-check", "observation": "AMF pod count", "expected": {"operator": "==", "value": 2}}],
        }
        monitoring_graph = {
            "version": "langgraph-style-v1",
            "subtask_id": "subtask-002",
            "subtask": "Monitor AMF for five minutes",
            "nodes": [
                runtime_agent_node("monitoring-agent", "Monitoring Agent"),
                {"id": "monitor-done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [{"source": "monitoring-agent", "target": "monitor-done"}],
            "entrypoint": "monitoring-agent",
            "terminal_nodes": ["monitor-done"],
            "representative_agent": "monitoring-agent",
            "compiled": False,
            "knowledge_context_used": True,
            "planning_basis": "test",
            "evaluation_request": expectation,
        }
        monitoring_plan = {
            "subtask_id": "subtask-002",
            "subtask": monitoring_graph["subtask"],
            "langgraph_spec": monitoring_graph,
            "representative_agent": monitoring_graph["nodes"][0],
            "attempts": [{"attempt_number": 1}],
            "status": "failed",
            "result": {"status": "replan_required"},
            "knowledge_context_used": True,
        }
        state = {
            "run_id": "planning-remaining-window-test",
            "intent": "Keep AMF at 2 for five minutes",
            "subtasks": ["Scale AMF to 2", monitoring_graph["subtask"]],
            "current_subtask_index": 1,
            "subtask_plans": [monitoring_plan],
            "pending_approval": None,
            "overall_status": "running",
        }
        self.assertTrue(agent._record_monitoring_window(
            state,
            "subtask-002",
            monitoring_graph["subtask"],
            monitoring_graph,
            now=1000.0,
        ))
        self.assertEqual(state["monitoring_windows"]["subtask-002"]["deadline_at_unix"], 1300.0)

        feedback = {
            "event_id": "monitoring-remaining-window-test",
            "source": "monitoring-agent",
            "run_id": state["run_id"],
            "subtask_id": "subtask-002",
            "plan_fingerprint": agent._plan_fingerprint(monitoring_plan),
            "status": "replan_required",
            "reason": "expected_network_state_not_observed",
            "expected": expectation,
            "observed": failed_threshold_evaluation(),
            "previous_action": None,
            "probe_errors": [],
        }
        self.assertTrue(agent._apply_monitoring_feedback_to_state(state, feedback))
        self.assertEqual(state["monitoring_recoveries"]["subtask-003"]["window_id"], "subtask-002")

        core_node = runtime_agent_node("representative-core-agent", "Representative Core Agent")
        pending_report = {"status": "pending_approval"}
        recovery_plan = {
            "subtask_id": "subtask-003",
            "subtask": state["subtasks"][2],
            "langgraph_spec": {
                "nodes": [core_node, {"id": "core-done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"}],
                "edges": [{"source": "representative-core-agent", "target": "core-done"}],
                "entrypoint": "representative-core-agent",
                "terminal_nodes": ["core-done"],
                "representative_agent": "representative-core-agent",
            },
            "representative_agent": core_node,
            "attempts": [{"attempt_number": 1, "status": "pending_approval"}],
            "status": "pending_approval",
            "result": {"status": "pending_approval", "representative_core_report": pending_report},
            "knowledge_context_used": True,
        }
        state["subtask_plans"].append(recovery_plan)
        state["pending_approval"] = {"subtask_id": "subtask-003", "report": pending_report}
        approval_result = {"status": "completed", "result": {"success": True}}

        with patch.dict(os.environ, {"PLANNING_REPRESENTATIVE_SETTLE_SECONDS": "30"}):
            with patch.object(agent.time, "time", return_value=1120.0):
                with patch.object(agent, "save_planning_state"):
                    agent.apply_approval_result_to_state(state, approval_result)

        self.assertEqual(state["current_subtask_index"], 3)
        self.assertEqual(len(state["subtasks"]), 4)
        self.assertEqual(state["representative_core_settle_until_unix"], 1150.0)
        self.assertIn("remaining 150 seconds", state["subtasks"][3])
        continuation = agent._monitoring_window_spec(
            state,
            "subtask-004",
            state["subtasks"][3],
            now=1150.0,
        )
        self.assertIsNotNone(continuation)
        self.assertEqual(continuation["evaluation_request"]["schedule"]["duration_seconds"], 150)
        self.assertEqual(
            [node["id"] for node in continuation["nodes"]],
            ["monitoring-agent", "monitor-done"],
        )
        self.assertEqual(state["overall_status"], "running")

    def test_representative_core_settle_wait_sleeps_once_and_clears_deadline(self) -> None:
        state = {
            "run_id": "planning-core-settle-test",
            "representative_core_settle_until_unix": 1030.0,
        }

        with patch.object(agent.time, "time", return_value=1000.0):
            with patch.object(agent.time, "sleep") as sleep:
                with patch.object(agent, "save_planning_state") as save:
                    agent._wait_for_representative_core_settle(state)

        sleep.assert_called_once_with(30.0)
        self.assertNotIn("representative_core_settle_until_unix", state)
        save.assert_called_once_with(state)

    def test_monitoring_only_mismatch_schedules_one_recovery(self) -> None:
        expectation = expected_threshold_request()
        graph = {
            "nodes": [
                runtime_agent_node("monitoring-agent", "Monitoring Agent"),
                {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [{"source": "monitoring-agent", "target": "done"}],
            "entrypoint": "monitoring-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "monitoring-agent",
            "evaluation_request": expectation,
        }
        plan = {
            "subtask_id": "subtask-001",
            "subtask": "Verify AMF remains at 2",
            "langgraph_spec": graph,
            "representative_agent": graph["nodes"][0],
            "attempts": [{"attempt_number": 1, "status": "failed"}],
            "status": "failed",
            "result": {},
            "knowledge_context_used": True,
        }
        observed = failed_threshold_evaluation()
        feedback = {
            "event_id": "monitoring-inline-recovery-test",
            "source": "monitoring-agent",
            "run_id": "planning-inline-recovery-test",
            "subtask_id": plan["subtask_id"],
            "plan_fingerprint": agent._plan_fingerprint(plan),
            "status": "replan_required",
            "reason": "expected_network_state_not_observed",
            "expected": expectation,
            "observed": observed,
            "previous_action": None,
            "probe_errors": [],
        }
        monitoring_report = {
            "run_id": feedback["run_id"],
            "subtask_id": feedback["subtask_id"],
            "status": "replan_required",
            "expected": expectation,
            "execution": {},
            "probe_report": {
                "status": "completed",
                "evaluation_request": expectation,
                "evaluation_result": observed,
            },
            "feedback": feedback,
            "planner_response": None,
            "errors": [],
            "source": {"probe": "probe-agent"},
        }
        result = {
            "status": "replan_required",
            "runtime": "monitoring-agent",
            "monitoring_agent_report": monitoring_report,
        }
        plan["result"] = result
        state = {
            "run_id": feedback["run_id"],
            "intent": "Keep AMF at 2",
            "subtasks": [plan["subtask"], "Report the final AMF state"],
            "current_subtask_index": 0,
            "subtask_plans": [plan],
            "pending_approval": None,
            "overall_status": "running",
        }

        mismatched_observed = {
            **observed,
            "checks": [{**observed["checks"][0], "actual": 0}],
            "query_values": {"opaque-check": 0},
        }
        mismatched_feedback = {**feedback, "observed": mismatched_observed}
        mismatched_result = {
            **result,
            "monitoring_agent_report": {**monitoring_report, "feedback": mismatched_feedback},
        }
        self.assertFalse(agent._consume_returned_monitoring_feedback(state, plan, mismatched_result))
        self.assertNotIn("monitoring_feedback_ids", state)

        for key, value in (("run_id", "wrong-run"), ("subtask_id", "subtask-999")):
            mismatched_feedback = {**feedback, key: value}
            mismatched_result = {
                **result,
                "monitoring_agent_report": {**monitoring_report, "feedback": mismatched_feedback},
            }
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError, f"{key} does not match"):
                    agent._consume_returned_monitoring_feedback(state, plan, mismatched_result)
                self.assertNotIn("monitoring_feedback_ids", state)

        self.assertTrue(agent._consume_returned_monitoring_feedback(state, plan, result))
        self.assertFalse(agent._consume_returned_monitoring_feedback(state, plan, result))
        self.assertEqual(len(state["subtasks"]), 3)
        self.assertIn("Restore the expected network outcome", state["subtasks"][1])
        self.assertEqual(state["subtasks"][2], "Report the final AMF state")
        self.assertEqual(state["current_subtask_index"], 1)
        self.assertNotIn("monitoring_replan_count", state)
        self.assertEqual(plan["status"], "completed")
        self.assertEqual(plan["result"]["handoff_status"], "recovery_scheduled")
        self.assertIs(plan["result"]["monitoring_agent_report"], result["monitoring_agent_report"])

        state["subtask_plans"].append({"status": "completed"})
        state["current_subtask_index"] = 2
        agent._set_state_status(state)
        self.assertEqual(state["overall_status"], "running")
        self.assertEqual(state["subtasks"][state["current_subtask_index"]], "Report the final AMF state")

        state["subtask_plans"].append({"status": "completed"})
        state["current_subtask_index"] = 3
        agent._set_state_status(state)
        self.assertEqual(state["overall_status"], "completed")

    def test_inline_core_graph_does_not_schedule_feedback_recovery(self) -> None:
        expectation = expected_threshold_request()
        graph = {
            "nodes": [
                runtime_agent_node("monitoring-agent", "Monitoring Agent"),
                runtime_agent_node("representative-core-agent", "Representative Core Agent"),
                {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "evaluation_request": expectation,
        }
        plan = {
            "subtask_id": "subtask-001",
            "subtask": "Verify and conditionally recover AMF",
            "langgraph_spec": graph,
            "status": "failed",
            "result": {},
        }
        feedback = {
            "event_id": "monitoring-inline-core-test",
            "source": "monitoring-agent",
            "run_id": "planning-inline-core-test",
            "subtask_id": plan["subtask_id"],
            "plan_fingerprint": agent._plan_fingerprint(plan),
            "status": "replan_required",
            "reason": "expected_network_state_not_observed",
            "expected": expectation,
            "observed": failed_threshold_evaluation(),
            "previous_action": None,
            "probe_errors": [],
        }
        result = {"node_results": [{"result": {"monitoring_agent_report": {"feedback": feedback}}}]}
        plan["result"] = result
        state = {
            "run_id": feedback["run_id"],
            "subtasks": [plan["subtask"]],
            "subtask_plans": [plan],
            "pending_approval": None,
            "overall_status": "running",
        }

        self.assertFalse(agent._consume_returned_monitoring_feedback(state, plan, result))
        self.assertEqual(state["subtasks"], [plan["subtask"]])
        self.assertNotIn("monitoring_replan_count", state)
        self.assertNotIn("monitoring_feedback_ids", state)

    def test_monitoring_only_graph_requests_response_delivery(self) -> None:
        expectation = expected_threshold_request()
        node = runtime_resolved_node("monitoring-agent", "Monitoring Agent")
        graph = {
            "nodes": [node, {"id": "done", "type": "agent", "role": "completion_checker"}],
            "evaluation_request": expectation,
        }
        report = {
            "run_id": "planning-response-delivery-test",
            "subtask_id": "subtask-001",
            "status": "healthy",
            "expected": expectation,
            "execution": {},
            "probe_report": None,
            "feedback": None,
            "planner_response": None,
            "errors": [],
            "source": {"probe": "probe-agent"},
        }

        with patch.object(subgraph_runtime, "invoke_monitoring_agent", return_value=report) as monitoring:
            subgraph_runtime._invoke_monitoring_node(
                original_intent="Keep AMF at 2",
                subtask_id="subtask-001",
                subtask="Verify AMF remains at 2",
                langgraph_spec=graph,
                node=node,
                previous_results=[],
                planning_report=None,
                run_id=report["run_id"],
                call_id="call-response-delivery-test",
            )

        self.assertTrue(monitoring.call_args.kwargs["notify_planner"])
        self.assertEqual(monitoring.call_args.kwargs["notification_delivery"], "response")

    def test_monitoring_response_rejects_wrong_run_or_subtask(self) -> None:
        expectation = expected_threshold_request()
        node = runtime_resolved_node("monitoring-agent", "Monitoring Agent")
        graph = {"nodes": [node], "evaluation_request": expectation}
        report = {
            "run_id": "planning-correlation-test",
            "subtask_id": "subtask-001",
            "status": "healthy",
            "expected": expectation,
            "execution": {},
            "probe_report": None,
            "feedback": None,
            "planner_response": None,
            "errors": [],
            "source": {"probe": "probe-agent"},
        }

        for key, value in (("run_id", "wrong-run"), ("subtask_id", "subtask-999")):
            with self.subTest(key=key):
                with patch.object(subgraph_runtime, "invoke_monitoring_agent", return_value={**report, key: value}):
                    result = subgraph_runtime._invoke_monitoring_node(
                        original_intent="Keep AMF at 2",
                        subtask_id="subtask-001",
                        subtask="Verify AMF remains at 2",
                        langgraph_spec=graph,
                        node=node,
                        previous_results=[],
                        planning_report=None,
                        run_id=report["run_id"],
                        call_id="call-correlation-test",
                    )
                self.assertEqual(result["status"], "blocked")
                self.assertIn(f"{key} does not match", result["message"])

    def test_monitoring_feedback_ignores_legacy_run_replan_count(self) -> None:
        expectation = expected_threshold_request()
        plan = {
            "subtask_id": "subtask-001",
            "subtask": "Observe AMF",
            "langgraph_spec": {"evaluation_request": expectation},
            "representative_agent": {"id": "monitoring-agent", "name": "Monitoring Agent"},
            "attempts": [{"attempt_number": 1}],
            "status": "failed",
            "result": {"status": "replan_required"},
            "knowledge_context_used": True,
        }
        state = {
            "run_id": "planning-feedback-limit-test",
            "intent": "Observe AMF",
            "subtasks": [plan["subtask"]],
            "current_subtask_index": 0,
            "subtask_plans": [plan],
            "pending_approval": None,
            "overall_status": "running",
            "monitoring_replan_count": 2,
        }
        feedback = {
            "event_id": "monitoring-feedback-limit-test",
            "source": "monitoring-agent",
            "run_id": state["run_id"],
            "subtask_id": "subtask-001",
            "plan_fingerprint": agent._plan_fingerprint(plan),
            "status": "replan_required",
            "reason": "expected_network_state_not_observed",
            "expected": expectation,
            "observed": failed_threshold_evaluation(),
            "previous_action": None,
        }

        with patch.dict(os.environ, {"PLANNING_MAX_ATTEMPTS": "2"}):
            self.assertTrue(agent._apply_monitoring_feedback_to_state(state, feedback))

        self.assertNotIn("monitoring_replan_count", state)
        self.assertEqual(state["current_subtask_index"], 1)

    def test_monitoring_feedback_rejects_stale_plan_snapshot(self) -> None:
        expectation = expected_threshold_request()
        plan = {
            "subtask_id": "subtask-001",
            "subtask": "Observe AMF",
            "langgraph_spec": {"evaluation_request": expectation},
            "representative_agent": {"id": "probe-agent", "name": "Probe Agent"},
            "attempts": [{"attempt_number": 1}],
            "status": "completed",
            "result": {"status": "completed"},
            "knowledge_context_used": True,
        }
        feedback = {
            "event_id": "monitoring-feedback-stale-test",
            "source": "monitoring-agent",
            "run_id": "planning-feedback-stale-test",
            "subtask_id": "subtask-001",
            "plan_fingerprint": agent._plan_fingerprint(plan),
            "status": "replan_required",
            "reason": "expected_network_state_not_observed",
            "expected": expectation,
            "observed": failed_threshold_evaluation(),
            "previous_action": None,
        }
        plan["attempts"].append({"attempt_number": 2})
        plan["status"] = "failed"
        plan["result"] = {"status": "replan_required"}
        self.assertEqual(agent._plan_fingerprint(plan), feedback["plan_fingerprint"])
        plan["langgraph_spec"] = {**plan["langgraph_spec"], "revision": 2}
        state = {
            "run_id": feedback["run_id"],
            "intent": "Observe AMF",
            "subtasks": ["Observe AMF"],
            "current_subtask_index": 1,
            "subtask_plans": [plan],
            "pending_approval": None,
            "overall_status": "completed",
        }

        with patch.object(agent, "load_planning_state", return_value=state):
            with self.assertRaisesRegex(ValueError, "stale"):
                agent.apply_monitoring_feedback(feedback)

    def test_monitoring_recovery_allows_repeating_successful_scale(self) -> None:
        expectation = expected_threshold_request()
        action = {
            "operation": "scale",
            "resource": "deployment",
            "namespace": "free5gc-v4",
            "target_scope": "single_nf",
            "nfs": ["AMF"],
            "parameters": {"replicas": 2},
            "rationale": "same action with different prose",
        }
        plan = {
            "subtask_id": "subtask-001",
            "subtask": "Scale AMF to 2 replicas",
            "langgraph_spec": {"evaluation_request": expectation},
            "status": "completed",
            "result": {"representative_core_report": {"command_proposal": action}},
        }
        state = {
            "run_id": "planning-repeat-scale-test",
            "subtasks": [plan["subtask"]],
            "current_subtask_index": 1,
            "subtask_plans": [plan],
            "pending_approval": None,
            "overall_status": "completed",
        }
        feedback = {
            "event_id": "monitoring-repeat-scale-test",
            "source": "monitoring-agent",
            "run_id": state["run_id"],
            "subtask_id": plan["subtask_id"],
            "plan_fingerprint": agent._plan_fingerprint(plan),
            "status": "replan_required",
            "reason": "expected_network_state_not_observed",
            "expected": expectation,
            "observed": failed_threshold_evaluation(),
            "previous_action": action,
            "probe_errors": [],
        }

        scheduled = agent._apply_monitoring_feedback_to_state(state, feedback)

        self.assertTrue(scheduled)
        self.assertNotIn("Do not repeat", state["subtasks"][1])
        self.assertNotIn("different registered", state["subtasks"][1])
        self.assertNotIn("monitoring_failed_action_signatures", state)
        prompt = build_llm_prompt(
            intent=state["run_id"],
            golden_goal_context_used=False,
            subtask_id="subtask-002",
            subtask=state["subtasks"][1],
            knowledge=knowledge_with_candidates(),
            completed_actions=[action],
        )
        self.assertIn("confirmed post-action drift is a new recovery need", prompt)
        self.assertIn("including restoration after confirmed drift", prompt)

    def test_decomposition_input_validation_accepts_minimal_contract(self) -> None:
        validated = validate_decomposition_input(sample_decomposition())

        self.assertEqual(validated.intent, "Launch a new mobile app")
        self.assertEqual(validated.subtasks, ["Define launch goals", "Prepare app store release"])
        self.assertFalse(validated.golden_goal_context_used)

    def test_decomposition_input_rejects_empty_subtasks(self) -> None:
        payload = {
            "intent": "Launch a new mobile app",
            "subtasks": [],
            "golden_goal_context_used": False,
        }

        with self.assertRaises(ValueError):
            validate_decomposition_input(payload)

    def test_unregistered_subtask_uses_llm_then_blocks_unregistered_placeholder(self) -> None:
        with patch.object(agent, "find_knowledge_for_subtask", return_value=knowledge_with_candidates()):
            with patch.object(agent, "build_langgraph_spec_with_llm", side_effect=fake_llm_spec) as planner:
                report = agent.plan_from_decomposition(sample_decomposition()).to_dict()

        planner.assert_called()
        validate_planning_report(report)
        self.assertEqual(report["overall_status"], "blocked")
        first_plan = report["subtask_plans"][0]
        self.assertEqual(first_plan["status"], "blocked")
        self.assertTrue(first_plan["knowledge_context_used"])
        self.assertEqual(first_plan["representative_agent"]["id"], "planning-agent")
        self.assertEqual(first_plan["representative_agent"]["source"], "local_runtime_registry")
        self.assertNotIn("agent-placeholder-subtask-001", [node["id"] for node in first_plan["langgraph_spec"]["nodes"]])
        self.assertEqual(first_plan["attempts"][0]["strategy"], "llm_langgraph_spec_with_runtime_adapter")

    def test_selected_agent_without_mongodb_contract_is_blocked(self) -> None:
        knowledge = knowledge_with_candidates()
        agents = [dict(document) for document in knowledge.agents]
        next(document for document in agents if document["agent_id"] == "probe-agent").pop("communication_contract")
        missing_contract = agent.KnowledgeRegistryResult(True, agents, knowledge.tools)
        graph = fake_llm_spec(subtask_id="subtask-001", subtask="Monitor AMF health")

        _, reason = agent.build_registered_subgraph(
            intent="Monitor AMF health",
            subtask_id="subtask-001",
            subtask="Monitor AMF health",
            knowledge=missing_contract,
            llm_langgraph_spec=graph,
        )

        self.assertEqual(reason, "knowledge_agent_communication_contract_missing")

    def test_selected_agent_with_unsupported_mongodb_schema_is_blocked(self) -> None:
        knowledge = knowledge_with_candidates()
        probe = next(document for document in knowledge.agents if document["agent_id"] == "probe-agent")
        probe["communication_contract"] = {
            **probe["communication_contract"],
            "planning": {"evaluation_request_schema": {"type": "object", "min_items": 1}},
        }
        graph = fake_llm_spec(subtask_id="subtask-001", subtask="Monitor AMF health")

        _, reason = agent.build_registered_subgraph(
            intent="Monitor AMF health",
            subtask_id="subtask-001",
            subtask="Monitor AMF health",
            knowledge=knowledge,
            llm_langgraph_spec=graph,
        )

        self.assertEqual(reason, "knowledge_agent_communication_contract_invalid")

    def test_mongodb_agent_identity_cannot_be_completion_checker(self) -> None:
        knowledge = knowledge_with_candidates()
        monitoring = next(document for document in knowledge.agents if document["agent_id"] == "monitoring-agent")
        monitoring["agent_id"] = "future-monitor-agent"
        monitoring["name"] = "Future Monitor Agent"
        graph = {
            "version": "langgraph-style-v1",
            "subtask_id": "subtask-001",
            "subtask": "Scale then monitor",
            "nodes": [
                {"id": "representative-core-agent", "type": "agent", "name": "Representative Core Agent"},
                {"id": "future-monitor-agent", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [{"source": "representative-core-agent", "target": "future-monitor-agent"}],
            "entrypoint": "representative-core-agent",
            "terminal_nodes": ["future-monitor-agent"],
            "representative_agent": "representative-core-agent",
            "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
        }

        _, reason = agent.build_registered_subgraph(
            intent="Scale then monitor",
            subtask_id="subtask-001",
            subtask="Scale then monitor",
            knowledge=knowledge,
            llm_langgraph_spec=graph,
        )

        self.assertEqual(reason, "invalid_subgraph_topology:registered_agent_cannot_be_completion_checker")

    def test_monitoring_request_missing_contract_schedule_is_blocked(self) -> None:
        graph = {
            "version": "langgraph-style-v1",
            "subtask_id": "subtask-001",
            "subtask": "Keep AMF stable",
            "nodes": [
                {"id": "monitoring-agent", "type": "agent", "name": "Monitoring Agent"},
                {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [{"source": "monitoring-agent", "target": "done"}],
            "entrypoint": "monitoring-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "monitoring-agent",
            "evaluation_request": {
                "type": "thresholds",
                "checks": [{"check_id": "opaque", "observation": "AMF pods", "expected": {"operator": "==", "value": 2}}],
            },
        }

        _, reason = agent.build_registered_subgraph(
            intent="Keep AMF stable",
            subtask_id="subtask-001",
            subtask="Keep AMF stable",
            knowledge=knowledge_with_candidates(),
            llm_langgraph_spec=graph,
        )

        self.assertEqual(reason, "knowledge_agent_communication_contract_violation")

    def test_monitoring_request_duplicate_check_ids_is_blocked(self) -> None:
        check = {"check_id": "duplicate", "observation": "AMF pods", "expected": {"operator": "==", "value": 2}}
        graph = {
            "version": "langgraph-style-v1",
            "subtask_id": "subtask-001",
            "subtask": "Keep AMF stable",
            "nodes": [
                {"id": "monitoring-agent", "type": "agent", "name": "Monitoring Agent"},
                {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [{"source": "monitoring-agent", "target": "done"}],
            "entrypoint": "monitoring-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "monitoring-agent",
            "evaluation_request": {
                "type": "thresholds",
                "schedule": {"duration_seconds": 300, "interval_seconds": 15},
                "checks": [check, dict(check)],
            },
        }

        _, reason = agent.build_registered_subgraph(
            intent="Keep AMF stable",
            subtask_id="subtask-001",
            subtask="Keep AMF stable",
            knowledge=knowledge_with_candidates(),
            llm_langgraph_spec=graph,
        )

        self.assertEqual(reason, "knowledge_agent_communication_contract_violation")

    def test_candidate_compaction_cannot_drop_monitoring_core_gate(self) -> None:
        graph = {
            "version": "langgraph-style-v1",
            "subtask_id": "subtask-001",
            "subtask": "Keep AMF stable",
            "nodes": [
                {"id": "monitoring-agent", "type": "agent", "name": "Monitoring Agent"},
                {"id": "unregistered-display-tool", "type": "tool", "name": "Unregistered Display Tool"},
                {"id": "representative-core-agent", "type": "agent", "name": "Representative Core Agent"},
                {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [
                {"source": "monitoring-agent", "target": "unregistered-display-tool"},
                {"source": "unregistered-display-tool", "target": "representative-core-agent", "condition": "only_if_mismatch"},
                {"source": "representative-core-agent", "target": "done"},
            ],
            "entrypoint": "monitoring-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "monitoring-agent",
            "evaluation_request": {
                "type": "thresholds",
                "schedule": {"duration_seconds": 300, "interval_seconds": 30},
                "checks": [{"check_id": "opaque", "observation": "AMF pods", "expected": {"operator": "==", "value": 2}}],
            },
        }

        _, reason = agent.build_registered_subgraph(
            intent="Keep AMF at 2",
            subtask_id="subtask-001",
            subtask="Keep AMF stable",
            knowledge=knowledge_with_candidates(),
            llm_langgraph_spec=graph,
        )

        self.assertEqual(reason, "invalid_subgraph_topology:monitoring_core_edge_requires_only_if_mismatch")

    def test_fuzzy_knowledge_candidates_can_match_related_subtasks(self) -> None:
        documents = [
            {
                "agent_id": "nf-health-monitor",
                "name": "NF Health Monitor",
                "description": "Collects network function health and performance metrics.",
                "capabilities": ["network-monitoring", "metric-collection"],
            },
            {
                "agent_id": "billing-agent",
                "name": "Billing Agent",
                "description": "Handles invoices and customer payments.",
                "capabilities": ["billing"],
            },
        ]

        matches = _annotate_and_filter(
            documents,
            "Collect health metrics for each network NF",
            limit=5,
            threshold=0.18,
        )

        self.assertEqual(matches[0]["agent_id"], "nf-health-monitor")
        self.assertGreaterEqual(matches[0]["_match_score"], 0.18)
        self.assertNotEqual(matches[0]["agent_id"], "billing-agent")

    def test_aliases_are_used_for_knowledge_matching(self) -> None:
        documents = [
            {
                "agent_id": "ran-agent",
                "name": "RAN Agent",
                "description": "Radio access network representative.",
                "aliases": ["radio", "gnb", "cell"],
            }
        ]

        matches = _annotate_and_filter(documents, "Tune gNB cell", limit=5, threshold=0.18)

        self.assertEqual(matches[0]["agent_id"], "ran-agent")
        self.assertIn("gnb", matches[0]["_matched_keywords"])

    def test_query_keywords_do_not_expand_domain_specific_terms(self) -> None:
        self.assertEqual(_query_keywords("Monitor AMF CPU"), ["monitor", "amf", "cpu"])

    def test_active_catalog_is_used_as_semantic_fallback(self) -> None:
        collection = MagicMock()
        collection.find.return_value.limit.return_value = [
            {"agent_id": "probe-agent", "status": "active"}
        ]

        candidates = _find_active_catalog(collection, 10)

        self.assertEqual(candidates[0]["agent_id"], "probe-agent")
        self.assertEqual(candidates[0]["_match_strategy"], "active_catalog_fallback")
        collection.find.return_value.limit.assert_called_once_with(10)

    def test_planning_prompt_uses_knowledge_db_candidates_only(self) -> None:
        knowledge = knowledge_with_candidates()
        knowledge.agents[0]["runtime_details"] = {"large": "must not reach the prompt"}
        knowledge.agents[0]["secret"] = "must not reach the prompt"
        prompt = build_llm_prompt(
            intent="Monitor AMF and SMF CPU",
            golden_goal_context_used=False,
            subtask_id="subtask-001",
            subtask="Identify the AMF and SMF components to monitor",
            knowledge=knowledge,
            completed_actions=[],
        )

        self.assertIn("Knowledge DB candidate context", prompt)
        self.assertIn("Use only Knowledge DB candidate agents/tools", prompt)
        self.assertIn("probe-agent", prompt)
        self.assertIn("Monitoring is selective", prompt)
        self.assertIn("contract_ref", prompt)
        self.assertNotIn("communication_contract", prompt)
        self.assertNotIn('"request_schema": {', prompt)
        self.assertNotIn('"response_schema": {', prompt)
        self.assertNotIn("json-schema-subset-v1", prompt)
        self.assertNotIn("runtime_details", prompt)
        self.assertNotIn("AMF_desired_replicas", prompt)
        self.assertIn("Never use probe-agent -> representative-core-agent", prompt)
        self.assertNotIn("_match_score", prompt)
        self.assertNotIn("Registered local runtime context", prompt)

        candidate_json = prompt.split("Knowledge DB candidate context:\n", 1)[1].split(
            "\n\nPreviously completed subtask actions:", 1
        )[0]
        self.assertLess(len(candidate_json.encode("utf-8")), 12000)
        self.assertLess(len(prompt.encode("utf-8")), 20000)
        candidate_context = json.loads(candidate_json)
        for candidate in candidate_context["candidate_agents"]:
            self.assertEqual(
                set(candidate),
                {"agent_id", "id", "type", "name", "planner_manifest", "contract_ref"},
            )
            self.assertEqual(set(candidate["planner_manifest"]), set(llm_subgraph_planner.PLANNER_MANIFEST_FIELDS))
        probe = next(item for item in candidate_context["candidate_agents"] if item["agent_id"] == "probe-agent")
        self.assertEqual(
            probe["contract_ref"],
            communication_contract_ref("probe-agent", knowledge.agents[0]["communication_contract"]),
        )
        self.assertNotIn("description", probe)
        self.assertNotIn("capabilities", probe)
        self.assertNotIn("secret", probe["planner_manifest"])
        self.assertNotIn("contract_ref", probe["planner_manifest"])

    def test_canonicalized_graph_uses_authoritative_contract_ref_only(self) -> None:
        knowledge = knowledge_with_candidates()
        graph = fake_llm_spec(subtask_id="subtask-001", subtask="Monitor AMF health")
        graph["nodes"][0]["communication_contract"] = {"version": "forged"}
        graph["nodes"][0]["contract_ref"] = {"version": "forged", "hash": "sha256:forged"}
        graph["nodes"][0]["planner_manifest"] = {"request_schema": {"type": "object"}}
        graph["nodes"][0]["request_schema"] = {"type": "object"}
        graph["nodes"][0]["response_schema"] = {"type": "object"}
        graph["nodes"][0]["_runtime_communication_contract"] = {"secret": True}

        canonical = canonicalize_knowledge_agent_identities(graph, knowledge)

        probe = next(node for node in canonical["nodes"] if node["id"] == "probe-agent")
        mongodb_contract = next(
            document["communication_contract"]
            for document in knowledge.agents
            if document["agent_id"] == "probe-agent"
        )
        self.assertNotIn("communication_contract", probe)
        self.assertEqual(probe["contract_ref"], communication_contract_ref("probe-agent", mongodb_contract))
        for forbidden in ("planner_manifest", "request_schema", "response_schema", "_runtime_communication_contract"):
            self.assertNotIn(forbidden, probe)

        registered, reason = agent.build_registered_subgraph(
            intent="Monitor AMF health",
            subtask_id="subtask-001",
            subtask="Monitor AMF health",
            knowledge=knowledge,
            llm_langgraph_spec=graph,
        )
        self.assertIsNone(reason)
        registered_probe = next(node for node in registered["nodes"] if node["id"] == "probe-agent")
        self.assertEqual(
            set(registered_probe),
            {"id", "agent_id", "type", "role", "name", "source", "capabilities", "contract_ref"},
        )

    def test_invalid_manifest_agent_is_excluded_from_prompt(self) -> None:
        knowledge = knowledge_with_candidates()
        knowledge.agents[0]["planner_manifest"] = {"contract_version": "stale"}

        prompt = build_llm_prompt(
            intent="Monitor AMF",
            golden_goal_context_used=False,
            subtask_id="subtask-001",
            subtask="Monitor AMF",
            knowledge=knowledge,
        )
        context = json.loads(
            prompt.split("Knowledge DB candidate context:\n", 1)[1].split(
                "\n\nPreviously completed subtask actions:", 1
            )[0]
        )

        self.assertNotIn("probe-agent", [item["agent_id"] for item in context["candidate_agents"]])
        self.assertNotIn("excluded_agent_ids", context)

    def test_nested_schema_manifest_is_excluded_and_cannot_be_selected(self) -> None:
        for forbidden in ("properties", "oneOf", "request_schema"):
            with self.subTest(forbidden=forbidden):
                knowledge = knowledge_with_candidates()
                knowledge.agents[0]["planner_manifest"]["planning_contract"]["nested"] = {
                    forbidden: {"secret": True}
                }
                prompt = build_llm_prompt(
                    intent="Monitor AMF",
                    golden_goal_context_used=False,
                    subtask_id="subtask-001",
                    subtask="Monitor AMF",
                    knowledge=knowledge,
                )
                context = json.loads(
                    prompt.split("Knowledge DB candidate context:\n", 1)[1].split(
                        "\n\nPreviously completed subtask actions:", 1
                    )[0]
                )
                self.assertNotIn("probe-agent", [item["agent_id"] for item in context["candidate_agents"]])
                _, reason = agent.build_registered_subgraph(
                    intent="Monitor AMF",
                    subtask_id="subtask-001",
                    subtask="Monitor AMF",
                    knowledge=knowledge,
                    llm_langgraph_spec=fake_llm_spec(subtask_id="subtask-001", subtask="Monitor AMF"),
                )
                self.assertEqual(reason, "knowledge_agent_planner_manifest_invalid")

    def test_live_contract_resolution_requires_exact_ref(self) -> None:
        contract = communication_contract(REPORT_VALUES_SCHEMA)
        reference = communication_contract_ref("probe-agent", contract)

        with patch("tools.knowledge_registry.find_agent_communication_contract", return_value=contract):
            self.assertEqual(resolve_agent_communication_contract("probe-agent", reference), contract)
            with self.assertRaisesRegex(ValueError, "ref_mismatch"):
                resolve_agent_communication_contract("probe-agent", {**reference, "version": "stale"})

    def test_llm_subgraph_repairs_invalid_response_format_once(self) -> None:
        class FakeMessage:
            def __init__(self, content: str) -> None:
                self.content = content

        class FakeAgent:
            def __init__(self) -> None:
                self.requests: list[dict[str, object]] = []

            def invoke(self, payload: dict[str, object]) -> dict[str, object]:
                self.requests.append(payload)
                graph = fake_llm_spec(subtask_id="subtask-001", subtask="Scale AMF deployment")
                content = {"graph": graph} if len(self.requests) == 1 else {"langgraph_spec": graph}
                return {"messages": [*payload["messages"], FakeMessage(json.dumps(content))]}

        fake_agent = FakeAgent()
        with patch.object(llm_subgraph_planner, "create_planning_llm_agent", return_value=fake_agent):
            spec = llm_subgraph_planner.build_langgraph_spec_with_llm(
                intent="Scale AMF deployment to 2 replicas",
                golden_goal_context_used=False,
                subtask_id="subtask-001",
                subtask="Scale AMF deployment",
                knowledge=knowledge_with_candidates(),
            )

        self.assertEqual(spec["representative_agent"], "representative-core-agent")
        self.assertEqual(len(fake_agent.requests), 2)
        self.assertIn("Fix only the response contract error", fake_agent.requests[1]["messages"][-1]["content"])

    def test_llm_subgraph_repairs_mongodb_contract_violation_once(self) -> None:
        class FakeMessage:
            def __init__(self, content: str) -> None:
                self.content = content

        class FakeAgent:
            def __init__(self) -> None:
                self.requests: list[dict[str, object]] = []

            def invoke(self, payload: dict[str, object]) -> dict[str, object]:
                self.requests.append(payload)
                graph = fake_llm_spec(subtask_id="subtask-001", subtask="Monitor AMF health")
                if len(self.requests) == 1:
                    graph["evaluation_request"] = {
                        "type": "thresholds",
                        "checks": [{"query_label": "AMF_ready", "operator": "==", "value": 2}],
                    }
                return {"messages": [*payload["messages"], FakeMessage(json.dumps({"langgraph_spec": graph}))]}

        fake_agent = FakeAgent()
        with patch.object(llm_subgraph_planner, "create_planning_llm_agent", return_value=fake_agent):
            spec = llm_subgraph_planner.build_langgraph_spec_with_llm(
                intent="Monitor AMF health",
                golden_goal_context_used=False,
                subtask_id="subtask-001",
                subtask="Monitor AMF health",
                knowledge=knowledge_with_candidates(),
            )

        self.assertEqual(spec["evaluation_request"], {"type": "report_values", "completion_rule": "data_returned"})
        self.assertEqual(len(fake_agent.requests), 2)
        repair_prompt = fake_agent.requests[1]["messages"][-1]["content"]
        self.assertIn("MongoDB communication_contract", repair_prompt)

    def test_normalize_langgraph_spec_accepts_object_node_references(self) -> None:
        spec = normalize_langgraph_spec(
            {
                "nodes": [
                    {"id": {"id": "probe-agent"}, "type": "agent", "name": "Probe Agent"},
                    {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
                ],
                "edges": [{"source": {"id": "probe-agent"}, "target": {"id": "done"}}],
                "entrypoint": {"id": "probe-agent"},
                "terminal_nodes": [{"id": "done"}],
                "representative_agent": {"agent_id": "probe-agent"},
                "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
            },
            "subtask-001",
            "Monitor AMF",
            True,
        )

        self.assertEqual(spec["entrypoint"], "probe-agent")
        self.assertEqual(spec["representative_agent"], "probe-agent")
        self.assertEqual(spec["terminal_nodes"], ["done"])

    def test_normalize_langgraph_spec_canonicalizes_monitoring_core_gate(self) -> None:
        spec = normalize_langgraph_spec(
            {
                "nodes": [
                    {"id": "monitoring-agent", "type": "agent", "name": "Monitoring Agent"},
                    {"id": "core-kubectl-command-tool", "type": "tool", "name": "Core Tool"},
                    {"id": "representative-core-agent", "type": "agent", "name": "Representative Core Agent"},
                    {"id": "done", "type": "terminal", "terminal": True},
                ],
                "edges": [
                    {"from": "monitoring-agent", "to": "core-kubectl-command-tool"},
                    {"from": "core-kubectl-command-tool", "to": "representative-core-agent", "condition": "when the observed state differs"},
                    {"from": "monitoring-agent", "to": "done", "condition": "healthy"},
                    {"source": "representative-core-agent", "target": "done"},
                ],
                "entrypoint": "monitoring-agent",
                "terminal_nodes": ["done"],
                "representative_agent": "monitoring-agent",
                "evaluation_request": {
                    "type": "thresholds",
                    "schedule": {"duration_seconds": 300, "interval_seconds": 30},
                    "checks": [{"check_id": "opaque", "observation": "AMF pods", "expected": {"operator": "==", "value": 2}}],
                },
            },
            "subtask-001",
            "Keep AMF stable",
            True,
        )

        self.assertEqual(spec["edges"][1]["condition"], "only_if_mismatch")
        completion = next(node for node in spec["nodes"] if node["id"] == "done")
        self.assertEqual(completion["role"], "completion_checker")
        subgraph_runtime.validate_subgraph_execution_topology(spec)

    def test_terminal_flag_cannot_hide_registered_monitoring_agent(self) -> None:
        spec = normalize_langgraph_spec(
            {
                "nodes": [
                    {"id": "representative-core-agent", "type": "agent", "name": "Representative Core Agent"},
                    {"id": "monitoring-agent", "type": "agent", "name": "Monitoring Agent", "terminal": True},
                ],
                "edges": [{"source": "representative-core-agent", "target": "monitoring-agent"}],
                "entrypoint": "representative-core-agent",
                "terminal_nodes": ["monitoring-agent"],
                "representative_agent": "representative-core-agent",
                "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
            },
            "subtask-001",
            "Scale then monitor",
            True,
        )

        monitoring = next(node for node in spec["nodes"] if node["id"] == "monitoring-agent")
        self.assertNotEqual(monitoring.get("role"), "completion_checker")
        with self.assertRaisesRegex(ValueError, "runtime_agent_node_contains_unsupported_fields"):
            subgraph_runtime.validate_subgraph_execution_topology(spec)

    def test_registered_agent_identity_cannot_spoof_completion_checker(self) -> None:
        spec = normalize_langgraph_spec(
            {
                "nodes": [
                    {"id": "representative-core-agent", "type": "agent", "name": "Representative Core Agent"},
                    {"id": "monitoring-agent", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
                ],
                "edges": [{"source": "representative-core-agent", "target": "monitoring-agent"}],
                "entrypoint": "representative-core-agent",
                "terminal_nodes": ["monitoring-agent"],
                "representative_agent": "representative-core-agent",
                "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
            },
            "subtask-001",
            "Scale then monitor",
            True,
        )

        with self.assertRaisesRegex(ValueError, "registered_agent_cannot_be_completion_checker"):
            subgraph_runtime.validate_subgraph_execution_topology(spec)

    def test_registered_agent_identity_cannot_spoof_terminal_type(self) -> None:
        spec = normalize_langgraph_spec(
            {
                "nodes": [
                    {"id": "representative-core-agent", "type": "agent", "name": "Representative Core Agent"},
                    {"id": "monitoring-agent", "type": "terminal"},
                ],
                "edges": [{"source": "representative-core-agent", "target": "monitoring-agent"}],
                "entrypoint": "representative-core-agent",
                "terminal_nodes": ["monitoring-agent"],
                "representative_agent": "representative-core-agent",
                "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
            },
            "subtask-001",
            "Scale then monitor",
            True,
        )

        with self.assertRaisesRegex(ValueError, "registered_agent_cannot_be_completion_checker"):
            subgraph_runtime.validate_subgraph_execution_topology(spec)

    def test_normalize_langgraph_spec_preserves_mongodb_agent_contract(self) -> None:
        raw = fake_llm_spec(subtask_id="subtask-003", subtask="Keep XNF42 stable")
        raw["evaluation_request"] = {
            "type": "thresholds",
            "schedule": {"duration_seconds": 7200, "interval_seconds": 600},
            "checks": [
                {"check_id": "dynamic-check", "observation": "XNF42 pod count", "expected": {"operator": "==", "value": 2}},
            ],
            "contract_extension": {"from_mongodb": True},
        }

        spec = normalize_langgraph_spec(raw, "subtask-003", "Keep XNF42 stable", True)

        evaluation = spec["evaluation_request"]
        self.assertEqual(evaluation, raw["evaluation_request"])

    def test_planning_does_not_invent_monitoring_schedule(self) -> None:
        raw = {
            "nodes": [
                {"id": "monitoring-agent", "type": "agent", "name": "Monitoring Agent"},
                {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [{"source": "monitoring-agent", "target": "done"}],
            "entrypoint": "monitoring-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "monitoring-agent",
            "evaluation_request": {"type": "thresholds", "target": "XNF42", "checks": []},
        }

        spec = normalize_langgraph_spec(raw, "subtask-001", "Verify XNF42", True)

        self.assertNotIn("schedule", spec["evaluation_request"])

    def test_knowledge_db_unavailable_blocks_without_llm(self) -> None:
        with patch.object(
            agent,
            "find_knowledge_for_subtask",
            return_value=agent.KnowledgeRegistryResult(False, [], [], available=False, error="connection refused"),
        ):
            with patch.object(agent, "build_langgraph_spec_with_llm", side_effect=AssertionError("LLM must not run without Knowledge DB")):
                report = agent.plan_from_decomposition(sample_decomposition()).to_dict()

        self.assertEqual(report["overall_status"], "blocked")
        self.assertEqual(report["subtask_plans"][0]["result"]["errors"][0]["code"], "knowledge_db_unavailable")

    def test_empty_knowledge_db_does_not_fallback_to_local_registry(self) -> None:
        monitoring_payload = {
            "intent": "Monitor AMF CPU",
            "subtasks": ["Monitor AMF CPU usage"],
            "golden_goal_context_used": False,
        }
        with patch.object(agent, "find_knowledge_for_subtask", return_value=agent.KnowledgeRegistryResult(False, [], [], available=True)):
            with patch.object(agent, "build_langgraph_spec_with_llm", side_effect=AssertionError("LLM must not run without candidates")):
                report = agent.plan_from_decomposition(monitoring_payload).to_dict()

        self.assertEqual(report["overall_status"], "blocked")
        self.assertEqual(report["subtask_plans"][0]["result"]["errors"][0]["code"], "knowledge_db_no_candidates")

    def test_monitoring_subtask_uses_runtime_adapter_result(self) -> None:
        monitoring_payload = {
            "intent": "Monitor AMF cpu",
            "subtasks": ["Monitor AMF CPU usage"],
            "golden_goal_context_used": False,
        }

        def fake_runtime(**_: object) -> dict[str, object]:
            return {
                "status": "completed",
                "runtime": "probe-agent",
                "monitoring_report": {
                    "intent": "Monitor AMF CPU usage",
                    "command": {
                        "operation": "cpu_average",
                        "metric": "cpu_usage",
                        "scope": "single_nf",
                        "nfs": ["AMF"],
                        "window_seconds": 60,
                    },
                    "scope": "single_nf",
                    "metric": "cpu_usage",
                    "window_seconds": 60,
                    "status": "completed",
                    "results": [{"target": "AMF", "average": 1.0}],
                    "errors": [],
                    "source": {"type": "test"},
                },
            }

        with patch.object(agent, "find_knowledge_for_subtask", return_value=knowledge_with_candidates()):
            with patch.object(agent, "build_langgraph_spec_with_llm", side_effect=fake_llm_spec):
                with patch.object(agent, "execute_subgraph", side_effect=fake_runtime):
                    report = agent.plan_from_decomposition(monitoring_payload).to_dict()

        validate_planning_report(report)
        self.assertEqual(report["overall_status"], "completed")
        first_plan = report["subtask_plans"][0]
        self.assertEqual(first_plan["status"], "completed")
        self.assertEqual(first_plan["result"]["runtime"], "probe-agent")
        self.assertEqual(first_plan["result"]["monitoring_report"]["results"][0]["target"], "AMF")

    def test_multiple_monitoring_subtasks_execute_independently(self) -> None:
        monitoring_payload = {
            "intent": "Monitor AMF and SMF CPU",
            "subtasks": ["Monitor AMF CPU usage", "Monitor SMF CPU usage"],
            "golden_goal_context_used": False,
        }
        calls: list[str] = []

        def fake_runtime(**kwargs: object) -> dict[str, object]:
            subtask = str(kwargs["subtask"])
            calls.append(subtask)
            target = "SMF" if "SMF" in subtask else "AMF"
            return {
                "status": "completed",
                "runtime": "probe-agent",
                "subtask_id": kwargs["subtask_id"],
                "subtask": subtask,
                "representative_agent_id": "agent-placeholder",
                "representative_agent_name": "placeholder",
                "message": "Probe Agent execution completed.",
                "compiled": False,
                "monitoring_report": {
                    "intent": subtask,
                    "command": {
                        "operation": "cpu_average",
                        "metric": "cpu_usage",
                        "scope": "single_nf",
                        "nfs": [target],
                        "window_seconds": 60,
                    },
                    "scope": "single_nf",
                    "metric": "cpu_usage",
                    "window_seconds": 60,
                    "status": "completed",
                    "results": [{"target": target, "average": 1.0}],
                    "errors": [],
                    "source": {"type": "test"},
                },
            }

        with patch.object(agent, "find_knowledge_for_subtask", return_value=knowledge_with_candidates()):
            with patch.object(agent, "build_langgraph_spec_with_llm", side_effect=fake_llm_spec):
                with patch.object(agent, "execute_subgraph", side_effect=fake_runtime):
                    report = agent.plan_from_decomposition(monitoring_payload).to_dict()

        validate_planning_report(report)
        self.assertEqual(calls, ["Monitor AMF CPU usage", "Monitor SMF CPU usage"])
        self.assertEqual(report["overall_status"], "completed")
        targets = [plan["result"]["monitoring_report"]["results"][0]["target"] for plan in report["subtask_plans"]]
        self.assertEqual(targets, ["AMF", "SMF"])
        self.assertNotIn("reused_monitoring_report", report["subtask_plans"][1]["result"])


    def test_registered_subgraph_uses_planning_llm_selected_representative(self) -> None:
        knowledge = knowledge_with_candidates()
        monitoring_spec = fake_llm_spec(
            subtask_id="subtask-001",
            subtask="Run the Probe Agent to monitor AMF after scaling",
            completed_actions=[{"operation": "scale", "nfs": ["AMF"]}],
        )
        core_spec = fake_llm_spec(subtask_id="subtask-002", subtask="Configure AMF to run with 2 replicas")

        monitoring_graph, monitoring_block = agent.build_registered_subgraph(
            intent="Scale AMF deployment to 2 replicas and then monitor AMF",
            subtask_id="subtask-001",
            subtask="Run the Probe Agent to monitor AMF after scaling",
            knowledge=knowledge,
            llm_langgraph_spec=monitoring_spec,
            completed_actions=[{"operation": "scale", "nfs": ["AMF"]}],
        )
        core_graph, core_block = agent.build_registered_subgraph(
            intent="Scale AMF deployment to 2 replicas",
            subtask_id="subtask-002",
            subtask="Configure AMF to run with 2 replicas",
            knowledge=knowledge,
            llm_langgraph_spec=core_spec,
        )

        self.assertIsNone(monitoring_block)
        self.assertEqual(monitoring_graph["representative_agent"], "probe-agent")
        self.assertIsNone(core_block)
        self.assertEqual(core_graph["representative_agent"], "representative-core-agent")



    def test_execute_subgraph_blocks_legacy_core_probe_post_action_path(self) -> None:
        graph = {
            "version": "langgraph-style-v1",
            "subtask_id": "subtask-001",
            "subtask": "Scale AMF then monitor AMF",
            "nodes": [
                runtime_agent_node("representative-core-agent", "Representative Core Agent"),
                {"id": "core-kubectl-command-tool", "type": "tool", "name": "Core kubectl Command Tool"},
                runtime_agent_node("probe-agent", "Probe Agent"),
                {"id": "grafana-prometheus-tool", "type": "tool", "name": "Grafana Prometheus Tool"},
                {"id": "agent-completion-check-subtask-001", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [
                {"source": "representative-core-agent", "target": "core-kubectl-command-tool"},
                {"source": "core-kubectl-command-tool", "target": "probe-agent"},
                {"source": "probe-agent", "target": "grafana-prometheus-tool"},
                {"source": "grafana-prometheus-tool", "target": "agent-completion-check-subtask-001"},
            ],
            "entrypoint": "representative-core-agent",
            "terminal_nodes": ["agent-completion-check-subtask-001"],
            "representative_agent": "representative-core-agent",
            "compiled": False,
            "knowledge_context_used": False,
            "planning_basis": "test",
            "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
        }
        calls: list[str] = []

        def fake_core(intent: str, **_: object) -> dict[str, object]:
            calls.append("core")
            self.assertIn("core-kubectl-command-tool", intent)
            return {
                "intent": intent,
                "command_proposal": {"operation": "scale"},
                "kubectl_preview": ["kubectl scale ..."],
                "risk_level": "medium",
                "approval_required": False,
                "approval": {"required": False, "state": "not_required", "approved": True},
                "status": "completed",
                "result": {"success": True},
                "errors": [],
                "source": {"type": "test"},
            }

        def fake_monitoring(intent: str, **_: object) -> dict[str, object]:
            calls.append("monitoring")
            self.assertIn("previous_node_results", intent)
            return {
                "intent": intent,
                "command": {"operation": "replica_health"},
                "evaluation_request": {"type": "report_values"},
                "evaluation_result": {"type": "report_values", "status": "passed"},
                "scope": "single_nf",
                "metric": "replica_health",
                "window_seconds": 60,
                "status": "completed",
                "results": [{"target": "AMF"}],
                "errors": [],
                "source": {"type": "test"},
            }

        with patch.object(subgraph_runtime, "invoke_representative_core_agent", side_effect=fake_core):
            with patch.object(subgraph_runtime, "invoke_probe_agent", side_effect=fake_monitoring):
                result = subgraph_runtime.execute_subgraph(
                    original_intent="Scale AMF then monitor AMF",
                    subtask_id="subtask-001",
                    subtask="Scale AMF then monitor AMF",
                    langgraph_spec=graph,
                    representative_agent={"id": "representative-core-agent", "name": "Representative Core Agent"},
                )

        self.assertEqual(calls, [])
        self.assertEqual(result["status"], "blocked")
        self.assertIn("legacy_core_probe_post_action_path_not_supported", result["message"])
        self.assertEqual(result["executed_node_ids"], [])

    def test_monitoring_gate_skips_core_when_expectation_is_healthy(self) -> None:
        expectation = expected_threshold_request()
        expectation["schedule"] = {"duration_seconds": 300, "interval_seconds": 15}
        expectation["checks"].append({
            "check_id": "optional-check",
            "observation": "optional AMF metric",
            "expected": {"operator": ">=", "value": 1},
            "optional": True,
        })
        healthy_evaluation = passed_threshold_evaluation()
        healthy_evaluation["checks"].append({
            "check_id": "optional-check",
            "observation": "optional AMF metric",
            "ok": None,
            "valid": False,
            "skipped": True,
            "actual": None,
            "expected": 1,
            "operator": ">=",
        })
        graph = {
            "nodes": [
                runtime_agent_node("monitoring-agent", "Monitoring Agent"),
                runtime_agent_node("representative-core-agent", "Representative Core Agent"),
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
            "evaluation_request": expectation,
        }
        report = {
            "run_id": "planning-monitoring-gate",
            "subtask_id": "subtask-002",
            "status": "healthy",
            "expected": graph["evaluation_request"],
            "execution": {},
            "probe_report": {
                "status": "completed",
                "evaluation_request": graph["evaluation_request"],
                "evaluation_result": healthy_evaluation,
            },
            "feedback": None,
            "planner_response": None,
            "errors": [],
            "source": {"probe": "probe-agent"},
        }
        with patch.object(
            subgraph_runtime,
            "resolve_agent_communication_contract",
            return_value=dict(RUNTIME_TEST_CONTRACT),
        ) as resolver:
            with patch.object(subgraph_runtime, "invoke_monitoring_agent", return_value=report) as monitoring:
                with patch.object(subgraph_runtime, "invoke_representative_core_agent") as core:
                    result = subgraph_runtime.execute_subgraph(
                        original_intent="Keep AMF at 2 for five minutes; reset only if it drifts",
                        subtask_id="subtask-002",
                        subtask="Verify AMF remains at 2 and recover only on mismatch",
                        langgraph_spec=graph,
                        representative_agent=graph["nodes"][0],
                        planning_report={"run_id": "planning-monitoring-gate", "intent": "test", "subtask_plans": [], "overall_status": "running"},
                    )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["runtime"], "monitoring-agent")
        self.assertEqual(result["executed_node_ids"], ["monitoring-agent"])
        self.assertFalse(monitoring.call_args.kwargs["notify_planner"])
        self.assertEqual(monitoring.call_args.kwargs["notification_delivery"], "response")
        self.assertEqual([call.args[0] for call in resolver.call_args_list], ["monitoring-agent"])
        snapshot_text = json.dumps(monitoring.call_args.args[0], ensure_ascii=False)
        for forbidden in ("communication_contract", "_runtime_communication_contract", "request_schema", "response_schema"):
            self.assertNotIn(forbidden, snapshot_text)
        core.assert_not_called()

        no_check_report = {
            **report,
            "probe_report": {**report["probe_report"], "evaluation_result": {"type": "thresholds", "status": "passed", "checks": []}},
        }
        with patch.object(subgraph_runtime, "invoke_monitoring_agent", return_value=no_check_report):
            with patch.object(subgraph_runtime, "invoke_representative_core_agent") as no_check_core:
                no_check_result = subgraph_runtime.execute_subgraph(
                    original_intent="Keep AMF at 2 for five minutes; reset only if it drifts",
                    subtask_id="subtask-002",
                    subtask="Verify AMF remains at 2 and recover only on mismatch",
                    langgraph_spec=graph,
                    representative_agent=graph["nodes"][0],
                )

        self.assertEqual(no_check_result["status"], "blocked")
        no_check_core.assert_not_called()

    def test_planning_runtime_allows_core_representative_after_monitoring_gate(self) -> None:
        graph = {
            "nodes": [
                runtime_agent_node("monitoring-agent", "Monitoring Agent"),
                runtime_agent_node("representative-core-agent", "Representative Core Agent"),
                {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [
                {"source": "monitoring-agent", "target": "representative-core-agent", "condition": "only_if_mismatch"},
                {"source": "representative-core-agent", "target": "done"},
            ],
            "entrypoint": "monitoring-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "representative-core-agent",
            "compiled": False,
            "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
        }

        order = subgraph_runtime.validate_subgraph_execution_topology(graph)

        self.assertEqual(
            [node["id"] for node in order],
            ["monitoring-agent", "representative-core-agent", "done"],
        )

    def test_disconnected_monitoring_node_cannot_satisfy_core_gate(self) -> None:
        graph = {
            "nodes": [
                runtime_agent_node("representative-core-agent", "Representative Core Agent"),
                runtime_agent_node("monitoring-agent", "Monitoring Agent"),
                {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [{"source": "representative-core-agent", "target": "done"}],
            "entrypoint": "representative-core-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "representative-core-agent",
            "compiled": False,
            "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
        }

        with self.assertRaisesRegex(ValueError, "unreachable_runtime_agent_node"):
            subgraph_runtime.validate_subgraph_execution_topology(graph)

    def test_monitoring_name_alias_is_canonicalized_before_core_gate_validation(self) -> None:
        graph = {
            "nodes": [
                {"id": "representative-core-agent", "type": "agent", "name": "Representative Core Agent"},
                {"id": "monitor", "type": "agent", "name": "Monitoring Agent"},
                {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [
                {"source": "representative-core-agent", "target": "monitor"},
                {"source": "monitor", "target": "done"},
            ],
            "entrypoint": "representative-core-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "representative-core-agent",
            "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
        }

        _, reason = agent.build_registered_subgraph(
            intent="Scale then monitor",
            subtask_id="subtask-001",
            subtask="Scale then monitor",
            knowledge=knowledge_with_candidates(),
            llm_langgraph_spec=graph,
        )

        self.assertEqual(reason, "invalid_subgraph_topology:conditional_core_must_be_immediately_gated_by_monitoring_agent")

    def test_conflicting_agent_id_and_name_are_rejected(self) -> None:
        graph = {
            "nodes": [
                {"id": "monitoring-agent", "type": "agent", "name": "Representative Core Agent"},
                {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [{"source": "monitoring-agent", "target": "done"}],
            "entrypoint": "monitoring-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "monitoring-agent",
            "evaluation_request": expected_threshold_request(),
        }

        _, reason = agent.build_registered_subgraph(
            intent="Monitor AMF",
            subtask_id="subtask-001",
            subtask="Monitor AMF",
            knowledge=knowledge_with_candidates(),
            llm_langgraph_spec=graph,
        )

        self.assertEqual(reason, "invalid_subgraph_topology:conflicting_knowledge_agent_identity")

    def test_persisted_registered_agent_alias_blocks_before_core_execution(self) -> None:
        graph = {
            "nodes": [
                runtime_agent_node("representative-core-agent", "Representative Core Agent"),
                runtime_agent_node("monitor", "Monitoring Agent"),
                {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [
                {"source": "representative-core-agent", "target": "monitor"},
                {"source": "monitor", "target": "done"},
            ],
            "entrypoint": "representative-core-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "representative-core-agent",
            "compiled": False,
            "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
        }

        with patch.object(subgraph_runtime, "invoke_representative_core_agent") as core:
            result = subgraph_runtime.execute_subgraph(
                original_intent="Scale then monitor",
                subtask_id="subtask-001",
                subtask="Scale then monitor",
                langgraph_spec=graph,
                representative_agent=graph["nodes"][0],
            )

        self.assertEqual(result["status"], "blocked")
        self.assertIn("registered_agent_id_must_be_canonical", result["message"])
        core.assert_not_called()

    def test_reserved_monitoring_id_cannot_be_hidden_as_tool(self) -> None:
        graph = {
            "nodes": [
                runtime_agent_node("representative-core-agent", "Representative Core Agent"),
                {"id": "monitoring-agent", "type": "tool", "name": "Monitoring Agent", "source": "planning_agent"},
                {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [
                {"source": "representative-core-agent", "target": "monitoring-agent"},
                {"source": "monitoring-agent", "target": "done"},
            ],
            "entrypoint": "representative-core-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "representative-core-agent",
            "compiled": False,
            "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
        }

        with patch.object(subgraph_runtime, "invoke_representative_core_agent") as core:
            result = subgraph_runtime.execute_subgraph(
                original_intent="Scale then monitor",
                subtask_id="subtask-001",
                subtask="Scale then monitor",
                langgraph_spec=graph,
                representative_agent=graph["nodes"][0],
            )

        self.assertEqual(result["status"], "blocked")
        self.assertIn("runtime_agent_id_requires_agent_node_type", result["message"])
        core.assert_not_called()

    def test_persisted_registered_agent_alias_cannot_be_hidden_as_tool(self) -> None:
        alias = runtime_agent_node("monitor", "Monitoring Agent")
        alias["type"] = "tool"
        graph = {
            "nodes": [
                runtime_agent_node("representative-core-agent", "Representative Core Agent"),
                alias,
                {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [
                {"source": "representative-core-agent", "target": "monitor"},
                {"source": "monitor", "target": "done"},
            ],
            "entrypoint": "representative-core-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "representative-core-agent",
            "compiled": False,
            "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
        }

        with patch.object(subgraph_runtime, "invoke_representative_core_agent") as core:
            result = subgraph_runtime.execute_subgraph(
                original_intent="Scale then monitor",
                subtask_id="subtask-001",
                subtask="Scale then monitor",
                langgraph_spec=graph,
                representative_agent=graph["nodes"][0],
            )

        self.assertEqual(result["status"], "blocked")
        self.assertIn("registered_agent_id_must_be_canonical", result["message"])
        core.assert_not_called()

    def test_persisted_contractless_agent_alias_cannot_be_hidden_as_tool(self) -> None:
        graph = {
            "nodes": [
                runtime_agent_node("representative-core-agent", "Representative Core Agent"),
                {"id": "monitor", "type": "tool", "name": "Monitoring Agent", "source": "planning_agent"},
                {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [
                {"source": "representative-core-agent", "target": "monitor"},
                {"source": "monitor", "target": "done"},
            ],
            "entrypoint": "representative-core-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "representative-core-agent",
            "compiled": False,
            "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
        }

        with patch.object(subgraph_runtime, "invoke_representative_core_agent") as core:
            result = subgraph_runtime.execute_subgraph(
                original_intent="Scale then monitor",
                subtask_id="subtask-001",
                subtask="Scale then monitor",
                langgraph_spec=graph,
                representative_agent=graph["nodes"][0],
            )

        self.assertEqual(result["status"], "blocked")
        self.assertIn("runtime_agent_name_requires_agent_node_type", result["message"])
        core.assert_not_called()

    def test_monitoring_gate_reaches_core_only_on_confirmed_mismatch(self) -> None:
        expectation = expected_threshold_request()
        expectation["checks"].append({
            "check_id": "optional-check",
            "observation": "optional AMF metric",
            "expected": {"operator": ">=", "value": 1},
            "optional": True,
        })
        graph = {
            "nodes": [
                runtime_agent_node("monitoring-agent", "Monitoring Agent"),
                runtime_agent_node("representative-core-agent", "Representative Core Agent"),
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
            "evaluation_request": expectation,
        }
        failed_evaluation = failed_threshold_evaluation()
        failed_evaluation["checks"].append({
            "check_id": "optional-check",
            "observation": "optional AMF metric",
            "ok": None,
            "valid": False,
            "skipped": True,
            "actual": None,
            "expected": 1,
            "operator": ">=",
        })
        monitoring_report = {
            "run_id": "planning-monitoring-recovery",
            "subtask_id": "subtask-002",
            "status": "replan_required",
            "expected": graph["evaluation_request"],
            "execution": {},
            "probe_report": {
                "status": "completed",
                "evaluation_request": graph["evaluation_request"],
                "evaluation_result": failed_evaluation,
            },
            "feedback": {
                "source": "monitoring-agent",
                "status": "replan_required",
                "reason": "expected_network_state_not_observed",
                "expected": graph["evaluation_request"],
                "observed": failed_evaluation,
            },
            "planner_response": None,
            "errors": [],
            "source": {"probe": "probe-agent"},
        }
        core_report = {
            "intent": "restore AMF",
            "command_proposal": {"operation": "scale"},
            "kubectl_preview": ["kubectl scale ..."],
            "risk_level": "medium",
            "approval_required": True,
            "approval": {"required": True, "state": "pending", "approved": False},
            "status": "pending_approval",
            "result": {"executed": False, "reason": "waiting_for_user_approval", "command_count": 1},
            "errors": [],
            "source": {"type": "test"},
        }
        with patch.object(
            subgraph_runtime,
            "resolve_agent_communication_contract",
            return_value=dict(RUNTIME_TEST_CONTRACT),
        ) as resolver:
            with patch.object(subgraph_runtime, "invoke_monitoring_agent", return_value=monitoring_report):
                with patch.object(subgraph_runtime, "invoke_representative_core_agent", return_value=core_report) as core:
                    result = subgraph_runtime.execute_subgraph(
                        original_intent="Recover AMF only if it is not 2",
                        subtask_id="subtask-002",
                        subtask="Verify and conditionally recover AMF",
                        langgraph_spec=graph,
                        representative_agent=graph["nodes"][0],
                        run_id=monitoring_report["run_id"],
                    )

        self.assertEqual(result["status"], "pending_approval")
        self.assertEqual(result["runtime"], "representative-core-agent")
        self.assertEqual(result["executed_node_ids"], ["monitoring-agent", "representative-core-agent"])
        self.assertEqual(
            [call.args[0] for call in resolver.call_args_list],
            ["monitoring-agent", "representative-core-agent"],
        )
        core_intent = core.call_args.args[0]
        for forbidden in ("communication_contract", "_runtime_communication_contract", "request_schema", "response_schema"):
            self.assertNotIn(forbidden, core_intent)
        core.assert_called_once()

        invalid_evaluation = failed_threshold_evaluation()
        invalid_evaluation["checks"][0]["check_id"] = "unrelated-check"
        invalid_evaluation["checks"][0]["observation"] = "SMF CPU usage"
        invalid_report = {
            **monitoring_report,
            "probe_report": {**monitoring_report["probe_report"], "evaluation_result": invalid_evaluation},
            "feedback": {**monitoring_report["feedback"], "observed": invalid_evaluation},
        }
        with patch.object(subgraph_runtime, "invoke_monitoring_agent", return_value=invalid_report):
            with patch.object(subgraph_runtime, "invoke_representative_core_agent") as invalid_core:
                invalid_result = subgraph_runtime.execute_subgraph(
                    original_intent="Recover AMF only if it is not 2",
                    subtask_id="subtask-002",
                    subtask="Verify and conditionally recover AMF",
                    langgraph_spec=graph,
                    representative_agent=graph["nodes"][0],
                    run_id=monitoring_report["run_id"],
                )

        self.assertEqual(invalid_result["status"], "blocked")
        invalid_core.assert_not_called()

    def test_langgraph_runtime_executes_a_branching_graph(self) -> None:
        expected = {"type": "report_values", "completion_rule": "data_returned"}
        probe = runtime_agent_node("probe-agent", "Probe Agent")
        graph = {
            "nodes": [
                probe,
                {"id": "tool-a", "type": "tool", "name": "Tool A"},
                {"id": "tool-b", "type": "tool", "name": "Tool B"},
                {"id": "done-a", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
                {"id": "done-b", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [
                {"source": "probe-agent", "target": "tool-a"},
                {"source": "probe-agent", "target": "tool-b"},
                {"source": "tool-a", "target": "done-a"},
                {"source": "tool-b", "target": "done-b"},
            ],
            "entrypoint": "probe-agent",
            "terminal_nodes": ["done-a", "done-b"],
            "representative_agent": "probe-agent",
            "compiled": False,
            "evaluation_request": expected,
        }
        report = {
            "intent": "Observe AMF",
            "command": {},
            "evaluation_request": expected,
            "evaluation_result": {"status": "passed"},
            "scope": "single_nf",
            "metric": "pods",
            "window_seconds": 60,
            "status": "completed",
            "results": [{}],
            "errors": [],
            "source": {"type": "test"},
        }

        with patch.object(subgraph_runtime, "invoke_probe_agent", return_value=report) as invoke:
            result = subgraph_runtime.execute_subgraph(
                original_intent="Observe AMF",
                subtask_id="subtask-001",
                subtask="Observe AMF through both declared tool branches",
                langgraph_spec=graph,
                representative_agent=probe,
            )

        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["compiled"])
        self.assertEqual(result["executed_node_ids"], ["probe-agent"])
        self.assertIn("tool-a", invoke.call_args.args[0])
        self.assertIn("tool-b", invoke.call_args.args[0])

    def test_initial_core_completed_response_is_blocked_before_approval(self) -> None:
        node = runtime_agent_node("representative-core-agent", "Representative Core Agent")
        graph = {
            "nodes": [node, {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"}],
            "edges": [{"source": "representative-core-agent", "target": "done"}],
            "entrypoint": "representative-core-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "representative-core-agent",
            "compiled": False,
            "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
        }
        completed_without_approval = {
            "intent": "Scale AMF",
            "command_proposal": {"operation": "scale"},
            "kubectl_preview": ["kubectl scale ..."],
            "risk_level": "medium",
            "approval_required": False,
            "approval": {"required": False, "state": "not_required", "approved": True},
            "status": "completed",
            "result": {"executed": True, "success": True},
            "errors": [],
            "source": {"type": "test"},
        }

        with patch.object(subgraph_runtime, "invoke_representative_core_agent", return_value=completed_without_approval) as core:
            result = subgraph_runtime.execute_subgraph(
                original_intent="Scale AMF",
                subtask_id="subtask-001",
                subtask="Scale AMF",
                langgraph_spec=graph,
                representative_agent=node,
            )

        self.assertEqual(result["status"], "blocked")
        self.assertIn("must not complete", result["message"])
        core.assert_called_once()

    def test_initial_core_pending_response_cannot_claim_success_without_execution(self) -> None:
        node = runtime_agent_node("representative-core-agent", "Representative Core Agent")
        graph = {
            "nodes": [node, {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"}],
            "edges": [{"source": "representative-core-agent", "target": "done"}],
            "entrypoint": "representative-core-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "representative-core-agent",
            "compiled": False,
            "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
        }
        contradictory_pending = {
            "intent": "Scale AMF",
            "command_proposal": {"operation": "scale"},
            "kubectl_preview": ["kubectl scale ..."],
            "risk_level": "medium",
            "approval_required": True,
            "approval": {"required": True, "state": "pending", "approved": False},
            "status": "pending_approval",
            "result": {"executed": False, "success": True},
            "errors": [],
            "source": {"type": "test"},
        }

        with patch.object(subgraph_runtime, "invoke_representative_core_agent", return_value=contradictory_pending):
            result = subgraph_runtime.execute_subgraph(
                original_intent="Scale AMF",
                subtask_id="subtask-001",
                subtask="Scale AMF",
                langgraph_spec=graph,
                representative_agent=node,
            )

        self.assertEqual(result["status"], "blocked")
        self.assertIn("not-executed result", result["message"])

    def test_dynamic_expected_value_is_bound_to_probe_query_value(self) -> None:
        expected = {
            "type": "thresholds",
            "checks": [{
                "check_id": "available-check",
                "observation": "AMF available replicas",
                "expected": {"operator": ">=", "value_from_check_id": "desired-check"},
            }],
        }
        forged = {
            "type": "thresholds",
            "status": "failed",
            "checks": [{
                "check_id": "available-check",
                "observation": "AMF available replicas",
                "ok": False,
                "valid": True,
                "actual": 2,
                "expected": 3,
                "operator": ">=",
            }],
            "query_values": {"available-check": 2, "desired-check": 2},
        }
        observed = {
            **forged,
            "checks": [{**forged["checks"][0], "actual": 1, "expected": 2}],
        }

        self.assertFalse(subgraph_runtime.has_confirmed_threshold_mismatch(forged, expected))
        self.assertTrue(subgraph_runtime.has_confirmed_threshold_mismatch(observed, expected))

    def test_monitoring_gate_blocks_core_when_healthy_evidence_is_missing(self) -> None:
        graph = {
            "nodes": [
                runtime_agent_node("monitoring-agent", "Monitoring Agent"),
                runtime_agent_node("representative-core-agent", "Representative Core Agent"),
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
            "evaluation_request": {"type": "thresholds", "checks": []},
        }
        report = {
            "run_id": "planning-invalid-gate",
            "subtask_id": "subtask-001",
            "status": "healthy",
            "expected": graph["evaluation_request"],
            "execution": {},
            "probe_report": None,
            "feedback": None,
            "planner_response": None,
            "errors": [],
            "source": {"probe": "probe-agent"},
        }

        with patch.object(subgraph_runtime, "invoke_monitoring_agent", return_value=report):
            with patch.object(subgraph_runtime, "invoke_representative_core_agent") as core:
                result = subgraph_runtime.execute_subgraph(
                    original_intent="Keep AMF at 2",
                    subtask_id="subtask-001",
                    subtask="Recover only on mismatch",
                    langgraph_spec=graph,
                    representative_agent=graph["nodes"][0],
                )

        self.assertEqual(result["status"], "blocked")
        core.assert_not_called()

    def test_monitoring_gate_rejects_reported_expected_state_drift(self) -> None:
        planned = {"type": "thresholds", "checks": [{"check_id": "planned", "observation": "AMF pods", "expected": {"operator": "==", "value": 2}}]}
        reported = {"type": "thresholds", "checks": [{"check_id": "reported", "observation": "AMF pods", "expected": {"operator": "==", "value": 1}}]}
        graph = {
            "nodes": [
                runtime_agent_node("monitoring-agent", "Monitoring Agent"),
                runtime_agent_node("representative-core-agent", "Representative Core Agent"),
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
            "evaluation_request": planned,
        }
        report = {
            "run_id": "planning-drift-gate",
            "subtask_id": "subtask-001",
            "status": "healthy",
            "expected": reported,
            "execution": {},
            "probe_report": {"status": "completed", "evaluation_request": reported, "evaluation_result": {"status": "passed"}},
            "feedback": None,
            "planner_response": None,
            "errors": [],
            "source": {"probe": "probe-agent"},
        }

        with patch.object(subgraph_runtime, "invoke_monitoring_agent", return_value=report):
            with patch.object(subgraph_runtime, "invoke_representative_core_agent") as core:
                result = subgraph_runtime.execute_subgraph(
                    original_intent="Keep AMF at 2",
                    subtask_id="subtask-001",
                    subtask="Recover only on mismatch",
                    langgraph_spec=graph,
                    representative_agent=graph["nodes"][0],
                )

        self.assertEqual(result["status"], "blocked")
        core.assert_not_called()

    def test_probe_response_must_echo_planning_evaluation_request(self) -> None:
        planned = {"type": "report_values", "completion_rule": "data_returned"}
        node = runtime_agent_node("probe-agent", "Probe Agent")
        graph = {
            "nodes": [node, {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"}],
            "edges": [{"source": "probe-agent", "target": "done"}],
            "entrypoint": "probe-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "probe-agent",
            "compiled": False,
            "evaluation_request": planned,
        }
        report = {
            "intent": "Observe AMF",
            "command": {},
            "evaluation_request": {"type": "thresholds", "checks": []},
            "evaluation_result": {"status": "passed"},
            "scope": "single_nf",
            "metric": "pods",
            "window_seconds": 60,
            "status": "completed",
            "results": [{}],
            "errors": [],
            "source": {"type": "test"},
        }

        with patch.object(subgraph_runtime, "invoke_probe_agent", return_value=report):
            result = subgraph_runtime.execute_subgraph(
                original_intent="Observe AMF",
                subtask_id="subtask-001",
                subtask="Observe AMF",
                langgraph_spec=graph,
                representative_agent=node,
            )

        self.assertEqual(result["status"], "blocked")

    def test_monitoring_core_graph_requires_explicit_mismatch_edge(self) -> None:
        graph = {
            "nodes": [
                runtime_agent_node("monitoring-agent", "Monitoring Agent"),
                runtime_agent_node("representative-core-agent", "Representative Core Agent"),
                {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [
                {"source": "monitoring-agent", "target": "representative-core-agent", "condition": "handoff_or_tool_use"},
                {"source": "representative-core-agent", "target": "done"},
            ],
            "entrypoint": "monitoring-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "monitoring-agent",
            "compiled": False,
            "evaluation_request": {"type": "thresholds", "checks": []},
        }

        with patch.object(subgraph_runtime, "invoke_monitoring_agent") as monitoring:
            with patch.object(subgraph_runtime, "invoke_representative_core_agent") as core:
                result = subgraph_runtime.execute_subgraph(
                    original_intent="Recover AMF only if mismatched",
                    subtask_id="subtask-001",
                    subtask="Conditional recovery",
                    langgraph_spec=graph,
                    representative_agent=graph["nodes"][0],
                )

        self.assertEqual(result["status"], "blocked")
        self.assertIn("only_if_mismatch", result["message"])
        monitoring.assert_not_called()
        core.assert_not_called()

    def test_legacy_probe_conditional_core_graph_is_blocked_before_execution(self) -> None:
        graph = {
            "nodes": [
                runtime_agent_node("probe-agent", "Probe Agent"),
                runtime_agent_node("representative-core-agent", "Representative Core Agent"),
                {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [
                {"source": "probe-agent", "target": "representative-core-agent", "condition": "only_if_mismatch"},
                {"source": "representative-core-agent", "target": "done"},
            ],
            "entrypoint": "probe-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "probe-agent",
            "compiled": False,
            "evaluation_request": {"type": "thresholds", "checks": []},
        }
        probe = {
            "intent": "verify AMF",
            "command": {},
            "evaluation_request": graph["evaluation_request"],
            "evaluation_result": {"type": "thresholds", "status": "passed"},
            "scope": "single_nf",
            "metric": "replica_health",
            "window_seconds": 60,
            "status": "completed",
            "results": [{}],
            "errors": [],
            "source": {"type": "test"},
        }
        with patch.object(subgraph_runtime, "invoke_probe_agent", return_value=probe) as probe_agent:
            with patch.object(subgraph_runtime, "invoke_representative_core_agent") as core:
                result = subgraph_runtime.execute_subgraph(
                    original_intent="Keep AMF at 2",
                    subtask_id="subtask-001",
                    subtask="Recover only if AMF is not 2",
                    langgraph_spec=graph,
                    representative_agent=graph["nodes"][0],
                )

        self.assertEqual(result["status"], "blocked")
        self.assertEqual(result["executed_node_ids"], [])
        self.assertIn("legacy_probe_conditional_recovery_not_supported", result["message"])
        probe_agent.assert_not_called()
        core.assert_not_called()

    def test_execute_subgraph_blocks_unknown_agent_without_guessing(self) -> None:
        graph = {
            "nodes": [
                runtime_agent_node("unknown-agent", "Unknown Agent"),
                {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
            ],
            "edges": [{"source": "unknown-agent", "target": "done"}],
            "entrypoint": "unknown-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "unknown-agent",
            "compiled": False,
        }

        result = subgraph_runtime.execute_subgraph(
            original_intent="Do something",
            subtask_id="subtask-001",
            subtask="Do something",
            langgraph_spec=graph,
            representative_agent={"id": "unknown-agent", "name": "Unknown Agent"},
        )

        self.assertEqual(result["status"], "blocked")
        self.assertIn("registered_agent_id_must_be_canonical", result["message"])

    def test_execute_subgraph_rejects_unsupported_contract_dialect(self) -> None:
        node = runtime_agent_node("probe-agent", "Probe Agent")
        graph = {
            "nodes": [node, {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"}],
            "edges": [{"source": "probe-agent", "target": "done"}],
            "entrypoint": "probe-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "probe-agent",
            "compiled": False,
            "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
        }

        with patch.object(
            subgraph_runtime,
            "resolve_agent_communication_contract",
            return_value={**RUNTIME_TEST_CONTRACT, "schema_dialect": "unsupported-v9"},
        ):
            result = subgraph_runtime.execute_subgraph(
                original_intent="Observe AMF",
                subtask_id="subtask-001",
                subtask="Observe AMF",
                langgraph_spec=graph,
                representative_agent=node,
            )

        self.assertEqual(result["status"], "blocked")
        self.assertIn("no resolved MongoDB communication contract dialect", result["message"])

    def test_execute_subgraph_rejects_schema_fields_embedded_in_runtime_node(self) -> None:
        node = {**runtime_agent_node("probe-agent", "Probe Agent"), "request_schema": {"type": "object"}}
        graph = {
            "nodes": [node, {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"}],
            "edges": [{"source": "probe-agent", "target": "done"}],
            "entrypoint": "probe-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "probe-agent",
            "compiled": False,
            "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
        }

        with patch.object(subgraph_runtime, "invoke_probe_agent") as probe:
            result = subgraph_runtime.execute_subgraph(
                original_intent="Observe AMF",
                subtask_id="subtask-001",
                subtask="Observe AMF",
                langgraph_spec=graph,
                representative_agent=node,
            )

        self.assertEqual(result["status"], "blocked")
        self.assertIn("runtime_agent_node_contains_unsupported_fields", result["message"])
        probe.assert_not_called()

    def test_execute_subgraph_rejects_invalid_response_schema_before_agent_call(self) -> None:
        node = runtime_agent_node("probe-agent", "Probe Agent")
        graph = {
            "nodes": [node, {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"}],
            "edges": [{"source": "probe-agent", "target": "done"}],
            "entrypoint": "probe-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "probe-agent",
            "compiled": False,
            "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
        }

        invalid_contract = {**RUNTIME_TEST_CONTRACT, "response_schema": {"type": "object", "min_items": 1}}
        with patch.object(subgraph_runtime, "resolve_agent_communication_contract", return_value=invalid_contract):
            with patch.object(subgraph_runtime, "invoke_probe_agent") as probe:
                result = subgraph_runtime.execute_subgraph(
                    original_intent="Observe AMF",
                    subtask_id="subtask-001",
                    subtask="Observe AMF",
                    langgraph_spec=graph,
                    representative_agent=node,
                )

        self.assertEqual(result["status"], "blocked")
        self.assertIn("unsupported schema keywords", result["message"])
        probe.assert_not_called()

    def test_execute_subgraph_blocks_stale_contract_ref_before_agent_call(self) -> None:
        node = runtime_agent_node("probe-agent", "Probe Agent")
        graph = {
            "nodes": [node, {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"}],
            "edges": [{"source": "probe-agent", "target": "done"}],
            "entrypoint": "probe-agent",
            "terminal_nodes": ["done"],
            "representative_agent": "probe-agent",
            "compiled": False,
            "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
        }

        with patch.object(
            subgraph_runtime,
            "resolve_agent_communication_contract",
            side_effect=ValueError("agent_communication_contract_ref_mismatch:probe-agent"),
        ):
            with patch.object(subgraph_runtime, "invoke_probe_agent") as probe:
                result = subgraph_runtime.execute_subgraph(
                    original_intent="Observe AMF",
                    subtask_id="subtask-001",
                    subtask="Observe AMF",
                    langgraph_spec=graph,
                    representative_agent=node,
                )

        self.assertEqual(result["status"], "blocked")
        self.assertIn("agent_communication_contract_unavailable", result["message"])
        probe.assert_not_called()

    def test_core_approval_request_is_validated_by_mongodb_contract(self) -> None:
        report = {
            "intent": "Scale AMF",
            "command_proposal": {"operation": "scale"},
            "kubectl_preview": ["kubectl scale ..."],
            "risk_level": "medium",
            "approval_required": True,
            "approval": {"required": True, "state": "pending", "approved": False},
            "status": "pending_approval",
            "result": {"executed": False},
            "errors": [],
            "source": {"type": "test"},
        }
        contract = {
            "schema_dialect": "json-schema-subset-v1",
            "approval_request_schema": {
                "type": "object",
                "required": ["report", "approved", "mongodb_only_token"],
                "properties": {
                    "report": {"type": "object"},
                    "approved": {"type": "boolean"},
                    "mongodb_only_token": {"type": "string"},
                },
            },
            "approval_response_schema": {"type": "object"},
        }

        with self.assertRaisesRegex(ValueError, "mongodb_only_token"):
            with patch.object(subgraph_runtime, "resolve_agent_communication_contract", return_value=contract):
                subgraph_runtime.approve_representative_core_report(
                    report,
                    approved=True,
                    contract_ref=runtime_contract_ref("representative-core-agent"),
                )

    def test_core_approval_response_is_bound_to_submitted_decision(self) -> None:
        pending = {
            "intent": "Scale AMF",
            "command_proposal": {"operation": "scale"},
            "kubectl_preview": ["kubectl scale ..."],
            "risk_level": "medium",
            "approval_required": True,
            "approval": {"required": True, "state": "pending", "approved": False},
            "status": "pending_approval",
            "result": {"executed": False, "reason": "waiting_for_user_approval", "command_count": 1},
            "errors": [],
            "source": {"type": "test"},
        }
        completed = {
            **pending,
            "approval": {"required": True, "state": "approved", "approved": True},
            "status": "completed",
            "result": {"executed": True, "success": True},
        }
        completed_failed = {**completed, "result": {"executed": True, "success": False}}
        rejected = {
            **pending,
            "approval": {"required": True, "state": "rejected", "approved": False},
            "status": "rejected",
        }
        blocked = {
            **pending,
            "approval": {"required": True, "state": "blocked", "approved": False},
            "status": "blocked",
        }
        failed_after_execution = {
            **pending,
            "approval": {"required": True, "state": "approved", "approved": True},
            "status": "blocked",
            "result": {"executed": True, "success": False},
        }
        contract = {
            "schema_dialect": "json-schema-subset-v1",
            "approval_request_schema": {"type": "object"},
            "approval_response_schema": {"type": "object"},
        }

        with patch.dict(os.environ, {"CORE_REPRESENTATIVE_AGENT_URL": ""}):
            with patch.object(subgraph_runtime, "_approve_representative_core_cli", return_value=completed):
                with self.assertRaisesRegex(ValueError, "Rejected Representative Core approval"):
                    subgraph_runtime.approve_representative_core_report(
                        pending,
                        approved=False,
                        contract_ref=runtime_contract_ref("representative-core-agent"),
                    )
            with patch.object(subgraph_runtime, "_approve_representative_core_cli", return_value=rejected):
                with self.assertRaisesRegex(ValueError, "Approved Representative Core request"):
                    subgraph_runtime.approve_representative_core_report(
                        pending,
                        approved=True,
                        contract_ref=runtime_contract_ref("representative-core-agent"),
                    )
            with patch.object(subgraph_runtime, "_approve_representative_core_cli", return_value=completed_failed):
                with self.assertRaisesRegex(ValueError, "successful executed result"):
                    subgraph_runtime.approve_representative_core_report(
                        pending,
                        approved=True,
                        contract_ref=runtime_contract_ref("representative-core-agent"),
                    )
            with patch.object(subgraph_runtime, "_approve_representative_core_cli", return_value=blocked):
                response = subgraph_runtime.approve_representative_core_report(
                    pending,
                    approved=True,
                    contract_ref=runtime_contract_ref("representative-core-agent"),
                )
        self.assertEqual(response["status"], "blocked")
        with patch.dict(os.environ, {"CORE_REPRESENTATIVE_AGENT_URL": ""}):
            with patch.object(subgraph_runtime, "_approve_representative_core_cli", return_value=failed_after_execution):
                response = subgraph_runtime.approve_representative_core_report(
                    pending,
                    approved=True,
                    contract_ref=runtime_contract_ref("representative-core-agent"),
                )
        self.assertEqual(response["result"], {"executed": True, "success": False})


    def test_pending_approval_stops_later_subtasks_until_approval(self) -> None:
        payload = {
            "intent": "Scale AMF deployment to 2 replicas and then monitor AMF",
            "subtasks": ["Set the AMF deployment replica count to 2", "Monitor AMF replica health"],
            "golden_goal_context_used": False,
        }
        executed_subtasks: list[str] = []

        def fake_runtime(**kwargs: object) -> dict[str, object]:
            subtask = str(kwargs["subtask"])
            executed_subtasks.append(subtask)
            return {
                "status": "pending_approval",
                "runtime": "representative-core-agent",
                "subtask_id": kwargs["subtask_id"],
                "subtask": subtask,
                "representative_agent_id": "representative-core-agent",
                "representative_agent_name": "Representative Core Agent",
                "message": "Representative Core Agent returned a kubectl approval request.",
                "compiled": False,
                "representative_core_report": {
                    "intent": payload["intent"],
                    "command_proposal": {
                        "operation": "scale",
                        "resource": "deployment",
                        "namespace": "free5gc-v4",
                        "target_scope": "single_nf",
                        "nfs": ["AMF"],
                        "parameters": {"replicas": 2},
                        "rationale": "scale AMF",
                    },
                    "kubectl_preview": ["kubectl scale deployment/free5gc-free5gc-amf -n free5gc-v4 --replicas=2"],
                    "risk_level": "medium",
                    "approval_required": True,
                    "approval": {"required": True, "state": "pending", "approved": False},
                    "status": "pending_approval",
                    "result": {"executed": False},
                    "errors": [],
                    "source": {"type": "test"},
                },
            }

        with patch.object(agent, "find_knowledge_for_subtask", return_value=knowledge_with_candidates()):
            with patch.object(agent, "build_langgraph_spec_with_llm", side_effect=fake_llm_spec):
                with patch.object(agent, "execute_subgraph", side_effect=fake_runtime):
                    report = agent.plan_from_decomposition(payload).to_dict()

        validate_planning_report(report)
        self.assertEqual(report["overall_status"], "pending_approval")
        self.assertEqual(executed_subtasks, ["Set the AMF deployment replica count to 2"])
        self.assertEqual(len(report["subtask_plans"]), 1)

    def test_malformed_planning_llm_spec_blocks_without_runtime_guessing(self) -> None:
        core_payload = {
            "intent": "Scale AMF deployment to 2 replicas",
            "subtasks": ["Set the AMF deployment replica count to 2"],
            "golden_goal_context_used": False,
        }

        with patch.object(agent, "find_knowledge_for_subtask", return_value=knowledge_with_candidates()):
            with patch.object(agent, "build_langgraph_spec_with_llm", side_effect=AssertionError("malformed llm output")):
                with patch.object(agent, "execute_subgraph", side_effect=AssertionError("runtime must not be guessed after LLM failure")):
                    report = agent.plan_from_decomposition(core_payload).to_dict()

        first_plan = report["subtask_plans"][0]
        self.assertEqual(report["overall_status"], "blocked")
        self.assertEqual(first_plan["representative_agent"]["id"], "planning-agent")
        self.assertEqual(first_plan["result"]["errors"][0]["code"], "llm_subgraph_generation_failed")


    def test_core_nf_action_spec_uses_existing_representative_core_agent(self) -> None:
        core_payload = {
            "intent": "Scale AMF deployment to 2 replicas",
            "subtasks": ["Set the AMF deployment replica count to 2"],
            "golden_goal_context_used": False,
        }

        def fake_scaling_spec(**kwargs: object) -> dict[str, object]:
            subtask_id = str(kwargs["subtask_id"])
            subtask = str(kwargs["subtask"])
            return {
                "version": "langgraph-style-v1",
                "subtask_id": subtask_id,
                "subtask": subtask,
                "nodes": [
                    {
                        "id": "representative-core-agent",
                        "type": "agent",
                        "role": "representative_agent",
                        "name": "Representative Core Agent",
                        "description": "LLM-selected registered Representative Core Agent.",
                        "source": "local_runtime_registry",
                        "capabilities": [],
                    },
                    {
                        "id": f"agent-completion-check-{subtask_id}",
                        "type": "agent",
                        "role": "completion_checker",
                        "name": "planning-completion-check",
                        "source": "planning_agent",
                        "description": "completion check",
                        "capabilities": ["completion-monitoring"],
                    },
                ],
                "edges": [{"source": "representative-core-agent", "target": f"agent-completion-check-{subtask_id}", "condition": "report_completion_status"}],
                "entrypoint": "representative-core-agent",
                "terminal_nodes": [f"agent-completion-check-{subtask_id}"],
                "representative_agent": "representative-core-agent",
                "compiled": False,
                "knowledge_context_used": False,
                "planning_basis": "fake llm scaling spec",
                "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
            }

        def fake_runtime(**kwargs: object) -> dict[str, object]:
            representative_agent = kwargs["representative_agent"]
            return {
                "status": "pending_approval",
                "runtime": "representative-core-agent",
                "subtask_id": kwargs["subtask_id"],
                "subtask": kwargs["subtask"],
                "representative_agent_id": representative_agent["id"],
                "representative_agent_name": representative_agent["name"],
                "message": "Representative Core Agent returned a kubectl approval request.",
                "compiled": False,
                "representative_core_report": {
                    "intent": "Scale AMF deployment to 2 replicas",
                    "command_proposal": {"operation": "scale", "resource": "deployment", "namespace": "free5gc-v4", "target_scope": "single_nf", "nfs": ["AMF"], "parameters": {"replicas": 2}, "rationale": "scale AMF"},
                    "kubectl_preview": ["kubectl scale deployment/free5gc-free5gc-amf -n free5gc-v4 --replicas=2"],
                    "risk_level": "medium",
                    "approval_required": True,
                    "approval": {"required": True, "state": "pending", "approved": False},
                    "status": "pending_approval",
                    "result": {"executed": False},
                    "errors": [],
                    "source": {"type": "test"},
                },
            }

        with patch.object(agent, "find_knowledge_for_subtask", return_value=knowledge_with_candidates()):
            with patch.object(agent, "build_langgraph_spec_with_llm", side_effect=fake_scaling_spec):
                with patch.object(agent, "execute_subgraph", side_effect=fake_runtime):
                    report = agent.plan_from_decomposition(core_payload).to_dict()

        first_plan = report["subtask_plans"][0]
        self.assertEqual(first_plan["representative_agent"]["id"], "representative-core-agent")
        self.assertEqual(first_plan["representative_agent"]["source"], "knowledge_db_candidate")
        self.assertEqual(first_plan["result"]["representative_agent_id"], "representative-core-agent")

    def test_core_nf_action_subtask_preserves_pending_approval(self) -> None:
        core_payload = {
            "intent": "Restart AMF",
            "subtasks": ["Restart AMF deployment through the Representative Core Agent"],
            "golden_goal_context_used": False,
        }

        def fake_runtime(**kwargs: object) -> dict[str, object]:
            return {
                "status": "pending_approval",
                "runtime": "representative-core-agent",
                "subtask_id": kwargs["subtask_id"],
                "subtask": kwargs["subtask"],
                "representative_agent_id": "representative-core-agent",
                "representative_agent_name": "RepresentativeCoreAgent",
                "message": "Representative Core Agent returned a kubectl approval request.",
                "compiled": False,
                "representative_core_report": {
                    "intent": "Restart AMF",
                    "command_proposal": {
                        "operation": "rollout_restart",
                        "resource": "deployment",
                        "namespace": "free5gc-v4",
                        "target_scope": "single_nf",
                        "nfs": ["AMF"],
                        "parameters": {},
                        "rationale": "restart AMF",
                    },
                    "kubectl_preview": ["kubectl rollout restart deployment/free5gc-free5gc-amf -n free5gc-v4"],
                    "risk_level": "high",
                    "approval_required": True,
                    "approval": {"required": True, "state": "pending", "approved": False},
                    "status": "pending_approval",
                    "result": {"executed": False},
                    "errors": [],
                    "source": {"type": "test"},
                },
            }

        with patch.object(agent, "find_knowledge_for_subtask", return_value=knowledge_with_candidates()):
            with patch.object(agent, "build_langgraph_spec_with_llm", side_effect=fake_llm_spec):
                with patch.object(agent, "execute_subgraph", side_effect=fake_runtime):
                    report = agent.plan_from_decomposition(core_payload).to_dict()

        validate_planning_report(report)
        self.assertEqual(report["overall_status"], "pending_approval")
        self.assertEqual(report["subtask_plans"][0]["status"], "pending_approval")
        self.assertEqual(report["subtask_plans"][0]["result"]["runtime"], "representative-core-agent")

    def test_cli_outputs_planning_report_from_file(self) -> None:
        input_path = PROJECT_ROOT / "sample-test-decomposition.json"
        input_path.write_text(json.dumps(sample_decomposition()), encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, "agent.py", "--json", "--input", str(input_path)],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PLANNING_USE_DETERMINISTIC_TEST_SPEC": "1"},
            )
        finally:
            input_path.unlink(missing_ok=True)

        report = json.loads(completed.stdout)
        validate_planning_report(report)
        self.assertEqual(report["overall_status"], "blocked")

    def test_cli_default_output_wraps_planning_report(self) -> None:
        input_path = PROJECT_ROOT / "sample-test-decomposition-marked.json"
        input_path.write_text(json.dumps(sample_decomposition()), encoding="utf-8")

        try:
            completed = subprocess.run(
                [sys.executable, "agent.py", "--input", str(input_path)],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "PLANNING_USE_DETERMINISTIC_TEST_SPEC": "1"},
            )
        finally:
            input_path.unlink(missing_ok=True)

        output = completed.stdout
        self.assertIn("------planning-agent------", output)
        self.assertIn('"overall_status": "blocked"', output)
        self.assertTrue(output.strip().endswith("----end----"))

    def test_approve_core_report_requires_stored_pending_run(self) -> None:
        pending_report = {
            "intent": "Restart AMF",
            "command_proposal": {
                "operation": "rollout_restart",
                "resource": "deployment",
                "namespace": "free5gc-v4",
                "target_scope": "single_nf",
                "nfs": ["AMF"],
                "parameters": {},
                "rationale": "restart AMF",
            },
            "kubectl_preview": ["kubectl rollout restart deployment/free5gc-free5gc-amf -n free5gc-v4"],
            "risk_level": "high",
            "approval_required": True,
            "approval": {"required": True, "state": "pending", "approved": False},
            "status": "pending_approval",
            "result": {"executed": False},
            "errors": [],
            "source": {"type": "test"},
        }
        with patch.object(agent, "approve_representative_core_report") as approve_runtime:
            with self.assertRaisesRegex(ValueError, "run_id"):
                agent.approve_core_report_from_payload(
                    {"report": pending_report},
                    approved=False,
                    approved_by="tester",
                    reason="not now",
                )

        approve_runtime.assert_not_called()

    def test_approval_rejects_tampered_report_before_core_call(self) -> None:
        stored_report = {
            "intent": "Scale AMF",
            "command_proposal": {"operation": "scale", "parameters": {"replicas": 2}},
            "kubectl_preview": ["kubectl scale ... --replicas=2"],
            "risk_level": "medium",
            "approval_required": True,
            "approval": {"required": True, "state": "pending", "approved": False},
            "status": "pending_approval",
            "result": {"executed": False},
            "errors": [],
            "source": {"type": "test"},
        }
        core_node = runtime_agent_node("representative-core-agent", "Representative Core Agent")
        state = {
            "run_id": "planning-tampered-approval",
            "pending_approval": {
                "subtask_id": "subtask-001",
                "representative_agent_id": "representative-core-agent",
                "report": stored_report,
                "graph_resume": None,
            },
            "subtask_plans": [{
                "subtask_id": "subtask-001",
                "status": "pending_approval",
                "langgraph_spec": {
                    "nodes": [core_node, {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"}],
                    "edges": [{"source": "representative-core-agent", "target": "done"}],
                    "entrypoint": "representative-core-agent",
                    "terminal_nodes": ["done"],
                    "representative_agent": "representative-core-agent",
                },
            }],
        }
        tampered = {**stored_report, "command_proposal": {"operation": "scale", "parameters": {"replicas": 9}}}

        with patch.object(agent, "load_planning_state", return_value=state):
            with patch.object(agent, "approve_representative_core_report") as core:
                with self.assertRaisesRegex(ValueError, "does not match"):
                    agent.approve_core_report_from_payload(
                        {"run_id": state["run_id"], "report": tampered},
                        approved=True,
                    )

        core.assert_not_called()

    def test_approval_rejects_contractless_stored_graph_before_core_call(self) -> None:
        stored_report = {
            "intent": "Scale AMF",
            "command_proposal": {"operation": "scale"},
            "kubectl_preview": ["kubectl scale ..."],
            "risk_level": "medium",
            "approval_required": True,
            "approval": {"required": True, "state": "pending", "approved": False},
            "status": "pending_approval",
            "result": {"executed": False},
            "errors": [],
            "source": {"type": "test"},
        }
        state = {
            "run_id": "planning-contractless-approval",
            "pending_approval": {"subtask_id": "subtask-001", "representative_agent_id": "representative-core-agent", "report": stored_report},
            "subtask_plans": [{
                "subtask_id": "subtask-001",
                "status": "pending_approval",
                "langgraph_spec": {
                    "nodes": [
                        {"id": "representative-core-agent", "type": "agent", "name": "Representative Core Agent"},
                        {"id": "done", "type": "agent", "role": "completion_checker", "name": "planning-completion-check"},
                    ],
                    "edges": [{"source": "representative-core-agent", "target": "done"}],
                    "entrypoint": "representative-core-agent",
                    "terminal_nodes": ["done"],
                    "representative_agent": "representative-core-agent",
                },
            }],
        }

        with patch.object(agent, "load_planning_state", return_value=state):
            with patch.object(agent, "approve_representative_core_report") as core:
                with self.assertRaisesRegex(ValueError, "contract reference"):
                    agent.approve_core_report_from_payload(
                        {"run_id": state["run_id"], "report": stored_report},
                        approved=True,
                    )

        core.assert_not_called()


    def test_approval_resume_routes_next_monitoring_subtask_by_current_subtask_only(self) -> None:
        run_id = "planning-test-route-after-approval"
        payload = {
            "run_id": run_id,
            "intent": "Scale AMF deployment to 2 replicas and monitor health",
            "subtasks": ["Set the AMF deployment replica count to 2", "Monitor AMF replica health"],
            "golden_goal_context_used": False,
        }
        route_calls: list[str] = []
        state_path = agent._state_path(run_id)

        pending_report = {
            "intent": "Set the AMF deployment replica count to 2",
            "command_proposal": {
                "operation": "scale",
                "resource": "deployment",
                "namespace": "free5gc-v4",
                "target_scope": "single_nf",
                "nfs": ["AMF"],
                "parameters": {"replicas": 2},
                "rationale": "scale AMF",
            },
            "kubectl_preview": ["kubectl scale deployment/free5gc-free5gc-amf -n free5gc-v4 --replicas=2"],
            "risk_level": "medium",
            "approval_required": True,
            "approval": {"required": True, "state": "pending", "approved": False},
            "status": "pending_approval",
            "result": {"executed": False},
            "errors": [],
            "source": {"type": "test"},
        }
        approved_report = dict(pending_report)
        approved_report["status"] = "completed"
        approved_report["approval"] = {"required": True, "state": "approved", "approved": True}
        approved_report["result"] = {"executed": True, "success": True, "stdout": "scaled"}

        def fake_runtime(**kwargs: object) -> dict[str, object]:
            representative_agent = kwargs["representative_agent"]
            representative_id = representative_agent["id"]
            route_calls.append(str(representative_id))
            if representative_id == "representative-core-agent":
                return {
                    "status": "pending_approval",
                    "runtime": "representative-core-agent",
                    "subtask_id": kwargs["subtask_id"],
                    "subtask": kwargs["subtask"],
                    "representative_agent_id": representative_id,
                    "representative_agent_name": representative_agent["name"],
                    "message": "Representative Core Agent returned a kubectl approval request.",
                    "compiled": False,
                    "representative_core_report": pending_report,
                }
            return {
                "status": "completed",
                "runtime": "probe-agent",
                "subtask_id": kwargs["subtask_id"],
                "subtask": kwargs["subtask"],
                "representative_agent_id": representative_id,
                "representative_agent_name": representative_agent["name"],
                "message": "Probe Agent execution completed.",
                "compiled": False,
                "monitoring_report": {
                    "intent": kwargs["subtask"],
                    "command": {"operation": "replica_health", "metric": "replica_health"},
                    "evaluation_request": {"type": "thresholds"},
                    "evaluation_result": {"type": "thresholds", "status": "passed", "health_status": "healthy", "healthy": True},
                    "scope": "single_nf",
                    "metric": "replica_health",
                    "window_seconds": 60,
                    "status": "completed",
                    "results": [{"target": "AMF", "evaluation": {"health_status": "healthy", "healthy": True}}],
                    "errors": [],
                    "source": {"type": "test"},
                },
            }

        try:
            with patch.object(agent, "find_knowledge_for_subtask", return_value=knowledge_with_candidates()):
                with patch.object(agent, "build_langgraph_spec_with_llm", side_effect=fake_llm_spec):
                    with patch.object(agent, "execute_subgraph", side_effect=fake_runtime):
                        report = agent.plan_from_decomposition(payload).to_dict()

            validate_planning_report(report)
            self.assertEqual(report["run_id"], run_id)
            self.assertEqual(report["overall_status"], "pending_approval")
            self.assertEqual(route_calls, ["representative-core-agent"])
            saved_state = agent.load_planning_state(run_id)
            self.assertEqual(saved_state["current_subtask_index"], 0)
            self.assertIsNotNone(saved_state["pending_approval"])
            with patch.object(agent, "execute_subgraph", side_effect=AssertionError("pending approval must not re-run")):
                paused = agent.resume_planning_run(run_id).to_dict()
            self.assertEqual(paused["overall_status"], "pending_approval")
            self.assertEqual(route_calls, ["representative-core-agent"])

            with patch.object(agent, "find_knowledge_for_subtask", return_value=knowledge_with_candidates()):
                with patch.object(agent, "build_langgraph_spec_with_llm", side_effect=fake_llm_spec):
                    with patch.object(agent, "execute_subgraph", side_effect=fake_runtime):
                        with patch.object(agent, "approve_representative_core_report", return_value=approved_report):
                            resumed = agent.approve_core_report_from_payload(
                                {"run_id": run_id, "report": pending_report},
                                approved=True,
                                approved_by="tester",
                                reason="approved",
                            )

            validate_planning_report(resumed)
            self.assertEqual(resumed["overall_status"], "completed")
            self.assertEqual(route_calls, ["representative-core-agent", "probe-agent"])
            self.assertEqual([plan["status"] for plan in resumed["subtask_plans"]], ["completed", "completed"])
            self.assertEqual(resumed["subtask_plans"][1]["result"]["runtime"], "probe-agent")
        finally:
            state_path.unlink(missing_ok=True)


    def test_rejected_approval_does_not_reinvoke_pending_subtask(self) -> None:
        run_id = "planning-test-reject-no-rerun"
        payload = {
            "run_id": run_id,
            "intent": "Scale AMF deployment to 2 replicas",
            "subtasks": ["Set the AMF deployment replica count to 2"],
            "golden_goal_context_used": False,
        }
        state_path = agent._state_path(run_id)
        pending_report = {
            "intent": "Set the AMF deployment replica count to 2",
            "command_proposal": {
                "operation": "scale",
                "resource": "deployment",
                "namespace": "free5gc-v4",
                "target_scope": "single_nf",
                "nfs": ["AMF"],
                "parameters": {"replicas": 2},
                "rationale": "scale AMF",
            },
            "kubectl_preview": ["kubectl scale deployment/free5gc-free5gc-amf -n free5gc-v4 --replicas=2"],
            "risk_level": "medium",
            "approval_required": True,
            "approval": {"required": True, "state": "pending", "approved": False},
            "status": "pending_approval",
            "result": {"executed": False},
            "errors": [],
            "source": {"type": "test"},
        }
        rejected_report = dict(pending_report)
        rejected_report["status"] = "rejected"
        rejected_report["approval"] = {"required": True, "state": "rejected", "approved": False}

        def fake_pending_runtime(**kwargs: object) -> dict[str, object]:
            return {
                "status": "pending_approval",
                "runtime": "representative-core-agent",
                "subtask_id": kwargs["subtask_id"],
                "subtask": kwargs["subtask"],
                "representative_agent_id": "representative-core-agent",
                "representative_agent_name": "Representative Core Agent",
                "message": "Representative Core Agent returned a kubectl approval request.",
                "compiled": False,
                "representative_core_report": pending_report,
            }

        try:
            with patch.object(agent, "find_knowledge_for_subtask", return_value=knowledge_with_candidates()):
                with patch.object(agent, "build_langgraph_spec_with_llm", side_effect=fake_llm_spec):
                    with patch.object(agent, "execute_subgraph", side_effect=fake_pending_runtime):
                        report = agent.plan_from_decomposition(payload).to_dict()
            self.assertEqual(report["overall_status"], "pending_approval")

            with patch.object(agent, "execute_subgraph", side_effect=AssertionError("rejected approval must not re-run")):
                with patch.object(agent, "approve_representative_core_report", return_value=rejected_report):
                    rejected = agent.approve_core_report_from_payload(
                        {"run_id": run_id, "report": pending_report},
                        approved=False,
                        approved_by="tester",
                        reason="reject",
                    )
            self.assertEqual(rejected["overall_status"], "rejected")
            self.assertEqual(rejected["subtask_plans"][0]["status"], "rejected")
        finally:
            state_path.unlink(missing_ok=True)

    def test_monitoring_report_values_completion_uses_evaluation_result_not_health_status(self) -> None:
        result = {
            "status": "completed",
            "runtime": "probe-agent",
            "monitoring_report": {
                "intent": "Monitor AMF and SMF CPU",
                "command": {"operation": "prometheus_query_plan", "metric": "cpu_usage"},
                "evaluation_request": {"type": "report_values", "completion_rule": "data_returned"},
                "evaluation_result": {"type": "report_values", "status": "passed", "values_present": True},
                "scope": "explicit_nfs",
                "metric": "cpu_usage",
                "window_seconds": 60,
                "status": "completed",
                "results": [
                    {
                        "target": "AMF,SMF",
                        "evaluation": {"health_status": "unknown", "healthy": False},
                        "query_results": [{"target": "AMF", "first_value": 1.0}, {"target": "SMF", "first_value": 2.0}],
                    }
                ],
                "errors": [],
                "source": {"type": "test"},
            },
        }

        self.assertEqual(agent.validate_subtask_completion(result), "completed")

    def test_unknown_probe_evaluation_is_not_completed(self) -> None:
        result = {
            "status": "completed",
            "runtime": "probe-agent",
            "monitoring_report": {
                "status": "completed",
                "evaluation_result": {"type": "thresholds", "status": "unknown"},
            },
        }

        self.assertEqual(agent.validate_subtask_completion(result), "failed")
        self.assertEqual(subgraph_runtime._adapter_completion(result), "failed")

    def test_partial_probe_collection_is_not_completed_by_passed_evaluation(self) -> None:
        result = {
            "status": "partial",
            "runtime": "probe-agent",
            "monitoring_report": {
                "status": "partial",
                "evaluation_result": {"type": "thresholds", "status": "passed"},
            },
        }

        self.assertEqual(agent.validate_subtask_completion(result), "failed")
        self.assertEqual(subgraph_runtime._adapter_completion(result), "failed")


    def test_monitoring_unhealthy_retries_until_max_attempts_then_blocks(self) -> None:
        run_id = "planning-test-monitoring-retry"
        payload = {
            "run_id": run_id,
            "intent": "Monitor AMF replica health",
            "subtasks": ["Monitor AMF replica health"],
            "golden_goal_context_used": False,
        }
        calls: list[str] = []
        state_path = agent._state_path(run_id)

        def fake_unhealthy_runtime(**kwargs: object) -> dict[str, object]:
            calls.append(str(kwargs["subtask"]))
            return {
                "status": "completed",
                "runtime": "probe-agent",
                "subtask_id": kwargs["subtask_id"],
                "subtask": kwargs["subtask"],
                "representative_agent_id": "probe-agent",
                "representative_agent_name": "Probe Agent",
                "message": "Probe Agent execution completed.",
                "compiled": False,
                "monitoring_report": {
                    "intent": kwargs["subtask"],
                    "command": {"operation": "replica_health", "metric": "replica_health"},
                    "evaluation_request": {"type": "thresholds"},
                    "evaluation_result": {"type": "thresholds", "status": "failed", "health_status": "unhealthy", "healthy": False},
                    "scope": "single_nf",
                    "metric": "replica_health",
                    "window_seconds": 60,
                    "status": "completed",
                    "results": [{"target": "AMF", "evaluation": {"health_status": "unhealthy", "healthy": False}}],
                    "errors": [],
                    "source": {"type": "test"},
                },
            }

        try:
            with patch.dict(os.environ, {"PLANNING_MAX_ATTEMPTS": "2"}):
                with patch.object(agent, "find_knowledge_for_subtask", return_value=knowledge_with_candidates()):
                    with patch.object(agent, "build_langgraph_spec_with_llm", side_effect=fake_llm_spec):
                        with patch.object(agent, "execute_subgraph", side_effect=fake_unhealthy_runtime):
                            report = agent.plan_from_decomposition(payload).to_dict()

            validate_planning_report(report)
            self.assertEqual(report["overall_status"], "blocked")
            self.assertEqual(len(calls), 2)
            first_plan = report["subtask_plans"][0]
            self.assertEqual(first_plan["status"], "blocked")
            self.assertEqual([attempt["status"] for attempt in first_plan["attempts"]], ["failed", "failed"])
        finally:
            state_path.unlink(missing_ok=True)


    def test_planning_writes_telemetry_events_for_pending_approval(self) -> None:
        from agent_ops.telemetry import read_events

        run_id = "planning-test-telemetry-pending"
        payload = {
            "run_id": run_id,
            "intent": "Scale SMF deployment to 2 replicas",
            "subtasks": ["Set the SMF deployment replica count to 2"],
            "golden_goal_context_used": False,
        }

        def fake_runtime(**kwargs: object) -> dict[str, object]:
            return {
                "status": "pending_approval",
                "runtime": "representative-core-agent",
                "subtask_id": kwargs["subtask_id"],
                "subtask": kwargs["subtask"],
                "representative_agent_id": "representative-core-agent",
                "representative_agent_name": "Representative Core Agent",
                "message": "Representative Core Agent returned a kubectl approval request.",
                "compiled": False,
                "representative_core_report": {
                    "intent": "Scale SMF deployment to 2 replicas",
                    "command_proposal": {
                        "operation": "scale",
                        "resource": "deployment",
                        "namespace": "free5gc-v4",
                        "target_scope": "single_nf",
                        "nfs": ["SMF"],
                        "parameters": {"replicas": 2},
                        "rationale": "scale SMF",
                    },
                    "kubectl_preview": ["kubectl scale deployment/free5gc-free5gc-smf -n free5gc-v4 --replicas=2"],
                    "risk_level": "medium",
                    "approval_required": True,
                    "approval": {"required": True, "state": "pending", "approved": False},
                    "status": "pending_approval",
                    "result": {"executed": False},
                    "errors": [],
                    "source": {"type": "test"},
                },
            }

        with patch.object(agent, "find_knowledge_for_subtask", return_value=knowledge_with_candidates()):
            with patch.object(agent, "build_langgraph_spec_with_llm", side_effect=fake_llm_spec):
                with patch.object(agent, "execute_subgraph", side_effect=fake_runtime):
                    report = agent.plan_from_decomposition(payload).to_dict()

        self.assertEqual(report["overall_status"], "pending_approval")
        event_types = [event["event_type"] for event in read_events(run_id)]
        self.assertIn("run_started", event_types)
        self.assertIn("subtask_started", event_types)
        self.assertIn("knowledge_lookup", event_types)
        self.assertIn("subgraph_selected", event_types)
        self.assertIn("pending_approval", event_types)

    def test_fastapi_resume_rejects_external_last_agent_result(self) -> None:
        client = TestClient(server.app)

        class DummyReport:
            def to_dict(self) -> dict[str, object]:
                return {
                    "run_id": "planning-test-resume",
                    "intent": "Monitor AMF",
                    "subtask_plans": [
                        {
                            "subtask_id": "subtask-001",
                            "subtask": "Monitor AMF",
                            "langgraph_spec": fake_llm_spec(subtask_id="subtask-001", subtask="Monitor AMF"),
                            "representative_agent": {"id": "probe-agent", "name": "Probe Agent"},
                            "attempts": [],
                            "status": "completed",
                            "result": {"status": "completed"},
                            "knowledge_context_used": False,
                        }
                    ],
                    "overall_status": "completed",
                }

        with patch.object(server, "resume_planning_run", return_value=DummyReport()) as resume_runtime:
            response = client.post(
                "/resume",
                json={"run_id": "planning-test-resume", "last_agent_result": {"status": "completed"}},
            )

        self.assertEqual(response.status_code, 400)
        self.assertIn("not accepted", response.json()["detail"])
        resume_runtime.assert_not_called()

    def test_fastapi_approve_matches_schema(self) -> None:
        client = TestClient(server.app)
        pending_report = {
            "intent": "Restart AMF",
            "command_proposal": {
                "operation": "rollout_restart",
                "resource": "deployment",
                "namespace": "free5gc-v4",
                "target_scope": "single_nf",
                "nfs": ["AMF"],
                "parameters": {},
                "rationale": "restart AMF",
            },
            "kubectl_preview": ["kubectl rollout restart deployment/free5gc-free5gc-amf -n free5gc-v4"],
            "risk_level": "high",
            "approval_required": True,
            "approval": {"required": True, "state": "pending", "approved": False},
            "status": "pending_approval",
            "result": {"executed": False},
            "errors": [],
            "source": {"type": "test"},
        }
        rejected_report = dict(pending_report)
        rejected_report["status"] = "rejected"
        rejected_report["approval"] = {"required": True, "state": "rejected", "approved": False}

        with patch.object(server, "approve_core_report_from_payload", return_value=rejected_report):
            response = client.post(
                "/approve",
                json={"run_id": "planning-test-approve", "report": pending_report, "approved": False, "approved_by": "tester"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "rejected")

    def test_fastapi_invoke_matches_schema(self) -> None:
        client = TestClient(server.app)

        with patch.object(agent, "build_langgraph_spec_with_llm", side_effect=fake_llm_spec):
            response = client.post("/invoke", json=sample_decomposition())

        self.assertEqual(response.status_code, 200)
        report = response.json()
        validate_planning_report(report)
        self.assertEqual(report["overall_status"], "blocked")


if __name__ == "__main__":
    unittest.main()
