from __future__ import annotations

import os
import unittest
from unittest.mock import patch

import agent
import server
from schemas import validate_monitoring_report


def expected_request() -> dict[str, object]:
    return {
        "type": "thresholds",
        "schedule": {"duration_seconds": 0},
        "checks": [{
            "check_id": "opaque-check",
            "observation": "AMF available replica count",
            "expected": {"operator": "==", "value": 2},
        }],
    }


def planning_report(status: str = "completed") -> dict[str, object]:
    return {
        "run_id": "planning-monitor-test",
        "intent": "Scale AMF to 2 replicas and verify it",
        "overall_status": status,
        "subtask_plans": [
            {
                "subtask_id": "subtask-001",
                "subtask": "Scale AMF to 2 replicas",
                "langgraph_spec": {
                    "evaluation_request": expected_request()
                },
                "representative_agent": {"id": "representative-core-agent"},
                "attempts": [{"attempt_number": 1}],
                "status": status,
                "result": {
                    "status": status,
                    "runtime": "representative-core-agent",
                    "representative_core_report": {
                        "status": status,
                        "command_proposal": {
                            "operation": "scale",
                            "resource": "deployment",
                            "namespace": "free5gc-v4",
                            "target_scope": "single_nf",
                            "nfs": ["AMF"],
                            "parameters": {"replicas": 2},
                            "rationale": "scale AMF",
                        },
                        "result": {"success": status == "completed"},
                    },
                },
            }
        ],
    }


def probe_report(evaluation_status: str = "passed", collection_status: str = "completed") -> dict[str, object]:
    return {
        "intent": "verify AMF",
        "command": {},
        "evaluation_request": expected_request(),
        "evaluation_result": {
            "type": "thresholds",
            "status": evaluation_status,
            "checks": [{"check_id": "opaque-check", "observation": "AMF available replica count", "ok": evaluation_status == "passed", "valid": True, "actual": 1, "expected": 2, "operator": "=="}],
            "query_values": {"opaque-check": 1},
        },
        "scope": "single_nf",
        "metric": "replica_health",
        "window_seconds": 60,
        "status": collection_status,
        "results": [{}] if collection_status != "blocked" else [],
        "errors": [],
        "source": {"type": "grafana"},
    }


class MonitoringAgentTests(unittest.TestCase):
    def test_http_invoke_requires_shared_token(self) -> None:
        with patch.dict(os.environ, {"MONITORING_AGENT_TOKEN": "test-token"}):
            with patch.object(server, "load_env_file"):
                with self.assertRaises(Exception) as raised:
                    server._require_token(None)

        self.assertEqual(raised.exception.status_code, 401)

    def test_failed_expectation_notifies_planning_with_probe_evidence(self) -> None:
        planner_response = {"run_id": "planning-monitor-test", "overall_status": "pending_approval", "subtask_plans": []}
        with patch.object(agent, "invoke_probe_agent", return_value=probe_report("failed")):
            with patch.object(agent, "notify_planning_agent", return_value=planner_response) as notify:
                report = agent.run_monitoring({"planning_report": planning_report()}).to_dict()

        validate_monitoring_report(report)
        self.assertEqual(report["status"], "replan_required")
        self.assertEqual(report["feedback"]["observed"]["status"], "failed")
        self.assertEqual(report["feedback"]["previous_action"]["operation"], "scale")
        self.assertEqual(report["planner_response"], planner_response)
        notify.assert_called_once_with(report["feedback"])

    def test_response_delivery_returns_feedback_without_notifying_planning(self) -> None:
        with patch.object(agent, "invoke_probe_agent", return_value=probe_report("failed")):
            with patch.object(agent, "notify_planning_agent") as notify:
                report = agent.run_monitoring({
                    "planning_report": planning_report(),
                    "notification_delivery": "response",
                }).to_dict()

        self.assertEqual(report["status"], "replan_required")
        self.assertEqual(report["feedback"]["observed"]["status"], "failed")
        self.assertIsNone(report["planner_response"])
        notify.assert_not_called()

    def test_notify_false_never_notifies_with_direct_delivery(self) -> None:
        with patch.object(agent, "invoke_probe_agent", return_value=probe_report("failed")):
            with patch.object(agent, "notify_planning_agent") as notify:
                report = agent.run_monitoring({
                    "planning_report": planning_report(),
                    "notify_planner": False,
                    "notification_delivery": "direct",
                }).to_dict()

        self.assertEqual(report["status"], "replan_required")
        self.assertIsNotNone(report["feedback"])
        self.assertIsNone(report["planner_response"])
        notify.assert_not_called()

    def test_invalid_notification_delivery_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "notification_delivery"):
            agent.run_monitoring({
                "planning_report": planning_report(),
                "notification_delivery": "queued",
            })

    def test_plan_fingerprint_ignores_runtime_fields_but_binds_graph(self) -> None:
        plan = planning_report()["subtask_plans"][0]
        fingerprint = agent.plan_fingerprint(plan)

        plan["status"] = "blocked"
        plan["attempts"].append({"attempt_number": 2})
        plan["result"] = {"status": "blocked"}
        self.assertEqual(agent.plan_fingerprint(plan), fingerprint)

        plan["langgraph_spec"]["evaluation_request"]["checks"][0]["expected"]["value"] = 3
        self.assertNotEqual(agent.plan_fingerprint(plan), fingerprint)

    def test_passed_expectation_is_healthy_and_does_not_notify(self) -> None:
        with patch.object(agent, "invoke_probe_agent", return_value=probe_report("passed")):
            with patch.object(agent, "notify_planning_agent") as notify:
                report = agent.run_monitoring({"planning_report": planning_report()}).to_dict()

        self.assertEqual(report["status"], "healthy")
        self.assertIsNone(report["feedback"])
        notify.assert_not_called()

    def test_duration_expectation_samples_the_full_forward_interval(self) -> None:
        planning = planning_report()
        expected = expected_request()
        expected["schedule"] = {"duration_seconds": 1, "interval_seconds": 1}
        planning["subtask_plans"][0]["langgraph_spec"]["evaluation_request"] = expected
        sample = probe_report("passed")
        sample["evaluation_request"] = expected

        with patch.object(agent, "invoke_probe_agent", side_effect=[sample, sample]) as probe:
            with patch.object(agent.time, "monotonic", side_effect=[0.0, 0.0, 1.0]):
                with patch.object(agent.time, "sleep") as sleep:
                    report = agent.run_monitoring({"planning_report": planning, "notify_planner": False}).to_dict()

        self.assertEqual(report["status"], "healthy")
        self.assertEqual(report["probe_report"]["monitoring_duration_seconds"], 1)
        self.assertEqual(len(report["probe_report"]["monitoring_samples"]), 2)
        self.assertEqual(probe.call_count, 2)
        sleep.assert_called_once_with(1)

    def test_missing_planning_schedule_does_not_probe_or_replan(self) -> None:
        planning = planning_report()
        expected = expected_request()
        expected.pop("schedule")
        planning["subtask_plans"][0]["langgraph_spec"]["evaluation_request"] = expected

        with patch.object(agent, "invoke_probe_agent") as probe:
            report = agent.run_monitoring({"planning_report": planning, "notify_planner": False}).to_dict()

        self.assertEqual(report["status"], "observation_unavailable")
        self.assertEqual(report["errors"][0]["stage"], "planning_contract")
        self.assertIn("schedule", report["errors"][0]["error"])
        self.assertIsNone(report["feedback"])
        probe.assert_not_called()

    def test_planning_schedule_has_no_monitoring_agent_maximum(self) -> None:
        self.assertEqual(
            agent._monitoring_schedule({"schedule": {"duration_seconds": 7200, "interval_seconds": 600}}),
            (7200, 600),
        )

    def test_pending_approval_does_not_probe_or_notify(self) -> None:
        with patch.object(agent, "invoke_probe_agent") as probe:
            with patch.object(agent, "notify_planning_agent") as notify:
                report = agent.run_monitoring({"planning_report": planning_report("pending_approval")}).to_dict()

        self.assertEqual(report["status"], "awaiting_execution")
        probe.assert_not_called()
        notify.assert_not_called()

    def test_unknown_probe_evaluation_does_not_replan(self) -> None:
        with patch.object(agent, "invoke_probe_agent", return_value=probe_report("unknown")):
            with patch.object(agent, "notify_planning_agent") as notify:
                report = agent.run_monitoring({"planning_report": planning_report()}).to_dict()

        self.assertEqual(report["status"], "observation_unavailable")
        self.assertIsNone(report["feedback"])
        notify.assert_not_called()

    def test_failed_execution_is_healthy_when_current_probe_passes(self) -> None:
        with patch.object(agent, "invoke_probe_agent", return_value=probe_report("passed")):
            report = agent.run_monitoring({"planning_report": planning_report("blocked"), "notify_planner": False}).to_dict()

        self.assertEqual(report["status"], "healthy")
        self.assertIsNone(report["feedback"])

    def test_failed_execution_replans_only_after_current_probe_mismatch(self) -> None:
        with patch.object(agent, "invoke_probe_agent", return_value=probe_report("failed")):
            report = agent.run_monitoring({"planning_report": planning_report("blocked"), "notify_planner": False}).to_dict()

        self.assertEqual(report["status"], "replan_required")
        self.assertEqual(report["feedback"]["reason"], "expected_network_state_not_observed")
        self.assertEqual(report["feedback"]["observed"]["checks"][0]["check_id"], "opaque-check")

    def test_blocked_probe_observation_is_not_an_execution_failure(self) -> None:
        planning = planning_report("blocked")
        plan = planning["subtask_plans"][0]
        plan["representative_agent"] = {"id": "probe-agent"}
        plan["result"] = {"status": "blocked", "runtime": "probe-agent"}
        with patch.object(agent, "invoke_probe_agent", return_value=probe_report("unknown", "blocked")):
            with patch.object(agent, "notify_planning_agent") as notify:
                report = agent.run_monitoring({"planning_report": planning}).to_dict()

        self.assertEqual(report["status"], "observation_unavailable")
        self.assertIsNone(report["feedback"])
        notify.assert_not_called()

    def test_mismatched_probe_evaluation_contract_does_not_replan(self) -> None:
        wrong_report = probe_report("failed")
        wrong_report["evaluation_request"] = {"type": "report_values", "completion_rule": "data_returned"}
        with patch.object(agent, "invoke_probe_agent", return_value=wrong_report):
            with patch.object(agent, "notify_planning_agent") as notify:
                report = agent.run_monitoring({"planning_report": planning_report()}).to_dict()

        self.assertEqual(report["status"], "observation_unavailable")
        self.assertIsNone(report["feedback"])
        notify.assert_not_called()


if __name__ == "__main__":
    unittest.main()
