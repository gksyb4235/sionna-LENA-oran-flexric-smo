from __future__ import annotations

import contextlib
import io
import os
import unittest
from unittest.mock import patch

import agent
import server
from fastapi.testclient import TestClient
from schemas import validate_core_agent_report, validate_core_command_proposal
from tools.kubectl_policy import evaluate_policy


class RepresentativeCoreTests(unittest.TestCase):
    def test_validate_command_proposal(self) -> None:
        proposal = validate_core_command_proposal(
            {
                "operation": "rollout_restart",
                "resource": "deployment",
                "namespace": "free5gc-v4",
                "target_scope": "single_nf",
                "nfs": ["AMF"],
                "parameters": {},
                "rationale": "restart AMF",
            }
        )

        self.assertEqual(proposal.operation, "rollout_restart")
        self.assertEqual(proposal.nfs, ["AMF"])

    def test_risk_classification(self) -> None:
        read_only = validate_core_command_proposal(
            {"operation": "get", "resource": "pod", "namespace": "free5gc-v4", "target_scope": "single_nf", "nfs": ["AMF"], "parameters": {}, "rationale": "read"}
        )
        medium = validate_core_command_proposal(
            {
                "operation": "annotate",
                "resource": "deployment",
                "namespace": "free5gc-v4",
                "target_scope": "single_nf",
                "nfs": ["AMF"],
                "parameters": {"key": "agent.test/owner", "value": "jaechan"},
                "rationale": "annotate",
            }
        )
        high = validate_core_command_proposal(
            {"operation": "delete", "resource": "pod", "namespace": "free5gc-v4", "target_scope": "single_nf", "nfs": ["AMF"], "parameters": {}, "rationale": "delete"}
        )
        scale_zero = validate_core_command_proposal(
            {"operation": "scale", "resource": "deployment", "namespace": "free5gc-v4", "target_scope": "single_nf", "nfs": ["AMF"], "parameters": {"replicas": 0}, "rationale": "stop"}
        )

        self.assertEqual(evaluate_policy(read_only).risk_level, "low")
        self.assertEqual(evaluate_policy(medium).risk_level, "medium")
        self.assertEqual(evaluate_policy(high).risk_level, "high")
        self.assertEqual(evaluate_policy(scale_zero).risk_level, "high")

    def test_scale_string_replicas_builds_preview(self) -> None:
        proposal = validate_core_command_proposal(
            {
                "operation": "scale",
                "resource": "deployment",
                "namespace": "free5gc-v4",
                "target_scope": "single_nf",
                "nfs": ["AMF"],
                "parameters": {"replicas": "2"},
                "rationale": "scale AMF to two replicas",
            }
        )

        policy = evaluate_policy(proposal)

        self.assertEqual(policy.errors, [])
        self.assertEqual(policy.risk_level, "medium")
        self.assertEqual(policy.kubectl_preview, ["kubectl scale deployment/free5gc-free5gc-amf -n free5gc-v4 --replicas=2"])


    def test_deterministic_scale_intent_extracts_requested_replicas(self) -> None:
        old = os.environ.get("REPRESENTATIVE_CORE_USE_DETERMINISTIC_TEST_COMMAND")
        os.environ["REPRESENTATIVE_CORE_USE_DETERMINISTIC_TEST_COMMAND"] = "1"
        try:
            report = agent.run_representative_core("Scale AMF deployment to 3 replicas").to_dict()
        finally:
            if old is None:
                os.environ.pop("REPRESENTATIVE_CORE_USE_DETERMINISTIC_TEST_COMMAND", None)
            else:
                os.environ["REPRESENTATIVE_CORE_USE_DETERMINISTIC_TEST_COMMAND"] = old

        validate_core_agent_report(report)
        self.assertEqual(report["command_proposal"]["parameters"]["replicas"], 3)
        self.assertEqual(report["kubectl_preview"], ["kubectl scale deployment/free5gc-free5gc-amf -n free5gc-v4 --replicas=3"])

    def test_deterministic_scale_intent_extracts_count_before_nf_name(self) -> None:
        old = os.environ.get("REPRESENTATIVE_CORE_USE_DETERMINISTIC_TEST_COMMAND")
        os.environ["REPRESENTATIVE_CORE_USE_DETERMINISTIC_TEST_COMMAND"] = "1"
        try:
            report = agent.run_representative_core("Scale with 2 AMFs").to_dict()
        finally:
            if old is None:
                os.environ.pop("REPRESENTATIVE_CORE_USE_DETERMINISTIC_TEST_COMMAND", None)
            else:
                os.environ["REPRESENTATIVE_CORE_USE_DETERMINISTIC_TEST_COMMAND"] = old

        validate_core_agent_report(report)
        self.assertEqual(report["command_proposal"]["parameters"]["replicas"], 2)
        self.assertEqual(report["kubectl_preview"], ["kubectl scale deployment/free5gc-free5gc-amf -n free5gc-v4 --replicas=2"])

    def test_pending_report_does_not_execute(self) -> None:
        proposal = validate_core_command_proposal(
            {"operation": "rollout_restart", "resource": "deployment", "namespace": "free5gc-v4", "target_scope": "single_nf", "nfs": ["AMF"], "parameters": {}, "rationale": "restart"}
        )

        report = agent.build_pending_report("restart AMF", proposal).to_dict()

        validate_core_agent_report(report)
        self.assertEqual(report["status"], "pending_approval")
        self.assertTrue(report["approval_required"])
        self.assertEqual(report["risk_level"], "high")
        self.assertFalse(report["result"]["executed"])
        self.assertIn("kubectl rollout restart deployment/free5gc-free5gc-amf", report["kubectl_preview"][0])

    def test_unknown_nf_blocks_as_unclear_target(self) -> None:
        proposal = validate_core_command_proposal(
            {"operation": "get", "resource": "pod", "namespace": "free5gc-v4", "target_scope": "single_nf", "nfs": ["WEBUI"], "parameters": {}, "rationale": "bad target"}
        )

        report = agent.build_pending_report("get WEBUI", proposal).to_dict()

        self.assertEqual(report["status"], "blocked")
        self.assertTrue(report["errors"])

    def test_rationale_punctuation_does_not_trigger_shell_injection_block(self) -> None:
        proposal = validate_core_command_proposal(
            {
                "operation": "get",
                "resource": "pod",
                "namespace": "free5gc-v4",
                "target_scope": "single_nf",
                "nfs": ["AMF"],
                "parameters": {},
                "rationale": "Inspect AMF pod metadata; this text is not executed.",
            }
        )

        report = agent.build_pending_report("get AMF", proposal).to_dict()

        self.assertEqual(report["status"], "pending_approval")
        self.assertEqual(report["kubectl_preview"], ["kubectl get pods -l nf=amf -n free5gc-v4"])

    def test_reject_does_not_execute(self) -> None:
        proposal = validate_core_command_proposal(
            {"operation": "get", "resource": "pod", "namespace": "free5gc-v4", "target_scope": "single_nf", "nfs": ["AMF"], "parameters": {}, "rationale": "read"}
        )
        pending = agent.build_pending_report("get AMF", proposal).to_dict()

        with patch("agent.execute_kubectl_commands") as executor:
            rejected = agent.approve_report(pending, approved=False, approved_by="tester", reason="no").to_dict()

        executor.assert_not_called()
        self.assertEqual(rejected["status"], "rejected")
        self.assertFalse(rejected["result"]["executed"])

    def test_approve_executes_via_adapter(self) -> None:
        proposal = validate_core_command_proposal(
            {"operation": "get", "resource": "pod", "namespace": "free5gc-v4", "target_scope": "single_nf", "nfs": ["AMF"], "parameters": {}, "rationale": "read"}
        )
        pending = agent.build_pending_report("get AMF", proposal).to_dict()

        with patch("agent.execute_kubectl_commands", return_value={"executed": True, "command_count": 1, "results": [], "success": True}) as executor:
            approved = agent.approve_report(pending, approved=True, approved_by="tester", reason="ok").to_dict()

        executor.assert_called_once()
        self.assertEqual(approved["status"], "completed")
        self.assertTrue(approved["result"]["executed"])

    def test_cli_wraps_output(self) -> None:
        old = os.environ.get("REPRESENTATIVE_CORE_USE_DETERMINISTIC_TEST_COMMAND")
        os.environ["REPRESENTATIVE_CORE_USE_DETERMINISTIC_TEST_COMMAND"] = "1"
        output = io.StringIO()
        try:
            with contextlib.redirect_stdout(output):
                self.assertEqual(agent.main(["restart AMF"]), 0)
        finally:
            if old is None:
                os.environ.pop("REPRESENTATIVE_CORE_USE_DETERMINISTIC_TEST_COMMAND", None)
            else:
                os.environ["REPRESENTATIVE_CORE_USE_DETERMINISTIC_TEST_COMMAND"] = old

        rendered = output.getvalue()
        self.assertIn("------representative-core-agent------", rendered)
        self.assertIn('"status": "pending_approval"', rendered)
        self.assertTrue(rendered.strip().endswith("----end----"))

    def test_fastapi_invoke_matches_schema(self) -> None:
        client = TestClient(server.app)
        old = os.environ.get("REPRESENTATIVE_CORE_USE_DETERMINISTIC_TEST_COMMAND")
        os.environ["REPRESENTATIVE_CORE_USE_DETERMINISTIC_TEST_COMMAND"] = "1"
        try:
            response = client.post("/invoke", json={"intent": "restart AMF"})
        finally:
            if old is None:
                os.environ.pop("REPRESENTATIVE_CORE_USE_DETERMINISTIC_TEST_COMMAND", None)
            else:
                os.environ["REPRESENTATIVE_CORE_USE_DETERMINISTIC_TEST_COMMAND"] = old

        self.assertEqual(response.status_code, 200)
        report = response.json()
        validate_core_agent_report(report)
        self.assertEqual(report["status"], "pending_approval")


if __name__ == "__main__":
    unittest.main()
