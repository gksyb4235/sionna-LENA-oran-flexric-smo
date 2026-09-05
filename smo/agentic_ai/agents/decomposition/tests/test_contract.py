from __future__ import annotations

import contextlib
import io
import json
import os
import unittest
from pathlib import Path

import agent
from schemas import DecompositionResult, OUTPUT_KEYS, validate_output_contract
from tools.decomposition_pattern_search import (
    DecompositionPatternContext,
    find_decomposition_pattern_context,
    validate_decomposition_semantics,
)
from tools.planning_handoff import invoke_planning_agent, validate_planning_report


class DecompositionContractTests(unittest.TestCase):
    def test_output_contract_has_exact_keys(self) -> None:
        result = DecompositionResult(
            intent="Build a report",
            subtasks=["Identify report requirements", "Draft the report"],
            golden_goal_context_used=False,
        )

        data = result.to_dict()

        self.assertEqual(tuple(data.keys()), OUTPUT_KEYS)
        validate_output_contract(data)
        self.assertEqual(json.loads(json.dumps(data)), data)

    def test_from_model_payload_discards_extra_fields(self) -> None:
        result = DecompositionResult.from_model_payload(
            {
                "intent": "ignored model intent",
                "subtasks": ["Collect inputs", "Create subtasks"],
                "golden_goal_context_used": True,
                "warnings": ["must not leak"],
            },
            intent="original user intent",
            golden_goal_context_used=False,
        )

        data = result.to_dict()

        self.assertEqual(tuple(data.keys()), OUTPUT_KEYS)
        self.assertEqual(data["intent"], "original user intent")
        self.assertFalse(data["golden_goal_context_used"])

    def test_extract_json_object_uses_contract_object(self) -> None:
        payload = agent.extract_json_object(
            '{"id": "reasoning", "content": []}\n'
            '{"intent": "x", "subtasks": ["a"], "golden_goal_context_used": false}\nextra'
        )

        self.assertEqual(payload["intent"], "x")
        self.assertEqual(payload["subtasks"], ["a"])

    def test_agent_instruction_files_are_loadable(self) -> None:
        instructions = agent.load_agent_instructions()
        skill_path = Path(agent.PROJECT_ROOT) / "skills" / "decomposition-tools" / "SKILL.md"
        monitoring_skill_path = Path(agent.PROJECT_ROOT) / "skills" / "monitoring-decomposition" / "SKILL.md"

        self.assertIn("Output Contract", instructions)
        self.assertNotIn("Agent Boundary Rules", instructions)
        self.assertTrue(skill_path.exists())
        self.assertIn("preloaded Decomposition Knowledge DB patterns", skill_path.read_text(encoding="utf-8"))
        self.assertTrue(monitoring_skill_path.exists())
        monitoring_skill = monitoring_skill_path.read_text(encoding="utf-8")
        self.assertIn("one requested monitoring measurement as one executable subtask", monitoring_skill)
        self.assertIn("Do not turn one monitoring request into procedural subtasks", monitoring_skill)
        self.assertIn("Monitoring Agent uses its observation tools", monitoring_skill)

    @staticmethod
    def maintain_pattern() -> dict[str, object]:
        return {
            "pattern_id": "maintain_expected_state-v1",
            "status": "active",
            "intent_class": "maintain_expected_state",
            "summary": "Maintain X in Y for N minutes.",
            "keywords": ["유지", "maintain"],
            "required_semantics": ["monitor", "notify"],
            "decomposition_rules": ["Monitor and notify Planning."],
            "validation": {
                "preserve_numbers": True,
                "preserve_uppercase_tokens": True,
                "required_concepts": {
                    "monitoring_agent": ["모니터링 에이전트", "monitoring agent"],
                    "observation_tools": ["관측 도구", "observation tools"],
                    "mismatch": ["만족하지", "mismatch"],
                    "planning_agent": ["플래닝 에이전트", "planning agent"],
                    "notification": ["알린", "notify"],
                },
            },
            "examples": [{
                "intent": "NF를 2개로 만들고 5분 유지",
                "expected_subtasks": ["NF를 2개로 만든다", "5분 동안 모니터링하고 불일치하면 알린다"],
            }],
            "version": "1.0",
        }

    def test_maintain_pattern_semantics_require_monitoring_and_planning_notification(self) -> None:
        valid = [
            "AMF를 2개로 구성한다.",
            "모니터링 에이전트가 관측 도구로 5분 동안 AMF가 2개인지 확인하고, 조건이 만족하지 않으면 플래닝 에이전트에 알린다.",
        ]
        validate_decomposition_semantics(
            "AMF 2개로 만들고 5분동안 유지시켜줘",
            valid,
            [self.maintain_pattern()],
        )

        with self.assertRaisesRegex(ValueError, "missing_concept:monitoring_agent"):
            validate_decomposition_semantics(
                "AMF 2개로 만들고 5분동안 유지시켜줘",
                ["AMF를 2개로 구성하고 5분 동안 유지한다."],
                [self.maintain_pattern()],
            )

    def test_run_decomposition_does_not_validate_pattern_semantics(self) -> None:
        responses = [
            '{"intent":"ignored","subtasks":["AMF를 2개로 만들고 5분 유지한다."],"golden_goal_context_used":true}',
        ]
        calls: list[dict[str, object]] = []

        class Message:
            def __init__(self, content: str) -> None:
                self.content = content

        class FakeAgent:
            def invoke(self, payload: dict[str, object]) -> dict[str, object]:
                calls.append(payload)
                return {"messages": [Message(responses[len(calls) - 1])]}

        original_find = agent.find_decomposition_pattern_context
        original_create = agent.create_decomposition_agent
        try:
            agent.find_decomposition_pattern_context = lambda _intent: DecompositionPatternContext(
                used=True,
                documents=[self.maintain_pattern()],
            )
            agent.create_decomposition_agent = FakeAgent
            result = agent.run_decomposition("AMF 2개로 만들고 5분동안 유지시켜줘")
        finally:
            agent.find_decomposition_pattern_context = original_find
            agent.create_decomposition_agent = original_create

        self.assertEqual(len(calls), 1)
        self.assertTrue(result.golden_goal_context_used)
        self.assertEqual(result.subtasks, ["AMF를 2개로 만들고 5분 유지한다."])

    def test_pending_representative_report_prompt_rejects(self) -> None:
        report = {
            "intent": "Scale AMF deployment to 2 replicas",
            "command_proposal": {"operation": "scale"},
            "kubectl_preview": ["kubectl scale deployment/free5gc-free5gc-amf -n free5gc-v4 --replicas=2"],
            "risk_level": "medium",
            "approval_required": True,
            "approval": {"required": True},
            "status": "pending_approval",
            "result": {"executed": False},
            "errors": [],
            "source": {"type": "test"},
        }
        calls = []
        rendered = []
        original = agent.approve_representative_core_report

        def fake_approve(pending_report, *, approved, approved_by=None, reason=None):
            calls.append((pending_report, approved, approved_by, reason))
            return {**pending_report, "status": "rejected", "result": {"executed": False}}

        try:
            agent.approve_representative_core_report = fake_approve
            results = agent.prompt_for_representative_approvals(
                [report],
                input_fn=lambda prompt: "n",
                output_fn=rendered.append,
            )
        finally:
            agent.approve_representative_core_report = original

        self.assertEqual(len(calls), 1)
        self.assertFalse(calls[0][1])
        self.assertEqual(results[0]["status"], "rejected")
        output = "\n".join(rendered)
        self.assertIn("------approval-request------", output)
        self.assertIn("------representative-core-agent-approval------", output)

    def test_duplicate_mutating_representative_report_is_not_prompted_twice(self) -> None:
        report = {
            "intent": "Scale AMF deployment to 2 replicas",
            "command_proposal": {"operation": "scale"},
            "kubectl_preview": ["kubectl scale deployment/free5gc-free5gc-amf -n free5gc-v4 --replicas=2"],
            "risk_level": "medium",
            "approval_required": True,
            "approval": {"required": True},
            "status": "pending_approval",
            "result": {"executed": False},
            "errors": [],
            "source": {"type": "test"},
        }
        calls = []
        rendered = []
        original = agent.approve_representative_core_report

        def fake_approve(pending_report, *, approved, approved_by=None, reason=None):
            calls.append((pending_report, approved, approved_by, reason))
            return {**pending_report, "status": "rejected", "result": {"executed": False}}

        try:
            agent.approve_representative_core_report = fake_approve
            agent.prompt_for_representative_approvals(
                [report, report],
                input_fn=lambda prompt: "n",
                output_fn=rendered.append,
            )
        finally:
            agent.approve_representative_core_report = original

        output = "\n".join(rendered)
        self.assertEqual(len(calls), 1)
        self.assertIn("------approval-request-duplicate-skipped------", output)


    def test_remaining_decomposition_after_pending_planning(self) -> None:
        decomposition_output = {
            "intent": "Scale and monitor AMF",
            "subtasks": ["Scale AMF to 2 replicas", "Monitor AMF replica health"],
            "golden_goal_context_used": False,
        }
        planning_output = {
            "intent": "Scale and monitor AMF",
            "subtask_plans": [{"subtask_id": "subtask-001", "subtask": "Scale AMF to 2 replicas"}],
            "overall_status": "pending_approval",
        }

        remaining = agent.remaining_decomposition_after_planning(decomposition_output, planning_output)

        self.assertEqual(
            remaining,
            {
                "intent": "Scale and monitor AMF",
                "subtasks": ["Monitor AMF replica health"],
                "golden_goal_context_used": False,
            },
        )

    def test_approval_results_executed_successfully(self) -> None:
        self.assertTrue(agent.approval_results_executed_successfully([{
            "status": "completed",
            "result": {"executed": True},
        }]))
        self.assertFalse(agent.approval_results_executed_successfully([{
            "status": "rejected",
            "result": {"executed": False},
        }]))

    def test_planning_handoff_invokes_local_planning_agent(self) -> None:
        old_flag = os.environ.get("PLANNING_USE_DETERMINISTIC_TEST_SPEC")
        os.environ["PLANNING_USE_DETERMINISTIC_TEST_SPEC"] = "1"
        try:
            planning_report = invoke_planning_agent(
                {
                    "intent": "Launch a new mobile app",
                    "subtasks": ["Define launch goals"],
                    "golden_goal_context_used": False,
                }
            )
        finally:
            if old_flag is None:
                os.environ.pop("PLANNING_USE_DETERMINISTIC_TEST_SPEC", None)
            else:
                os.environ["PLANNING_USE_DETERMINISTIC_TEST_SPEC"] = old_flag

        validate_planning_report(planning_report)
        self.assertEqual(planning_report["overall_status"], "blocked")
        self.assertEqual(planning_report["subtask_plans"][0]["status"], "blocked")

    def test_cli_defaults_to_planning_handoff(self) -> None:
        calls: list[str] = []
        original_decomposition = agent.run_decomposition
        original_planning = agent.invoke_planning_agent

        class DummyDecomposition:
            def to_dict(self) -> dict[str, object]:
                return {
                    "intent": "Launch a service",
                    "subtasks": ["Define scope"],
                    "golden_goal_context_used": False,
                }

        def fake_decomposition(intent: str) -> DummyDecomposition:
            calls.append(f"decomposition:{intent}")
            return DummyDecomposition()

        def fake_planning(payload: dict[str, object]) -> dict[str, object]:
            calls.append(f"planning:{payload['intent']}")
            return {
                "intent": payload["intent"],
                "subtask_plans": [],
                "overall_status": "mock_completed",
            }

        output = io.StringIO()
        try:
            agent.run_decomposition = fake_decomposition  # type: ignore[assignment]
            agent.invoke_planning_agent = fake_planning  # type: ignore[assignment]
            with contextlib.redirect_stdout(output):
                self.assertEqual(agent.main(["Launch a service"]), 0)
        finally:
            agent.run_decomposition = original_decomposition  # type: ignore[assignment]
            agent.invoke_planning_agent = original_planning  # type: ignore[assignment]

        rendered = output.getvalue()
        self.assertIn("------decomposition-agent------", rendered)
        self.assertIn("------planning-agent------", rendered)
        self.assertEqual(rendered.count("----end----"), 2)
        self.assertEqual(calls, ["decomposition:Launch a service", "planning:Launch a service"])

    def test_cli_decompose_only_preserves_decomposition_contract(self) -> None:
        calls: list[str] = []
        original_decomposition = agent.run_decomposition
        original_planning = agent.invoke_planning_agent

        class DummyDecomposition:
            def to_dict(self) -> dict[str, object]:
                return {
                    "intent": "Launch a service",
                    "subtasks": ["Define scope"],
                    "golden_goal_context_used": False,
                }

        def fake_decomposition(intent: str) -> DummyDecomposition:
            calls.append(f"decomposition:{intent}")
            return DummyDecomposition()

        def fake_planning(payload: dict[str, object]) -> dict[str, object]:
            calls.append(f"planning:{payload['intent']}")
            return {
                "intent": payload["intent"],
                "subtask_plans": [],
                "overall_status": "mock_completed",
            }

        output = io.StringIO()
        try:
            agent.run_decomposition = fake_decomposition  # type: ignore[assignment]
            agent.invoke_planning_agent = fake_planning  # type: ignore[assignment]
            with contextlib.redirect_stdout(output):
                self.assertEqual(agent.main(["--decompose-only", "Launch a service"]), 0)
        finally:
            agent.run_decomposition = original_decomposition  # type: ignore[assignment]
            agent.invoke_planning_agent = original_planning  # type: ignore[assignment]

        rendered = output.getvalue()
        self.assertIn("------decomposition-agent------", rendered)
        self.assertNotIn("------planning-agent------", rendered)
        self.assertEqual(rendered.count("----end----"), 1)
        self.assertEqual(calls, ["decomposition:Launch a service"])

    def test_extract_monitoring_reports_from_planning_output(self) -> None:
        report = {
            "intent": "Monitor AMF",
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
            "results": [{"target": "AMF"}],
            "errors": [],
            "source": {"type": "test"},
        }
        planning_output = {
            "intent": "Monitor AMF",
            "subtask_plans": [{"result": {"monitoring_report": report}}],
            "overall_status": "completed",
        }

        self.assertEqual(agent.extract_monitoring_reports(planning_output), [report])

    def test_extract_representative_core_reports_from_planning_output(self) -> None:
        report = {
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
        planning_output = {
            "intent": "Restart AMF",
            "subtask_plans": [{"result": {"representative_core_report": report}}],
            "overall_status": "pending_approval",
        }

        self.assertEqual(agent.extract_representative_core_reports(planning_output), [report])

    def test_planning_handoff_accepts_pending_approval_status(self) -> None:
        planning_report = {
            "intent": "Scale out AMF",
            "subtask_plans": [
                {
                    "subtask_id": "subtask-001",
                    "subtask": "Scale out AMF pod",
                    "status": "pending_approval",
                }
            ],
            "overall_status": "pending_approval",
        }

        validate_planning_report(planning_report)

    def test_missing_mongodb_uri_returns_empty_context(self) -> None:
        old_uri = os.environ.get("DECOMPOSITION_KNOWLEDGE_MONGODB_URI")
        os.environ["DECOMPOSITION_KNOWLEDGE_MONGODB_URI"] = ""
        try:
            context = find_decomposition_pattern_context("test intent")
        finally:
            if old_uri is None:
                os.environ.pop("DECOMPOSITION_KNOWLEDGE_MONGODB_URI", None)
            else:
                os.environ["DECOMPOSITION_KNOWLEDGE_MONGODB_URI"] = old_uri

        self.assertFalse(context.used)
        self.assertEqual(context.documents, [])


if __name__ == "__main__":
    unittest.main()
