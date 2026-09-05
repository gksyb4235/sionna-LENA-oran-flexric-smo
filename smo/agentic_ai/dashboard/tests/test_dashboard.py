from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import ANY, patch

from fastapi.testclient import TestClient

ROOT = Path("/home/ubuntu/jaechan")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(ROOT / "dashboard") not in sys.path:
    sys.path.insert(0, str(ROOT / "dashboard"))

from agent_ops.telemetry import log_event  # noqa: E402
import server  # noqa: E402


class DashboardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(server.app)
        self.run_id = "dashboard-test-run"
        state_path = ROOT / "agents" / "planning" / ".state" / "planning-runs" / f"{self.run_id}.json"
        state_path.parent.mkdir(parents=True, exist_ok=True)
        state_path.write_text(
            json.dumps(
                {
                    "run_id": self.run_id,
                    "intent": "Monitor AMF",
                    "subtasks": ["Monitor AMF CPU"],
                    "current_subtask_index": 1,
                    "subtask_plans": [],
                    "pending_approval": None,
                    "overall_status": "completed",
                }
            ),
            encoding="utf-8",
        )
        log_event(run_id=self.run_id, agent="planning-agent", event_type="run_completed", status="completed")

    def test_runs_and_detail_work_without_mongo(self) -> None:
        runs = self.client.get("/api/runs")
        self.assertEqual(runs.status_code, 200)
        self.assertTrue(any(item["run_id"] == self.run_id for item in runs.json()["runs"]))

        detail = self.client.get(f"/api/runs/{self.run_id}")
        self.assertEqual(detail.status_code, 200)
        payload = detail.json()
        self.assertEqual(payload["summary"]["overall_status"], "completed")
        self.assertTrue(payload["events"])


    def test_invoke_intent_uses_decomposition_agent(self) -> None:
        with patch.object(
            server,
            "_run_decomposition_intent",
            return_value={"run_id": self.run_id, "overall_status": "completed", "subtask_plans": []},
        ) as invoke:
            response = self.client.post("/api/invoke", json={"intent": "Monitor AMF"})
        self.assertEqual(response.status_code, 200)
        invoke.assert_called_once_with("Monitor AMF", call_id=ANY, client_host="testclient", request_path="/api/invoke")
        payload = response.json()
        self.assertEqual(payload["result"]["run_id"], self.run_id)
        self.assertEqual(payload["run"]["summary"]["run_id"], self.run_id)

    def test_invoke_rejects_blank_intent(self) -> None:
        response = self.client.post("/api/invoke", json={"intent": "   "})
        self.assertEqual(response.status_code, 400)


    def test_run_decomposition_keeps_venv_python_path(self) -> None:
        stdout = json.dumps({"run_id": self.run_id, "overall_status": "completed", "subtask_plans": []})
        completed = server.subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")
        with patch.object(server, "_run_import_preflight") as preflight, patch.object(
            server.subprocess, "run", return_value=completed
        ) as run:
            result = server._run_decomposition_intent(
                "Monitor AMF",
                call_id="invoke-venv-path-test",
                client_host="testclient",
                request_path="/api/invoke",
            )
        self.assertEqual(result["run_id"], self.run_id)
        self.assertEqual(preflight.call_args.kwargs["python_path"], server.DECOMPOSITION_PYTHON)
        self.assertEqual(run.call_args.args[0][0], str(server.DECOMPOSITION_PYTHON))


    def test_invoke_preflight_failure_returns_call_id(self) -> None:
        failure = server.InvokeFailure(call_id="invoke-test", stage="preflight", message="preflight failed")
        with patch.object(server, "_run_decomposition_intent", side_effect=failure):
            response = self.client.post("/api/invoke", json={"intent": "Monitor AMF"})
        self.assertEqual(response.status_code, 400)
        detail = response.json()["detail"]
        self.assertEqual(detail["call_id"], "invoke-test")
        self.assertEqual(detail["stage"], "preflight")
        self.assertIn("preflight failed", detail["message"])

    def test_import_preflight_succeeds_without_llm_call(self) -> None:
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONNOUSERSITE"] = "1"
        server._run_import_preflight(
            call_id="invoke-preflight-test",
            python_path=server.DECOMPOSITION_PYTHON,
            cwd=server.DECOMPOSITION_ROOT,
            env=env,
            client_host="testclient",
            request_path="/api/invoke",
        )

    def test_approve_uses_planning_agent_path(self) -> None:
        with patch.object(server, "approve_core_report_from_payload", return_value={"overall_status": "completed"}) as approve:
            response = self.client.post(
                f"/api/runs/{self.run_id}/approve",
                json={"report": {"status": "pending_approval"}, "approved": False, "approved_by": "dashboard"},
            )
        self.assertEqual(response.status_code, 200)
        approve.assert_called_once()


if __name__ == "__main__":
    unittest.main()
