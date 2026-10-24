"""Application role routing must preserve reviewed values and A1 provenance."""

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import httpx
from fastapi.testclient import TestClient
from smo.agentic_ai.agora_offline.applications import application_for, applications
from smo.agentic_ai.agora_offline.provider import configuration
from smo.ric_control import dispatch, gateway, state
from smo.ric_control.messages import Submission, submission_digest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "agora_a1_test_main", ROOT / "non-rt-ric/A1_Mediator/app/main.py"
)
mediator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mediator)
from non_rt_ric import a1


def body(objective="energy_saving"):
    return {
        "live_run_id": "b" * 32,
        "plan": {
            "schema_version": "agora-ric-queue-v1",
            "deployment_id": "a" * 32,
            "session_id": "c" * 32,
            "candidate_id": "reviewed",
            "target_ric": "non_rt",
            "run_id": "gnn_0130",
            "intent": {"objective": objective},
            "current_parameters": {"txp_5g": 43, "ret_5g": 9},
            "candidate": {
                "changes": {"txp_5g": 40},
                "interval_start_s": 180,
                "interval_end_s": 360,
            },
        },
    }


class ApplicationTests(unittest.TestCase):
    def test_policy_package_is_owned_by_non_rt_server(self):
        self.assertEqual(Path(a1.__file__).resolve(), ROOT / "non-rt-ric/non_rt_ric/a1.py")
        self.assertIs(mediator.agora_router, a1.router)
        self.assertEqual(a1.ROOT, ROOT)

    def test_same_control_routes_by_application_purpose(self):
        apps = applications()
        self.assertEqual(len(apps), 5)
        self.assertEqual(sum(app.role == "rApp" for app in apps.values()), 2)
        self.assertEqual(apps["energy_saving"].target_ric, "non_rt")
        self.assertEqual(apps["coverage"].target_ric, "non_rt")
        self.assertEqual(apps["throughput"].target_ric, "near_rt")
        plan = body()["plan"]
        self.assertEqual(apps["energy_saving"].prepare(plan), plan)
        with self.assertRaises(ValueError):
            apps["throughput"].prepare(plan)

    def test_rapp_cannot_change_an_unowned_control_or_increase_txp(self):
        plan = body()["plan"]
        app = application_for(plan["intent"])
        plan["candidate"]["changes"]["txp_5g"] = 44
        with self.assertRaises(ValueError):
            app.prepare(plan)
        plan["candidate"]["changes"] = {"ttt_ms": 340}
        with self.assertRaises(ValueError):
            app.prepare(plan)

    def test_provider_no_longer_reads_legacy_agent_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            legacy = root / "agents/planning"
            legacy.mkdir(parents=True)
            (legacy / ".env").write_text("OPENAI_API_KEY=legacy-dummy\nOPENAI_MODEL=legacy\n")
            (root / ".env").write_text(
                "OPENAI_API_KEY=independent-dummy\nOPENAI_MODEL=independent\n"
            )
            with patch.dict("os.environ", {"AGENTIC_AI_ROOT": str(root)}, clear=True):
                self.assertEqual(configuration()["key"], "independent-dummy")
                (root / ".env").unlink()
                self.assertEqual(configuration()["key"], "")


class MediatorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        directory = Path(temporary.name)
        for module, name, value in (
            (a1, "RUNTIME", directory / "a1"),
            (state, "RUNTIME", directory / "control"),
            (state, "RUNS", directory / "runs"),
        ):
            self.enterContext(patch.object(module, name, value))
        self.enterContext(patch.object(dispatch, "live_run", return_value={"process_alive": True}))
        self.enterContext(
            patch.object(dispatch, "encode", return_value={"start_s": 180, "end_s": 360})
        )
        self.sent = self.enterContext(patch.object(dispatch, "transmit"))
        real_client = httpx.AsyncClient

        async def transport(request):
            app = mediator.app if request.url.port == 9000 else gateway.app
            async with real_client(
                transport=httpx.ASGITransport(app=app),
                base_url=f"http://127.0.0.1:{request.url.port}",
                trust_env=False,
            ) as client:
                return await client.request(
                    request.method,
                    request.url.path,
                    content=request.content,
                    headers=request.headers,
                )

        def client_factory(*args, **kwargs):
            kwargs["transport"] = httpx.MockTransport(transport)
            return real_client(*args, **kwargs)

        self.enterContext(patch("httpx.AsyncClient", side_effect=client_factory))
        self.client = self.enterContext(TestClient(mediator.app))
        self.execution = self.enterContext(TestClient(gateway.app))
        self.url = "/a1-p/policytypes/401/policies/" + "a" * 32

    def test_standalone_mediator_forwards_exact_plan_once_and_records_rapp(self):
        submitted = body()
        response = self.client.put(self.url, json=submitted)
        self.assertEqual(response.status_code, 202, response.text)
        result = response.json()
        self.assertTrue(result["policy_stored"])
        self.assertFalse(result["applied"])
        self.assertEqual(result["path"][:2], ["Non_RT_RIC", "Energy Saving rApp"])
        self.assertEqual(result["plan"], submitted["plan"])
        stored = self.client.get(self.url).json()
        self.assertEqual(stored["submission"], submitted)
        self.assertEqual(stored["rapp_invocation"]["status"], "forwarded")
        self.assertEqual(self.client.put(self.url, json=submitted).status_code, 202)
        self.assertEqual(self.sent.call_count, 1)
        changed = deepcopy(submitted)
        changed["plan"]["candidate"]["changes"]["txp_5g"] = 37
        self.assertEqual(self.client.put(self.url, json=changed).status_code, 409)
        self.assertEqual(self.sent.call_count, 1)

    def test_role_mismatch_cannot_bypass_standalone_a1(self):
        submitted = body()
        submitted["plan"]["target_ric"] = "near_rt"
        self.assertEqual(
            self.execution.post("/api/near-rt/deployments", json=submitted).status_code, 409
        )
        submitted["plan"]["target_ric"] = "non_rt"
        self.assertEqual(self.execution.put(self.url, json=submitted).status_code, 410)
        submitted["plan"]["intent"]["objective"] = "throughput"
        self.assertEqual(self.client.put(self.url, json=submitted).status_code, 422)
        self.sent.assert_not_called()

    def test_missing_or_changed_stored_policy_cannot_execute(self):
        submitted = body()
        forwarded = {
            **submitted,
            "policy_id": "a" * 32,
            "policy_sha256": submission_digest(Submission(**submitted)),
        }
        self.assertEqual(
            self.execution.post("/api/rapp/deployments", json=forwarded).status_code, 503
        )
        self.assertEqual(self.client.put(self.url, json=submitted).status_code, 202)
        forwarded["plan"]["candidate"]["changes"]["txp_5g"] = 37
        self.assertEqual(
            self.execution.post("/api/rapp/deployments", json=forwarded).status_code, 409
        )
        self.assertEqual(self.sent.call_count, 1)

    def test_policy_survives_failed_delivery_without_claiming_application(self):
        with patch.object(a1, "gateway", side_effect=a1.HTTPException(503, "offline")):
            response = self.client.put(self.url, json=body())
            self.assertEqual(response.status_code, 503)
            saved = self.client.get(self.url).json()
            self.assertEqual(saved["status"], "dispatch_failed")
            status = self.client.get(self.url + "/status").json()
            self.assertTrue(status["policy_stored"])
            self.assertFalse(status["applied"])
            self.assertFalse(status["execution_status_available"])
        self.assertEqual(self.client.put(self.url, json=body()).status_code, 202)
        self.assertEqual(self.sent.call_count, 1)

    def test_real_readbacks_reach_a1_status(self):
        self.assertEqual(self.client.put(self.url, json=body()).status_code, 202)
        directory = state.RUNS / ("b" * 32)
        directory.mkdir(parents=True)
        events = [
            {"run_id": "b" * 32, "command_id": "a" * 32, "status": name, "sim_time_s": time}
            for name, time in (("scheduled", 100), ("applied", 180), ("restored", 360))
        ]
        (directory / "receipts.jsonl").write_text("\n".join(json.dumps(row) for row in events))
        response = self.client.get(self.url + "/status").json()
        self.assertEqual(response["status"], "restored")
        self.assertTrue(response["applied"] and response["restored"])
        self.assertEqual(response["receipts"], events)
