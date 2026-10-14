"""Real frozen-model orchestration plus isolated provider/protocol tests."""

import asyncio
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import httpx
from fastapi.testclient import TestClient
from agora_offline import server
from agora_offline.runtime import Runtime
from agora_offline.orchestrator import Session, compact_snapshot
from agora_offline.planning import parse_deterministic
from agora_offline.provider import Provider
from agora_offline.settings import PROMPTS
from agora_offline.contracts import DEFAULT_POLICY
from agora_offline.intent_policy import compile_spec, default_spec, entry

SETTINGS = {"prompts": PROMPTS, "policy": DEFAULT_POLICY, "provider": {"model": "mock-test-model"}}


class WorkbenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = Runtime()

    def test_parser_preserves_delayed_duration_and_units(self):
        parsed = parse_deterministic("90초 뒤부터 3분 동안 Cell 5G TxP 43 dBm -> 33 dBm", {}, 90)
        self.assertEqual((parsed["horizon_s"], parsed["delay_s"]), (180, 90))
        self.assertEqual(parsed["changes"], {"txp_5g": 33.0})
        self.assertEqual(parsed["expected_current"], {"txp_5g": 43.0})
        observed = parse_deterministic(
            "최근 180초 관측 기준 Cell5G TxP43dBm에서33dBm으로90초간 유지", {}, 90
        )
        self.assertEqual(observed["horizon_s"], 90)
        fractional = parse_deterministic("Cell5G TxP43 ->33 for1.51minutes", {}, 180)
        self.assertEqual(fractional["horizon_s"], 90.6)

    def test_ui_reload_references_new_assets_after_a_static_update(self):
        import re

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("agora_offline.server.ROOT", Path(directory)),
        ):
            static = Path(directory) / "static"
            static.mkdir()
            (static / "index.html").write_text(
                '<link href="/static/styles.css"><script src="/static/app.js"></script>',
                encoding="utf-8",
            )
            (static / "styles.css").write_text("body{}", encoding="utf-8")
            (static / "app.js").write_text("old_ui()", encoding="utf-8")
            client = TestClient(server.app)
            before = client.get("/")
            (static / "app.js").write_text("new_ui()", encoding="utf-8")
            after = client.get("/")
            self.assertEqual(before.headers["cache-control"], "no-store")
            self.assertEqual(after.headers["cache-control"], "no-store")
            pattern = r"/static/app\.js\?v=[a-f0-9]+"
            self.assertNotEqual(
                re.search(pattern, before.text).group(), re.search(pattern, after.text).group()
            )
            css_pattern = r"/static/styles\.css\?v=[a-f0-9]+"
            self.assertEqual(
                re.search(css_pattern, before.text).group(),
                re.search(css_pattern, after.text).group(),
            )

    def test_original_first_and_combined_refinement_real_model(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("agora_offline.orchestrator.STATE", Path(directory)),
        ):
            session = Session(
                dict(
                    prompt="Cell 5G TxP43 ->33 for3minutes",
                    run_id="gnn_0079",
                    issue_time_s=360,
                    mode="deterministic",
                    horizon_s=180,
                ),
                self.runtime,
                SETTINGS,
            )
            asyncio.run(session.run())
            self.assertEqual(session.status, "completed", session.events[-2:])
            batches = [
                e
                for e in session.events
                if e["type"] == "tool_call" and e["content"] == "estimate_batch"
            ]
            self.assertEqual(batches[0]["data"]["arguments"]["candidate_count"], 1)
            self.assertEqual(
                batches[0]["data"]["arguments"]["candidates"][0]["changes"], {"txp_5g": 33.0}
            )
            self.assertGreater(len(batches), 1)
            self.assertFalse(session.result["applied"])
            self.assertFalse(session.result["ns3_validated"])
            self.assertTrue(session.result["selected"]["feasible"])
            self.assertEqual(
                session.result["execution_plan"]["observed_history"]["end_s_exclusive"], 360
            )
            self.assertEqual(session.result["original"]["conflict_status"], "constraint_conflict")
            self.assertGreater(len(session.shared_evidence), 0)

    def test_all_seventeen_observed_kpis_are_available_without_expanding_agent_context(self):
        snapshot = self.runtime.snapshot("gnn_0130", 180)
        history = snapshot["kpi_history"]
        self.assertEqual(len(history["metric_names"]), 17)
        self.assertLess(max(history["times_s"]), 180)
        self.assertEqual(len(history["values"]), 180)
        for cell_index, cell in enumerate(history["cell_labels"]):
            for metric_index, metric in enumerate(history["metric_names"]):
                self.assertEqual(
                    history["values"][-1][cell_index][metric_index],
                    snapshot["latest_kpis"][cell][metric],
                )
        self.assertTrue(
            {"power_w", "energy_eff_mbit_per_j", "traffic_share_pct"}.issubset(
                history["metric_names"]
            )
        )
        self.assertEqual(len(snapshot["history"]["input_names"]), 15)
        context = compact_snapshot(snapshot)
        self.assertNotIn("history", context)
        self.assertNotIn("kpi_history", context)
        self.assertIn("observed_90s", context)

    def test_wrong_start_value_fails_before_forecast(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("agora_offline.orchestrator.STATE", Path(directory)),
        ):
            session = Session(
                dict(
                    prompt="Cell5G TxP40 ->33 for3minutes",
                    run_id="gnn_0079",
                    issue_time_s=360,
                    mode="deterministic",
                    horizon_s=180,
                ),
                self.runtime,
                SETTINGS,
            )
            asyncio.run(session.run())
            self.assertEqual(session.status, "failed")
            self.assertEqual(session.batch_count, 0)
            self.assertIn(
                "differs from replay",
                next(e["content"] for e in session.events if e["type"] == "error"),
            )

    def test_multiagent_tool_handoff_with_explicit_test_double(self):
        seen = []

        def tool(name, arguments):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "mock_call",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ],
            }

        async def fake_complete(provider, role, messages, emit, tools=None, forced=None):
            seen.append((role, messages))
            await emit("summary_delta", role, "TEST DOUBLE public summary")
            if role == "intent":
                return tool(
                    "submit_intent",
                    {
                        "changes": {"txp_5g": 33},
                        "horizon_s": 180,
                        "delay_s": 0,
                        "objective": "energy_saving",
                        "target_fraction": 0.01,
                        "constraints": [],
                        "interpretation": "Test double",
                    },
                )
            if role == "analyst" and len(messages) == 1:
                return tool("Analyze", {"candidate_ids": ["requested"]})
            if role == "mitigation" and len(messages) == 1:
                return tool(
                    "estimate_batch",
                    {
                        "axis": "combined",
                        "cells": ["4g2"],
                        "values": [42],
                        "delays_s": [0],
                        "summary": "Test double candidate",
                    },
                )
            return {"role": "assistant", "content": "TEST DOUBLE summary"}

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("agora_offline.orchestrator.STATE", Path(directory)),
            patch.object(Provider, "complete", fake_complete),
        ):
            session = Session(
                dict(
                    prompt="Cell5G TxP43 ->33 for3minutes",
                    run_id="gnn_0079",
                    issue_time_s=360,
                    mode="llm",
                    horizon_s=180,
                ),
                self.runtime,
                SETTINGS,
            )
            asyncio.run(session.run())
            self.assertEqual(session.status, "completed", session.events[-2:])
            self.assertEqual(len(session.intent["constraints"]), 5)
            planner = next(messages for role, messages in seen if role == "mitigation")
            self.assertIn("shared_numeric_and_scoped_context", planner[0]["content"])
            self.assertIn("violated_constraints", planner[0]["content"])
            self.assertEqual(
                {role for role, _ in seen},
                {"intent", "conflict", "analyst", "mitigation", "deployment"},
            )
            deployment = json.loads(
                next(messages for role, messages in seen if role == "deployment")[0]["content"]
            )
            self.assertEqual(len(deployment["operator_candidates"]), 3)
            self.assertIn("operator_ranking", deployment)
            self.assertEqual(deployment["current_parameters"], session.snapshot["current_rcp"])
            conflict = json.loads(
                next(messages for role, messages in seen if role == "conflict")[0]["content"]
            )
            self.assertEqual(conflict["current_parameters"], session.snapshot["current_rcp"])

    def test_editor_goals_and_thresholds_cannot_be_replaced_by_the_agent(self):
        spec = default_spec(180)
        spec["goals"].append(entry("goodput", 0.07, ident="second_goal"))
        spec["base_constraints"][0]["target"] = 98
        session = Session(
            dict(
                prompt="Cell5G TxP43 ->33",
                run_id="gnn_0130",
                issue_time_s=180,
                mode="deterministic",
                horizon_s=180,
                intent_spec=spec,
            ),
            self.runtime,
            SETTINGS,
        )
        session.snapshot = self.runtime.snapshot("gnn_0130", 180)
        parsed = dict(
            changes={"txp_5g": 33},
            horizon_s=180,
            objective="load_balancing",
            target_fraction=0.001,
            goals=[entry("power", -0.001)],
            constraints=[entry("rlf", 999, "extra")],
        )
        session.prepare_intent(parsed)
        expected = compile_spec(spec, 180)
        self.assertEqual(session.intent, expected)
        self.assertEqual(session.intent["constraints"][0]["target"], 98)
        self.assertEqual(session.intent["objective"], "energy_saving")

    def test_policy_catalog_and_validation_before_starting_agents(self):
        client = TestClient(server.app)
        data = client.get("/api/intent-policy?horizon_s=90").json()
        self.assertEqual(data["templates"]["energy_saving"]["base_constraints"][1]["target"], 2)
        self.assertEqual(client.get("/api/intent-policy?horizon_s=150").status_code, 422)
        body = dict(
            prompt="Cell5G TxP43 ->33",
            run_id="gnn_0130",
            issue_time_s=180,
            mode="deterministic",
            horizon_s=90,
            intent_spec=default_spec(180),
        )
        response = client.post("/api/sessions", json=body)
        self.assertEqual(response.status_code, 422)
        self.assertIn("horizon", response.json()["detail"])

    def test_streamed_content_and_completed_tool_arguments(self):
        events = []
        chunks = [
            {"type": "response.output_text.delta", "delta": "Public summary"},
            {"type": "response.function_call_arguments.delta", "delta": '{"candidate_'},
            {"type": "response.function_call_arguments.delta", "delta": 'ids":["requested"]}'},
            {
                "type": "response.completed",
                "response": {
                    "output": [
                        {
                            "type": "function_call",
                            "call_id": "c1",
                            "name": "Analyze",
                            "arguments": '{"candidate_ids":["requested"]}',
                        }
                    ]
                },
            },
        ]
        data = "".join("data: " + json.dumps(c) + "\n\n" for c in chunks) + "data: [DONE]\n\n"
        actual_client = httpx.AsyncClient

        def handler(request):
            return httpx.Response(200, text=data, headers={"content-type": "text/event-stream"})

        async def emit(*args):
            events.append(args)

        async def run():
            provider = Provider(SETTINGS)
            provider.conf = {
                "key": "synthetic-test-key",
                "model": "mock",
                "base_url": "https://unit-test.invalid",
            }
            with patch(
                "agora_offline.provider.httpx.AsyncClient",
                lambda **kwargs: actual_client(transport=httpx.MockTransport(handler)),
            ):
                return await provider.complete(
                    "analyst", [{"role": "user", "content": "test"}], emit
                )

        result = asyncio.run(run())
        self.assertEqual(result["content"], "Public summary")
        self.assertEqual(
            json.loads(result["tool_calls"][0]["function"]["arguments"]),
            {"candidate_ids": ["requested"]},
        )
        self.assertTrue(any(e[0] == "summary_delta" for e in events))

    def test_http_sse_reconnect_and_export(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("agora_offline.orchestrator.STATE", Path(directory)),
            patch("agora_offline.settings.STATE", Path(directory)),
            patch("agora_offline.server.STATE", Path(directory)),
        ):
            with TestClient(server.app) as client:
                self.assertEqual(
                    client.get("/api/status").json()["model"]["registered_parameters"], 153537
                )
                self.assertEqual(len(client.get("/api/runs?split=test").json()["runs"]), 100)
                response = client.post(
                    "/api/sessions",
                    json={
                        "prompt": "Cell5G TxP43 ->33 for3minutes",
                        "run_id": "gnn_0079",
                        "issue_time_s": 360,
                        "mode": "deterministic",
                    },
                )
                self.assertEqual(response.status_code, 200, response.text)
                ident = response.json()["session_id"]
                for _ in range(100):
                    state = client.get(f"/api/sessions/{ident}").json()
                    if state["status"] in ("completed", "failed"):
                        break
                    time.sleep(0.1)
                self.assertEqual(state["status"], "completed")
                streamed = client.get(
                    f"/api/sessions/{ident}/events", headers={"Last-Event-ID": "3"}
                ).text
                self.assertTrue(streamed.startswith("id: 4\n"))
                export = client.get(f"/api/sessions/{ident}/export")
                self.assertEqual(export.status_code, 200)
                self.assertFalse(export.json()["result"]["applied"])
                del server.sessions[ident]
                archived = client.get(f"/api/sessions/{ident}").json()
                self.assertEqual(archived["status"], "completed")
                self.assertEqual(archived["result"]["session_id"], ident)
                self.assertEqual(archived["request"]["prompt"], "Cell5G TxP43 ->33 for3minutes")
                self.assertEqual(
                    client.post("/api/predict/batch", json={"plans": []}).status_code, 422
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
