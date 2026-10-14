"""Regression tests for incomplete axis searches and actual DAG handoff."""

import asyncio
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agora_offline.contracts import DEFAULT_POLICY, DOMAINS, validate_intent
from agora_offline.orchestrator import Session
from agora_offline.provider import Provider
from agora_offline.runtime import Runtime
from agora_offline.search import AxisSearch
from agora_offline.settings import PROMPTS
from agora_offline.tests.test_estimator import CURRENT, FakeRuntime, intent

SETTINGS = {"prompts": PROMPTS, "policy": DEFAULT_POLICY, "provider": {"model": "test-double"}}


def search(changes=None, current=None, policy=None, t=180):
    return AxisSearch(
        {"changes": changes or {"txp_5g": 33.0}, "delay_s": 0, "horizon_s": 90},
        current or CURRENT,
        validate_intent(intent()),
        policy or DEFAULT_POLICY,
        t,
    )


class SearchTests(unittest.TestCase):
    def test_weaker_values_and_all_axes_are_not_dependent_on_llm_labels(self):
        grid = search()
        self.assertEqual({a for p in grid.first for a in p["axes"]}, {"time", "intensity", "cell"})
        values = {p["changes"]["txp_5g"] for p in grid.first if p["axes"] == ["intensity"]}
        self.assertTrue({36.0, 37.0, 40.0, 41.0, 42.0}.issubset(values))
        grid.record(
            {
                "all_candidates": [
                    {
                        "changes": {"txp_5g": 37.0},
                        "delay_s": d,
                        "axes": ["time", "intensity", "cell"],
                        "feasible": False,
                    }
                    for d in [0, 30, 60]
                ]
            }
        )
        self.assertFalse(grid.report()["independent_axes_complete"])
        self.assertEqual(grid.report()["axes"]["cell"]["evaluated"], 0)

    def test_blocked_cell_and_shared_controls_are_explained(self):
        grid = search(current=dict(CURRENT, txp_4g2=30.0))
        blocked = next(c for c in grid.report()["cell_targets"] if c["target"] == "txp_4g2")
        self.assertEqual(blocked["status"], "blocked")
        self.assertIn("supported bound", blocked["reason"])
        for name, target in [("ttt_ms", 128.0), ("hys_db", 0.5)]:
            report = search({name: target}).report()
            self.assertEqual(report["axes"]["cell"]["status"], "not_applicable")
            self.assertIn("shared controls", report["cell_targets"][0]["reason"])

    def test_control_families_and_trained_only_policy(self):
        for name, value in [("ret_5g", 3.0), ("cio_5g", 5.0), ("ttt_ms", 128.0), ("hys_db", 0.5)]:
            grid = search({name: value})
            for plan in grid.first + grid.combined:
                self.assertTrue(all(v in DOMAINS[k] for k, v in plan["changes"].items()))
                self.assertLessEqual(
                    set(k.split("_")[0] for k in plan["changes"]), {name.split("_")[0]}
                )
        grid = search(
            policy=dict(DEFAULT_POLICY, allow_continuous_txp=False),
            current=dict(CURRENT, txp_4g1=40.0),
        )
        self.assertTrue(
            all(
                v in DOMAINS[k] for p in grid.first + grid.combined for k, v in p["changes"].items()
            )
        )

    def test_combined_round_and_time_bounds(self):
        grid = search(t=690)
        self.assertEqual(grid.delays, [0, 30])
        grid.record(
            {"all_candidates": [{**p, "feasible": False} for p in [grid.base, *grid.first]]}
        )
        self.assertEqual(grid.report()["next_stage"], "combinations")
        combined = grid.pending("combinations")
        self.assertTrue(
            any(p["changes"].get("txp_4g2") == 40 and p["delay_s"] == 30 for p in combined)
        )
        self.assertEqual(len(combined), len({grid.key(p) for p in combined}))
        self.assertFalse(grid.report()["stop_allowed"])
        self.assertEqual(
            grid.report(budget_available=False)["stop_reason"],
            "budget_exhausted_with_unsearched_candidates",
        )

    def test_duplicate_explicit_llm_plans_need_no_axis_labels(self):
        grid = search()
        extra = {"changes": {"txp_5g": 41.5}, "delay_s": 30}
        plans = grid.deduplicate([extra, deepcopy(extra)])
        self.assertEqual(len(plans), 1)
        self.assertEqual(plans[0]["changes"], extra["changes"])

    def test_impossible_goal_continues_gnn_analysis_until_declared_grid_exhausted(self):
        async def run():
            session = Session(
                dict(prompt="fixture", run_id="fixture", issue_time_s=180, mode="deterministic"),
                FakeRuntime(),
                SETTINGS,
            )
            session.snapshot = session.runtime.snapshot("fixture", 180)
            session.prepare_intent(
                dict(
                    changes={"txp_5g": 33.0},
                    horizon_s=90,
                    objective="energy_saving",
                    constraints=[
                        dict(
                            id="impossible",
                            metric="goodput_mbps",
                            cells=["5g"],
                            aggregation="sum",
                            operator=">=",
                            target=10000.0,
                        )
                    ],
                )
            )
            await session.evaluate([deepcopy(session.requested)], "conflict")
            while not session.search_report()["stop_allowed"]:
                await session.tool(
                    "estimate_batch",
                    {"axis": "all", "candidates": [], "summary": "test fixture"},
                    "mitigation",
                )
            return session

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("agora_offline.orchestrator.STATE", Path(directory)),
        ):
            session = asyncio.run(run())
        report = session.search_report()
        self.assertEqual(report["stop_reason"], "declared_candidate_grid_exhausted")
        self.assertEqual(report["combinations_evaluated"], report["combination_candidates"])
        self.assertGreater(session.batch_count, 2)
        self.assertLessEqual(session.batch_count, 5)
        self.assertLessEqual(session.candidate_count, 160)
        self.assertEqual(len(session.search.seen), session.candidate_count)
        analyses = [
            e for e in session.events if e["type"] == "tool_result" and e["content"] == "Analyze"
        ]
        self.assertEqual(len(analyses), session.batch_count - 1)

    def test_budget_exhaustion_remains_incomplete_for_large_compound_search(self):
        async def run():
            session = Session(
                dict(prompt="fixture", run_id="fixture", issue_time_s=180, mode="deterministic"),
                FakeRuntime(),
                SETTINGS,
            )
            session.snapshot = session.runtime.snapshot("fixture", 180)
            session.prepare_intent(
                dict(
                    changes={
                        "txp_5g": 30.0,
                        "txp_4g1": 30.0,
                        "txp_4g2": 30.0,
                        "ret_5g": 3.0,
                        "cio_5g": 5.0,
                        "ttt_ms": 128.0,
                        "hys_db": 0.5,
                    },
                    horizon_s=90,
                    objective="energy_saving",
                    constraints=[
                        dict(
                            id="impossible",
                            metric="goodput_mbps",
                            cells=["5g"],
                            aggregation="sum",
                            operator=">=",
                            target=10000.0,
                        )
                    ],
                )
            )
            await session.evaluate([deepcopy(session.requested)], "conflict")
            while not session.search_report()["stop_allowed"]:
                await session.tool(
                    "estimate_batch",
                    {"axis": "all", "candidates": [], "summary": "test fixture"},
                    "mitigation",
                )
            return session

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("agora_offline.orchestrator.STATE", Path(directory)),
        ):
            session = asyncio.run(run())
        report = session.search_report()
        self.assertEqual(report["stop_reason"], "budget_exhausted_with_unsearched_candidates")
        self.assertIsNotNone(report["next_stage"])
        self.assertLessEqual(session.batch_count, 5)
        self.assertLessEqual(session.candidate_count, 160)


class RealRegressionTests(unittest.TestCase):
    def test_reported_failure_is_repaired_even_when_llm_repeats_narrow_proposals(self):
        received = []

        def call(name, arguments):
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "fixture",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(arguments)},
                    }
                ],
            }

        async def fake_complete(provider, role, messages, emit, tools=None, forced=None):
            if role == "intent":
                return call(
                    "submit_intent",
                    dict(
                        changes={"txp_5g": 33.0},
                        expected_current={"txp_5g": 43.0},
                        horizon_s=180,
                        delay_s=0,
                        objective="energy_saving",
                        target_fraction=0.01,
                        interpretation="fixture",
                    ),
                )
            if role == "analyst" and forced == "Analyze":
                return call("Analyze", {"candidate_ids": ["requested"]})
            if role == "analyst" and tools and len(messages) == 3:
                return call("lookup_context", {"feature": "txp_5g", "kpi": "goodput_mbps"})
            if role == "mitigation":
                received.append(deepcopy(messages))
                if tools:
                    return call(
                        "estimate_batch",
                        {
                            "axis": "time",
                            "summary": "Reproduce narrow proposals",
                            "candidates": [
                                {"changes": {"txp_5g": value}, "delay_s": delay}
                                for value, delay in [
                                    (33, 0),
                                    (33, 30),
                                    (33, 60),
                                    (37, 0),
                                    (37, 30),
                                    (37, 60),
                                ]
                            ],
                        },
                    )
            return {"role": "assistant", "content": "Explicit test double summary"}

        with (
            tempfile.TemporaryDirectory() as directory,
            patch("agora_offline.orchestrator.STATE", Path(directory)),
            patch.object(Provider, "complete", fake_complete),
        ):
            session = Session(
                dict(
                    prompt="Cell5G TxP43 ->33 for3minutes",
                    run_id="gnn_0130",
                    issue_time_s=180,
                    mode="llm",
                    horizon_s=180,
                ),
                Runtime(),
                SETTINGS,
            )
            asyncio.run(session.run())
        self.assertEqual(session.status, "completed", session.events[-2:])
        # Frozen Q75 thresholds are stricter than the historical per-cell demo.
        # Exhausting the declared grid is correct; the planner may not relax it.
        self.assertEqual(session.result["status"], "no_feasible_candidate")
        self.assertTrue(session.result["search_coverage"]["independent_axes_complete"])
        self.assertIsNone(session.result["selected"])
        review = session.result["operator_review"]
        self.assertEqual(len(review["candidates"]), 3)
        self.assertTrue(
            all(not row["feasible"] and row["deployment_eligible"] for row in review["candidates"])
        )
        scores = [row["operator_score"]["normalized_violation_sum"] for row in review["candidates"]]
        self.assertEqual(scores, sorted(scores))
        self.assertTrue(all(row["requires_constraint_override"] for row in review["candidates"]))
        self.assertTrue(
            all(
                next(c for c in row["constraints"] if c["metric"] == "rlfs")["effective_threshold"]
                == 4
                for row in review["candidates"]
            )
        )
        self.assertFalse(
            any(
                r["feasible"]
                for e in session.evaluations
                for r in e["all_candidates"]
                if r["status"] == "evaluated"
            )
        )
        batches = [
            e
            for e in session.events
            if e["type"] == "tool_call" and e["content"] == "estimate_batch"
        ]
        self.assertEqual(batches[0]["data"]["arguments"]["candidate_count"], 1)
        plans = batches[1]["data"]["arguments"]["candidates"]
        self.assertTrue(any(p["changes"].get("txp_5g") == 40 for p in plans))
        self.assertTrue(any(p["changes"].get("txp_5g") == 41 for p in plans))
        self.assertTrue(any("txp_4g1" in p["changes"] for p in plans))
        context = json.loads(received[0][0]["content"])
        self.assertEqual(context["context_provenance"]["dag_features"], ["txp_4g1", "txp_5g"])
        evidence = [
            e for e in context["shared_numeric_and_scoped_context"] if e["tool"] == "lookup_context"
        ]
        self.assertTrue(all("kpi" not in e["arguments"] for e in evidence))
        self.assertTrue(
            all("out_of_scope" in json.dumps(e["result"]["applicability"]) for e in evidence)
        )
        audit = next(
            e
            for e in session.events
            if e["type"] == "tool_result" and e["content"] == "prepare_mitigation_context"
        )
        self.assertEqual(context, json.loads(json.dumps(audit["data"]["result"])))
        self.assertGreaterEqual(len(received), 2)
        self.assertIn('"search_coverage"', received[-1][-1]["content"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
