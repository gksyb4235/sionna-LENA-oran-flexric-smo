"""Operator tradeoffs, immutable evidence and a durable non-executing RIC outbox."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from agora_offline import server
from agora_offline.deployment_queue import enqueue, get_receipt, list_receipts, QueueConflict
from agora_offline.operator_review import ensure_review, shortlist


def candidate(ident, value, violations, *, feasible=False, goal=1, unknown=False):
    constraints = [
        dict(
            id=f"c{i}",
            metric="rlfs",
            label="RLF",
            predicted_value=threshold + violation,
            effective_threshold=threshold,
            reference_value=99,
            violation=violation,
            satisfied=violation == 0,
            evaluation_status="evaluated",
            operator="<=",
        )
        for i, (violation, threshold) in enumerate(violations)
    ]
    if unknown:
        constraints[0].update(
            predicted_value=None,
            violation=None,
            satisfied=False,
            evaluation_status="insufficient_evidence",
        )
    return dict(
        candidate_id=ident,
        status="evaluated",
        parameters={"txp_5g": value, "ttt": 256},
        current_parameters={"txp_5g": 43, "ttt": 256},
        changes={"txp_5g": value},
        interval_start_s=210,
        interval_end_s=390,
        delay_s=30,
        horizon_s=180,
        feasible=feasible,
        goals_met=True,
        all_expectations_met=feasible,
        constraints=constraints,
        goals=[dict(id="g", predicted_value=2, evaluation_status="evaluated", satisfied=True)],
        goal_achievement=goal,
        request_deviation=0,
    )


def make_session():
    rows = [
        candidate("a", 40, [(1, 3)]),
        candidate("b", 39, [(2, 3)]),
        candidate("c", 38, [(3, 3)]),
    ]
    result = {
        "operator_review": shortlist(rows),
        "intent": {"constraints": [{"metric": "rlfs", "target": 3}]},
        "execution_plan": {
            "current_parameters": {"txp_5g": 43, "ttt": 256},
            "run_id": "gnn_0130",
            "issue_time_s": 180,
            "model": {"checkpoint": "frozen-model"},
        },
        "applied": False,
        "ns3_validated": False,
    }
    session = SimpleNamespace(id="a" * 32, status="completed", result=result, events=[], task=None)
    session.state = lambda: {
        "session_id": session.id,
        "status": session.status,
        "result": session.result,
        "events": [],
    }
    return session


class ReviewTests(unittest.TestCase):
    def test_less_total_violation_outranks_fewer_but_severe_violations(self):
        severe = candidate("severe", 33, [(6, 3)])
        mild = candidate("mild", 39, [(0.3, 3), (1, 100)])
        medium = candidate("medium", 37, [(1, 3)])
        original = deepcopy(mild)
        result = shortlist([severe, mild, medium])
        self.assertEqual(
            [r["candidate_id"] for r in result["candidates"]], ["mild", "medium", "severe"]
        )
        self.assertAlmostEqual(
            result["candidates"][0]["operator_score"]["normalized_violation_sum"], 0.11
        )
        self.assertEqual(mild, original)
        self.assertTrue(
            all(
                not r["feasible"] and r["requires_constraint_override"]
                for r in result["candidates"]
            )
        )

    def test_known_feasible_first_unknown_last_and_no_duplicate_actions(self):
        rows = [
            candidate("unknown", 33, [(0, 3)], unknown=True),
            candidate("mild", 37, [(1, 3)]),
            candidate("met", 40, [(0, 3)], feasible=True, goal=1),
            candidate("unmet", 39, [(0, 3)], feasible=True, goal=0.5),
        ]
        rows.append(dict(deepcopy(rows[2]), candidate_id="duplicate"))
        result = shortlist(rows)
        self.assertEqual(
            [r["candidate_id"] for r in result["candidates"]], ["met", "unmet", "mild"]
        )
        self.assertEqual(result["evaluated_unique_count"], 4)
        self.assertFalse(shortlist(rows[:1])["candidates"][0]["deployment_eligible"])

    def test_scales_do_not_reward_a_larger_delay_baseline(self):
        first = candidate("first", 39, [(1, 3)])
        second = candidate("second", 40, [(1, 3)])
        second["constraints"][0]["reference_value"] = 999999
        result = shortlist([first, second])
        self.assertEqual(
            result["candidates"][0]["operator_score"], result["candidates"][1]["operator_score"]
        )

    def test_archive_recovers_all_candidates_without_forecasting(self):
        rows = [candidate(str(i), 33 + i, [(i + 1, 3)]) for i in range(5)]
        result = {"infeasible_diagnostics": rows[-2:], "selected": None, "applied": False}
        events = [{"type": "candidates", "data": {"all_candidates": rows}}]
        ensure_review(result, events)
        self.assertEqual(
            [r["candidate_id"] for r in result["operator_review"]["candidates"]], ["0", "1", "2"]
        )
        fingerprint = result["operator_review"]["review_sha256"]
        ensure_review(result, events)
        self.assertEqual(result["operator_review"]["review_sha256"], fingerprint)
        self.assertIsNone(result["selected"])


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.state = Path(self.temp.name)
        self.session = make_session()
        self.args = {
            "candidate_id": "a",
            "target_ric": "non_rt",
            "review_sha256": self.session.result["operator_review"]["review_sha256"],
            "accept_constraint_violations": True,
        }

    def test_saved_selection_preserves_thresholds_and_exact_simulation_restore(self):
        before = deepcopy(self.session.result)
        with patch(
            "socket.create_connection", side_effect=AssertionError("No RIC network calls allowed")
        ):
            receipt = enqueue(self.state, self.session, **self.args)
        self.assertEqual(self.session.result, before)
        self.assertFalse(receipt["applied"])
        self.assertFalse(receipt["forwarded"])
        self.assertEqual(receipt["status"], "queued_offline")
        self.assertTrue(receipt["operator_decision"]["constraint_override"])
        self.assertEqual(receipt["intent"]["constraints"][0]["target"], 3)
        schedule = receipt["routes"][0]["schedule"]
        self.assertEqual([r["time_s"] for r in schedule], [210, 390])
        self.assertEqual(schedule[0]["changes"], {"txp_5g": 40})
        self.assertEqual(schedule[1]["changes"], {"txp_5g": 43})
        self.assertEqual(get_receipt(self.state, self.session.id), receipt)

    def test_repeated_concurrent_clicks_create_one_receipt_and_different_selection_conflicts(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            receipts = list(
                pool.map(lambda _: enqueue(self.state, self.session, **self.args), range(8))
            )
        self.assertEqual(len({r["deployment_id"] for r in receipts}), 1)
        self.assertEqual(len(list_receipts(self.state)), 1)
        with self.assertRaises(QueueConflict):
            enqueue(self.state, self.session, **dict(self.args, candidate_id="b"))
        with self.assertRaises(QueueConflict):
            enqueue(self.state, self.session, **dict(self.args, target_ric="near_rt"))

    def test_invalid_stale_unacknowledged_and_incomplete_reviews_do_not_queue(self):
        for args in [
            dict(self.args, candidate_id="invented"),
            dict(self.args, review_sha256="0" * 64),
            dict(self.args, accept_constraint_violations=False),
        ]:
            with self.assertRaises(ValueError):
                enqueue(self.state, self.session, **args)
        self.session.status = "running"
        with self.assertRaises(QueueConflict):
            enqueue(self.state, self.session, **self.args)
        self.assertEqual(list_receipts(self.state), [])

    def test_unknown_evidence_is_not_a_low_violation_deployment(self):
        self.session.result["operator_review"] = shortlist(
            [candidate("a", 40, [(0, 3)], unknown=True)]
        )
        self.args["review_sha256"] = self.session.result["operator_review"]["review_sha256"]
        with self.assertRaises(ValueError):
            enqueue(self.state, self.session, **self.args)

    def test_split_routes_preserve_shared_controls_and_only_restore_changed_controls(self):
        row = candidate("a", 40, [(1, 3)])
        row["changes"]["ttt"] = 128
        row["parameters"]["ttt"] = 128
        self.session.result["operator_review"] = shortlist([row])
        self.args.update(
            target_ric="split",
            review_sha256=self.session.result["operator_review"]["review_sha256"],
        )
        receipt = enqueue(self.state, self.session, **self.args)
        self.assertEqual([r["ric"] for r in receipt["routes"]], ["non_rt", "near_rt"])
        self.assertEqual(receipt["routes"][1]["changes"], {"ttt": 128})
        self.assertEqual(receipt["routes"][1]["schedule"][1]["changes"], {"ttt": 256})
        self.assertEqual(len(list_receipts(self.state, "near_rt")), 1)

    def test_http_queue_export_and_restored_receipt_and_untrusted_parameters_rejected(self):
        with (
            patch("agora_offline.server.STATE", self.state),
            patch.dict(server.sessions, {self.session.id: self.session}, clear=True),
        ):
            client = TestClient(server.app)
            url = f"/api/sessions/{self.session.id}"
            self.assertEqual(
                client.post(
                    url + "/deploy", json=dict(self.args, parameters={"txp_5g": 90})
                ).status_code,
                422,
            )
            response = client.post(url + "/deploy", json=self.args)
            self.assertEqual(response.status_code, 200, response.text)
            self.assertEqual(client.get(url + "/deployment").json(), response.json())
            self.assertEqual(client.get(url).json()["deployment"], response.json())
            self.assertEqual(client.get(url + "/export").json()["deployment"], response.json())
            self.assertEqual(
                len(client.get("/api/deployments?ric=non_rt").json()["deployments"]), 1
            )
            self.assertEqual(client.get("/api/deployments?ric=near_rt").json()["deployments"], [])


if __name__ == "__main__":
    unittest.main()
