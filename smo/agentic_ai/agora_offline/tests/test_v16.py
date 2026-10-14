"""Physical KPI definitions and portable inference against frozen test fixtures."""

import json
from pathlib import Path
import unittest
import numpy as np
from agora_offline.contracts import CELLS, ContractError
from agora_offline.estimator import aggregate, MetricUnavailable
from agora_offline.intent_policy import default_spec, compile_spec
from agora_offline.planning import parse_deterministic
from agora_offline.runtime import Runtime


class V16Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = Runtime()

    def test_cpu_service_matches_frozen_gpu_evaluation(self):
        fixture = json.loads(Path(__file__).with_name("inference_fixture_v16.json").read_text())
        self.assertEqual(fixture["checkpoint_sha256"], self.runtime.checkpoint_sha256)
        for case in fixture["cases"]:
            rows = self.runtime.predict_batch(case["run_id"], case["issue_time_s"], case["plans"])
            numeric = [
                [
                    [
                        row["predictions"][c][self.runtime.target_names[k]]
                        for k in fixture["target_indices"]
                    ]
                    for c in CELLS
                ]
                for row in rows
            ]
            scale = np.stack(
                [fixture["target_std"][[90, 180].index(p["horizon_s"])] for p in case["plans"]]
            )
            self.assertLess(
                float(np.max(np.abs(np.array(numeric) - case["expected"]) / scale)), 0.001
            )

    def test_future_replay_is_not_available_to_forecast(self):
        runtime = self.runtime
        record = runtime._run("gnn_0130")
        saved = record["x"].copy()
        params = runtime._index["gnn_0130"]["current_rcp"]
        plan = dict(
            candidate_id="test", parameters=dict(params, txp_5g=36.0), delay_s=90, horizon_s=180
        )
        before = runtime.predict_batch("gnn_0130", 180, [plan])
        try:
            record["x"][180:] = 1e9
            after = runtime.predict_batch("gnn_0130", 180, [plan])
            self.assertEqual(before, after)
        finally:
            record["x"][:] = saved

    def test_snapshot_p10_pools_the_whole_window(self):
        r = self.runtime._run("gnn_0130")
        snap = self.runtime.snapshot("gnn_0130", 180)
        for c, cell in enumerate(CELLS):
            use = (r["rsrp_times"] >= 90) & (r["rsrp_times"] < 180) & (r["rsrp_cells"] == c + 1)
            values = r["rsrp_values"][use]
            row = snap["observed_90s"]["kpis"][cell]
            self.assertAlmostEqual(row["rsrp_p10_dbm"], float(np.quantile(values, 0.1)), places=4)
            self.assertAlmostEqual(row["rsrp_interval_mean_dbm"], float(values.mean()), places=4)

    def test_pp_ratio_pools_successes_and_checks_minimum_evidence(self):
        p = {
            "5g": dict(pingpongs=2.0, ho_successes=10.0),
            "4g1": dict(pingpongs=0.0, ho_successes=30.0),
        }
        e = dict(metric="pingpong_per_success_pct", cells=["5g", "4g1"], aggregation="sum")
        self.assertEqual(aggregate(p, e), 5.0)
        p["5g"]["ho_successes"] = 2.0
        p["4g1"]["ho_successes"] = 0.0
        with self.assertRaises(MetricUnavailable):
            aggregate(p, e)

    def test_gap_does_not_replace_absolute_p10_protection(self):
        policy = compile_spec(default_spec(180, "coverage"), 180)
        self.assertEqual(
            {g["metric"] for g in policy["goals"]}, {"rsrp_p10_dbm", "rsrp_interval_mean_dbm"}
        )
        self.assertTrue(
            any(
                c["metric"] == "rsrp_p10_dbm" and c["operator"] == ">="
                for c in policy["constraints"]
            )
        )
        p = {"5g": dict(rsrp_interval_mean_dbm=-105.0, rsrp_p10_dbm=-103.0)}
        self.assertEqual(
            aggregate(p, dict(metric="rsrp_mean_p10_gap_db", cells=["5g"], aggregation="max")), -2.0
        )
        bad = default_spec(180, "coverage")
        bad["goals"][0]["target_kind"] = "relative_fraction"
        with self.assertRaises(ContractError):
            compile_spec(bad, 180)

    def test_five_apps_and_compound_mobility_parse(self):
        current = self.runtime._index["gnn_0130"]["current_rcp"]
        cases = [
            ("Cell5G TxP43 ->33", "energy_saving"),
            ("Cell5G TxP43 ->43", "throughput"),
            ("Cell5G RET7 ->11", "coverage"),
            ("Cell5G CIO7 ->9", "load_balancing"),
            ("TTT512 ->256 ms HYS2 ->3 dB", "handover_control"),
        ]
        for prompt, objective in cases:
            self.assertEqual(parse_deterministic(prompt, current)["objective"], objective)
        mobility = parse_deterministic(cases[-1][0], current)
        self.assertEqual(mobility["changes"], {"ttt_ms": 256.0, "hys_db": 3.0})
        meta = self.runtime.metadata()
        self.assertEqual(
            (len(meta["apps"]), meta["graph_nodes"], len(meta["predicted_kpis"])), (5, 46, 11)
        )


if __name__ == "__main__":
    unittest.main()
