"""Policy/metric edge cases that can otherwise turn a conflict into a pass."""

from copy import deepcopy
import json
import unittest

from agora_offline.contracts import CELLS, ContractError
from agora_offline.estimator import Estimator, MetricUnavailable, aggregate
from agora_offline.intent_policy import catalog, compile_spec, default_spec, entry
from agora_offline.tests.test_estimator import FakeRuntime, CURRENT, plan


class PolicyTests(unittest.TestCase):
    def test_multiple_goals_and_constraints_remain_independent(self):
        spec = default_spec(180)
        spec["goals"].append(entry("goodput", 0.1, ident="more_goodput"))
        spec["base_constraints"][0]["target"] = 97
        spec["base_constraints"][1]["enabled"] = False
        spec["additional_constraints"].append(entry("prb_max", 70, "extra", ident="prb"))
        result = compile_spec(spec, 180)
        self.assertEqual(len(result["goals"]), 2)
        self.assertEqual(len(result["constraints"]), 5)
        self.assertEqual(result["constraints"][0]["target"], 97)
        self.assertTrue(all(e["target_kind"] == "absolute" for e in result["constraints"]))
        self.assertEqual(result["constraints"][0]["recommendation"]["target"], 95)
        spec["base_constraints"][0]["target"] = 5
        self.assertEqual(result["constraints"][0]["target"], 97)

    def test_invalid_and_mismatched_policy_cannot_reach_forecast(self):
        spec = default_spec(180)
        with self.assertRaises(ContractError):
            compile_spec(spec, 90)
        spec["base_constraints"][0]["target_kind"] = "relative_fraction"
        with self.assertRaisesRegex(ContractError, "threshold"):
            compile_spec(spec, 180)
        spec = default_spec(180, "coverage")
        spec["goals"][0]["target_kind"] = "relative_fraction"
        with self.assertRaisesRegex(ContractError, "dB"):
            compile_spec(spec, 180)
        spec = default_spec(180, "load_balancing")
        spec["goals"][0]["cells"] = ["5g"]
        with self.assertRaisesRegex(ContractError, "at least two"):
            compile_spec(spec, 180)
        spec = default_spec(180)
        spec["goals"][0]["target"] = -0.1
        with self.assertRaisesRegex(ContractError, "direction"):
            compile_spec(spec, 180)

    def test_network_retention_allows_traffic_to_move_between_cells(self):
        keep = FakeRuntime._prediction(CURRENT, 0)
        prediction = deepcopy(keep)
        prediction["5g"]["goodput_mbps"] -= 15
        prediction["4g2"]["goodput_mbps"] += 15
        spec = compile_spec(default_spec(180), 180)
        guard = spec["constraints"][0]
        result = Estimator(FakeRuntime())._expectation(guard, prediction, keep, keep, 180, 0)
        self.assertEqual(result["predicted_value"], 100)
        self.assertEqual(result["effective_threshold"], 95)
        self.assertTrue(result["satisfied"])

    def test_ratios_pool_counts_and_zero_denominator_is_not_a_pass(self):
        prediction = {
            "5g": {"pingpongs": 1.0, "ho_attempts": 10.0},
            "4g1": {"pingpongs": 9.0, "ho_attempts": 30.0},
        }
        guard = entry("pp_ratio", 26, "base", cells=["5g", "4g1"])
        self.assertEqual(aggregate(prediction, guard), 25)
        prediction["4g1"]["ho_attempts"] = 0
        prediction["5g"]["ho_attempts"] = 0
        prediction["4g1"]["pingpongs"] = prediction["5g"]["pingpongs"] = 0
        result = Estimator(FakeRuntime())._expectation(
            guard, prediction, prediction, prediction, 180, 0
        )
        self.assertFalse(result["satisfied"])
        self.assertEqual(result["evaluation_status"], "insufficient_evidence")
        json.dumps(result, allow_nan=False)

    def test_low_sample_ratio_and_zero_baseline_improvement_are_not_claimed(self):
        pred = {"5g": {"pingpongs": 0.0, "ho_attempts": 9.0, "rlfs": 0.0}}
        with self.assertRaises(MetricUnavailable):
            aggregate(pred, entry("pp_ratio", 13.0, "base", cells=["5g"]))
        result = Estimator(FakeRuntime())._expectation(
            entry("rlf", -0.2, cells=["5g"]), pred, pred, pred, 180, 0
        )
        self.assertFalse(result["satisfied"])
        self.assertEqual(result["evaluation_status"], "insufficient_evidence")

    def test_energy_is_analytic_and_efficiency_is_ratio_of_totals(self):
        spec = default_spec(180)
        spec["base_constraints"] = []
        spec["additional_constraints"] = []
        result = Estimator(FakeRuntime()).evaluate(
            "run", 180, compile_spec(spec, 180), plan("request"), [plan("request")]
        )
        row = result["all_candidates"][0]
        expected_power = 130 + 4.7 * 10**0.3
        self.assertAlmostEqual(row["predictions"]["5g"]["power_w"], expected_power)
        self.assertEqual(row["model_predictions"]["5g"]["power_w"], 33)
        predicted = row["predictions"]
        expected = sum(predicted[c]["goodput_mbps"] for c in CELLS) / sum(
            predicted[c]["power_w"] for c in CELLS
        )
        self.assertAlmostEqual(aggregate(predicted, entry("ee", 1)), expected)
        self.assertAlmostEqual(row["goals"][0]["predicted_value"], expected)

    def test_jain_and_coverage_delta_use_physical_units(self):
        equal = {
            c: {"prb_pct": 50, "rsrp_interval_mean_dbm": -80, "rsrp_p10_dbm": -90} for c in CELLS
        }
        self.assertEqual(aggregate(equal, entry("prb_jain", 0.9)), 1.0)
        spec = compile_spec(default_spec(180, "coverage"), 180)
        predicted = {
            c: {"prb_pct": 50, "rsrp_interval_mean_dbm": -77, "rsrp_p10_dbm": -87} for c in CELLS
        }
        result = Estimator(FakeRuntime())._expectation(
            spec["goals"][0], predicted, equal, equal, 180, 0
        )
        self.assertEqual(result["effective_threshold"], -87)
        self.assertTrue(result["satisfied"])

    def test_recommendations_are_train_only_and_no_fake_absolute_loss_limit(self):
        data = catalog(180)
        self.assertEqual(data["reference"]["partition"], "train")
        self.assertEqual(data["reference"]["runs"], 720)
        self.assertNotIn("loss", {m["key"] for m in data["metrics"]})
        self.assertIn("rsrp_p10", {m["key"] for m in data["metrics"]})


if __name__ == "__main__":
    unittest.main()
