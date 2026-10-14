"""Decision-semantic tests independent of model accuracy or private data."""

from copy import deepcopy
import unittest

from agora_offline.contracts import (
    CELLS,
    METRICS,
    ContractError,
    make_research_intent,
    validate_parameters,
)
from agora_offline.estimator import Estimator


CURRENT = {
    **{f"txp_{c}": 43.0 for c in CELLS},
    **{f"ret_{c}": 7.0 for c in CELLS},
    "cio_5g": 7.0,
    "cio_4g1": 0.0,
    "cio_4g2": 0.0,
    "ttt_ms": 512.0,
    "hys_db": 2.0,
}


class FakeRuntime:
    """Synthetic controlled forecasts, clearly separated from real model results."""

    def __init__(self):
        self.calls = []

    def metadata(self):
        return {"test_double": True}

    def snapshot(self, run_id, t):
        predictions = self._prediction(CURRENT, 0)
        return {
            "current_rcp": deepcopy(CURRENT),
            "observed_90s": {"start_s": t - 90, "end_s": t, "kpis": predictions},
        }

    @staticmethod
    def _prediction(parameters, delay):
        result = {}
        for cell in CELLS:
            delta = CURRENT[f"txp_{cell}"] - parameters[f"txp_{cell}"]
            values = {metric: 10.0 for metric in METRICS}
            values.update(
                power_w=parameters[f"txp_{cell}"],
                goodput_mbps=30.0 + delay / 10 - delta,
                rlfs=delta,
                pingpongs=5.0,
                prb_pct=50.0 - delta,
            )
            result[cell] = values
        return result

    def predict_batch(self, run_id, t, plans):
        self.calls.append(deepcopy(plans))
        return [
            {
                "candidate_id": p["candidate_id"],
                "predictions": self._prediction(p["parameters"], p["delay_s"]),
                "interval_start_s": t + p["delay_s"],
                "interval_end_s": t + p["delay_s"] + p["horizon_s"],
                "warnings": [],
            }
            for p in plans
        ]


def plan(ident, value=33, delay=0, cell="5g"):
    return dict(candidate_id=ident, changes={f"txp_{cell}": value}, delay_s=delay)


def intent():
    return {
        "horizon_s": 90,
        "goals": [
            {
                "id": "power",
                "metric": "power_w",
                "cells": ["5g"],
                "aggregation": "sum",
                "operator": "<=",
                "target_kind": "absolute",
                "target": 40.0,
                "reference_kind": "keep_forecast_same_window",
                "weight": 1.0,
            }
        ],
        "constraints": [
            {
                "id": "rlf",
                "metric": "rlfs",
                "cells": ["5g"],
                "aggregation": "sum",
                "operator": "<=",
                "target": 8.0,
            }
        ],
        "safety": {"multiplier": 0.0},
    }


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        self.runtime = FakeRuntime()
        self.estimator = Estimator(self.runtime)

    def test_lexicographic_hard_feasibility_then_saturated_goal_then_deviation(self):
        result = self.estimator.evaluate(
            "run",
            180,
            intent(),
            plan("request"),
            [plan("request"), plan("weak", 40), plan("middle", 36)],
        )
        self.assertEqual(result["feasible_count"], 2)
        self.assertEqual([r["candidate_id"] for r in result["top_candidates"]], ["middle", "weak"])
        # More goal achievement cannot buy a violation; beyond-target progress
        # cannot outweigh less request deviation once both goals are saturated.
        self.assertFalse(result["all_candidates"][0]["recommended"])
        self.assertAlmostEqual(result["top_candidates"][0]["request_deviation"], 0.3)
        self.assertEqual(result["top_candidates"][0]["goal_achievement"], 1.0)

    def test_exact_request_is_one_candidate_with_internal_keep_baseline(self):
        result = self.estimator.evaluate(
            "run", 180, intent(), plan("request", 40), [plan("request", 40)]
        )
        self.assertEqual(result["requested_count"], 1)
        self.assertEqual(result["baseline_count"], 1)
        self.assertEqual(len(self.runtime.calls[-1]), 2)
        self.assertEqual(result["top_candidates"][0]["request_deviation"], 0)

    def test_continuous_txp_is_marked_unvalidated_and_not_snapped(self):
        result = self.estimator.evaluate(
            "run", 180, intent(), plan("request"), [plan("fraction", 36)]
        )
        row = result["top_candidates"][0]
        self.assertEqual(row["parameters"]["txp_5g"], 36.0)
        self.assertTrue(any(w["code"] == "interpolated_unvalidated" for w in row["warnings"]))
        with self.assertRaises(ContractError):
            validate_parameters(dict(CURRENT, ret_5g=6))

    def test_matched_delay_keep_and_observed_reference_remain_distinct(self):
        spec = intent()
        for name, reference in (
            ("observed", "observed_at_issue"),
            ("future", "keep_forecast_same_window"),
        ):
            spec["constraints"].append(
                dict(
                    id=name,
                    metric="goodput_mbps",
                    cells=["5g"],
                    aggregation="sum",
                    operator=">=",
                    target_kind="relative_fraction",
                    target=-0.2,
                    reference_kind=reference,
                )
            )
        result = self.estimator.evaluate("run", 180, spec, plan("request"), [plan("later", 40, 90)])
        row = result["all_candidates"][0]
        by_id = {c["id"]: c for c in row["constraints"]}
        self.assertEqual(by_id["observed"]["reference_value"], 30.0)
        self.assertEqual(by_id["future"]["reference_value"], 39.0)
        self.assertEqual(row["delta_vs_keep"]["5g"]["goodput_mbps"], -3.0)

    def test_unsafe_diagnostic_is_never_recommended(self):
        result = self.estimator.evaluate("run", 180, intent(), plan("request"), [plan("request")])
        self.assertEqual(result["top_candidates"], [])
        self.assertFalse(result["all_expectations_met"])
        self.assertEqual(len(result["infeasible_diagnostics"]), 1)
        self.assertFalse(result["infeasible_diagnostics"][0]["recommended"])

    def test_signed_attenuation_and_domain_relocation(self):
        result = self.estimator.evaluate(
            "run",
            180,
            intent(),
            plan("request"),
            [plan("relocate", 40, cell="4g2"), plan("too_strong", 30)],
        )
        rows = {r["candidate_id"]: r for r in result["all_candidates"]}
        self.assertEqual(rows["too_strong"]["status"], "invalid")
        self.assertAlmostEqual(rows["relocate"]["request_deviation"], 1.7)
        separate = Estimator(
            self.runtime, {"cell_ric_domains": {"5g": "a", "4g1": "a", "4g2": "b"}}
        )
        blocked = separate.evaluate(
            "run", 180, intent(), plan("request"), [plan("relocate", 40, cell="4g2")]
        )
        self.assertEqual(blocked["invalid_count"], 1)
        self.assertEqual(blocked["evaluated_count"], 0)

    def test_unsupported_delay_and_shared_cell_control_are_rejected(self):
        result = self.estimator.evaluate(
            "run",
            180,
            intent(),
            plan("request"),
            [
                plan("late", 40, 120),
                {"candidate_id": "shared", "changes": {"ttt_5g": 256}, "delay_s": 0},
            ],
        )
        self.assertEqual(result["invalid_count"], 2)
        self.assertEqual(result["all_candidates"][0]["candidate_id"], "late")

    def test_validation_margin_tightens_in_correct_direction(self):
        calibration = {
            "partition": "val",
            "calibration_id": "fixture",
            "delays_s": [0],
            "rmse": {"90": {"0": {"5g": {"rlfs": 1.0}}}},
        }
        estimator = Estimator(self.runtime, calibration=calibration)
        spec = intent()
        spec["safety"] = {"multiplier": 2.0, "calibration_id": "fixture"}
        result = estimator.evaluate(
            "run", 180, spec, plan("request"), [plan("a", 36), plan("b", 37)]
        )
        self.assertEqual(result["top_candidates"][0]["candidate_id"], "b")
        self.assertEqual(result["all_candidates"][0]["constraints"][0]["effective_threshold"], 6.0)
        spec["constraints"][0].update(metric="goodput_mbps", operator=">=", target=20.0)
        calibration["rmse"]["90"]["0"]["5g"]["goodput_mbps"] = 2.0
        result = Estimator(self.runtime, calibration=calibration).evaluate(
            "run", 180, spec, plan("request"), [plan("b", 37)]
        )
        self.assertEqual(result["all_candidates"][0]["constraints"][0]["effective_threshold"], 24.0)

    def test_no_test_calibration_or_uncalibrated_margin(self):
        with self.assertRaises(ContractError):
            Estimator(self.runtime, calibration={"partition": "test"})
        spec = intent()
        spec["safety"]["multiplier"] = 1.0
        with self.assertRaises(ContractError):
            self.estimator.evaluate("run", 180, spec, plan("request"), [plan("request")])

    def test_hold_calibration_cannot_be_claimed_for_changed_actions(self):
        calibration = {
            "partition": "val",
            "scope": "keep_current_only",
            "switched_action_calibrated": False,
            "delays_s": [0],
            "rmse": {"90": {"0": {"5g": {"rlfs": 1.0}}}},
        }
        spec = intent()
        spec["safety"]["multiplier"] = 1.0
        with self.assertRaisesRegex(ContractError, "unchanged settings only"):
            Estimator(self.runtime, calibration=calibration).evaluate(
                "run", 180, spec, plan("request"), [plan("weak", 40)]
            )

    def test_nonlinear_contribution_aggregates_reset_prediction(self):
        row = {
            "predictions": {"5g": {"prb_pct": 10.0}, "4g1": {"prb_pct": 9.0}},
            "keep_predictions": {"5g": {"prb_pct": 8.0}, "4g1": {"prb_pct": 9.0}},
            "changes": {"txp_5g": 40.0},
            "attribution": {
                "contrasts": {"txp_5g": {"5g": {"prb_pct": 2.0}, "4g1": {"prb_pct": 0.0}}}
            },
        }
        spec = {
            "goals": [
                {
                    "id": "balance",
                    "metric": "prb_pct",
                    "cells": ["5g", "4g1"],
                    "aggregation": "spread",
                    "operator": "<=",
                }
            ],
            "constraints": [],
        }
        self.estimator._attribution_diagnostics(row, spec)
        # spread(10,9)-spread(8,9)=0, whereas spread(2,0)=2 is wrong.
        self.assertEqual(row["attribution_diagnostics"][0]["effects"][0]["contribution"], 0.0)

    def test_research_thresholds_are_frozen_by_horizon_not_past_failures(self):
        snapshot = self.runtime.snapshot("run", 180)
        snapshot["observed_90s"]["kpis"]["5g"]["rlfs"] = 2.0
        generated = make_research_intent(snapshot, horizon_s=180)
        guard = next(g for g in generated["constraints"] if g["id"] == "base_rlf")
        self.assertEqual(guard["target"], 4.0)
        self.assertEqual(guard["cells"], list(CELLS))
        shorter = make_research_intent(snapshot, horizon_s=90)
        self.assertEqual(
            next(g for g in shorter["constraints"] if g["id"] == "base_rlf")["target"], 2.0
        )
        self.assertEqual(generated["policy_version"], "threshold-v3-app5")

    def test_negative_goal_achievement_is_not_clipped_to_zero(self):
        entry = intent()["goals"][0]
        observed = self.runtime.snapshot("run", 180)["observed_90s"]["kpis"]
        prediction = deepcopy(observed)
        prediction["5g"]["power_w"] = 46.0
        result = self.estimator._expectation(entry, prediction, observed, observed, 90, 0)
        self.assertEqual(result["achievement"], -1.0)


if __name__ == "__main__":
    unittest.main()
