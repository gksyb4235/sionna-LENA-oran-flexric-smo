"""Meaningful regression checks for frozen-model parity and history isolation.

Portable checks:
  python -m agora_offline.runtime.test_runtime
Optional exact parity against original training-workspace implementation:
  python -m agora_offline.runtime.test_runtime --workspace /path/to/workspace
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import unittest
import numpy as np
from .api import Runtime

WORKSPACE = None
REPORT = {}


class RuntimeChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = Runtime()
        cls.row = cls.runtime.list_runs("val")[0]

    def plan(self, **changes):
        parameters = {**self.row["current_rcp"], **changes}
        return {"candidate_id": "test", "parameters": parameters, "delay_s": 0, "horizon_s": 180}

    def test_frozen_checkpoint_and_split(self):
        meta = self.runtime.metadata()
        self.assertEqual(meta["registered_parameters"], 147284)
        self.assertEqual(meta["horizons_s"], [90, 180])
        self.assertEqual([len(self.runtime.list_runs(x)) for x in ("train", "val", "test")], [720, 180, 100])
        self.assertTrue(all("original_cell_csv_sha256" in x for x in self.runtime.list_runs()))
        json.dumps(meta, allow_nan=False)

    def test_no_future_observation_leakage(self):
        runtime, run_id, t = self.runtime, self.row["run_id"], 180
        plans = [self.plan(txp_5g=33)]
        before_snapshot = runtime.snapshot(run_id, t)
        before = runtime.predict_batch(run_id, t, plans)
        source = runtime._run(run_id)
        backups = {k: v[t:].copy() for k, v in source.items()}
        try:
            for key, value in source.items():
                value[t:] = False if key == "valid" else np.nan
            after_snapshot = runtime.snapshot(run_id, t)
            after = runtime.predict_batch(run_id, t, plans)
            self.assertEqual(before_snapshot, after_snapshot)
            self.assertEqual(before, after)
            self.assertLess(max(after_snapshot["history"]["times_s"]), t)
        finally:
            for key, values in backups.items():
                source[key][t:] = values
        REPORT["future_data_mutation_has_no_effect"] = True

    def test_native_counter_aggregation(self):
        run_id = self.row["run_id"]
        source = self.runtime._run(run_id)
        snap = self.runtime.snapshot(run_id, 180)["observed_90s"]
        self.assertEqual([snap["start_s"], snap["end_s"]], [90, 180])
        for ci, cell in enumerate(self.runtime.cell_labels):
            actual = snap["kpis"][cell]
            for ki in (7, 8, 9):
                self.assertEqual(actual[self.runtime.target_names[ki]], float(source["num"][90:180, ci, ki].sum()))
            for ki in (0, 2, 4, 5, 6, 10, 11, 12, 13):
                expected = source["num"][90:180, ci, ki].sum() / source["den"][90:180, ci, ki].sum()
                self.assertAlmostEqual(actual[self.runtime.target_names[ki]], float(expected), places=9)
        REPORT["native_counter_aggregation_verified"] = True

    def test_out_of_domain_and_interpolation(self):
        run_id = self.row["run_id"]
        for key, value in [("txp_5g", 46), ("ttt_ms", 1024), ("cio_5g", -1)]:
            with self.assertRaises(ValueError):
                self.runtime.predict_batch(run_id, 180, [self.plan(**{key: value})])
        for value in (36, 41):
            result = self.runtime.predict_batch(run_id, 180, [self.plan(txp_5g=value)])[0]
            self.assertTrue(result["interpolation"])
            self.assertIn("txp_5g", result["warnings"][0])
        for delay, horizon, t in [(120, 180, 180), (0, 150, 180), (90, 180, 660)]:
            plan = {**self.plan(), "delay_s": delay, "horizon_s": horizon}
            with self.assertRaises(ValueError):
                self.runtime.predict_batch(run_id, t, [plan])
        with self.assertRaises(ValueError):
            self.runtime.predict_batch(run_id, 180, [self.plan(), self.plan()])

    def test_masked_warmup_and_json(self):
        run_id = self.row["run_id"]
        snapshot = self.runtime.snapshot(run_id, 30)
        self.assertEqual(snapshot["history_rows"], 30)
        self.assertFalse(snapshot["observed_90s"]["complete_window"])
        outputs = self.runtime.predict_batch(run_id, 30, [self.plan()])
        json.dumps({"snapshot": snapshot, "outputs": outputs}, allow_nan=False)

    def test_coordinate_reset_explanation(self):
        plan = self.plan(txp_5g=36)
        result = self.runtime.explain_batch(self.row["run_id"], 180, [plan])[0]
        self.assertEqual(set(result["contrasts"]), set(self.runtime.parameter_names))
        for key in self.runtime.parameter_names:
            if key != "txp_5g":
                self.assertTrue(all(v == 0 for cell in result["contrasts"][key].values() for v in cell.values()))
        keep = self.runtime.explain_batch(self.row["run_id"], 180, [self.plan()])[0]
        self.assertTrue(all(v == 0 for effect in keep["contrasts"].values() for cell in effect.values() for v in cell.values()))

    def test_validation_calibration_provenance(self):
        calibration = self.runtime.hold_action_calibration
        if calibration is None:
            self.skipTest("Optional descriptive keep-action calibration has not been generated")
        self.assertEqual(calibration["partition"], "val")
        self.assertEqual(calibration["scope"], "keep_current_only")
        self.assertFalse(calibration["switched_action_calibrated"])
        self.assertFalse(calibration["formal_coverage_guarantee"])
        path = self.runtime.root / calibration["residual_file"]
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), calibration["residual_file_sha256"])
        with np.load(path, allow_pickle=False) as source:
            self.assertEqual(source["prediction"].shape, (540, 2, 3, 14))
            self.assertEqual(len(set(source["run_ids"])), 180)
            self.assertEqual(set(source["run_ids"]), {r["run_id"] for r in self.runtime.list_runs("val")})
            self.assertEqual(set(source["issue_times_s"]), {180, 360, 540})
            residual = source["residual_truth_minus_prediction"]
            for hi, h in enumerate((90, 180)):
                for ci, cell in enumerate(self.runtime.cell_labels):
                    for ki, kpi in enumerate(self.runtime.target_names):
                        expected = float(np.sqrt(np.nanmean(residual[:, hi, ci, ki] ** 2)))
                        self.assertAlmostEqual(calibration["rmse"][str(h)]["0"][cell][kpi], expected, places=9)
        REPORT["validation_only_calibration_provenance_verified"] = True

    def test_portable_reference_predictions(self):
        path = Path(__file__).resolve().parent / "reference_predictions.json"
        if not path.exists():
            self.skipTest("Optional original-model cross-platform fixture absent")
        fixture = json.loads(path.read_text(encoding="utf-8"))
        maxima = []
        for case in fixture["cases"]:
            x, mask, current, _ = self.runtime._observed_inputs(case["run_id"], case["issue_time_s"])
            actual = self.runtime.forecast.predict_plans(x, mask, current, case["issue_time_s"],
                                                        case["parameters"], case["delays_s"])["kpi"]
            expected = np.asarray(case["predictions"])
            maxima.append(float(np.max(np.abs(actual - expected))))
            np.testing.assert_allclose(actual, expected, atol=fixture["absolute_tolerance"],
                                       rtol=fixture["relative_tolerance"])
        REPORT["portable_reference"] = {"max_abs_difference": max(maxima), "cases": len(maxima),
                                        "atol": fixture["absolute_tolerance"], "rtol": fixture["relative_tolerance"]}

    def test_original_90_180_bit_exact_parity(self):
        if WORKSPACE is None:
            self.skipTest("Original workspace optional; portable checks run without it")
        sys.path.insert(0, str(WORKSPACE))
        from agora_final_90_180.inference import Forecast as Original
        original = Original(WORKSPACE / "agora_final_90_180/model/parameter5_backbone.pt")
        maxima = []
        started = time.perf_counter()
        for split, t in (("train", 30), ("val", 180), ("test", 630)):
            row = self.runtime.list_runs(split)[0]
            x, mask, current, _ = self.runtime._observed_inputs(row["run_id"], t)
            alternative = current.copy()
            alternative[0] = 36
            candidates = np.stack([current, alternative])
            expected = original.predict_plans(x, mask, current, t, candidates, [0, 30, 60, 90])
            actual = self.runtime.forecast.predict_plans(x, mask, current, t, candidates, [0, 30, 60, 90])
            maxima.append(float(np.max(np.abs(expected["kpi"] - actual["kpi"]))))
            np.testing.assert_array_equal(actual["kpi"], expected["kpi"])
            np.testing.assert_array_equal(actual["predicted_pre_action_state"], expected["predicted_pre_action_state"])
        REPORT["parity"] = {"max_abs_difference": max(maxima), "cases": 3,
                            "candidates_per_case": 2, "delays": [0, 30, 60, 90],
                            "horizons": [90, 180], "elapsed_s": time.perf_counter() - started}


def main():
    global WORKSPACE
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path)
    args = parser.parse_args()
    WORKSPACE = args.workspace.resolve() if args.workspace else None
    if WORKSPACE:
        for relative in ("experiment/khu/studies/legacy", "experiments/legacy"):
            if (WORKSPACE / relative).is_dir():
                WORKSPACE = WORKSPACE / relative
                break
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(RuntimeChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    REPORT.update(tests_run=result.testsRun, failures=len(result.failures), errors=len(result.errors),
                  skipped=len(result.skipped), all_passed=result.wasSuccessful())
    path = Path(__file__).resolve().parent / "parity_report.json"
    path.write_text(json.dumps(REPORT, indent=2), encoding="utf-8")
    print(json.dumps(REPORT, indent=2))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
