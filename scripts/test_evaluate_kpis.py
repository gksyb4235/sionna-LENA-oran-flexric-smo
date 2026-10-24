"""Tests for interval weighting, failure exclusion, and evaluation provenance."""

import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import evaluate_kpis as evaluate


class EvaluationTests(unittest.TestCase):
    def write_rows(self, path, rows):
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    def row(self, timestamp=.5, interval=1, **extras):
        return dict(time_s=timestamp, interval_s=interval, cell_id=1,
                    tx_packets_interval=10, rx_packets_interval=8,
                    tx_bytes_interval=1000, rx_bytes_interval=800, **extras)

    def test_ratios_use_summed_counts_and_delay_samples(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cell_kpi.csv"
            self.write_rows(path, [
                self.row(delay_samples_interval=1, delay_sum_ms_interval=100,
                         delay_p95_ms=100, prb_used_reg=1, prb_capacity_reg=2,
                         interval_energy_j=10, ho_success_count=10, ho_successes_interval=1),
                self.row(2.5, 2, delay_samples_interval=9, delay_sum_ms_interval=90,
                         delay_p95_ms=10, prb_used_reg=9, prb_capacity_reg=90,
                         interval_energy_j=20, ho_success_count=11, ho_successes_interval=1)])
            rows, _ = evaluate.aggregate_cell_csv(path)
            row = rows[0]
            self.assertAlmostEqual(row["goodput_mbps"], 1600 * 8 / 3e6)
            self.assertAlmostEqual(row["prb_utilization_fraction"], 10 / 92)
            self.assertEqual(row["delay_mean_ms"], 19)
            self.assertEqual(row["ho_successes_interval"], 2)
            self.assertNotIn("ho_success_count", row)
            self.assertNotIn("delay_p95_ms", row)
            self.assertAlmostEqual(row["energy_efficiency_mbit_per_j"], .0128 / 30)

    def test_windows_preserve_half_second_report_origin(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cell_kpi.csv"
            self.write_rows(path, [self.row(index + .5) for index in range(61)])
            _, windows = evaluate.aggregate_cell_csv(path, 60)
            self.assertEqual([row["rows"] for row in windows], [60, 1])
            self.assertEqual(windows[0]["window_start_s"], -.5)
            self.assertEqual(windows[0]["window_end_s"], 59.5)
            self.assertEqual(windows[1]["nominal_duration_s"], 1)

    def test_rejects_intervals_crossing_window_boundaries_and_missing_ticks(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cell_kpi.csv"
            self.write_rows(path, [self.row(40, 40), self.row(80, 40)])
            with self.assertRaisesRegex(ValueError, "crosses"):
                evaluate.aggregate_cell_csv(path, 60)
            self.write_rows(path, [self.row(.5), self.row(2.5)])
            with self.assertRaisesRegex(ValueError, "missing interval"):
                evaluate.aggregate_cell_csv(path)

    def test_raw_opt_in_never_includes_failed_receipts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertFalse(evaluate.raw_run(root, False)["included"])
            self.assertTrue(evaluate.raw_run(root, True)["included"])
            receipt = root / "run_status.json"
            receipt.write_text(json.dumps({"status": "failed", "exit_code": 134}))
            self.assertFalse(evaluate.raw_run(root, True)["included"])
            receipt.write_text(json.dumps({"status": "complete", "exit_code": 0}))
            self.assertEqual(evaluate.raw_run(root, False)["verification"], "completion_receipt_only")

    def test_collect_hashes_detect_stale_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            artifacts = {}
            for name in ("cell_kpi.csv", "ue_kpi.csv", "parameter_changes.csv"):
                path = root / name
                path.write_text("test")
                artifacts[name] = {"sha256": evaluate.digest(path)}
            report = root / "collect.json"
            report.write_text(json.dumps({"jobs": [{"job_id": "conflict_0501",
                "output": str(root), "verified_complete": True, "status": "complete",
                "returncode": 0, "verification": {"passed": True, "artifacts": artifacts}}]}))
            self.assertTrue(evaluate.collection_runs(report)[0]["included"])
            (root / "cell_kpi.csv").write_text("changed")
            self.assertFalse(evaluate.collection_runs(report)[0]["included"])

    def test_recorded_local_paths_still_validate_after_alias_removal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            studies = root / "experiment/khu/studies"
            results = root / "experiment/khu/results/agora_conflict_sweep"
            studies.mkdir(parents=True)
            results.mkdir(parents=True)
            (studies / "agora_conflict_sweep").symlink_to("../results/agora_conflict_sweep")
            artifacts = {}
            for name in ("cell_kpi.csv", "ue_kpi.csv", "parameter_changes.csv"):
                (results / name).write_text("unchanged data")
                artifacts[name] = {"sha256": evaluate.digest(results / name)}
            legacy = root / "experiments/agora_conflict_sweep"
            report = results / "collect.json"
            report.write_text(json.dumps({"jobs": [{"job_id": "conflict_0501",
                "output": str(legacy), "verified_complete": True, "status": "complete",
                "returncode": 0, "verification": {"passed": True, "artifacts": artifacts}}]}))
            before = report.read_bytes()
            with patch.object(evaluate, "ROOT", root):
                item = evaluate.collection_runs(legacy / "collect.json")[0]
                self.assertTrue(item["included"])
                self.assertEqual(item["directory"], str(results))
                self.assertEqual(evaluate.raw_run(legacy, True)["directory"], str(results))
                external = root / "other_checkout/experiments/run1"
                self.assertEqual(evaluate.local_run_path(external), external)
            self.assertFalse((root / "experiments").exists())
            self.assertEqual(report.read_bytes(), before)

    def test_cli_marks_raw_and_does_not_overwrite_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_rows(root / "cell_kpi.csv", [self.row()])
            args = ["--experiment-kind", "fixed", "--run", str(root),
                    "--include-unverified", "--output-dir", str(root / "evaluation")]
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(evaluate.main(args), 0)
            report = evaluate.read_json(root / "evaluation/evaluation.json")
            self.assertEqual(report["runs"][0]["verification"], "raw_unverified")
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                evaluate.main(args)

    def test_zero_denominator_is_null_and_bad_headers_fail(self):
        self.assertIsNone(evaluate.ratio(0, 0))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cell_kpi.csv"
            path.write_text("time_s,time_s,cell_id\n1,1,1\n")
            with self.assertRaisesRegex(ValueError, "header"):
                evaluate.aggregate_cell_csv(path)


if __name__ == "__main__":
    unittest.main()
