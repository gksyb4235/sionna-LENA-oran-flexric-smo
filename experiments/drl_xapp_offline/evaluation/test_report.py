"""Small synthetic checks for aggregation and validation/test separation."""
import csv
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path


SPEC = importlib.util.spec_from_file_location("drl_report", Path(__file__).with_name("report.py"))
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def raw_rows(seconds=2):
    rows = []
    for second in range(seconds):
        for index, name in enumerate(("gNB_5G", "gNB_4G_1", "gNB_4G_2")):
            row = {key: 0 for key in report.REQUIRED}
            row.update(time_s=second + .5, cell=name, num_ues=10,
                       delivered_megabits_interval=2, generated_load_mbps=5,
                       prb_used_reg=20, prb_capacity_reg=100, interval_s=1)
            if seconds == 2:
                row["prb_used_reg"] = ((10, 20, 40), (30, 20, 40))[second][index]
                if index == 0:
                    row.update(ho_attempts_interval=(10, 100)[second],
                               ho_successes_interval=(8, 90)[second],
                               pingpongs_interval=(1, 9)[second],
                               rlfs_interval=(0, 1)[second],
                               delay_sum_ms_interval=(100, 90)[second],
                               delay_samples_interval=(1, 9)[second])
            rows.append(row)
    return rows


def parsed_rows(seconds=2):
    rows = raw_rows(seconds)
    return [{**row, "_raw": {k: str(v) for k, v in row.items()},
             "interval_energy_j": None, "tb_total_interval": None,
             "tb_corrupt_interval": None} for row in rows]


def write_run(root, phase="validation", status="complete", seconds=900):
    folder = Path(root) / "results" / "synthetic"
    folder.mkdir(parents=True)
    (folder / "execution.json").write_text(json.dumps({"status": status, "job": {
        "job_id": "synthetic", "phase": phase, "lb_start_s": 180,
        "mo_start_s": 540, "trace_seed": 9999, "arm": "none"}}), encoding="utf-8")
    rows = raw_rows(seconds)
    with (folder / "cell_kpi.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return folder


class ReportTests(unittest.TestCase):
    def test_ratio_of_sums_and_two_distinct_jain_aggregations(self):
        metrics = report.aggregate(parsed_rows(), 0, 2)
        self.assertAlmostEqual(metrics["pp_per_attempt_pct"], 100 * 10 / 110)
        self.assertAlmostEqual(metrics["pp_per_success_pct"], 100 * 10 / 98)
        self.assertAlmostEqual(metrics["prb_jain_of_interval_means"], 8 / 9)
        self.assertAlmostEqual(metrics["mean_per_second_prb_jain"], (7 / 9 + 27 / 29) / 2)
        self.assertAlmostEqual(metrics["delay_ms"], 19)
        self.assertAlmostEqual(metrics["rlf_per_1000_ue_s"], 1000 / 60)
        self.assertAlmostEqual(metrics["delivery_pct"], 40)

    def test_sparse_and_zero_load_metrics_are_undefined(self):
        rows = parsed_rows()
        for row in rows:
            row["ho_attempts_interval"] = row["ho_successes_interval"] = 0
            row["pingpongs_interval"] = row["prb_used_reg"] = 0
        metrics = report.aggregate(rows, 0, 2)
        self.assertIsNone(metrics["pp_per_attempt_pct"])
        self.assertIsNone(metrics["pp_per_success_pct"])
        self.assertIsNone(metrics["prb_jain_of_interval_means"])
        self.assertIsNone(metrics["mean_per_second_prb_jain"])
        self.assertFalse(report.score_metrics(metrics, "mo")["eligible"])

    def test_missing_duplicate_and_incomplete_windows_rejected(self):
        with self.assertRaises(ValueError):
            report.aggregate(parsed_rows()[:-1], 0, 2)
        rows = parsed_rows()
        with self.assertRaises(ValueError):
            report.aggregate(rows + [rows[0]], 0, 2)
        with self.assertRaises(ValueError):
            report.aggregate(rows, 0, 3)

    def test_failed_handovers_cannot_win_by_inflating_pp_denominator(self):
        rows = parsed_rows()
        for row in rows:
            row["ho_successes_interval"] = row["pingpongs_interval"] = 0
            row["ho_failures_interval"] = row["ho_attempts_interval"]
        score = report.score_metrics(report.aggregate(rows, 0, 2), "mo")
        self.assertEqual(score["components"]["objective"], 0)
        self.assertGreater(score["components"]["ho_failure_penalty"], 0)
        self.assertLess(score["score"], 0)

    def test_score_run_refuses_test_and_incomplete_data(self):
        with tempfile.TemporaryDirectory() as folder:
            run = write_run(folder, phase="test")
            with self.assertRaisesRegex(ValueError, "validation"):
                report.score_run(run, "lb")
        with tempfile.TemporaryDirectory() as folder:
            run = write_run(folder, status="running")
            with self.assertRaisesRegex(ValueError, "incomplete"):
                report.score_run(run, "lb")
        with tempfile.TemporaryDirectory() as folder:
            run = write_run(folder)
            score = report.score_run(run, "mo")
            self.assertEqual((score["start_s"], score["end_s"]), (540, 900))
            self.assertTrue(math.isfinite(score["score"]))

    def test_report_ignores_partial_and_records_invalid_completed_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            write_run(folder, status="running", seconds=2)
            result = report.build_report(folder)
            self.assertEqual(result["completed_valid_runs"], 0)
            self.assertEqual(len(result["skipped_incomplete_runs"]), 1)
        with tempfile.TemporaryDirectory() as folder:
            write_run(folder, status="complete", seconds=2)
            result = report.build_report(folder)
            self.assertEqual(len(result["invalid_complete_runs"]), 1)

    def test_completed_25_second_smoke_is_excluded_not_invalid(self):
        with tempfile.TemporaryDirectory() as folder:
            write_run(folder, phase="smoke", status="complete", seconds=25)
            result = report.build_report(folder)
            self.assertEqual(result["completed_valid_runs"], 0)
            self.assertEqual(result["invalid_complete_runs"], [])
            self.assertEqual(result["skipped_incomplete_runs"], [])
            self.assertEqual(len(result["excluded_runs"]), 1)
            self.assertEqual(result["excluded_runs"][0]["phase"], "smoke")
            self.assertIn("outside", result["excluded_runs"][0]["reason"])

    def test_prefix_comparison_is_before_onset_and_checks_raw_rows(self):
        a = {"job_id": "none", "_rows": parsed_rows(900)}
        b = {"job_id": "mo", "_rows": parsed_rows(900)}
        b["_rows"][3 * 540]["_raw"]["prb_used_reg"] = "99"
        self.assertTrue(report.prefix_check(a, b, 540)["identical"])
        self.assertFalse(report.prefix_check(a, b, 900)["identical"])

    def test_all_four_frozen_arms_have_expected_factorial_and_prefixes(self):
        runs = []
        for arm, rate in (("none", 2), ("lb_local", 3), ("mo_local", 4), ("both_local", 6)):
            rows = parsed_rows(900)
            for row in rows:
                t = row['time_s']
                value = rate if t >= 540 else 3 if t >= 180 and arm in ('lb_local', 'both_local') else 2
                row['delivered_megabits_interval'] = value
                row['_raw']['delivered_megabits_interval'] = str(value)
            runs.append({'job_id': arm, 'job': {'phase': 'test', 'trace_seed': 1419, 'arm': arm}, '_rows': rows})
        checks, interactions, warnings = report.factorial_analysis(runs)
        self.assertEqual(warnings, [])
        self.assertEqual(len(checks), 5)
        self.assertTrue(all(check['identical'] for check in checks))
        item = next(row for row in interactions if row['metric'] == 'goodput_mbps' and row['start_s'] == 540 and row['end_s'] == 720)
        self.assertEqual(item['interaction_both_minus_lb_minus_mo_plus_none'], 3)
        self.assertEqual(item['both_minus_lb'], 9)

    def test_latency_excludes_terminal_and_distinguishes_app_latency(self):
        with tempfile.TemporaryDirectory() as folder:
            control = Path(folder) / 'control'
            control.mkdir()
            values = [dict(terminal=False, controller_ms=5, updates={'ttt_ms': 512}, apps={'mo': {'decision_ms': 2}}),
                      dict(terminal=False, controller_ms=7, updates={}, apps={}),
                      dict(terminal=True, controller_ms=999, updates={}, apps={})]
            (control / 'actions.jsonl').write_text(''.join(json.dumps(value) + '\n' for value in values))
            result = report.control_statistics(folder)
            self.assertEqual(result['control_observations'], 2)
            self.assertEqual(result['controller']['median_ms'], 6)
            self.assertEqual(result['apps']['mo']['median_ms'], 2)
            self.assertEqual(result['apps']['lb']['samples'], 0)
            self.assertEqual(result['nonempty_update_count'], 1)


if __name__ == "__main__":
    unittest.main()
