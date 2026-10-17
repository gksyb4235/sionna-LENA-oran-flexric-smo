"""Synthetic process integration checks; not evidence of ns-3 radio correctness."""
from copy import deepcopy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

BUNDLE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BUNDLE))
import sweep
import combine_reports


def fixture_job():
    initial = {p: values[0] for p, values in sweep.DOMAINS.items()}
    initial["txp_5g"] = 43
    initial["ret_4g2"] = 9
    return {"job_id": "fixture", "trace_seed": 1000, "trace_path": "trace.csv", "initial_app": "TM",
            "initial_parameters": initial, "screening_priority": 1, "screening_tier": "A",
            "events": [{"time_s": 180, "app": "ES", "updates": {"txp_5g": 33}},
                       {"time_s": 360, "app": "CV", "updates": {"ret_4g2": 12}},
                       {"time_s": 540, "app": "LB", "updates": {"cio_4g1": 2}},
                       {"time_s": 720, "app": "MO", "updates": {"ttt_ms": 720, "hys_db": 5}}]}


def fixture_manifest(job):
    return {"schema": sweep.SCHEMA, "experiment_id": "fixture", "duration_s": 900,
            "kpi_interval_s": 1, "n_ues": 300, "rng_seed": 1, "rng_run": 1, "jobs": [job]}


class ProcessContractTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "site").mkdir()
        self.job = fixture_job()
        self.manifest = fixture_manifest(self.job)
        self.config = {"ns3_root": str(self.root), "python_site_packages": "site"}

    def tearDown(self):
        self.temporary.cleanup()

    def run_fixture(self):
        def command(manifest, config, job, attempt):
            return [sys.executable, str(BUNDLE / "tests" / "dummy_backend.py"),
                    str(attempt / "job.json"), str(attempt / "data" / job["job_id"])]
        with patch.object(sweep, "argv_for", side_effect=command):
            item = sweep.launch(self.manifest, self.config, self.job, self.root / "results", {"fixture": True})
        item["process"].wait(timeout=20)
        result = sweep.finish(item)
        return item, result

    def test_real_dummy_process_receipts_and_2700_parameter_readbacks(self):
        sweep.validate_manifest(self.manifest)
        item, result = self.run_fixture()
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["verification"]["receipt_count"], 5)
        self.assertEqual(result["verification"]["cell_rows"], 2700)
        self.assertEqual(result["verification"]["last_cell_time_s"], 899.5)
        completed = sweep.completed_attempt(self.root / "results", self.job,
                                             sweep.canonical_hash(self.manifest), verify=True)
        self.assertEqual(completed, item["attempt"])

    def test_delayed_receipt_and_unexpected_parameter_change_are_rejected(self):
        item, _ = self.run_fixture()
        folder = item["attempt"] / "data" / self.job["job_id"]
        receipt = folder / "parameter_changes.csv"
        text = receipt.read_text(encoding="utf-8")
        receipt.write_text(text.replace("180,180,", "180,181,"), encoding="utf-8")
        self.assertFalse(sweep.verify_job(self.job, folder)["passed"])
        receipt.write_text(text, encoding="utf-8")
        path = folder / "cell_kpi.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        values = lines[400].split(",")
        values[2] = "34"
        lines[400] = ",".join(values)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        report = sweep.verify_job(self.job, folder)
        self.assertFalse(report["passed"])
        self.assertTrue(any("txp_" in issue for issue in report["issues"]))

    def test_failed_attempt_is_preserved_and_successful_data_never_reused(self):
        first, _ = self.run_fixture()
        first_bytes = (first["attempt"] / "execution.json").read_bytes()
        second, _ = self.run_fixture()
        self.assertNotEqual(first["attempt"], second["attempt"])
        self.assertEqual(first_bytes, (first["attempt"] / "execution.json").read_bytes())
        self.assertEqual(second["attempt"].name, "attempt0002")

    def test_same_timestamp_receipt_order_is_irrelevant_but_duplicates_fail(self):
        item, result = self.run_fixture()
        folder = item["attempt"] / "data" / self.job["job_id"]
        receipt = folder / "parameter_changes.csv"
        lines = receipt.read_text(encoding="utf-8").splitlines()
        self.assertIn(",hys_db,", lines[-2])
        self.assertIn(",ttt_ms,", lines[-1])
        lines[-2], lines[-1] = lines[-1], lines[-2]
        receipt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertTrue(sweep.verify_job(self.job, folder)["passed"])
        lines[-1] = lines[-2]
        receipt.write_text("\n".join(lines) + "\n", encoding="utf-8")
        report = sweep.verify_job(self.job, folder)
        self.assertFalse(report["passed"])
        self.assertTrue(any("Duplicate receipt" in issue for issue in report["issues"]))

    def test_noop_and_wrong_app_family_are_rejected(self):
        invalid = deepcopy(self.manifest)
        invalid["jobs"][0]["events"][0]["updates"]["txp_5g"] = 43
        with self.assertRaises(ValueError):
            sweep.validate_manifest(invalid)
        invalid = deepcopy(self.manifest)
        invalid["jobs"][0]["events"][0]["updates"] = {"cio_5g": 9}
        with self.assertRaises(ValueError):
            sweep.validate_manifest(invalid)

    def test_nine_shards_cover_500_without_overlap_and_pilot_keeps_assignment(self):
        manifest = deepcopy(self.manifest)
        manifest["jobs"] = []
        for i in range(500):
            job = deepcopy(self.job)
            job.update(job_id=f"case_{i:04d}", screening_priority=500-i)
            manifest["jobs"].append(job)
        shards = [sweep.selected_jobs(manifest, i, 9) for i in range(9)]
        ids = [job["job_id"] for shard in shards for job in shard]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(len(ids), 500)
        pilot = [job["job_id"] for job in manifest["jobs"][:20]]
        for i, full in enumerate(shards):
            subset = sweep.selected_jobs(manifest, i, 9, pilot_ids=pilot)
            self.assertEqual({j["job_id"] for j in subset}, {j["job_id"] for j in full} & set(pilot))

    def test_cross_pc_shared_source_mismatch_rejected_binary_build_difference_allowed(self):
        manifest = deepcopy(self.manifest)
        second_job = deepcopy(self.job)
        second_job["job_id"] = "fixture2"
        manifest["jobs"].append(second_job)
        paths = []
        for index, job in enumerate(manifest["jobs"]):
            report = {"manifest_hash": sweep.canonical_hash(manifest),
                      "jobs": [{"job_id": job["job_id"], "verified_complete": True}],
                      "input_provenance_available": True,
                      "shared_input_sha256": {"scratch/backend.cc": "same-source", "cache.h5": "same-cache"},
                      "frozen_inputs": {"scratch/backend.cc": "same-source", "cache.h5": "same-cache",
                                        "build/ns3": f"different-build-{index}"}}
            path = self.root / f"report{index}.json"
            sweep.atomic_json(path, report)
            paths.append(path)
        self.assertTrue(combine_reports.combine(manifest, paths)["all_verified_complete"])
        altered = sweep.read_json(paths[1])
        altered["shared_input_sha256"]["scratch/backend.cc"] = "different-source"
        altered["frozen_inputs"]["scratch/backend.cc"] = "different-source"
        sweep.atomic_json(paths[1], altered)
        with self.assertRaisesRegex(ValueError, "Cross-PC"):
            combine_reports.combine(manifest, paths)

    @unittest.skipUnless(sys.platform == "linux", "fcntl/process-group integration requires Linux")
    def test_linux_full_runner_resumes_complete_and_preserves_failed_attempt(self):
        self.config.update(output_root="distributed_results", memory_reserve_gib=0,
                           estimated_job_gib=.000001, disk_reserve_gib=0,
                           estimated_output_gib_per_job=.000001, startup_stagger_s=0, timeout_s=60)
        def command(manifest, config, job, attempt):
            return [sys.executable, str(BUNDLE / "tests" / "dummy_backend.py"),
                    str(attempt / "job.json"), str(attempt / "data" / job["job_id"])]
        def execute(retry=False):
            provenance = {"input_fingerprints": [{"path": "tests/dummy_backend.py", "sha256": "fixture-source", "roles": ["source"]}],
                          "manifest_hash": sweep.canonical_hash(self.manifest), "fixture": True}
            with patch.object(sweep, "preflight", return_value=provenance), \
                 patch.object(sweep, "argv_for", side_effect=command):
                return sweep.run_batch(self.manifest, self.config, [self.job], workers=1,
                                       shard_index=0, shard_count=1, resume=True, retry_failed=retry)
        first = execute()
        self.assertEqual(first["status"], "complete")
        collected = sweep.status(self.manifest, self.config, [self.job], verify=True)
        self.assertTrue(collected["all_verified_complete"])
        self.assertEqual(collected["shared_input_sha256"], {"tests/dummy_backend.py": "fixture-source"})
        root = sweep.output_root(self.config)
        attempts = sweep.attempt_paths(root, self.job)
        self.assertEqual(len(attempts), 1)
        second = execute()
        self.assertEqual(second["skipped_complete"], ["fixture"])
        self.assertEqual(len(sweep.attempt_paths(root, self.job)), 1)
        old_file = attempts[0] / "execution.json"
        record = sweep.read_json(old_file)
        record.update(status="failed", returncode=1)
        sweep.atomic_json(old_file, record)
        prior = old_file.read_bytes()
        with self.assertRaises(ValueError):
            execute()
        third = execute(retry=True)
        self.assertEqual(third["status"], "complete")
        self.assertEqual(len(sweep.attempt_paths(root, self.job)), 2)
        self.assertEqual(old_file.read_bytes(), prior)


if __name__ == "__main__":
    unittest.main()
