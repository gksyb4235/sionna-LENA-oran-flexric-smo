"""Standard-library mocked tests; never execute an ns-3 binary."""
from __future__ import annotations

import copy
import csv
import io
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import two_switch_batch as runner


class BatchTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="two_switch_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.manifest = {
            "schema": runner.SCHEMA, "experiment_id": "test_two_switch",
            "backend_root": str(self.root), "binary": "build/mock-ns3",
            "ret_cache_root": "scenarios/khu-real/ret_caches",
            "output_root": "scenarios/results/new_batch", "duration_s": 900,
            "kpi_interval_s": 1, "n_ues": 300, "rng_seed": 1, "rng_run": 1,
            "baseline_parameters": {
                "txp_5g": 43, "txp_4g1": 43, "txp_4g2": 43,
                "ret_5g": 9, "ret_4g1": 9, "ret_4g2": 9,
                "cio_5g": 6, "cio_4g1": 0, "cio_4g2": -1,
                "ttt_ms": 512, "hys_db": 1,
            },
            "jobs": [{"job_id": "pair01_txp_then_ret", "trace_seed": 1405,
                      "trace_path": "scenarios/khu-real/ue_position/ue_positions_seed1405.csv",
                      "order": "txp_then_ret", "actions": [
                          {"time_s": 180, "parameter": "txp_5g", "value": 37},
                          {"time_s": 540, "parameter": "ret_4g1", "value": 12}]}],
        }
        self.job = self.manifest["jobs"][0]
        self.output = self.root / self.manifest["output_root"]
        (self.root / ".venv/lib/python3.12/site-packages").mkdir(parents=True)
        paths = [self.manifest["binary"], "scratch/khu-real-dual-nr-sionna-final.cc",
                 self.job["trace_path"], "scenarios/khu-real/gnbs-ret.csv"]
        for relative in paths:
            path = self.root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture content, never executed\n", encoding="utf-8")
            path.chmod(0o700)
        cache_root = self.root / self.manifest["ret_cache_root"]
        for angle, cell in ((9, "5g"), (9, "4g1"), (12, "4g1")):
            path = runner.cache_for_angle(cache_root, angle, cell)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fixture-cache-existence-and-hash-only")
        self.manifest_path = self.root / "manifest.json"
        self.manifest_path.write_text(json.dumps(self.manifest), encoding="utf-8")

    def make_outputs(self, job=None):
        job = job or self.job
        directory = self.output / job["job_id"]
        directory.mkdir(parents=True)
        with (directory / "parameter_changes.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=runner.RECEIPT_FIELDS)
            writer.writeheader()
            state = dict(self.manifest["baseline_parameters"])
            for action in job["actions"]:
                writer.writerow({"scheduled_time_s": action["time_s"], "applied_time_s": action["time_s"],
                                 "parameter": action["parameter"], "before": state[action["parameter"]],
                                 "after": action["value"], "status": "applied"})
                state[action["parameter"]] = action["value"]
        fields = ("time_s", "cell_id", "tx_power_dbm", "ret_tilt_deg", "cio_bias_db", "ttt_ms", "hysteresis_db")
        with (directory / "cell_kpi.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for second in range(900):
                state = runner.expected_parameters(self.manifest, job, second + .5)
                for cell, _, cell_id in runner.CELLS:
                    writer.writerow({"time_s": second + .5, "cell_id": cell_id,
                                     "tx_power_dbm": state[f"txp_{cell}"], "ret_tilt_deg": state[f"ret_{cell}"],
                                     "cio_bias_db": state[f"cio_{cell}"], "ttt_ms": state["ttt_ms"],
                                     "hysteresis_db": state["hys_db"]})
        (directory / "ue_kpi.csv").write_text("time_s,ue_id\n899.5,1\n", encoding="utf-8")
        return directory

    def test_argv_matches_static_backend_and_disables_external_interfaces(self):
        argv = runner.build_argv(self.manifest, self.job, self.output / "schedule.csv")
        expected = ["--bandwidth18=10e6", "--bandwidth35=10e6", "--numerology=0",
                    "--udpBaseIntervalMs=200", "--trafficMultiplierEnabled=true", "--sionnaUpdatePeriod=1s",
                    "--simTime=900", "--N_Ues=300",
                    "--ns3::NrHelper::E2ModeNr=false", "--runTag=pair01_txp_then_ret"]
        for item in expected:
            self.assertIn(item, argv)
        self.assertEqual(sum(value.startswith("--parameterSchedule=") for value in argv), 1)
        self.assertIn("--cellTxPowerDbm=gNB_5G:43,gNB_4G_1:43,gNB_4G_2:43", argv)
        self.assertNotIn("shell", argv)
        self.assertFalse(any(value.startswith(("--guiSrc=", "--influxSrc=")) for value in argv))

    def test_old_manifest_output_resolves_to_new_layout_without_mutation(self):
        studies = self.root / "experiment/khu/studies"
        studies.mkdir(parents=True)
        (studies / "two_switch_smoke").symlink_to("../results/two_switch_smoke")
        self.manifest["output_root"] = "experiments/two_switch_smoke/run1"
        before = runner.canonical_hash(self.manifest)
        target = self.root / "experiment/khu/results/two_switch_smoke/run1"
        self.assertEqual(runner.child_path(self.root, self.manifest["output_root"], "output_root"), target)
        argv = runner.build_argv(self.manifest, self.job, target / "schedule.csv")
        self.assertIn(f"--resultsRoot={target}", argv)
        self.assertEqual(runner.canonical_hash(self.manifest), before)
        self.assertFalse((self.root / "experiments").exists())

    def test_environment_removes_live_control_and_limits_threads(self):
        env = runner.clean_environment({"AGORA_CONTROL_DIR": "secret", "AGORA_RUN_ID": "a",
                                        "RIC_CONFIG": "b", "PATH": "keep", "OMP_NUM_THREADS": "64",
                                        "PYTHONPATH": "untrusted-prior-path", "PYTHONHOME": "wrong-home"},
                                       python_site_packages=self.root / ".venv/lib/python3.12/site-packages")
        self.assertEqual(env["PATH"], "keep")
        self.assertFalse(any(name.startswith(("AGORA_", "RIC_")) for name in env))
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            self.assertEqual(env[name], "1")
        self.assertEqual(env["PYTHONPATH"], str(self.root / ".venv/lib/python3.12/site-packages"))
        self.assertNotIn("PYTHONHOME", env)

    def test_missing_embedded_python_packages_fails_preflight(self):
        self.manifest["python_site_packages"] = ".venv/missing-site-packages"
        with self.assertRaises(FileNotFoundError):
            runner.preflight(self.manifest)

    def test_preflight_hashes_all_inputs_without_creating_output_or_launching(self):
        with patch.object(runner.subprocess, "Popen") as popen:
            result = runner.preflight(self.manifest, self.manifest_path)
        popen.assert_not_called()
        self.assertFalse(self.output.exists())
        roles = {role for row in result["input_fingerprints"] for role in row["roles"]}
        self.assertTrue({"binary", "source", "initial_cache_18", "initial_cache_35"} <= roles)
        self.assertIn("scheduled_ret_cache:" + self.job["job_id"], roles)
        self.assertTrue(all(len(row["sha256"]) == 64 for row in result["input_fingerprints"]))

    def test_existing_output_root_is_preserved_and_rejected(self):
        self.output.mkdir(parents=True)
        marker = self.output / "important.txt"
        marker.write_text("keep", encoding="utf-8")
        with self.assertRaises(FileExistsError):
            runner.preflight(self.manifest)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_missing_later_job_input_prevents_any_launch(self):
        second = copy.deepcopy(self.job)
        second.update(job_id="pair02", trace_path="missing-trace.csv")
        self.manifest["jobs"].append(second)
        with patch.object(runner.sys, "platform", "linux"), patch.object(runner, "launch_job") as launch:
            with self.assertRaises(FileNotFoundError):
                runner.run_batch(self.manifest, self.manifest_path)
        launch.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_missing_exact_ret_angle_never_falls_back_to_nearest_cache(self):
        self.job["actions"][1]["value"] = 28
        with self.assertRaises(FileNotFoundError):
            runner.preflight(self.manifest)

    def test_manifest_rejects_traversal_duplicate_jobs_and_wrong_direction(self):
        mutations = [lambda m: m.update(output_root="../escape"),
                     lambda m: m["jobs"].append(copy.deepcopy(m["jobs"][0])),
                     lambda m: m["jobs"][0]["actions"][0].update(value=45),
                     lambda m: m["jobs"][0]["actions"][1].update(time_s=180),
                     lambda m: m["jobs"][0].update(order="ret_then_txp")]
        for mutation in mutations:
            manifest = copy.deepcopy(self.manifest)
            mutation(manifest)
            with self.subTest(manifest=manifest), self.assertRaises(ValueError):
                runner.validate_manifest(manifest)

    def test_frozen_expected_hash_mismatch_is_rejected(self):
        self.manifest["expected_sha256"] = {self.manifest["binary"]: "0" * 64}
        with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
            runner.preflight(self.manifest)

    def test_expected_state_is_cumulative_and_persistent_both_orders(self):
        at179 = runner.expected_parameters(self.manifest, self.job, 179.5)
        at180 = runner.expected_parameters(self.manifest, self.job, 180.5)
        at540 = runner.expected_parameters(self.manifest, self.job, 540.5)
        self.assertEqual((at179["txp_5g"], at180["txp_5g"], at540["txp_5g"]), (43, 37, 37))
        self.assertEqual((at180["ret_4g1"], at540["ret_4g1"]), (9, 12))
        reverse = copy.deepcopy(self.job)
        reverse["actions"] = [dict(self.job["actions"][1], time_s=180), dict(self.job["actions"][0], time_s=540)]
        early = runner.expected_parameters(self.manifest, reverse, 180.5)
        final = runner.expected_parameters(self.manifest, reverse, 899.5)
        self.assertEqual((early["txp_5g"], early["ret_4g1"]), (43, 12))
        self.assertEqual(final, at540)

    def test_full_grid_and_receipts_verify_and_missing_tail_fails(self):
        directory = self.make_outputs()
        result = runner.verify_job(self.manifest, self.job, directory)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["last_cell_time_s"], 899.5)
        path = directory / "cell_kpi.csv"
        lines = path.read_text(encoding="utf-8").splitlines()
        path.write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")
        result = runner.verify_job(self.manifest, self.job, directory)
        self.assertFalse(result["passed"])
        self.assertFalse(result["complete_900s_cell_grid"])

    def test_reverting_first_action_after_second_action_is_detected(self):
        directory = self.make_outputs()
        path = directory / "cell_kpi.csv"
        text = path.read_text(encoding="utf-8")
        self.assertIn("899.5,1,37.0", text)
        path.write_text(text.replace("899.5,1,37.0", "899.5,1,43.0"), encoding="utf-8")
        result = runner.verify_job(self.manifest, self.job, directory)
        self.assertFalse(result["passed"])
        self.assertTrue(any("txp_5g" in issue for issue in result["issues"]))

    def test_bad_applied_time_or_status_is_detected(self):
        directory = self.make_outputs()
        path = directory / "parameter_changes.csv"
        text = path.read_text(encoding="utf-8").replace("180,180,", "180,181,").replace("applied", "failed", 1)
        path.write_text(text, encoding="utf-8")
        result = runner.verify_job(self.manifest, self.job, directory)
        self.assertFalse(result["passed"])
        self.assertTrue(any("applied_time_s" in issue for issue in result["issues"]))

    def test_batch_verification_also_requires_successful_exit_and_same_manifest(self):
        self.make_outputs()
        runner.atomic_json(self.output / "manifest.json", self.manifest)
        metadata = self.output / "_runner" / self.job["job_id"]
        metadata.mkdir(parents=True)
        runner.atomic_json(metadata / "execution.json", {"returncode": 1, "status": "failed"})
        self.assertFalse(runner.verify(self.manifest)["all_passed"])
        runner.atomic_json(metadata / "execution.json", {"returncode": 0, "status": "complete"})
        self.assertTrue(runner.verify(self.manifest)["all_passed"])
        changed = copy.deepcopy(self.manifest)
        changed["rng_run"] = 2
        result = runner.verify(changed)
        self.assertFalse(result["all_passed"])
        self.assertIn("requested_manifest_differs_from_launch_manifest", result["batch_issues"])

    def test_launch_uses_file_logs_and_new_session_and_refuses_reuse(self):
        self.output.mkdir(parents=True)
        process = Mock(pid=12345)
        process.poll.return_value = None
        factory = Mock(return_value=process)
        running = runner.launch_job(self.manifest, self.job, self.output, {"input_fingerprints": []}, factory)
        self.addCleanup(running["stdout"].close)
        self.addCleanup(running["stderr"].close)
        kwargs = factory.call_args.kwargs
        self.assertTrue(kwargs["start_new_session"])
        self.assertTrue(hasattr(kwargs["stdout"], "fileno"))
        self.assertTrue(hasattr(kwargs["stderr"], "fileno"))
        self.assertNotEqual(kwargs["stdout"], subprocess.PIPE)
        schedule = running["metadata"] / "parameter_schedule.csv"
        self.assertEqual(schedule.read_text(encoding="utf-8").splitlines()[0], "time_s,parameter,value")
        with self.assertRaises(FileExistsError):
            runner.launch_job(self.manifest, self.job, self.output, {"input_fingerprints": []}, factory)
        self.assertEqual(factory.call_count, 1)

    def test_launch_failure_closes_file_logs_and_records_error(self):
        self.output.mkdir(parents=True)
        handles = []

        def failed(*args, **kwargs):
            handles.extend([kwargs["stdout"], kwargs["stderr"]])
            raise OSError("mock exec failure")

        with self.assertRaises(OSError):
            runner.launch_job(self.manifest, self.job, self.output, {"input_fingerprints": []}, failed)
        self.assertTrue(all(handle.closed for handle in handles))
        record = json.loads((self.output / "_runner" / self.job["job_id"] / "execution.json").read_text())
        self.assertEqual(record["status"], "launch_failed")

    def test_cleanup_targets_only_owned_process_group_and_escalates_after_timeout(self):
        process = Mock(pid=24680)
        process.poll.return_value = None
        process.wait.side_effect = [subprocess.TimeoutExpired("mock", 1), 0]
        with patch.object(runner.os, "killpg", create=True) as kill, patch.object(signal, "SIGKILL", 9, create=True):
            runner.stop_owned_process(process, grace_s=1)
            self.assertEqual([call.args for call in kill.call_args_list],
                             [(24680, signal.SIGTERM), (24680, signal.SIGKILL)])

    def test_completed_process_is_not_signaled(self):
        process = Mock(pid=24680)
        process.poll.return_value = 0
        with patch.object(runner.os, "killpg", create=True) as kill:
            runner.stop_owned_process(process)
        kill.assert_not_called()

    def test_memory_guard_preserves_reserve_plus_estimated_new_job(self):
        for available, can_launch in ((4.74, False), (4.75, True), (8, True)):
            gate = runner.memory_launch_gate({"available": True, "available_bytes": available * 1024**3}, 3, 1.75)
            self.assertEqual(gate["can_launch"], can_launch)
        self.assertEqual(runner.memory_launch_gate({"available": False}, 3, 1.75)["reason"],
                         "waiting_for_memory_measurement")

    def test_sigterm_handler_cleans_owned_children_and_restores_previous_handler(self):
        process = Mock(pid=13579)
        process.poll.return_value = None
        record = {"job_id": self.job["job_id"], "linux_pid": process.pid, "status": "running", "timed_out": False}
        running = {"process": process, "record": record, "started_monotonic": runner.time.monotonic(),
                   "output": self.output / self.job["job_id"], "peak_rss_bytes": 0}
        prior = signal.getsignal(signal.SIGTERM)

        def send_sigterm(_):
            handler = signal.getsignal(signal.SIGTERM)
            self.assertNotEqual(handler, prior)
            handler(signal.SIGTERM, None)

        with patch.object(runner.sys, "platform", "linux"), \
             patch.object(runner, "memory_report", return_value={"available": True, "available_bytes": 50 * 1024**3}), \
             patch.object(runner, "launch_job", return_value=running), \
             patch.object(runner, "stop_owned_process") as stop, \
             patch.object(runner, "finish_job", return_value={"job_id": self.job["job_id"], "status": "interrupted"}) as finish, \
             patch.object(runner.time, "sleep", side_effect=send_sigterm):
            with self.assertRaises(KeyboardInterrupt):
                runner.run_batch(self.manifest, self.manifest_path, startup_stagger_s=0)
        stop.assert_called_once_with(process)
        finish.assert_called_once_with(self.manifest, running, interrupted=True)
        self.assertEqual(signal.getsignal(signal.SIGTERM), prior)
        state = json.loads((self.output / "batch_status.json").read_text())
        self.assertEqual(state["status"], "interrupted")


if __name__ == "__main__":
    unittest.main()
