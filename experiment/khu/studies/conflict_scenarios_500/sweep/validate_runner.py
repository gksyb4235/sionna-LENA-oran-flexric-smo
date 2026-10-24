"""Validate packaged trajectories/sharding plus synthetic subprocess contracts."""
import io
import json
from pathlib import Path
import sys
import unittest
import prepare_manifest
import sweep

bundle = Path(__file__).resolve().parent
manifest = sweep.validate_manifest(sweep.read_json(bundle / "data/execution_manifest.json"))
rebuilt = prepare_manifest.prepare(bundle / "data/conflict_scenario_params_reviewed_500.csv",
                                   manifest["trace_assignment"]["seed_start"], manifest["experiment_id"])
if rebuilt != manifest:
    raise ValueError("Packaged CSV and execution manifest differ")
pilot = sweep.read_json(bundle / "data/pilot_ids.json")["scenario_ids"]
shards = [sweep.selected_jobs(manifest, i, 9) for i in range(9)]
flattened = [job["job_id"] for jobs in shards for job in jobs]
if len(flattened) != len(set(flattened)) or len(flattened) != len(manifest["jobs"]):
    raise ValueError("Shard assignment overlaps or omits jobs")
suite = unittest.defaultTestLoader.discover(str(bundle / "tests"))
stream = io.StringIO()
result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
report = {"passed": result.wasSuccessful(), "platform": sys.platform,
          "tests_run": result.testsRun, "tests_skipped": len(result.skipped), "test_output": stream.getvalue(),
          "real_ns3_launched": False, "dummy_subprocesses_only": True,
          "csv_manifest_roundtrip_identical": True, "scenario_count": len(manifest["jobs"]),
          "event_count": sum(len(job["events"]) for job in manifest["jobs"]),
          "full_shard_counts": list(map(len, shards)),
          "pilot_shard_counts": [len(sweep.selected_jobs(manifest, i, 9, pilot_ids=pilot)) for i in range(9)],
          "manifest_hash": sweep.canonical_hash(manifest), "runner_sha256": sweep.sha256(bundle / "sweep.py")}
sweep.atomic_json(bundle / "runner_validation.json", report)
print(json.dumps({key: value for key, value in report.items() if key != "test_output"}))
raise SystemExit(0 if result.wasSuccessful() else 1)
