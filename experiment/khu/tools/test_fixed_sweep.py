"""Fast fixed-sweep checks using synthetic CSVs and a stand-in simulator."""

import csv
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

TOOLS = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("fixed_run_status", TOOLS / "fixed_run_status.py")
status = importlib.util.module_from_spec(spec)
spec.loader.exec_module(status)
ROW = "gnn_0000,43,43,43,9,5,7,0,0,0,256,2.5"


class StatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.run = Path(self.temporary.name) / "gnn_0000"
        self.run.mkdir()
        self.expected = status.metadata(ROW, 3, 1)
        (self.run / "run.log").write_text("normal completion\n")
        with (self.run / "cell_kpi.csv").open("w") as output:
            output.write("time_s,interval_s,cell_id\n")
            for t in (1.5, 2.5):
                for cell in (1, 2, 3):
                    output.write(f"{t},1,{cell}\n")
        (self.run / "ue_kpi.csv").write_text("time_s,interval_s,ue\n1.5,1,ue1\n2.5,1,ue1\n")

    def test_exit_and_complete_windows_required_for_resume(self):
        self.assertFalse(status.completed(self.run, self.expected))
        status.finish(self.run, self.expected, 0, "2")
        self.assertTrue(status.completed(self.run, self.expected))
        changed = status.metadata(ROW, 4, 1)
        self.assertFalse(status.completed(self.run, changed))
        status.finish(self.run, self.expected, 134, "2")
        self.assertFalse(status.completed(self.run, self.expected))

    def test_malloc_and_truncated_output_are_failures_even_with_exit_zero(self):
        (self.run / "run.log").write_text("malloc(): invalid size (unsorted)\n")
        self.assertEqual(status.finish(self.run, self.expected, 0, "2")["status"], "failed")
        (self.run / "run.log").write_text("normal completion\n")
        (self.run / "cell_kpi.csv").write_text("time_s,interval_s,cell_id\n1.5,1,1\n")
        self.assertEqual(status.finish(self.run, self.expected, 0, "2")["status"], "failed")

    def test_archive_preserves_results_and_launcher_log(self):
        original = (self.run / "cell_kpi.csv").read_bytes()
        launcher = self.run.parent / "gnn_0000.launcher.log"
        launcher.write_text("old launcher\n")
        status.archive(self.run)
        archived = list(self.run.parent.glob("_attempts/gnn_0000/*"))
        self.assertEqual(len(archived), 1)
        self.assertEqual((archived[0] / "gnn_0000/cell_kpi.csv").read_bytes(), original)
        self.assertEqual((archived[0] / launcher.name).read_text(), "old launcher\n")
        self.assertFalse(self.run.exists())


@unittest.skipUnless(len(os.sched_getaffinity(0)) >= 2 and shutil.which("taskset") and shutil.which("prlimit"), "needs two CPUs and Linux affinity utilities")
class RunnerTests(unittest.TestCase):
    def test_dry_run_refill_failures_resume_and_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tools = root / "experiment/khu/tools"
            tools.mkdir(parents=True)
            for name in ("run_gnn_dataset_sweep.sh", "fixed_run_status.py"):
                shutil.copy2(TOOLS / name, tools / name)
            (tools / "build_merged_ret_cache.py").write_text("print('SIONNA_CACHE_18=/mock/cache.h5')\n")
            data = root / "experiment/khu/datasets"
            (data / "ue_position").mkdir(parents=True)
            with (data / "gnn_dataset_params.csv").open("w") as output:
                output.write(",".join(status.FIELDS) + "\n")
                for index in range(3):
                    output.write(ROW.replace("gnn_0000", f"gnn_{index:04d}") + "\n")
                    (data / f"ue_position/ue_positions_seed{index}.csv").touch()
            for tilt, band in ((9, "3p5ghz"), (5, "1p8ghz"), (7, "1p8ghz")):
                cache = root / f"experiment/khu/caches/ret_caches/tilt_{tilt}deg"
                cache.mkdir(parents=True, exist_ok=True)
                (cache / f"sionna_rt_cache_{band}_2x2_1x1_lzf.h5").touch()
            binary = root / "build/scratch/ns3.48-khu-real-dual-nr-sionna-final-default"
            binary.parent.mkdir(parents=True)
            binary.write_text(f"#!{sys.executable}\n" + '''import json, os, sys, time
from pathlib import Path
args = dict(value[2:].split('=', 1) for value in sys.argv[1:])
directory = Path(args['resultsRoot']) / args['runTag']
directory.mkdir()
started = time.monotonic()
(directory / 'run.log').write_text('normal start\\n')
time.sleep(1.4 if args['runTag'] == 'gnn_0000' else 0.05)
(directory / 'timing.json').write_text(json.dumps({'start': started, 'end': time.monotonic(), 'cpus': sorted(os.sched_getaffinity(0))}))
if args['runTag'] == 'gnn_0002':
    (directory / 'run.log').write_text('malloc(): invalid size (unsorted)\\n')
    sys.exit(134)
(directory / 'cell_kpi.csv').write_text('time_s,interval_s,cell_id\\n' + ''.join(f'{t},1,{c}\\n' for t in (1.5, 2.5) for c in (1, 2, 3)))
(directory / 'ue_kpi.csv').write_text('time_s,interval_s,ue\\n1.5,1,u1\\n2.5,1,u1\\n')
''')
            binary.chmod(0o755)
            env = dict(os.environ, GNN_SWEEP_SIM_TIME="3", GNN_SWEEP_KPI_REPORT_INTERVAL="1",
                       GNN_SWEEP_INFLUX="false", GNN_SWEEP_LIVE_LOGS="true")
            for name in ("GNN_SWEEP_RESULTS_ROOT", "RET_CACHE_ROOT"):
                env.pop(name, None)
            command = ["bash", str(tools / "run_gnn_dataset_sweep.sh"), "0", "2", "1", "2"]
            dry = subprocess.run(command + ["--dry-run"], env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(dry.returncode, 0, dry.stderr)
            results = root / "experiment/khu/results/fixed"
            self.assertFalse(results.exists())
            run = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(run.returncode, 1, run.stdout + run.stderr)
            self.assertTrue((results / "gnn_0000/timing.json").exists(), run.stdout + run.stderr)
            first = json.loads((results / "gnn_0000/timing.json").read_text())
            second = json.loads((results / "gnn_0001/timing.json").read_text())
            third = json.loads((results / "gnn_0002/timing.json").read_text())
            self.assertLess(third["start"], first["end"], "next job must refill before slowest job ends")
            self.assertEqual(third["cpus"], second["cpus"])
            self.assertNotEqual(first["cpus"], second["cpus"])
            self.assertEqual(json.loads((results / "gnn_0000/run_status.json").read_text())["status"], "complete")
            self.assertEqual(json.loads((results / "gnn_0002/run_status.json").read_text())["exit_code"], 134)
            retry = subprocess.run(command, env=env, capture_output=True, text=True, timeout=20)
            self.assertEqual(retry.returncode, 1, retry.stdout + retry.stderr)
            self.assertIn("[skip] gnn_0000", retry.stdout)
            self.assertIn("[skip] gnn_0001", retry.stdout)
            self.assertTrue(list((results / "_attempts/gnn_0002").glob("*/gnn_0002/run_status.json")))


if __name__ == "__main__":
    unittest.main()
