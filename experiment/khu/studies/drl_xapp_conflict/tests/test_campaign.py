"""Critical experiment invariants without launching or modifying any simulator."""
import copy
import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("isolated_drl_campaign", ROOT / "campaign.py")
campaign = importlib.util.module_from_spec(spec)
spec.loader.exec_module(campaign)
sys.path.insert(0, str(ROOT / "python"))
from drl_xapps.controller import parser as controller_parser


class DesignTests(unittest.TestCase):
    def test_training_chains_and_heldout_factorial_activation(self):
        plan = campaign.make_plan()
        jobs = {j["id"]: j for j in plan["jobs"]}
        self.assertEqual(len(jobs), 64)
        for phase, policy in (("train", "train"), ("validation", "eval")):
            allowed = set(plan["training_traces" if phase == "train" else "validation_traces"])
            for job in jobs.values():
                if job["phase"] != phase:
                    continue
                self.assertIn(job["trace_seed"], allowed)
                self.assertEqual(job["policy"], policy)
                self.assertEqual(job["mode"], job["app"])
                if job.get("load_job"):
                    previous = jobs[job["load_job"]]
                    self.assertEqual(previous["phase"], "train")
                    for key in ("app", "reward", "learning_seed"):
                        self.assertEqual(previous[key], job[key])
                    if phase == "train":
                        self.assertEqual(previous["episode"] + 1, job["episode"])
                    else:
                        self.assertEqual(previous["episode"], len(plan["training_traces"]))
        for trace in plan["test_traces"]:
            arms = [j for j in jobs.values() if j["phase"] == "test" and j["trace_seed"] == trace]
            self.assertEqual({j["arm"] for j in arms}, set(campaign.ARMS))
            for job in arms:
                self.assertEqual((job["lb_start_s"], job["mo_start_s"]), (180, 540))
                self.assertNotEqual(job["policy"], "train")
                self.assertNotIn("load_job", job)
                self.assertNotIn(trace, plan["training_traces"] + plan["validation_traces"])

    def test_commands_preserve_explicit_output_and_activation_contract(self):
        plan = campaign.make_plan()
        job = next(j for j in plan["jobs"] if j["id"] == "test_both_local_t1419")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            with patch.object(campaign, "ROOT", root):
                args = campaign.sim_argv(plan, job, root / "results/test_both_local_t1419")
                options = dict(a[2:].split("=", 1) for a in args[1:])
                self.assertEqual(options["cellKpiCsvPath"], str(root / "results/test_both_local_t1419") + "/cell_kpi.csv")
                self.assertEqual(options["ueKpiCsvPath"], str(root / "results/test_both_local_t1419") + "/ue_kpi.csv")
                self.assertEqual(options["ns3::NrHelper::E2ModeNr"], "false")
                job = dict(job, selected_load_dir=str(root / "selected/local"))
                ctrl = campaign.controller_argv(job, root / "control")
                parsed = controller_parser().parse_args(ctrl[3:])
                self.assertEqual((parsed.lb_start_s, parsed.mo_start_s), (180, 540))
                self.assertEqual((parsed.mode, parsed.policy, parsed.reward), ("both", "eval", "local"))
                self.assertEqual(parsed.load_dir, str(root / "selected/local"))
                self.assertEqual(parsed.epsilon_decay, 96)

    def test_validation_selection_never_reads_test_outcomes(self):
        plan = campaign.make_plan()
        scored = []

        def score(path, app):
            name = path.name
            self.assertTrue(name.startswith("val_"))
            self.assertTrue(name.endswith(("_t1417", "_t1418")))
            scored.append(name)
            return {"eligible": True, "score": 2.0 if "_s202_" in name else 1.0}

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for app in ("lb", "mo"):
                for reward in ("local", "guarded"):
                    for seed in (101, 202):
                        checkpoint = root / "results" / f"train_{app}_{reward}_s{seed}_e03/control/checkpoints/{app}.pt"
                        checkpoint.parent.mkdir(parents=True)
                        checkpoint.write_bytes(f"{app}:{reward}:{seed}".encode())
            # Tempting test-derived output exists, but selection must never open it.
            test = root / "results/test_both_local_t1419"
            test.mkdir(parents=True)
            (test / "score.json").write_text('{"score":1e99}')
            with patch.object(campaign, "ROOT", root), patch.object(campaign, "load_analysis", return_value=types.SimpleNamespace(score_run=score)):
                result = campaign.select_policies(plan)
            self.assertEqual(len(scored), 16)
            for app in ("lb", "mo"):
                for reward in ("local", "guarded"):
                    selected = root / f"selected/{reward}/{app}.pt"
                    self.assertEqual(selected.read_bytes(), f"{app}:{reward}:202".encode())
                    self.assertEqual(result["selections"][f"{app}_{reward}"]["chosen"]["learning_seed"], 202)


def make_verified_run(root):
    control = root / "control"
    control.mkdir()
    before = {"cio_5g": 0, "cio_4g1": 0, "cio_4g2": 0, "ttt_ms": 128, "hys_db": 1}
    after = dict(before, cio_4g1=2)
    labels = {1: "gNB_5G", 2: "gNB_4G_1", 3: "gNB_4G_2"}
    fields = ("time_s", "cell_id", "cell", "cio_bias_db", "ttt_ms", "hysteresis_db")
    with (root / "cell_kpi.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for second in range(3):
            for cell in (1, 2, 3):
                writer.writerow({"time_s": second + .5, "cell_id": cell, "cell": labels[cell],
                    "cio_bias_db": 2 if second >= 1 and cell == 2 else 0,
                    "ttt_ms": 128, "hysteresis_db": 1})
    receipts = [
        {"seq": 0, "time_s": 1, "status": "applied", "before": before, "updates": {"cio_4g1": 2}, "after": after},
        {"seq": 1, "time_s": 2, "status": "held", "before": after, "updates": {}, "after": after},
        {"seq": 2, "time_s": 3, "status": "terminal", "before": after, "updates": {}, "after": after},
    ]
    actions = [{"seq": r["seq"], "time_s": r["time_s"], "terminal": r["status"] == "terminal",
                "parameters": r["before"], "updates": r["updates"]} for r in receipts]
    (control / "receipts.jsonl").write_text("\n".join(json.dumps(r) for r in receipts) + "\n")
    (control / "actions.jsonl").write_text("\n".join(json.dumps(a) for a in actions) + "\n")
    (control / "observation.json").write_text(json.dumps({"seq": 2, "time_s": 3, "terminal": True, "parameters": after}))
    (control / "summary.json").write_text(json.dumps({"status": "complete", "last_time_s": 3}))
    return receipts


class ReceiptTests(unittest.TestCase):
    def test_valid_hold_and_applied_receipts_and_failed_readback(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            receipts = make_verified_run(root)
            result = campaign.verify(root, 3)
            self.assertTrue(result["passed"], result["issues"])
            corrupt = copy.deepcopy(receipts)
            corrupt[0]["after"]["cio_4g1"] = -2
            (root / "control/receipts.jsonl").write_text("\n".join(json.dumps(r) for r in corrupt) + "\n")
            result = campaign.verify(root, 3)
            self.assertFalse(result["passed"])
            self.assertIn("action_receipt_readback_mismatch", result["issues"])

    def test_incomplete_kpi_grid_is_not_a_completed_experiment(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            make_verified_run(root)
            path = root / "cell_kpi.csv"
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n")
            result = campaign.verify(root, 3)
            self.assertFalse(result["passed"])
            self.assertTrue(any(issue.startswith("incomplete_cell_grid") for issue in result["issues"]))


if __name__ == "__main__":
    unittest.main()
