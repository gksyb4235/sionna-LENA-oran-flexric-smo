import unittest
from smo.ric_control.protocol import encode


def fixture():
    current = {
        "txp_5g": 43,
        "txp_4g1": 37,
        "txp_4g2": 30,
        "ret_5g": 9,
        "ret_4g1": 3,
        "ret_4g2": 9,
        "cio_5g": 9,
        "cio_4g1": 1,
        "cio_4g2": 0,
        "ttt_ms": 128,
        "hys_db": 4,
    }
    plan = {
        "deployment_id": "b" * 32,
        "run_id": "gnn_0130",
        "target_ric": "near_rt",
        "current_parameters": current,
        "candidate": {
            "interval_start_s": 180,
            "interval_end_s": 360,
            "changes": {"txp_5g": 40, "ret_4g2": 5, "cio_4g1": 2, "ttt_ms": 340, "hys_db": 3},
        },
    }
    rows = []
    for i, c in enumerate(("5g", "4g1", "4g2"), 1):
        rows.append(
            {
                "label": c,
                "cell_id": i,
                "e2_enabled": True,
                "ideal_rrc": True,
                "parameters": {
                    k: current[k if k in ("ttt_ms", "hys_db") else f"{k}_{c}"]
                    for k in ("txp", "ret", "cio", "ttt_ms", "hys_db")
                },
            }
        )
    live = {
        "run_id": "a" * 32,
        "replay_run_id": "gnn_0130",
        "sim_time_s": 10,
        "stop_time_s": 400,
        "cells": rows,
        "lease": "",
    }
    return plan, live


class ContractTest(unittest.TestCase):
    def test_all_five_controls_full_baseline(self):
        p, l = fixture()
        out = encode(p, l)
        values = out["payload"].split()
        self.assertEqual(values[:6], ["AGORA1", "a" * 32, "b" * 32, "180", "360", "15"])
        entries = [values[i : i + 4] for i in range(6, len(values), 4)]
        self.assertIn(["1", "1", "40", "43"], entries)
        self.assertIn(["3", "2", "5", "9"], entries)
        for cell in ("1", "2", "3"):
            self.assertIn([cell, "4", "340", "128"], entries)
            self.assertIn([cell, "5", "3", "4"], entries)

    def test_rejects_mismatched_baseline(self):
        p, l = fixture()
        l["cells"][2]["parameters"]["txp"] = 31
        with self.assertRaisesRegex(ValueError, "Baseline mismatch"):
            encode(p, l)

    def test_rejects_other_trace_binding(self):
        p, l = fixture()
        l["replay_run_id"] = "gnn_0100"
        with self.assertRaisesRegex(ValueError, "bound"):
            encode(p, l)

    def test_no_implicit_time_shift(self):
        p, l = fixture()
        l["sim_time_s"] = 180
        with self.assertRaisesRegex(ValueError, "interval"):
            encode(p, l)

    def test_real_rrc_rejects_340(self):
        p, l = fixture()
        l["cells"][0]["ideal_rrc"] = False
        with self.assertRaisesRegex(ValueError, "represented"):
            encode(p, l)

    def test_ret_missing_cache_rejected(self):
        p, l = fixture()
        p["candidate"]["changes"]["ret_4g2"] = 6
        with self.assertRaisesRegex(ValueError, "cache"):
            encode(p, l)

    def test_split_is_not_silently_relabelled(self):
        p, l = fixture()
        p["target_ric"] = "split"
        with self.assertRaisesRegex(ValueError, "one Near-RT"):
            encode(p, l)

    def test_nan_rejected(self):
        p, l = fixture()
        p["candidate"]["changes"]["txp_5g"] = float("nan")
        with self.assertRaises(ValueError):
            encode(p, l)

    def test_no_concurrent_lease(self):
        p, l = fixture()
        l["lease"] = "c" * 32
        with self.assertRaisesRegex(ValueError, "lease"):
            encode(p, l)

    def test_requires_restoration_before_simulation_stops(self):
        p, l = fixture()
        l["stop_time_s"] = 360
        with self.assertRaisesRegex(ValueError, "interval"):
            encode(p, l)


if __name__ == "__main__":
    unittest.main()
