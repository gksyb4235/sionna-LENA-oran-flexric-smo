import copy
import csv
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch

from drl_xapps.actions import catalog, valid_mask
from drl_xapps.observations import observe, REQUIRED
from drl_xapps.rewards import reward
from drl_xapps.dqn import DoubleDQN
from drl_xapps.controller import Controller, parser


PARAMETERS = {"cio_5g": 6., "cio_4g1": 0., "cio_4g2": -1., "ttt_ms": 512., "hys_db": 1.}


def write_kpis(path, seconds=120, future=True):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED)
        writer.writeheader()
        for second in range(seconds + (1 if future else 0)):
            for cell, utilization in (("gNB_5G", .2), ("gNB_4G_1", .6), ("gNB_4G_2", .4)):
                writer.writerow({"time_s": second + .5, "cell": cell, "num_ues": 20,
                    "delivered_megabits_interval": 1, "generated_load_mbps": 1,
                    "prb_used_reg": utilization * 100, "prb_capacity_reg": 100,
                    "ho_attempts_interval": 1, "ho_successes_interval": 1,
                    "ho_failures_interval": 0, "pingpongs_interval": 0,
                    "rlfs_interval": 10000 if second == seconds else 0,
                    "delay_sum_ms_interval": 50, "delay_samples_interval": 10})


class ActionTests(unittest.TestCase):
    def test_physical_levels_no_duplicates_no_clipped_noop(self):
        for app in ("lb", "mo"):
            actions = catalog(app)
            self.assertEqual(len(actions), 37)
            mask = valid_mask(actions, PARAMETERS)
            self.assertTrue(mask[0])
            identities = [tuple(sorted(a.updates(PARAMETERS).items())) for a, valid in zip(actions, mask) if valid]
            self.assertEqual(len(identities), len(set(identities)))
        for a, valid in zip(catalog("lb"), valid_mask(catalog("lb"), PARAMETERS)):
            if a.cell == "5g" and a.delta_db > 0:
                self.assertFalse(valid)


class ObservationRewardTests(unittest.TestCase):
    def test_causal_ratios_and_zero_ho_not_perfect_reward(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kpi.csv"
            write_kpis(path)
            state, metrics = observe({"time_s": 120, "cell_kpi_path": str(path), "parameters": PARAMETERS})
            self.assertTrue(np.isfinite(state).all())
            self.assertEqual(len(state), 54)
            self.assertEqual(metrics["short"]["duration_s"], 30)
            self.assertEqual(metrics["short"]["rlf"], 0)  # Huge future RLF never leaks.
            self.assertEqual(metrics["short"]["ho_attempts"], 90)
            self.assertAlmostEqual(metrics["short"]["jain"], 1.2 ** 2 / (3 * (.04 + .36 + .16)))
            no_ho = copy.deepcopy(metrics)
            no_ho["short"].update(ho_attempts=0, ho_successes=0, ho_success_support=0,
                pingpongs=0, pp_posterior=.1, ho_confidence=0)
            value, terms = reward("mo", "local", metrics, no_ho)
            self.assertEqual(terms["objective"], 0)
            self.assertEqual(value, 0)
            harmed = copy.deepcopy(no_ho)
            harmed["short"].update(rlf=2, rlf_per_1000_ue_s=1, delivery_ratio=.5)
            self.assertLess(reward("mo", "local", metrics, harmed)[0], 0)
            self.assertLess(reward("mo", "guarded", metrics, harmed)[0], reward("mo", "local", metrics, harmed)[0])

    def test_failed_attempts_cannot_create_positive_mobility_reward(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kpi.csv"
            write_kpis(path)
            _, previous = observe({"time_s": 120, "cell_kpi_path": str(path), "parameters": PARAMETERS})
            failed = copy.deepcopy(previous)
            failed["short"].update(ho_successes=0, ho_failures=90, ho_failure_ratio=1,
                ho_success_support=0, pingpongs=0, pp_posterior=.01)
            value, terms = reward("mo", "local", previous, failed)
            self.assertEqual(terms["objective"], 0)
            self.assertAlmostEqual(terms["ho_failure_penalty"], .45)
            self.assertLess(value, 0)
            failed["short"].update(ho_attempts=0, ho_failures=3, ho_confidence=0)
            value, terms = reward("mo", "local", previous, failed)
            self.assertAlmostEqual(terms["ho_failure_penalty"], .15)
            self.assertLess(value, 0)

    def test_missing_cell_is_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kpi.csv"
            write_kpis(path, 2, future=False)
            lines = path.read_text().splitlines()
            path.write_text("\n".join(lines[:-1]) + "\n")
            with self.assertRaises(ValueError):
                observe({"time_s": 2, "cell_kpi_path": str(path), "parameters": PARAMETERS})


class LearningTests(unittest.TestCase):
    def test_terminal_does_not_bootstrap(self):
        state, valid = np.zeros(4, np.float32), np.ones(2, bool)
        agent = DoubleDQN(4, 2, 1, batch_size=1, warmup=1, gradient_steps=1)
        with torch.no_grad():
            for param in agent.online.parameters():
                param.zero_()
            for param in agent.target.parameters():
                param.zero_()
            agent.target.layers[-1].bias.fill_(10)
        losses = agent.learn(state, 0, 0, state, True, valid)
        self.assertEqual(losses, [0.0])

    def test_persist_resume_and_frozen_eval(self):
        state, valid = np.arange(4, dtype=np.float32), np.array([True, False, True])
        agent = DoubleDQN(4, 3, 2, batch_size=2, warmup=2, gradient_steps=2)
        for _ in range(3):
            agent.learn(state, 0, 1, state + .1, False, valid)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.pt"
            agent.save(path, {"app": "test"})
            resumed = DoubleDQN(4, 3, 999)
            resumed.load(path, {"app": "test"})
            self.assertEqual(resumed.transitions, 3)
            self.assertEqual(resumed.rng.bit_generator.state, agent.rng.bit_generator.state)
            self.assertEqual(resumed.select(state, valid), agent.select(state, valid))
            frozen = DoubleDQN(4, 3, 3, train=False)
            frozen.load(path, {"app": "test"})
            before = copy.deepcopy(frozen.online.state_dict())
            steps = frozen.gradient_updates
            self.assertFalse(frozen.select(state, valid)[1])
            self.assertEqual(frozen.learn(state, 0, 99, state, False, valid), [])
            self.assertEqual(frozen.gradient_updates, steps)
            self.assertEqual(len(frozen.replay), 0)
            self.assertIsNone(frozen.optimizer)
            self.assertTrue(all(torch.equal(before[k], v) for k, v in frozen.online.state_dict().items()))


class ProtocolTests(unittest.TestCase):
    def test_delayed_activation_same_observation_and_terminal_learning(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "kpi.csv"
            write_kpis(path, 8)
            args = parser().parse_args(["--control-dir", str(Path(tmp) / "control"), "--mode", "both",
                "--lb-start-s", "2", "--mo-start-s", "4", "--batch-size", "1", "--warmup", "1"])
            controller = Controller(args)
            params = dict(PARAMETERS)
            for seq, now in enumerate((2, 4, 6, 8)):
                response = controller.process({"seq": seq, "time_s": now, "terminal": now == 8,
                    "cell_kpi_path": str(path), "parameters": params})
                if now == 2:
                    self.assertNotIn("mo", controller.pending)
                if now == 4:
                    np.testing.assert_array_equal(controller.pending["lb"]["state"], controller.pending["mo"]["state"])
                params.update(response["updates"])
            self.assertEqual(controller.local_transitions, {"lb": 3, "mo": 2})
            self.assertTrue(controller.finished)
            logs = [json.loads(line) for line in controller.log_path.read_text().splitlines()]
            self.assertEqual(logs[-1]["apps"]["lb"]["feedback"]["outcome_window_s"], [6.5, 7.5])
            self.assertTrue(logs[-1]["apps"]["lb"]["feedback"]["terminal"])
            self.assertEqual(json.loads((controller.directory / "action.json").read_text())["seq"], 2)


if __name__ == "__main__":
    unittest.main()
