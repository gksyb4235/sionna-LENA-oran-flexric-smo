"""One controller process per isolated ns-3 instance; atomic file handshake."""
import argparse
import json
import os
from pathlib import Path
import time
import traceback
import numpy as np
from .actions import catalog, describe_catalog, valid_mask, movement_cost, TTT_LEVELS, HYS_LEVELS
from .observations import observe, STATE_SCHEMA
from .rewards import reward
from .dqn import DoubleDQN


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def metadata(app, variant):
    return {"app": app, "reward": variant, "state_schema": STATE_SCHEMA,
            "actions": describe_catalog(app)}


def heuristic(app, parameters, metrics, actions, valid):
    """Transparent non-learning baseline; never label this policy DRL."""
    m = metrics["long"]
    if app == "lb":
        ordered = sorted(m["cells"], key=lambda c: c["prb"])
        gap = ordered[-1]["prb"] - ordered[0]["prb"]
        if gap < 0.1 or not m["active_load"]:
            return 0
        magnitude = min(6, max(1, int(np.ceil(gap / 0.1))))
        desired = ((ordered[0]["name"], magnitude), (ordered[-1]["name"], -magnitude))
        for cell, delta in desired:
            choices = [(abs(a.delta_db - delta), i) for i, a in enumerate(actions)
                       if valid[i] and a.app == "lb" and a.cell == cell and a.delta_db * delta > 0]
            if choices:
                return min(choices)[1]
        return 0
    direction = -1 if m["rlf"] > 0 else (1 if m["ho_attempts"] >= 10 and m["pp_posterior"] > 0.12 else 0)
    if direction == 0:
        return 0
    ti = min(range(len(TTT_LEVELS)), key=lambda i: abs(TTT_LEVELS[i] - parameters["ttt_ms"]))
    hi = min(range(len(HYS_LEVELS)), key=lambda i: abs(HYS_LEVELS[i] - parameters["hys_db"]))
    target = (TTT_LEVELS[int(np.clip(ti + direction, 0, 5))],
              HYS_LEVELS[int(np.clip(hi + direction, 0, 5))])
    for i, action in enumerate(actions):
        if valid[i] and (action.ttt_ms, action.hys_db) == target:
            return i
    return 0


class Controller:
    def __init__(self, args):
        self.args = args
        self.directory = Path(args.control_dir).resolve()
        self.directory.mkdir(parents=True, exist_ok=True)
        self.checkpoints = Path(args.checkpoint_dir or self.directory / "checkpoints")
        self.names = {"none": (), "lb": ("lb",), "mo": ("mo",), "both": ("lb", "mo")}[args.mode]
        self.actions = {name: catalog(name) for name in self.names}
        self.agents = {}
        self.pending = {}
        self.reward_totals = {name: 0.0 for name in self.names}
        self.local_transitions = {name: 0 for name in self.names}
        self.observations = 0
        self.started_wall = time.time()
        self.last_seq = -1
        self.last_time = -1.0
        self.previous_updates = {}
        self.initial_stats = {}
        self.finished = False
        self.log_path = self.directory / "actions.jsonl"
        if self.log_path.exists():
            raise ValueError(f"Refusing to append a different run to existing {self.log_path}")
        atomic_json(self.directory / "controller_config.json", {**vars(args),
            "action_catalogs": {a: describe_catalog(a) for a in self.names},
            "algorithm": "Independent Double DQN; two agents use the same causal observation",
            "training_environment": "Actual ns-3 next-interval KPIs only; no GNN/surrogate rewards",
            "metrics_contract": "Contiguous 1-second cell reports; ratios of sums; 30/90s histories",
            "strength_contract": "CIO signed delta magnitude 1..6dB; TTT and HYS each six absolute levels"})

    def initialize(self, state):
        if self.agents or self.args.policy == "heuristic":
            return
        for index, name in enumerate(self.names):
            agent = DoubleDQN(len(state), len(self.actions[name]), self.args.seed + (17 if name == "mo" else 0),
                train=self.args.policy == "train", gradient_steps=self.args.gradient_steps,
                warmup=self.args.warmup, batch_size=self.args.batch_size,
                epsilon_decay=self.args.epsilon_decay)
            checkpoint = getattr(self.args, name + "_load")
            if not checkpoint and self.args.load_dir:
                checkpoint = str(Path(self.args.load_dir) / (name + ".pt"))
            if checkpoint:
                agent.load(checkpoint, metadata(name, self.args.reward))
            if self.args.policy == "eval" and (not checkpoint or agent.gradient_updates == 0):
                raise ValueError(f"Frozen evaluation requires a trained {name} checkpoint with gradient updates")
            self.agents[name] = agent
            self.initial_stats[name] = agent.stats()

    def save(self):
        if self.args.policy == "train":
            for name, agent in self.agents.items():
                agent.save(self.checkpoints / (name + ".pt"), metadata(name, self.args.reward))

    def process(self, payload):
        began = time.perf_counter()
        seq, now = int(payload["seq"]), float(payload["time_s"])
        terminal = bool(payload.get("terminal", False))
        if seq <= self.last_seq or now <= self.last_time:
            raise ValueError("Observation sequence/time must strictly increase")
        if self.last_seq >= 0 and seq != self.last_seq + 1:
            raise ValueError("Observation sequence skipped; reward attribution is ambiguous")
        parameters = {k: float(v) for k, v in payload["parameters"].items()}
        for key, expected in self.previous_updates.items():
            if abs(parameters[key] - expected) > 1e-6:
                raise ValueError(f"Previous action was not applied: {key} expected {expected}, actual {parameters[key]}")
        state, metrics = observe(payload, now - self.last_time if self.pending else None)
        observation_ms = 1000 * (time.perf_counter() - began)
        self.initialize(state)
        updates, records = {}, {}
        for name in self.names:
            action_list = self.actions[name]
            valid = valid_mask(action_list, parameters)
            record = {"active": now >= getattr(self.args, name + "_start_s"), "feedback": None}
            if name in self.pending:
                old = self.pending.pop(name)
                outcome = {**metrics, "short": metrics["outcome"]}
                value, terms = reward(name, self.args.reward, old["metrics"], outcome, old["movement"])
                losses = self.agents[name].learn(old["state"], old["action_id"], value, state, terminal, valid) if name in self.agents else []
                self.reward_totals[name] += value
                self.local_transitions[name] += 1
                record["feedback"] = {"for_seq": old["seq"], "reward": value, "terms": terms,
                    "losses": losses, "terminal": terminal,
                    "outcome_window_s": [metrics["outcome"]["first_time_s"], metrics["outcome"]["last_time_s"]]}
            if not terminal and record["active"]:
                decision_start = time.perf_counter()
                if self.args.policy == "heuristic":
                    action_id, explored = heuristic(name, parameters, metrics, action_list, valid), False
                else:
                    action_id, explored = self.agents[name].select(state, valid)
                selected = action_list[action_id]
                proposed = selected.updates(parameters)
                if updates.keys() & proposed.keys():
                    raise ValueError("Independent xApps unexpectedly control an overlapping parameter")
                updates.update(proposed)
                self.pending[name] = {"seq": seq, "state": state.copy(), "action_id": action_id,
                    "metrics": metrics, "movement": movement_cost(name, parameters, proposed)}
                record.update({"action_id": action_id, "label": selected.label(), "updates": proposed,
                    "explored": explored, "valid_action_count": int(valid.sum()),
                    "decision_ms": 1000 * (time.perf_counter() - decision_start)})
            if name in self.agents:
                record["agent"] = self.agents[name].stats()
            records[name] = record
        latency_ms = 1000 * (time.perf_counter() - began)
        response = {"seq": seq, "updates": updates}
        if not terminal:
            atomic_json(self.directory / "action.json", response)
        # Persist checkpoints after responding so disk I/O does not stall ns-3.
        self.save()
        event = {"seq": seq, "time_s": now, "terminal": terminal,
            "parameters": parameters, "updates": updates, "metrics": metrics,
            "apps": records, "observation_ms": observation_ms, "controller_ms": latency_ms,
            "wall_time_unix": time.time()}
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, allow_nan=False) + "\n")
        self.observations += 1
        self.last_seq, self.last_time = seq, now
        self.previous_updates = updates
        self.finished = terminal
        atomic_json(self.directory / "summary.json", self.summary())
        return response

    def summary(self):
        return {"status": "complete" if self.finished else "running", "policy": self.args.policy,
            "mode": self.args.mode, "reward_variant": self.args.reward,
            "seed": self.args.seed, "last_seq": self.last_seq, "last_time_s": self.last_time,
            "observations": self.observations, "episode_transitions": self.local_transitions,
            "episode_reward_sums": self.reward_totals, "initial_agents": self.initial_stats,
            "final_agents": {name: agent.stats() for name, agent in self.agents.items()},
            "elapsed_wall_s": time.time() - self.started_wall,
            "convergence_claim": False,
            "note": "Online training reward is not a held-out policy-improvement estimate."}

    def run(self):
        last_new_wall = time.monotonic()
        path = self.directory / "observation.json"
        while not self.finished:
            if path.exists():
                payload = json.loads(path.read_text(encoding="utf-8"))
                if int(payload["seq"]) > self.last_seq:
                    self.process(payload)
                    last_new_wall = time.monotonic()
                    print(f"seq={self.last_seq} t={self.last_time:g} policy={self.args.policy} complete={self.finished}", flush=True)
            if time.monotonic() - last_new_wall > self.args.timeout_s:
                raise TimeoutError("No new ns-3 observation within controller timeout")
            if not self.finished:
                time.sleep(self.args.poll_s)
        return self.summary()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--control-dir", required=True)
    p.add_argument("--mode", choices=("none", "lb", "mo", "both"), default="both")
    p.add_argument("--reward", choices=("local", "guarded"), default="local")
    p.add_argument("--policy", choices=("train", "eval", "heuristic"), default="train")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--checkpoint-dir")
    p.add_argument("--load-dir")
    p.add_argument("--lb-load")
    p.add_argument("--mo-load")
    p.add_argument("--lb-start-s", type=float, default=180)
    p.add_argument("--mo-start-s", type=float, default=180)
    p.add_argument("--gradient-steps", type=int, default=4)
    p.add_argument("--warmup", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epsilon-decay", type=int, default=1000)
    p.add_argument("--timeout-s", type=float, default=86400)
    p.add_argument("--poll-s", type=float, default=0.01)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    controller = Controller(args)
    try:
        controller.run()
    except Exception as exc:
        atomic_json(controller.directory / "controller_error.json", {"type": type(exc).__name__,
            "message": str(exc), "last_seq": controller.last_seq, "traceback": traceback.format_exc()})
        raise


if __name__ == "__main__":
    main()
