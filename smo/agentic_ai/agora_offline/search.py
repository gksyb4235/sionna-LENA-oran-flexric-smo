"""Explicit finite candidate coverage for time, intensity and cell axes.

Coverage is computed from actual controls and predictions, never an LLM label.
It certifies completion of the declared candidate grid, not global feasibility
over the continuous action space or causal validity of the frozen model.
"""

from copy import deepcopy
from itertools import product
import math
from .contracts import CELLS, DOMAINS

AXES = ("time", "intensity", "cell")


class AxisSearch:
    def __init__(self, requested, current, intent, policy, issue_time_s):
        self.requested = deepcopy(requested)
        self.current = dict(current)
        self.intent = intent
        self.policy = policy
        self.delays = sorted(d for d in intent["allowed_delays_s"] if issue_time_s + d <= 720)
        self.origin_delay = requested["delay_s"]
        self.changes = {
            k: float(v) for k, v in requested["changes"].items() if abs(v - current[k]) > 1e-7
        }
        self.seen = {}
        self.first = []
        self.combined = []
        self.cell_checks = []
        self.base = self.plan(self.changes, self.origin_delay, ["original"])
        self._build()

    def key(self, plan):
        parameters = plan.get("parameters") or dict(self.current, **plan.get("changes", {}))
        return (
            tuple((k, round(float(parameters[k]), 7)) for k in sorted(self.current)),
            plan.get("delay_s", self.origin_delay),
            plan.get("horizon_s", self.intent["horizon_s"]),
        )

    def plan(self, changes, delay, axes):
        changes = {k: float(v) for k, v in changes.items() if abs(v - self.current[k]) > 1e-7}
        if not changes:
            return None
        return {
            "changes": changes,
            "delay_s": delay,
            "horizon_s": self.intent["horizon_s"],
            "axes": list(axes),
        }

    def strengths(self, key, target):
        current = self.current[key]
        low, high = sorted((current, target))
        values = [float(v) for v in DOMAINS[key] if low < v < high]
        if key.startswith("txp_") and self.policy["allow_continuous_txp"]:
            # A finite, disclosed 1 dB grid includes 36/40/41 when applicable.
            values += [
                float(v) for v in range(math.ceil(low), math.floor(high) + 1) if low < v < high
            ]
            if not values and abs(target - current) > 1e-6:
                values.append((current + target) / 2)
        return sorted(set(values), key=lambda value: abs(value - target))

    def deduplicate(self, plans, exclude=()):
        excluded = set(exclude)
        unique = {}
        for plan in plans:
            if plan is None:
                continue
            key = self.key(plan)
            if key in excluded:
                continue
            if key not in unique:
                unique[key] = deepcopy(plan)
            else:
                unique[key]["axes"] = sorted(
                    set(unique[key].get("axes", []) + plan.get("axes", []))
                )
        return list(unique.values())

    def _build(self):
        if self.base is None:
            return
        time_plans = [
            self.plan(self.changes, d, ["time"]) for d in self.delays if d != self.origin_delay
        ]
        intensity = []
        locations = []
        for key, target in self.changes.items():
            for value in self.strengths(key, target):
                intensity.append(
                    self.plan({**self.changes, key: value}, self.origin_delay, ["intensity"])
                )
            kind = key.split("_")[0]
            if kind in ("ttt", "hys"):
                self.cell_checks.append(
                    {
                        "source": key,
                        "status": "not_applicable",
                        "reason": "TTT/HYS are shared controls; cell relocation is undefined",
                    }
                )
                continue
            source_cell = key.split("_", 1)[1]
            for cell in CELLS:
                if cell == source_cell:
                    continue
                destination = kind + "_" + cell
                check = {
                    "source": key,
                    "target": destination,
                    "current_value": self.current[destination],
                }
                reason = None
                if cell not in self.intent["allowed_cells"]:
                    reason = "Target cell is excluded by the intent"
                elif (
                    self.policy["cell_ric_domains"][cell]
                    != self.policy["cell_ric_domains"][source_cell]
                ):
                    reason = "Relocation crosses the declared RIC domain"
                elif destination in self.changes:
                    reason = "Target already has a requested change; collapsing two controls is ambiguous"
                delta = target - self.current[key]
                lower, upper = DOMAINS[destination][0], DOMAINS[destination][-1]
                shifted = max(lower, min(upper, self.current[destination] + delta))
                if reason is None and (kind != "txp" or not self.policy["allow_continuous_txp"]):
                    points = [
                        v
                        for v in DOMAINS[destination]
                        if (v - self.current[destination]) * delta > 0
                        and abs(v - self.current[destination]) <= abs(delta) + 1e-7
                    ]
                    if points:
                        shifted = min(points, key=lambda v: abs(v - shifted))
                    else:
                        reason = "No trained value preserves direction and requested magnitude"
                if reason is None and abs(shifted - self.current[destination]) < 1e-7:
                    reason = "Target is already at the supported bound in the requested direction"
                if reason:
                    self.cell_checks.append({**check, "status": "blocked", "reason": reason})
                    continue
                relocated = {k: v for k, v in self.changes.items() if k != key}
                relocated[destination] = shifted
                plan = self.plan(relocated, self.origin_delay, ["cell"])
                locations.append(plan)
                self.cell_checks.append(
                    {
                        **check,
                        "status": "available",
                        "proposed_value": shifted,
                        "candidate_key": self.key(plan),
                    }
                )
        # Put every applicable axis in the first tensor batch even for compound
        # requests with more than 48 total baseline alternatives.
        groups = [list(time_plans), list(intensity), list(locations)]
        interleaved = []
        while any(groups):
            for group in groups:
                if group:
                    interleaved.append(group.pop(0))
        self.first = self.deduplicate(interleaved, [self.key(self.base)])
        combined = []
        for location in [self.base, *locations]:
            magnitudes = [location]
            for key, target in location["changes"].items():
                for value in self.strengths(key, target):
                    magnitudes.append(
                        self.plan(
                            {**location["changes"], key: value}, self.origin_delay, ["intensity"]
                        )
                    )
            for magnitude, delay in product(magnitudes, self.delays):
                axes = []
                if location is not self.base:
                    axes.append("cell")
                if magnitude is not location:
                    axes.append("intensity")
                if delay != self.origin_delay:
                    axes.append("time")
                combined.append(self.plan(magnitude["changes"], delay, axes))
        self.combined = self.deduplicate(
            combined, [self.key(self.base)] + [self.key(p) for p in self.first]
        )

    def record(self, evaluation):
        for row in evaluation["all_candidates"]:
            if row.get("status") != "invalid":
                self.seen[self.key(row)] = row

    def pending(self, stage):
        pool = self.first if stage == "axes" else self.combined
        return [deepcopy(plan) for plan in pool if self.key(plan) not in self.seen]

    def next_stage(self, all_expectations_met=False):
        if self.pending("axes"):
            return "axes"
        if not all_expectations_met and self.pending("combinations"):
            return "combinations"
        return None

    def report(self, all_expectations_met=False, budget_available=True):
        axes = {}
        for axis in AXES:
            pool = [p for p in self.first if axis in p["axes"]]
            done = [self.seen[self.key(p)] for p in pool if self.key(p) in self.seen]
            status = "complete" if len(done) == len(pool) else "pending"
            reason = None
            if not pool:
                status = "not_applicable"
                reason = {
                    "time": "No alternative delay remains within the intent and trained time bounds",
                    "intensity": "No weaker nonzero value exists in the declared candidate grid",
                    "cell": "No permitted directional relocation is available; see per-target checks",
                }[axis]
            axes[axis] = {
                "status": status,
                "planned": len(pool),
                "evaluated": len(done),
                "feasible": sum(bool(r["feasible"]) for r in done),
                "reason": reason,
            }
        checks = deepcopy(self.cell_checks)
        for check in checks:
            key = check.pop("candidate_key", None)
            if key is not None:
                check["status"] = "evaluated" if key in self.seen else "pending"
        stage = self.next_stage(all_expectations_met)
        return {
            "axes": axes,
            "cell_targets": checks,
            "independent_axes_complete": not self.pending("axes"),
            "combination_candidates": len(self.combined),
            "combinations_evaluated": len(self.combined) - len(self.pending("combinations")),
            "next_stage": stage,
            "stop_allowed": stage is None or not budget_available,
            "stop_reason": "budget_exhausted_with_unsearched_candidates"
            if stage and not budget_available
            else "goal_met_after_all_axes_checked"
            if stage is None and all_expectations_met
            else "declared_candidate_grid_exhausted"
            if stage is None
            else "search_incomplete",
            "scope": "Trained control values plus weaker integer-dBm TxP values when interpolation is enabled. Each compound-request alternative weakens or relocates one control while holding the others as requested; each is combined with supported delays. This finite grid does not exhaust joint changes to multiple controls or all continuous values.",
            "execution": "All three applicable axes share one vectorized GNN batch when they fit the 48-candidate limit",
        }
