"""History-only replay adapter and batch API for the selected frozen model.

No candidate ranking occurs here. The adapter reads recorded observations ending
strictly before issue_time_s and never substitutes observations from another run
for a candidate's predicted future. Replay runs are fixed-configuration records;
they are not a simulator of a policy trajectory or of branched interventions.
"""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
import hashlib
import io
import json
import math
import os
from pathlib import Path
from threading import RLock
from typing import Any
import numpy as np
import torch
from .forecast import Forecast

CHECKPOINT_SHA256 = "b9fc9cbdb2b1245948428386d52969cc93537410e54d6c161b0a471ea0d9c324"
PRIMARY = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12]
COUNT_INDICES = [7, 8, 9]
HORIZONS = [90, 180]
DELAYS = [0, 30, 60, 90]


def json_safe(value: Any):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


class Runtime:
    def __init__(self, root: Path | None = None, device: str = "cpu"):
        self.root = Path(root or Path(__file__).resolve().parent / "assets").resolve()
        manifest_path = self.root / "manifest.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Replay assets missing: {manifest_path}; run runtime/build_assets.py once")
        self.manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        checkpoint = self.root / "parameter5_backbone.pt"
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != CHECKPOINT_SHA256:
            raise ValueError("Checkpoint integrity check failed")
        torch.set_num_threads(max(1, min(4, int(os.environ.get("AGORA_TORCH_THREADS", "4")))))
        self.forecast = Forecast(checkpoint, device=device)
        self._index = {r["run_id"]: r for r in self.manifest["runs"]}
        self._file_index = {r["file"]: r for r in self.manifest["files"]}
        self._cache = OrderedDict()
        self._lock = RLock()
        self.parameter_names = self.manifest["parameter_names"]
        self.target_names = self.manifest["target_names"]
        self.cell_labels = self.manifest["cell_labels"]
        self._training_configurations = {
            tuple(r["current_rcp"][k] for k in self.parameter_names)
            for r in self._index.values() if r["split"] == "train"
        }
        calibration_path = self.root / "calibration_hold.json"
        self.hold_action_calibration = None
        if calibration_path.exists():
            calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
            if calibration.get("checkpoint_sha256") != CHECKPOINT_SHA256 or calibration.get("partition", calibration.get("split")) != "val":
                raise ValueError("Calibration provenance mismatch")
            if any(run_id not in self._index or self._index[run_id]["split"] != "val"
                   for run_id in calibration.get("run_ids", [])):
                raise ValueError("Calibration contains a non-validation run")
            self.hold_action_calibration = calibration

    def metadata(self):
        result = deepcopy(self.forecast.metadata)
        result.update({
            "model_name": "AGORA final Temporal GNN v15, frozen 90/180 output selection",
            "checkpoint_sha256": CHECKPOINT_SHA256,
            "registered_parameters": sum(p.numel() for p in self.forecast.model.parameters()),
            "training_horizons_s": [60, 90, 150, 180],
            "horizons_s": HORIZONS, "serving_horizons_s": HORIZONS,
            "delays_s": DELAYS, "history_seconds": 180,
            "issue_time_range_s": [30, 720], "max_action_start_s": 720,
            "parameter_values": deepcopy(self.manifest["parameter_values"]),
            "parameter_bounds": {k: [min(v), max(v)] for k, v in self.manifest["parameter_values"].items()},
            "run_id_namespace": self.manifest["run_id_namespace"],
            "replay_runs": len(self._index),
            "uncertainty_calibration": {"available": False, "reason": "No calibrated predictive interval is bundled; no confidence percentage is inferred"},
            "hold_action_calibration": deepcopy(self.hold_action_calibration),
            "counter_semantics": "ho_attempts, pingpongs and rlfs are interval event counts; other targets are native interval means/ratios",
            "parameter_policy": "Reject outside the observed envelope; flag in-envelope non-grid values as unvalidated interpolation",
            "execution_mode": "offline_prediction_only",
            "limitations": [
                "Fixed-run same-clock replay training does not identify online switched-action causal effects",
                "No future observed KPIs are used in input or ranking",
                "One fixed three-cell topology; TTT/HYS are shared across cells",
                "Predicted event counts are expectations, not guaranteed integer outcomes",
                "Cell labels 4g1/4g2 refer to 1.8 GHz NR simulation cells, not LTE",
            ],
        })
        return json_safe(result)

    def list_runs(self, split: str | None = None):
        if split is not None and split not in ("train", "val", "test"):
            raise ValueError("split must be train, val, or test")
        return deepcopy([r for r in self._index.values() if split is None or r["split"] == split])

    @staticmethod
    def _time(issue_time_s):
        if isinstance(issue_time_s, bool) or not isinstance(issue_time_s, (int, float)):
            raise ValueError("issue_time_s must be an integer simulation second")
        if not math.isfinite(issue_time_s) or int(issue_time_s) != issue_time_s or not 30 <= issue_time_s <= 720:
            raise ValueError("issue_time_s must be an integer in [30,720]")
        return int(issue_time_s)

    def _run(self, run_id):
        if not isinstance(run_id, str) or run_id not in self._index:
            raise ValueError("Unknown original training run_id; do not substitute a Final_Dataset renumbered ID")
        with self._lock:
            if run_id not in self._cache:
                name = f"runs/{run_id}.npz"
                data = (self.root / name).read_bytes()
                if hashlib.sha256(data).hexdigest() != self._file_index[name]["sha256"]:
                    raise ValueError(f"Replay file integrity check failed: {run_id}")
                with np.load(io.BytesIO(data), allow_pickle=False) as source:
                    record = {k: source[k].copy() for k in ("y", "valid", "aux", "num", "den")}
                if record["y"].shape != (900, 3, 14) or record["valid"].shape != (900, 3, 14):
                    raise ValueError("Unexpected replay shapes")
                self._cache[run_id] = record
                while len(self._cache) > 4:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(run_id)
            return self._cache[run_id]

    def _observed_inputs(self, run_id, t):
        record = self._run(run_id)
        start = max(0, t - 180)
        # Crucial boundary: index t is the FUTURE interval [t,t+1). Every read
        # passed to the GNN below ends at t (exclusive), regardless of horizon.
        y = record["y"][start:t]
        valid = record["valid"][start:t]
        ues = record["aux"][start:t, :, 5:6]
        x = np.concatenate((ues, y[..., PRIMARY]), axis=-1)
        mask = np.concatenate((np.ones_like(ues, dtype=bool), valid[..., PRIMARY]), axis=-1)
        current = np.asarray([self._index[run_id]["current_rcp"][k] for k in self.parameter_names], dtype="float32")
        return x, mask, current, start

    def _named(self, values):
        return json_safe({cell: dict(zip(self.target_names, row.tolist()))
                          for cell, row in zip(self.cell_labels, np.asarray(values))})

    def _past_aggregate(self, record, t, requested_length):
        start = max(0, t - requested_length)
        num = record["num"][start:t].sum(0)
        den = record["den"][start:t].sum(0)
        den[:, COUNT_INDICES] = 1
        kpis = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
        kpis[:, 3] = np.where((kpis[:, 3] >= 0) & (kpis[:, 3] <= 100), kpis[:, 3], np.nan)
        return {"start_s": start, "end_s": t, "duration_s": t - start,
                "complete_window": t - start == requested_length,
                "aggregation": "native sufficient-statistic sums; events summed, ratios weighted by their native denominators",
                "kpis": self._named(kpis)}

    def snapshot(self, run_id: str, issue_time_s: int):
        t = self._time(issue_time_s)
        x, mask, _, start = self._observed_inputs(run_id, t)
        record = self._run(run_id)
        latest = np.where(record["valid"][t - 1], record["y"][t - 1], np.nan)
        warnings = []
        if t < 180:
            warnings.append(f"Only {t} seconds observed; unavailable earlier history is left-padded with masked zeros")
        if t % 10:
            warnings.append("Issue time is between training origins (10-second stride); temporal interpolation is unvalidated")
        return json_safe({**deepcopy(self._index[run_id]), "issue_time_s": t,
                          "history_start_s": start, "history_end_s": t,
                          "history_rows": len(x), "latest_kpis": self._named(latest),
                          "observed_90s": self._past_aggregate(record, t, 90),
                          "observed_180s": self._past_aggregate(record, t, 180),
                          "history": {"times_s": np.arange(start, t) + .5,
                                      "values": np.where(mask, x, np.nan), "valid": mask,
                                      "input_names": self.manifest["input_names"], "cell_labels": self.cell_labels},
                          "kpi_history": {"times_s": np.arange(start, t) + .5,
                                          "values": np.where(record["valid"][start:t], record["y"][start:t], np.nan),
                                          "valid": record["valid"][start:t],
                                          "metric_names": self.target_names, "cell_labels": self.cell_labels,
                                          "purpose": "past-only display of all fourteen observed KPIs; not an additional model input"},
                          "warnings": warnings})

    def validate_parameters(self, parameters):
        if not isinstance(parameters, dict) or set(parameters) != set(self.parameter_names):
            raise ValueError(f"parameters must contain exactly all 11 keys: {self.parameter_names}")
        values, warnings = [], []
        for key in self.parameter_names:
            value = parameters[key]
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)):
                raise ValueError(f"{key} must be a finite number")
            value = float(value)
            grid = self.manifest["parameter_values"][key]
            if not min(grid) <= value <= max(grid):
                raise ValueError(f"{key}={value} outside training envelope [{min(grid)},{max(grid)}]")
            if not any(abs(value - observed) <= 1e-7 for observed in grid):
                warnings.append(f"{key}={value:g}: interpolation within [{min(grid):g},{max(grid):g}], not separately validated on this value")
            values.append(value)
        return values, warnings

    def _validate_plans(self, plans, t):
        if not isinstance(plans, list) or not 1 <= len(plans) <= 256:
            raise ValueError("plans must contain 1 through 256 candidates")
        validated, ids = [], set()
        for plan in plans:
            if not isinstance(plan, dict):
                raise ValueError("Each plan must be an object")
            candidate_id = plan.get("candidate_id")
            if not isinstance(candidate_id, str) or not 1 <= len(candidate_id) <= 128 or candidate_id in ids:
                raise ValueError("candidate_id must be a unique nonempty string, at most 128 characters")
            ids.add(candidate_id)
            delay, horizon = plan.get("delay_s", 0), plan.get("horizon_s", 180)
            if isinstance(delay, bool) or delay not in DELAYS:
                raise ValueError("delay_s must be one of 0,30,60,90; longer delays require another validated model")
            if isinstance(horizon, bool) or horizon not in HORIZONS:
                raise ValueError("horizon_s must be 90 or 180")
            if t + delay > 720:
                raise ValueError("issue_time_s + delay_s must not exceed 720")
            values, warnings = self.validate_parameters(plan.get("parameters"))
            if t % 10:
                warnings.append("Issue time lies between training origins; temporal interpolation is unvalidated")
            validated.append((candidate_id, values, int(delay), int(horizon), warnings))
        return validated

    def predict_batch(self, run_id: str, issue_time_s: int, plans: list[dict]):
        t = self._time(issue_time_s)
        validated = self._validate_plans(plans, t)
        x, mask, current, start = self._observed_inputs(run_id, t)
        # One history encoding, then one tensor batch across candidates/delays.
        # Deduplication avoids recomputing the same parameters for each horizon.
        vectors = list(dict.fromkeys(tuple(p[1]) for p in validated))
        vector_index = {v: i for i, v in enumerate(vectors)}
        delays = sorted(set(p[2] for p in validated))
        with self._lock:
            result = self.forecast.predict_plans(x, mask, current, t, np.asarray(vectors), delays)
        outputs = []
        for candidate_id, values, delay, horizon, warnings in validated:
            kpis = result["kpi"][delays.index(delay), vector_index[tuple(values)], HORIZONS.index(horizon)]
            if not np.isfinite(kpis).all():
                raise RuntimeError("Frozen model produced a nonfinite prediction")
            outputs.append({"candidate_id": candidate_id, "run_id": run_id, "issue_time_s": t,
                            "parameters": dict(zip(self.parameter_names, values)),
                            "delay_s": delay, "horizon_s": horizon,
                            "interval_start_s": t + delay, "interval_end_s": t + delay + horizon,
                            "history_start_s": start, "history_end_s": t,
                            "predictions": self._named(kpis),
                            "interpolation": any("not separately validated" in w for w in warnings),
                            "configuration_seen_in_training_data": tuple(values) in self._training_configurations,
                            "warnings": warnings,
                            "checkpoint_sha256": CHECKPOINT_SHA256,
                            "evidence_type": "frozen_model_prediction_not_observed_intervention"})
        return outputs

    def explain_batch(self, run_id: str, issue_time_s: int, plans: list[dict]):
        """One-coordinate reset contrasts relative to this run's current RCP.

        These are signed frozen-model sensitivities, not SHAP values or identified
        causal effects. Nonlinear interactions mean they need not sum to a total.
        Only changed coordinates need an additional prediction; unchanged effects
        are exactly zero. Intended for a short list of finalist candidates.
        """
        t = self._time(issue_time_s)
        checked = self._validate_plans(plans, t)
        if len(plans) > 16:
            raise ValueError("Explain at most 16 finalists at once")
        current = self._index.get(run_id, {}).get("current_rcp")
        if current is None:
            raise ValueError("Unknown run_id")
        expanded = []
        reset_ids = {}
        for i, (candidate_id, values, delay, horizon, _) in enumerate(checked):
            parameters = dict(zip(self.parameter_names, values))
            expanded.append({"candidate_id": f"base_{i}", "parameters": parameters,
                             "delay_s": delay, "horizon_s": horizon})
            for key in self.parameter_names:
                if parameters[key] != current[key]:
                    reset_id = f"reset_{i}_{key}"
                    reset_ids[(i, key)] = reset_id
                    expanded.append({"candidate_id": reset_id, "parameters": {**parameters, key: current[key]},
                                     "delay_s": delay, "horizon_s": horizon})
        predictions = {r["candidate_id"]: r["predictions"] for r in self.predict_batch(run_id, t, expanded)}
        outputs = []
        for i, (candidate_id, _, _, _, _) in enumerate(checked):
            base = predictions[f"base_{i}"]
            contrasts = {}
            for key in self.parameter_names:
                reset = predictions[reset_ids[(i, key)]] if (i, key) in reset_ids else base
                contrasts[key] = {cell: {kpi: base[cell][kpi] - reset[cell][kpi] for kpi in base[cell]
                                        if base[cell][kpi] is not None and reset[cell].get(kpi) is not None}
                                  for cell in self.cell_labels}
            outputs.append({"candidate_id": candidate_id, "contrasts": contrasts,
                            "definition": "prediction(candidate) - prediction(candidate with one coordinate reset to current)",
                            "reference_parameters": deepcopy(current),
                            "evidence_type": "model_relative_coordinate_reset_not_SHAP_not_identified_causality"})
        return outputs
