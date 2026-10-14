"""Five-app model adapter; legacy observations remain available for diagnostics."""

from collections import OrderedDict
import hashlib
import io
import json
import os
from pathlib import Path
from threading import RLock
import numpy as np
import torch
from .api import Runtime as LegacyRuntime, json_safe
from .engine.inference import Forecast
from .engine.schema import APPS, CORE, COUNTS, HORIZONS, TARGETS


class Runtime(LegacyRuntime):
    def __init__(self, root=None, device="cpu"):
        self.root = Path(root or Path(__file__).parent / "assets_v16").resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        checkpoint = self.root / "model.pt"
        self.checkpoint_sha256 = self.manifest["checkpoint_sha256"]
        if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != self.checkpoint_sha256:
            raise ValueError("V16 checkpoint integrity check failed")
        torch.set_num_threads(max(1, min(4, int(os.environ.get("AGORA_TORCH_THREADS", "4")))))
        self.forecast = Forecast(checkpoint, device)
        self._index = {r["run_id"]: r for r in self.manifest["runs"]}
        self._file_index = {r["file"]: r for r in self.manifest["files"]}
        self._cache = OrderedDict()
        self._lock = RLock()
        self.parameter_names = self.manifest["parameter_names"]
        self.target_names = TARGETS
        self.cell_labels = ["5g", "4g1", "4g2"]
        self._training_configurations = {
            tuple(r["current_rcp"][k] for k in self.parameter_names)
            for r in self._index.values()
            if r["split"] == "train"
        }
        self.hold_action_calibration = None

    def metadata(self):
        return json_safe(
            dict(
                **self.forecast.metadata,
                model_name="AGORA Temporal GNN v16 · 5 Apps · 90/180s",
                checkpoint_sha256=self.checkpoint_sha256,
                training_horizons_s=HORIZONS,
                serving_horizons_s=HORIZONS,
                delays_s=[0, 30, 60, 90],
                issue_time_range_s=[30, 720],
                max_action_start_s=720,
                observation_stride_s=1,
                training_origin_stride_s=30,
                evaluation_origin_stride_s=10,
                parameter_values=self.manifest["parameter_values"],
                parameter_bounds={
                    k: [min(v), max(v)] for k, v in self.manifest["parameter_values"].items()
                },
                run_id_namespace=self.manifest["run_id_namespace"],
                replay_runs=len(self._index),
                apps=APPS,
                graph_nodes=5 + 5 + 3 + 3 * len(CORE),
                predicted_kpis=[TARGETS[i] for i in CORE],
                observed_kpis=TARGETS,
                derived_kpis=[
                    "power_w",
                    "energy_eff_mbit_per_j",
                    "traffic_share_pct",
                    "rsrp_mean_p10_gap_db",
                ],
                rsrp_mean_definition=self.manifest["rsrp_mean_definition"],
                rsrp_p10_definition=self.manifest["rsrp_p10_definition"],
                uncertainty_calibration={
                    "available": False,
                    "reason": "Risk scores are auxiliary model outputs, not calibrated guarantees",
                },
                hold_action_calibration=None,
                execution_mode="offline_prediction_only",
                counter_semantics="Interval counts; received-packet delay exceedance excludes lost packets",
                limitations=[
                    "Same-clock fixed-run replay, not branched intervention validation",
                    "Unavailable RSRP is masked; P10 is not an all-UE coverage rate",
                    "Mean delay, packet loss and UE share are diagnostic observations, not v16 forecast outputs",
                    "App graph describes declared control capabilities; actual app policies were not observed",
                    "TTT and HYS remain shared across all three cells",
                    "4g1/4g2 are historical labels for 1.8 GHz NR cells",
                ],
            )
        )

    def _run(self, run_id):
        if run_id not in self._index:
            raise ValueError("Unknown original training run ID")
        with self._lock:
            if run_id not in self._cache:
                name = f"runs/{run_id}.npz"
                raw = (self.root / name).read_bytes()
                if hashlib.sha256(raw).hexdigest() != self._file_index[name]["sha256"]:
                    raise ValueError("Replay integrity failed")
                with np.load(io.BytesIO(raw), allow_pickle=False) as z:
                    record = {k: z[k] for k in z.files}
                if record["y"].shape != (900, 3, 17):
                    raise ValueError("Unexpected v16 replay shape")
                self._cache[run_id] = record
                while len(self._cache) > 4:
                    self._cache.popitem(last=False)
            self._cache.move_to_end(run_id)
            return self._cache[run_id]

    def _observed_inputs(self, run_id, t):
        record = self._run(run_id)
        start = max(0, t - 180)
        current = np.array(
            [self._index[run_id]["current_rcp"][k] for k in self.parameter_names], dtype="float32"
        )
        return record["x"][start:t], record["x_valid"][start:t], current, start

    def _past_aggregate(self, record, t, requested_length):
        start = max(0, t - requested_length)
        n = record["num"][start:t].sum(0)
        d = record["den"][start:t].sum(0)
        d[:, COUNTS] = 1
        y = np.divide(n, d, out=np.full_like(n, np.nan), where=d > 0)
        use = (record["rsrp_times"] >= start) & (record["rsrp_times"] < t)
        for c in range(3):
            values = record["rsrp_values"][use & (record["rsrp_cells"] == c + 1)]
            y[c, 14] = np.quantile(values, 0.1) if len(values) else np.nan
        y[:, 3] = np.where((y[:, 3] >= 0) & (y[:, 3] <= 100), y[:, 3], np.nan)
        named = self._named(y)
        for cell, row in named.items():
            a, b = row["rsrp_interval_mean_dbm"], row["rsrp_p10_dbm"]
            row["rsrp_mean_p10_gap_db"] = a - b if a is not None and b is not None else None
        return dict(
            start_s=start,
            end_s=t,
            duration_s=t - start,
            complete_window=t - start == requested_length,
            aggregation="Native counter sums; RSRP mean and P10 pool UE-time observations over the entire interval",
            kpis=named,
        )

    def snapshot(self, run_id, issue_time_s):
        result = super().snapshot(run_id, issue_time_s)
        result["warnings"] = [
            w.replace("training origins (10-second stride)", "evaluated origins (10-second stride)")
            for w in result["warnings"]
        ]
        result["kpi_history"]["purpose"] = (
            "Observed KPIs; diagnostic-only fields are not v16 prediction outputs"
        )
        result["model_version"] = "app5-v16"
        return result

    def _validate_plans(self, plans, t):
        return [
            (
                ident,
                values,
                delay,
                horizon,
                [
                    w.replace("between training origins", "between evaluated origins")
                    for w in warnings
                ],
            )
            for ident, values, delay, horizon, warnings in super()._validate_plans(plans, t)
        ]

    def predict_batch(self, run_id, issue_time_s, plans):
        t = self._time(issue_time_s)
        validated = self._validate_plans(plans, t)
        x, m, current, start = self._observed_inputs(run_id, t)
        vectors = list(dict.fromkeys(tuple(v[1]) for v in validated))
        ds = sorted(set(v[2] for v in validated))
        with self._lock:
            forecast = self.forecast.predict_plans(x, m, current, t, np.array(vectors), ds)
        results = []
        for ident, values, delay, horizon, warnings in validated:
            vi = vectors.index(tuple(values))
            di = ds.index(delay)
            hi = HORIZONS.index(horizon)
            physical = forecast["kpi"][di, vi, hi]
            if not np.isfinite(physical).all():
                raise RuntimeError("Nonfinite GNN output")
            rows = {
                cell: {TARGETS[k]: float(physical[c, j]) for j, k in enumerate(CORE)}
                for c, cell in enumerate(self.cell_labels)
            }
            total = sum(row["goodput_mbps"] for row in rows.values())
            for c, cell in enumerate(self.cell_labels):
                row = rows[cell]
                power = 130 + 4.7 * 10 ** ((values[c] - 30) / 10)
                row.update(
                    power_w=power,
                    energy_eff_mbit_per_j=row["goodput_mbps"] / power,
                    traffic_share_pct=100 * row["goodput_mbps"] / total if total > 0 else None,
                    rsrp_mean_p10_gap_db=row["rsrp_interval_mean_dbm"] - row["rsrp_p10_dbm"],
                )
            risk = forecast["rlf_risk"][di, vi, hi]
            results.append(
                dict(
                    candidate_id=ident,
                    run_id=run_id,
                    issue_time_s=t,
                    parameters=dict(zip(self.parameter_names, values)),
                    delay_s=delay,
                    horizon_s=horizon,
                    interval_start_s=t + delay,
                    interval_end_s=t + delay + horizon,
                    history_start_s=start,
                    history_end_s=t,
                    predictions=rows,
                    warnings=warnings,
                    interpolation=any("not separately validated" in w for w in warnings),
                    configuration_seen_in_training_data=tuple(values)
                    in self._training_configurations,
                    checkpoint_sha256=self.checkpoint_sha256,
                    evidence_type="frozen_model_prediction_not_observed_intervention",
                    rlf_risk=dict(
                        cell_occurrence=dict(zip(self.cell_labels, map(float, risk[:3]))),
                        network_violation_probability=float(risk[3]),
                        network_count_threshold=2 if horizon == 90 else 3,
                        calibrated=False,
                    ),
                )
            )
        return json_safe(results)
