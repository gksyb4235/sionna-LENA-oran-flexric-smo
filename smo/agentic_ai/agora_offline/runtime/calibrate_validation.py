"""Audit prediction residuals using validation runs and KEEP actions only.

This offline evaluation script is the only calibration path allowed to inspect
future observations. Runtime prediction still consumes strictly past history.
Residual quantiles are descriptive: checkpoint selection reused the validation
split, observations within a run are correlated, and switched/interpolated/delayed
actions have not been calibrated. These are NOT a general uncertainty guarantee.

Run from the installed package's parent directory:
    python -m agora_offline.runtime.calibrate_validation
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
from .api import Runtime, CHECKPOINT_SHA256, json_safe


def native_future_truth(record, start, horizon):
    # Only this offline calibration script accesses the future interval. This
    # function is never imported into api.py or the online candidate estimator.
    numerator = record["num"][start:start + horizon].sum(0)
    denominator = record["den"][start:start + horizon].sum(0)
    denominator[:, [7, 8, 9]] = 1
    truth = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    valid = np.isfinite(truth)
    valid[:, 3] &= (truth[:, 3] >= 0) & (truth[:, 3] <= 100)
    return truth, valid


def main():
    runtime = Runtime()
    rows = runtime.list_runs("val")
    if len(rows) != 180 or any(row["split"] != "val" for row in rows):
        raise ValueError("Expected exactly the frozen 180 validation runs")
    origins, horizons = [180, 360, 540], [90, 180]
    run_ids, times, truths, predictions, masks = [], [], [], [], []
    started = time.perf_counter()
    for i, row in enumerate(rows):
        run_id = row["run_id"]
        for t in origins:
            plans = [{"candidate_id": f"keep_{h}", "parameters": row["current_rcp"],
                      "delay_s": 0, "horizon_s": h} for h in horizons]
            output = runtime.predict_batch(run_id, t, plans)
            predicted = np.asarray([[[p["predictions"][c][k] for k in runtime.target_names]
                                     for c in runtime.cell_labels] for p in output])
            # Explicit evaluation access after inference; truth cannot influence
            # the observations, parameters, or issue time passed to the model.
            source = runtime._run(run_id)
            actual = [native_future_truth(source, t, h) for h in horizons]
            run_ids.append(run_id)
            times.append(t)
            predictions.append(predicted)
            truths.append(np.stack([a[0] for a in actual]))
            masks.append(np.stack([a[1] for a in actual]))
        if (i + 1) % 60 == 0:
            print(f"Validation keep-action residuals {i + 1}/180 runs", flush=True)
    predicted, truth, valid = np.stack(predictions), np.stack(truths), np.stack(masks)
    residual = np.where(valid, truth - predicted, np.nan)
    path = runtime.root / "calibration_hold.npz"
    np.savez_compressed(path, run_ids=np.asarray(run_ids), issue_times_s=np.asarray(times),
                        horizons_s=np.asarray(horizons), prediction=predicted, truth=truth,
                        residual_truth_minus_prediction=residual, valid=valid)
    statistics = {}
    for hi, h in enumerate(horizons):
        statistics[str(h)] = {}
        for ci, cell in enumerate(runtime.cell_labels):
            statistics[str(h)][cell] = {}
            for ki, kpi in enumerate(runtime.target_names):
                r = residual[:, hi, ci, ki]
                r = r[np.isfinite(r)]
                statistics[str(h)][cell][kpi] = {
                    "samples": len(r), "rmse": float(np.sqrt(np.mean(r * r))),
                    "mae": float(np.abs(r).mean()), "mean_truth_minus_prediction": float(r.mean()),
                    "residual_q05": float(np.quantile(r, .05)), "residual_q95": float(np.quantile(r, .95)),
                    "absolute_error_q90": float(np.quantile(np.abs(r), .9)),
                } if len(r) else {"samples": 0}
    summary = {
        "format": "agora-validation-keep-residuals-v1", "checkpoint_sha256": CHECKPOINT_SHA256,
        "split": "val", "partition": "val", "scope": "keep_current_only", "delay_s": 0, "delays_s": [0],
        "switched_action_calibrated": False, "delayed_action_calibrated": False,
        "interpolation_calibrated": False, "formal_coverage_guarantee": False,
        "validation_used_for_checkpoint_selection": True,
        "runs": 180, "cases": len(times), "origins_s": origins, "horizons_s": horizons,
        "run_ids": [row["run_id"] for row in rows],
        "residual_definition": "native future interval truth minus frozen prediction",
        "aggregation": "Native numerator/denominator sums; event counts summed; invalid loss ratios masked",
        "dependence": "Three origins from each run are correlated; 540 cases are not 540 independent simulation seeds",
        "intended_use": "Descriptive baseline forecast error reference; not a calibrated safety bound for candidate switches",
        "future_observations": "Read only by offline calibrate_validation.py, never passed to inference or candidate ranking",
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "residual_file": path.name, "residual_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "statistics": statistics, "elapsed_s": time.perf_counter() - started,
    }
    summary["calibration_id"] = "val-keep-" + summary["residual_file_sha256"][:16]
    summary["rmse"] = {str(h): {"0": {cell: {kpi: statistics[str(h)][cell][kpi]["rmse"]
                                                     for kpi in runtime.target_names}
                                             for cell in runtime.cell_labels}}
                       for h in horizons}
    (runtime.root / "calibration_hold.json").write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
    print(json.dumps({"runs": summary["runs"], "cases": summary["cases"],
                      "elapsed_s": summary["elapsed_s"], "file": str(path)}, indent=2))


if __name__ == "__main__":
    main()
