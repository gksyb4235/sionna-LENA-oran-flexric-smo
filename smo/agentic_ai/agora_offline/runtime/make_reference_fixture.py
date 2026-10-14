"""Generate checked cross-platform reference values using the original Forecast.

Run only from the original training workspace with PYTHONPATH including both
that workspace and agora_offline_stage. Never regenerate this fixture on deploy.
"""
import json
from pathlib import Path
import platform
import sys
import numpy as np
import torch
from .api import Runtime, CHECKPOINT_SHA256


def main():
    from agora_final_90_180.inference import Forecast as Original
    runtime = Runtime()
    original = Original(runtime.root / "parameter5_backbone.pt")
    cases = []
    for split, t in (("train", 30), ("val", 180), ("test", 630)):
        row = runtime.list_runs(split)[0]
        observed, valid, current, _ = runtime._observed_inputs(row["run_id"], t)
        alternative = current.copy()
        alternative[0] = 36
        candidates = np.stack([current, alternative])
        delays = [0, 30, 60, 90]
        expected = original.predict_plans(observed, valid, current, t, candidates, delays)
        actual = runtime.forecast.predict_plans(observed, valid, current, t, candidates, delays)
        np.testing.assert_array_equal(actual["kpi"], expected["kpi"])
        cases.append({"run_id": row["run_id"], "split": split, "issue_time_s": t,
                      "parameters": candidates.tolist(), "delays_s": delays,
                      "predictions_shape": list(expected["kpi"].shape),
                      "predictions": expected["kpi"].tolist()})
    fixture = {"checkpoint_sha256": CHECKPOINT_SHA256, "reference": "original agora_final_90_180.Forecast",
               "python": sys.version.split()[0], "platform": platform.platform(),
               "torch": torch.__version__, "numpy": np.__version__,
               "horizons_s": [90, 180], "absolute_tolerance": 0.001, "relative_tolerance": 1e-5,
               "reason": "Float32 operations may vary slightly across OS and torch CPU kernels; tolerance applies to raw KPI units",
               "cases": cases}
    path = Path(__file__).resolve().parent / "reference_predictions.json"
    path.write_text(json.dumps(fixture, indent=2), encoding="utf-8")
    print(json.dumps({"path": str(path), "cases": len(cases), "bytes": path.stat().st_size}))


if __name__ == "__main__":
    main()
