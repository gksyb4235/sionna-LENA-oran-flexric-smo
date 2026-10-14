"""Minimal final-model inference with unchanged affine scaling and clipping."""
from pathlib import Path
import numpy as np
import torch
from .edge_model import PaperGNN


class Preprocessor:
    def __init__(self, stats):
        self.stats = stats
        for key, value in stats.items():
            if key.endswith(("_mean", "_std")):
                setattr(self, key, np.asarray(value, dtype="float32"))

    def inverse(self, z):
        y = np.asarray(z, dtype="float64") * self.y_std + self.y_mean
        ids = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13]
        y[..., ids] = np.maximum(y[..., ids], 0)
        ids = [1, 3, 4, 12, 13]
        y[..., ids] = np.minimum(y[..., ids], 100)
        return y


def summaries(x, m):
    count = m.sum(1)
    mean = (x * m).sum(1) / np.maximum(count, 1)
    std = np.sqrt(np.maximum(((x - mean[:, None]) ** 2 * m).sum(1) / np.maximum(count, 1), 0))
    return np.concatenate([a.reshape(len(x), -1) for a in (mean, std, x[:, -1], m.mean(1))], -1).astype("float32")


class Forecast:
    def __init__(self, checkpoint: Path, device="cpu"):
        self.device = device
        raw = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.prep = Preprocessor(raw["preprocessing"])
        self.metadata = raw["metadata"]
        self.model = PaperGNN(**raw["config"]).to(device)
        self.model.load_state_dict(raw["state_dict"], strict=True)
        self.model.eval()
        if self.model.config["horizons_s"] != [60, 90, 150, 180]:
            raise ValueError("Unexpected frozen model horizons")

    def inputs(self, observed, valid, current_rcp, issue_time_s):
        x = np.asarray(observed, dtype="float32")
        m = np.asarray(valid, dtype=bool)
        p = np.asarray(current_rcp, dtype="float32")
        if x.ndim != 3 or x.shape[1:] != (3, 12) or m.shape != x.shape or p.shape != (11,):
            raise ValueError("Expected history (seconds,3,12), matching mask, and 11 RCPs")
        if not 0 < len(x) <= issue_time_s or not np.isfinite(p).all():
            raise ValueError("Invalid time/history/RCP")
        m = m & np.isfinite(x)
        length = self.model.history
        x = np.where(m, (x - self.prep.x_mean) / self.prep.x_std, 0)[-length:]
        m = m[-length:]
        missing = length - len(x)
        x = np.pad(x, ((missing, 0), (0, 0), (0, 0)))[None]
        m = np.pad(m, ((missing, 0), (0, 0), (0, 0)))[None]
        p = ((p - self.prep.p_mean) / self.prep.p_std)[None]
        return p, x, m.astype("float32"), np.array([[issue_time_s / 900]], dtype="float32"), summaries(x, m)

    @torch.inference_mode()
    def predict_plans(self, observed, valid, current_rcp, issue_time_s, candidates, delays_s):
        batch = self.inputs(observed, valid, current_rcp, issue_time_s)
        c, ds = np.asarray(candidates, dtype="float32"), np.asarray(delays_s)
        if c.ndim != 2 or c.shape[1] != 11 or not len(c) or not np.isfinite(c).all():
            raise ValueError("Expected finite candidates C x 11")
        if ds.ndim != 1 or not len(ds) or not np.isin(ds, [0, 30, 60, 90]).all():
            raise ValueError("Supported delays: 0,30,60,90 seconds")
        if issue_time_s < 30 or np.any(issue_time_s + ds > 720):
            raise ValueError("Validated action start range is 30 through 720 seconds")
        inp = tuple(torch.as_tensor(v, device=self.device, dtype=torch.float32) for v in batch)
        pc = torch.as_tensor((c - self.prep.p_mean) / self.prep.p_std, device=self.device)[None]
        p, s = self.model.multi(*inp, pc, delays=ds, return_state=True)
        return {"kpi": np.take(self.prep.inverse(p.cpu().numpy())[0], [1, 3], axis=2),
                "predicted_pre_action_state": s.cpu().numpy()[0] * self.prep.x_std + self.prep.x_mean,
                "state_forecast_valid": ds > 0, "delays_s": ds.astype("int64"),
                "start_times_s": issue_time_s + ds, "horizons_s": np.array([90, 180])}
