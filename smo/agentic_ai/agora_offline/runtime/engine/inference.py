"""Portable inference: only schema, model, PyTorch and NumPy are required."""

from pathlib import Path
import numpy as np
import torch
from .model import TemporalGNN
from .schema import HORIZONS, INPUTS, TARGETS


def physical(z, stats, indices):
    y = (
        np.asarray(z, dtype="float64") * stats["y_std"][..., indices]
        + stats["y_mean"][..., indices]
    )
    for i, index in enumerate(indices):
        if index not in [5, 6, 14]:
            y[..., i] = np.maximum(y[..., i], 0)
        if index in [1, 3, 4, 12, 13, 16]:
            y[..., i] = np.minimum(y[..., i], 100)
    return y


class Forecast:
    def __init__(self, checkpoint, device="cpu"):
        self.device = torch.device(device)
        obj = torch.load(Path(checkpoint), map_location=self.device, weights_only=True)
        self.model = TemporalGNN(**obj["config"]).to(self.device).eval()
        self.model.load_state_dict(obj["state_dict"])
        self.stats = {k: np.asarray(v, dtype="float32") for k, v in obj["stats"].items()}
        self.indices = obj["config"]["target_indices"]
        self.metadata = dict(
            version=obj["version"],
            target_names=[TARGETS[i] for i in self.indices],
            input_names=INPUTS,
            horizons_s=HORIZONS,
            history_seconds=180,
            registered_parameters=sum(p.numel() for p in self.model.parameters()),
            app_nodes=obj["config"]["apps"],
            parameter_nodes=5,
            cell_nodes=3,
            kpi_nodes=3 * len(self.indices),
        )

    @torch.inference_mode()
    def predict_plans(self, history, valid, current, issue_time_s, candidates, delays):
        history = np.asarray(history, dtype="float32")
        valid = np.asarray(valid, dtype=bool)
        if history.shape != valid.shape or history.shape[1:] != (3, len(INPUTS)):
            raise ValueError("Expected history and mask with shape [seconds,3,15]")
        if not 1 <= len(history) <= 180:
            raise ValueError("History must contain 1..180 completed seconds")
        x = np.zeros((1, 180, 3, len(INPUTS)), dtype="float32")
        m = np.zeros_like(x)
        x[0, -len(history) :] = np.where(
            valid, (history - self.stats["x_mean"]) / self.stats["x_std"], 0
        ).clip(-12, 12)
        m[0, -len(history) :] = valid
        cp = (np.asarray(current, dtype="float32") - self.stats["p_mean"]) / self.stats["p_std"]
        selected = (np.asarray(candidates, dtype="float32") - self.stats["p_mean"]) / self.stats[
            "p_std"
        ]

        def tensor(a):
            return torch.as_tensor(a, dtype=torch.float32, device=self.device)

        current = tensor(cp[None])
        clock = tensor([[issue_time_s / 900]])
        state = self.model.encode(tensor(x), tensor(m), current, clock)
        c = len(selected)
        n = c * 2
        chosen = tensor(selected)[:, None].expand(-1, 2, -1).reshape(n, 11)
        hs = tensor(HORIZONS)[None].expand(c, -1).flatten()
        states = tuple(v.expand(n, *v.shape[1:]) for v in state)
        pp = []
        rr = []
        ss = []
        for delay in delays:
            p, r, s = self.model.decode(
                states,
                current.expand(n, -1),
                chosen,
                clock.expand(n, -1),
                torch.full((n,), float(delay), device=self.device),
                hs,
            )
            pp.append(p.cpu().numpy().reshape(c, 2, 3, -1))
            rr.append(r.sigmoid().cpu().numpy().reshape(c, 2, 4))
            ss.append(s.cpu().numpy().reshape(c, 2, 3, -1))
        return dict(
            kpi=physical(np.stack(pp), self.stats, self.indices),
            rlf_risk=np.stack(rr),
            predicted_pre_action_state=np.stack(ss),
            horizons_s=np.array(HORIZONS),
            target_indices=self.indices,
        )
