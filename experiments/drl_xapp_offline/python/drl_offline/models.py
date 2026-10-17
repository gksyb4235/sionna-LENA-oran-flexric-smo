"""Small Q policy; the separate response model is used only for offline training."""
import numpy as np
import torch
from torch import nn


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_count):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(state_dim, 64), nn.ReLU(),
                                    nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, action_count))

    def forward(self, state):
        return self.layers(state)


class ResponseNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(14, 128), nn.SiLU(),
                                    nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, 15))

    def forward(self, x):
        return self.layers(x)


def encode_targets(y, exog):
    """PRB and UE shares use logits; positive rates/counts/delay use log1p."""
    z = np.asarray(y[..., :15], dtype=np.float64).copy()
    z[..., 3:6] /= np.maximum(exog[..., 1:2], 1e-6)
    q = np.clip(z[..., :6], .001, .999)
    z[..., :6] = np.log(q / (1 - q))
    z[..., 6:] = np.log1p(np.maximum(z[..., 6:], 0))
    return z.astype(np.float32)


def decode_targets(z, exog):
    y = np.zeros(z.shape[:-1] + (18,), dtype=np.float32)
    y[..., :6] = 1 / (1 + np.exp(-np.clip(z[..., :6], -20, 20)))
    shares = y[..., 3:6]
    y[..., 3:6] = shares / np.maximum(shares.sum(-1, keepdims=True), 1e-12) * exog[..., 1:2]
    y[..., 6:15] = np.maximum(np.expm1(np.clip(z[..., 6:15], 0, 10)), 0)
    # Buffered delivery can exceed generation within a reporting window.
    scale = np.minimum(1., 1.2 * exog[..., 2] / np.maximum(y[..., 6:9].sum(-1), 1e-6))
    y[..., 6:9] *= scale[..., None]
    y[..., 10] = np.minimum(y[..., 10], y[..., 9])
    y[..., 12] = np.minimum(y[..., 12], y[..., 10])
    y[..., 15:17] = exog[..., [2, 1]]
    y[..., 17] = y[..., 6:9].sum(-1)
    return y


class ResponseEnsemble:
    def __init__(self, models, x_mean, x_std, z_mean, z_std, device):
        self.models = models
        self.x_mean, self.x_std = x_mean, x_std
        self.z_mean, self.z_std = z_mean, z_std
        self.device = device

    @torch.inference_mode()
    def predict(self, p, exog):
        x = (np.column_stack([p, exog]) - self.x_mean) / self.x_std
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        predictions = []
        for model in self.models:
            chunks = [model(v).cpu().numpy() for v in x.split(4096)]
            z = np.concatenate(chunks) * self.z_std + self.z_mean
            predictions.append(decode_targets(z, exog))
        return np.stack(predictions)
