"""Deterministic GNN training script for the Training_Manager ``train_model`` stage.

Invoked as an isolated subprocess (design.md: "학습 실행은
``asyncio.create_subprocess_exec``로 격리된 학습 스크립트(train_gnn.py)를
호출"), this script trains a small graph neural network that predicts every
Target_KPI (design.md GNN_Model glossary entry) from a cell graph, the
Time_Step, and each cell's 5 Control_Parameter values.

Input contract (``--records-file``): a JSON list of objects shaped like
``FeatureRecord.model_dump(mode="json")`` (see
``smo.aimlfw.common.models.FeatureRecord``) — each entry has ``time_step``,
``cell_id``, ``control_parameters`` (5 keys), and ``features`` (Target_KPI ->
float | null). The distinct ``cell_id`` values observed across the input
form the cell graph's node set; a fully-connected (clique) topology is used
because the framework does not model inter-cell adjacency beyond "same
Feature_Group scenario" (design.md, Data Models comment on ``ParameterSet``
being a per-cell map rather than a single scalar tuple).

Output contract:
  * ``--artifact-output``: a ``torch.save`` checkpoint (state_dict + the
    normalization/graph metadata needed to reload the model), written to a
    plain filesystem path chosen by the caller. The Training_Manager's
    ``save_artifact`` stage is responsible for moving this file into
    Model_Storage's versioned location — this script does not know about
    model names or versions.
  * ``--metrics-output``: a JSON object with the fields Requirement 3.4
    requires the Training_Manager to persist to Model_Registry: split
    ratio, train/validation sample counts, seed, and per-Target_KPI MAPE
    (percent, rounded to 2 decimal places).

Determinism (Requirement 3.6, Property 18): every source of randomness
(the train/validation split, and PyTorch's weight initialization and
training-loop randomness) is derived solely from ``--seed``. The script
runs on CPU only and avoids any nondeterministic CUDA kernels or unordered
reductions, so repeated invocations with the same Feature_Group snapshot
and seed reproduce sample counts exactly and Target_KPI MAPE within the
1.0 percentage-point tolerance Requirement 3.6 allows (bit-identical here,
since no GPU nondeterminism is involved).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
from torch import nn

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, TARGET_KPIS, TTT_ALLOWED_MS

MAPE_EPSILON = 1e-6
MAPE_ROUND_NDIGITS = 2
HIDDEN_DIM = 16
DEFAULT_MAX_EPOCHS = 200
DEFAULT_PATIENCE = 10
DEFAULT_LEARNING_RATE = 0.05


def _normalize_parameter(name: str, value: float) -> float:
    """Map one Control_Parameter value to a roughly [0, 1] scale for the model input."""
    if name == "ttt_ms":
        index = TTT_ALLOWED_MS.index(int(value))
        return index / (len(TTT_ALLOWED_MS) - 1)
    ranges = {
        "tx_power_dbm": (30.0, 46.0),
        "ret_tilt_deg": (0.0, 15.0),
        "cio_bias_db": (-6.0, 6.0),
        "hysteresis_db": (0.0, 10.0),
    }
    minimum, maximum = ranges[name]
    return (float(value) - minimum) / (maximum - minimum)


def _node_input_vector(control_parameters: dict[str, Any], time_step: int) -> list[float]:
    values = [_normalize_parameter(name, control_parameters[name]) for name in CONTROL_PARAMETER_NAMES]
    values.append(time_step / 4.0)
    return values


class _Row:
    """One (time_step, cell_id) training row: input features + masked KPI targets."""

    __slots__ = ("time_step", "cell_id", "input_vector", "targets", "mask")

    def __init__(self, time_step: int, cell_id: str, input_vector: list[float], targets: list[float], mask: list[bool]):
        self.time_step = time_step
        self.cell_id = cell_id
        self.input_vector = input_vector
        self.targets = targets
        self.mask = mask


def _load_rows(records: list[dict[str, Any]]) -> list[_Row]:
    rows: list[_Row] = []
    for record in records:
        targets = [record["features"].get(kpi) for kpi in TARGET_KPIS]
        mask = [value is not None for value in targets]
        rows.append(
            _Row(
                time_step=int(record["time_step"]),
                cell_id=str(record["cell_id"]),
                input_vector=_node_input_vector(record["control_parameters"], int(record["time_step"])),
                targets=[float(value) if value is not None else 0.0 for value in targets],
                mask=mask,
            )
        )
    return rows


def _split_rows(rows: list[_Row], seed: int, train_split: float) -> tuple[list[_Row], list[_Row]]:
    """Deterministically shuffle-and-split rows using only ``seed`` as entropy."""
    indices = list(range(len(rows)))
    random.Random(seed).shuffle(indices)
    train_count = round(len(rows) * train_split)
    train_count = max(1, min(len(rows) - 1, train_count)) if len(rows) > 1 else len(rows)
    train_indices, val_indices = indices[:train_count], indices[train_count:]
    return [rows[i] for i in train_indices], [rows[i] for i in val_indices]


def _group_by_time_step(rows: list[_Row], node_order: list[str]) -> list[list[_Row]]:
    """Group rows into per-Time_Step graph snapshots, ordered by time_step then node order."""
    node_rank = {cell_id: index for index, cell_id in enumerate(node_order)}
    by_step: dict[int, list[_Row]] = {}
    for row in rows:
        by_step.setdefault(row.time_step, []).append(row)
    snapshots = []
    for step in sorted(by_step):
        snapshots.append(sorted(by_step[step], key=lambda row: node_rank[row.cell_id]))
    return snapshots


class CellGraphGnn(nn.Module):
    """A minimal two-layer graph-convolutional network over a fixed cell clique.

    Each node aggregates the mean hidden state of every other node in the
    same Time_Step snapshot (a fully-connected graph over the observed
    cells) alongside its own state, mirroring a standard GCN update rule
    while staying dependency-free (no ``torch_geometric`` requirement).
    """

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.self1 = nn.Linear(in_dim, hidden_dim)
        self.neighbor1 = nn.Linear(in_dim, hidden_dim)
        self.self2 = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor2 = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, out_dim)
        self.activation = nn.ReLU()

    def _propagate(self, self_layer: nn.Linear, neighbor_layer: nn.Linear, h: torch.Tensor) -> torch.Tensor:
        num_nodes = h.shape[0]
        if num_nodes > 1:
            neighbor_sum = h.sum(dim=0, keepdim=True) - h
            neighbor_mean = neighbor_sum / (num_nodes - 1)
        else:
            neighbor_mean = torch.zeros_like(h)
        return self.activation(self_layer(h) + neighbor_layer(neighbor_mean))

    def forward(self, node_inputs: torch.Tensor) -> torch.Tensor:
        """``node_inputs``: (num_nodes_in_snapshot, in_dim) -> (num_nodes_in_snapshot, out_dim)."""
        h = self._propagate(self.self1, self.neighbor1, node_inputs)
        h = self._propagate(self.self2, self.neighbor2, h)
        return self.output(h)


def _forward_snapshot(model: CellGraphGnn, snapshot: list[_Row]) -> torch.Tensor:
    inputs = torch.tensor([row.input_vector for row in snapshot], dtype=torch.float32)
    return model(inputs)


def _masked_mse_loss(predictions: torch.Tensor, snapshot: list[_Row]) -> torch.Tensor:
    targets = torch.tensor([row.targets for row in snapshot], dtype=torch.float32)
    mask = torch.tensor([row.mask for row in snapshot], dtype=torch.float32)
    squared_error = (predictions - targets) ** 2 * mask
    denominator = mask.sum()
    if denominator.item() == 0:
        return torch.zeros((), dtype=torch.float32)
    return squared_error.sum() / denominator


def _epoch_loss(model: CellGraphGnn, snapshots: list[list[_Row]]) -> float:
    if not snapshots:
        return 0.0
    total, count = 0.0, 0
    with torch.no_grad():
        for snapshot in snapshots:
            predictions = _forward_snapshot(model, snapshot)
            loss = _masked_mse_loss(predictions, snapshot)
            total += float(loss.item())
            count += 1
    return total / count


def _train(
    model: CellGraphGnn,
    train_snapshots: list[list[_Row]],
    val_snapshots: list[list[_Row]],
    *,
    max_epochs: int,
    patience: int,
    learning_rate: float,
) -> int:
    """Train with early stopping on validation loss; returns epochs actually run."""
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    best_val_loss = float("inf")
    best_state = {key: value.clone() for key, value in model.state_dict().items()}
    epochs_without_improvement = 0
    epochs_run = 0

    for epoch in range(max_epochs):
        epochs_run = epoch + 1
        model.train()
        for snapshot in train_snapshots:
            optimizer.zero_grad()
            predictions = _forward_snapshot(model, snapshot)
            loss = _masked_mse_loss(predictions, snapshot)
            loss.backward()
            optimizer.step()

        model.eval()
        evaluation_snapshots = val_snapshots or train_snapshots
        val_loss = _epoch_loss(model, evaluation_snapshots)
        if val_loss < best_val_loss - 1e-9:
            best_val_loss = val_loss
            best_state = {key: value.clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return epochs_run


def _evaluate_mape(model: CellGraphGnn, snapshots: list[list[_Row]]) -> dict[str, float]:
    """Per-Target_KPI MAPE (percent, 2 decimals) over every masked target in ``snapshots``."""
    absolute_percentage_errors: dict[str, list[float]] = {kpi: [] for kpi in TARGET_KPIS}
    model.eval()
    with torch.no_grad():
        for snapshot in snapshots:
            predictions = _forward_snapshot(model, snapshot).tolist()
            for row, prediction in zip(snapshot, predictions, strict=True):
                for index, kpi in enumerate(TARGET_KPIS):
                    if not row.mask[index]:
                        continue
                    actual = row.targets[index]
                    denominator = max(abs(actual), MAPE_EPSILON)
                    error = abs(prediction[index] - actual) / denominator * 100.0
                    absolute_percentage_errors[kpi].append(error)
    metrics: dict[str, float] = {}
    for kpi, errors in absolute_percentage_errors.items():
        if errors:
            metrics[kpi] = round(sum(errors) / len(errors), MAPE_ROUND_NDIGITS)
    return metrics


def run_training(
    records: list[dict[str, Any]],
    *,
    seed: int,
    train_split: float = 0.8,
    max_epochs: int = DEFAULT_MAX_EPOCHS,
    patience: int = DEFAULT_PATIENCE,
    learning_rate: float = DEFAULT_LEARNING_RATE,
) -> tuple[CellGraphGnn, list[str], dict[str, Any]]:
    """Train the cell-graph GNN and return the model, node order, and metrics payload."""
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    random.seed(seed)

    rows = _load_rows(records)
    if not rows:
        raise ValueError("training requires at least one Feature_Record")
    node_order = sorted({row.cell_id for row in rows})

    train_rows, val_rows = _split_rows(rows, seed, train_split)
    train_snapshots = _group_by_time_step(train_rows, node_order)
    val_snapshots = _group_by_time_step(val_rows, node_order)

    in_dim = len(CONTROL_PARAMETER_NAMES) + 1
    out_dim = len(TARGET_KPIS)
    model = CellGraphGnn(in_dim, out_dim)

    epochs_run = _train(
        model,
        train_snapshots,
        val_snapshots,
        max_epochs=max_epochs,
        patience=patience,
        learning_rate=learning_rate,
    )
    metrics = _evaluate_mape(model, val_snapshots or train_snapshots)

    payload = {
        "metrics": metrics,
        "train_split": train_split,
        "validation_split": round(1.0 - train_split, 6),
        "train_samples": len(train_rows),
        "validation_samples": len(val_rows),
        "seed": seed,
        "epochs_run": epochs_run,
        "node_order": node_order,
    }
    return model, node_order, payload


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records-file", required=True, type=Path)
    parser.add_argument("--artifact-output", required=True, type=Path)
    parser.add_argument("--metrics-output", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--train-split", type=float, default=0.8)
    parser.add_argument("--max-epochs", type=int, default=DEFAULT_MAX_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    return parser.parse_args(argv)


# Public aliases used by Inference_Service (Requirement 5) so the node-input
# encoding stays identical between training and inference without either
# component importing the other's private helpers.
normalize_parameter = _normalize_parameter
node_input_vector = _node_input_vector


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        records = json.loads(args.records_file.read_text(encoding="utf-8"))
        model, node_order, payload = run_training(
            records,
            seed=args.seed,
            train_split=args.train_split,
            max_epochs=args.max_epochs,
            patience=args.patience,
            learning_rate=args.learning_rate,
        )
        args.artifact_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "node_order": node_order,
                "target_kpis": list(TARGET_KPIS),
                "control_parameter_names": list(CONTROL_PARAMETER_NAMES),
            },
            args.artifact_output,
        )
        args.metrics_output.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_output.write_text(json.dumps(payload), encoding="utf-8")
    except Exception as exc:  # noqa: BLE001 - subprocess boundary: report and exit non-zero
        print(f"train_gnn failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
