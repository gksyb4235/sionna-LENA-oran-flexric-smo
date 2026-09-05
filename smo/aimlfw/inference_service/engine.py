"""Model loading and deterministic batch forward passes for Inference_Service.

Design.md (Inference_Service, Requirement 5): load one registered GNN
artifact within 60 seconds of startup and serve deterministic batch
predictions with full Target_KPI and per-cell completeness. This module
holds no HTTP concerns — see ``server.py`` for the FastAPI boundary.
"""

from __future__ import annotations

import asyncio
import logging
from decimal import Decimal
from typing import Any

import torch

from smo.aimlfw.common.constants import (
    CONTROL_PARAMETER_NAMES,
    CONTROL_PARAMETER_RANGES,
    MAX_TIME_STEP,
    MIN_TIME_STEP,
    TARGET_KPIS,
    TTT_ALLOWED_MS,
)
from smo.aimlfw.model_registry import ModelRegistry
from smo.aimlfw.training_manager.train_gnn import CellGraphGnn, node_input_vector

from .errors import InferenceServiceError
from .schemas import MAX_BATCH_SIZE, ParameterSetInput

logger = logging.getLogger(__name__)

DEFAULT_LOAD_TIMEOUT_SECONDS = 60.0
_IN_DIM = len(CONTROL_PARAMETER_NAMES) + 1  # +1 for the normalized Time_Step input
_OUT_DIM = len(TARGET_KPIS)
_FORWARD_SEED = 0  # design.md 6: "forward pass는 순수 함수로 래핑하고 난수 시드를 고정" (5.8)


def control_parameter_violations(
    parameter_sets: list[ParameterSetInput],
) -> list[dict[str, Any]]:
    """Return every (Parameter_Set index, cell_id, parameter, value) range/step violation.

    Requirement 5.5 requires every violation across the whole batch to be
    collected atomically, so no ParameterSet is short-circuited on its own
    first violation.
    """
    violations: list[dict[str, Any]] = []
    for index, parameter_set in enumerate(parameter_sets):
        for cell_id, cell_parameters in parameter_set.cells.items():
            for name in CONTROL_PARAMETER_NAMES:
                value = getattr(cell_parameters, name)
                if name == "ttt_ms":
                    if value not in TTT_ALLOWED_MS:
                        violations.append(
                            {"parameter_set_index": index, "cell_id": cell_id, "parameter": name, "value": value}
                        )
                    continue
                rule = CONTROL_PARAMETER_RANGES[name]
                minimum = Decimal(str(rule["min"]))
                maximum = Decimal(str(rule["max"]))
                step = Decimal(str(rule["step"]))
                decimal_value = Decimal(str(value))
                if not (minimum <= decimal_value <= maximum) or (decimal_value - minimum) % step != 0:
                    violations.append(
                        {"parameter_set_index": index, "cell_id": cell_id, "parameter": name, "value": value}
                    )
    return violations


class InferenceEngine:
    """Load one GNN artifact and serve deterministic batch predictions."""

    def __init__(
        self,
        model_registry: ModelRegistry,
        model_name: str,
        model_version: int | None = None,
        *,
        load_timeout_seconds: float = DEFAULT_LOAD_TIMEOUT_SECONDS,
    ) -> None:
        self.model_registry = model_registry
        self.model_name = model_name
        self.requested_version = model_version
        self.load_timeout_seconds = load_timeout_seconds

        self.model_load_status: str = "not_loaded"
        self.loaded_model_name: str | None = None
        self.loaded_model_version: int | None = None
        self._model: CellGraphGnn | None = None
        self._load_lock = asyncio.Lock()

    async def load(self) -> None:
        """Load the configured model artifact within ``load_timeout_seconds`` (5.1, 5.11).

        Idempotent: a second call while already ``loaded`` is a no-op, so
        callers may safely invoke this from both a FastAPI startup hook and
        test setup. Any failure (registry lookup, missing/corrupt artifact,
        or a timeout) leaves ``model_load_status == "not_loaded"`` rather
        than raising, so the service can still start and answer ``/status``.
        """
        async with self._load_lock:
            if self.model_load_status == "loaded":
                return
            try:
                model, version = await asyncio.wait_for(
                    asyncio.to_thread(self._load_sync), timeout=self.load_timeout_seconds
                )
            except Exception:  # noqa: BLE001 - startup boundary: any failure means not_loaded
                logger.exception(
                    "Inference_Service failed to load model_name=%r within %.1fs",
                    self.model_name,
                    self.load_timeout_seconds,
                )
                self.model_load_status = "not_loaded"
                return
            self._model = model
            self.loaded_model_name = self.model_name
            self.loaded_model_version = version
            self.model_load_status = "loaded"

    def _load_sync(self) -> tuple[CellGraphGnn, int]:
        record = (
            self.model_registry.latest(self.model_name)
            if self.requested_version is None
            else self.model_registry.get(self.model_name, self.requested_version)
        )
        return self._load_record(record)

    @staticmethod
    def _load_record(record: Any) -> tuple[CellGraphGnn, int]:
        # The artifact at record.artifact_uri is produced exclusively by this
        # framework's own train_gnn.py -> Model_Storage pipeline (never a
        # user upload), so loading it with the default unpickling behavior
        # is within this service's trust boundary.
        checkpoint = torch.load(record.artifact_uri, map_location="cpu", weights_only=False)
        model = CellGraphGnn(_IN_DIM, _OUT_DIM)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        return model, record.version

    async def switch_version(self, model_name: str, model_version: int) -> None:
        """Load a candidate fully, then atomically swap the in-process serving model."""
        if model_name != self.model_name:
            raise InferenceServiceError(
                "model_not_found",
                "Inference_Service is configured for a different model name",
                {"configured_model_name": self.model_name, "requested_model_name": model_name},
            )
        async with self._load_lock:
            try:
                record = await asyncio.to_thread(self.model_registry.get, model_name, model_version)
                candidate, version = await asyncio.wait_for(
                    asyncio.to_thread(self._load_record, record),
                    timeout=self.load_timeout_seconds,
                )
            except InferenceServiceError:
                raise
            except Exception as exc:
                raise InferenceServiceError(
                    "model_load_failed",
                    "Candidate model could not be loaded; the current serving version was retained",
                    {"model_name": model_name, "model_version": model_version},
                ) from exc
            self._model = candidate
            self.loaded_model_name = model_name
            self.loaded_model_version = version
            self.requested_version = version
            self.model_load_status = "loaded"

    def predict_batch(self, parameter_sets: list[ParameterSetInput], time_step: Any) -> list[dict[str, Any]]:
        """Run a deterministic forward pass for every Parameter_Set (5.2, 5.3, 5.4, 5.8, 5.9).

        Checks run in this fixed order, each atomic (no partial results):
        model-loaded (5.11) -> batch size (5.6, 5.7) -> Time_Step validity
        (5.12, 5.13) -> Control_Parameter range/step (5.5). Requirement 5.11
        states that *every* Batch_Prediction_Request must fail with
        ``model_not_loaded`` while the model is unavailable, so that check
        is checked first regardless of what else is wrong with the request.
        """
        if self.model_load_status != "loaded" or self._model is None:
            raise InferenceServiceError(
                "model_not_loaded",
                "Inference_Service has no loaded model artifact",
                {"model_name": self.model_name},
            )
        if len(parameter_sets) == 0:
            raise InferenceServiceError(
                "empty_batch",
                "Batch_Prediction_Request must contain at least one Parameter_Set",
                {"received": 0},
            )
        if len(parameter_sets) > MAX_BATCH_SIZE:
            raise InferenceServiceError(
                "batch_too_large",
                f"Batch_Prediction_Request must contain at most {MAX_BATCH_SIZE} Parameter_Set entries",
                {"received": len(parameter_sets)},
            )
        resolved_time_step = MIN_TIME_STEP if time_step is None else time_step
        if (
            not isinstance(resolved_time_step, int)
            or isinstance(resolved_time_step, bool)
            or not MIN_TIME_STEP <= resolved_time_step <= MAX_TIME_STEP
        ):
            raise InferenceServiceError(
                "invalid_time_step",
                "time_step must be an integer between 0 and 4",
                {"time_step": time_step},
            )
        violations = control_parameter_violations(parameter_sets)
        if violations:
            raise InferenceServiceError(
                "parameter_out_of_range",
                "One or more Control_Parameter values are outside the allowed range or step",
                {"violations": violations},
            )

        predictions: list[dict[str, Any]] = []
        for index, parameter_set in enumerate(parameter_sets):
            try:
                predictions.append(self._predict_one(index, parameter_set, resolved_time_step))
            except InferenceServiceError:
                raise
            except Exception as exc:  # noqa: BLE001 - Requirement 5.10: any mid-batch forward-pass
                # failure (e.g. a corrupt/incompatible loaded artifact) must surface as a
                # well-formed InferenceServiceError carrying the model name/version and a
                # stable error_code, never as a bare unhandled exception, and must never
                # leak the ``predictions`` already computed for earlier items in this batch.
                raise InferenceServiceError(
                    "prediction_failed",
                    "Inference_Service failed to produce a prediction for one Parameter_Set",
                    {
                        "model_name": self.loaded_model_name,
                        "model_version": self.loaded_model_version,
                        "parameter_set_index": index,
                    },
                ) from exc
        return predictions

    def _predict_one(self, index: int, parameter_set: ParameterSetInput, time_step: int) -> dict[str, Any]:
        cell_ids = sorted(parameter_set.cells)  # fixed order: independent of request JSON key order (5.8)
        node_inputs = [
            node_input_vector(
                {name: getattr(parameter_set.cells[cell_id], name) for name in CONTROL_PARAMETER_NAMES},
                time_step,
            )
            for cell_id in cell_ids
        ]
        # Requirement 5.8/Property 33: the forward pass has no learned randomness
        # (no dropout, eval() disables batchnorm running-stat updates), but the
        # RNG is still pinned so any future stochastic layer stays deterministic.
        torch.manual_seed(_FORWARD_SEED)
        with torch.no_grad():
            output = self._model(torch.tensor(node_inputs, dtype=torch.float32))
        rows = output.tolist()

        cell_kpi: dict[str, dict[str, float]] = {}
        for cell_id, row in zip(cell_ids, rows, strict=True):
            cell_kpi[cell_id] = {kpi: float(value) for kpi, value in zip(TARGET_KPIS, row, strict=True)}

        target_kpi = {
            kpi: sum(cell_kpi[cell_id][kpi] for cell_id in cell_ids) / len(cell_ids) for kpi in TARGET_KPIS
        }
        return {"index": index, "target_kpi": target_kpi, "cell_kpi": cell_kpi}


__all__ = ["InferenceEngine", "control_parameter_violations"]
