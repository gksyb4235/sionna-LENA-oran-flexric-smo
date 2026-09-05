"""Deterministic Retraining Controller test clients."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from smo.aimlfw.common.constants import TARGET_KPIS
from smo.aimlfw.common.models import CellParameters, FeatureRecord, SampleCount


def record(
    *,
    target_kpi: str = "cell_goodput_mbps",
    actual: float = 100.0,
    source_dir: str = "/results/run-1",
) -> FeatureRecord:
    return FeatureRecord(
        feature_group="default",
        time_step=0,
        cell_id="gNB_5G",
        control_parameters=CellParameters(
            tx_power_dbm=40.0,
            ret_tilt_deg=5.0,
            cio_bias_db=0.0,
            hysteresis_db=2.0,
            ttt_ms=160,
        ),
        features={target_kpi: actual},
        sample_counts={target_kpi: SampleCount(valid=1, excluded=0)},
        data_quality="complete",
        source_dir=source_dir,
        seed=0,
    )


class PredictionClient:
    def __init__(
        self,
        values: dict[str, float] | None = None,
        *,
        failure: Exception | None = None,
    ) -> None:
        self.values = values or {}
        self.failure = failure
        self.calls = 0

    async def predict(self, _parameter_set: Any, _time_step: int) -> dict[str, Any]:
        self.calls += 1
        if self.failure:
            raise self.failure
        return {
            "model_name": "gnn",
            "model_version": 1,
            "target_kpi": dict(self.values),
            "cell_kpi": {"gNB_5G": dict(self.values)},
        }


class TrainingClient:
    def __init__(self, status: str = "running", candidate_version: int = 2) -> None:
        self.status = status
        self.candidate_version = candidate_version
        self.created: list[dict[str, Any]] = []

    async def create_job(
        self, feature_group: str, model_name: str, seed: int, metadata: dict[str, Any]
    ) -> str:
        self.created.append(
            {
                "feature_group": feature_group,
                "model_name": model_name,
                "seed": seed,
                "metadata": metadata,
            }
        )
        return f"job-{len(self.created)}"

    async def get_job(self, _job_id: str) -> dict[str, Any]:
        if self.status == "completed":
            return {
                "status": "completed",
                "training_summary": {"model_version": self.candidate_version},
            }
        if self.status == "failed":
            return {"status": "failed", "failure_reason": "training failed"}
        return {"status": self.status}


class Evaluator:
    def __init__(
        self,
        current: dict[str, float] | None = None,
        candidate: dict[str, float] | None = None,
    ) -> None:
        self.current = current or {name: 10.0 for name in TARGET_KPIS}
        self.candidate = candidate or {name: 9.0 for name in TARGET_KPIS}
        self.calls: list[tuple[int, list[FeatureRecord]]] = []

    async def errors(
        self, _model_name: str, version: int, records: list[FeatureRecord]
    ) -> dict[str, float]:
        self.calls.append((version, records))
        return dict(self.current if version == 1 else self.candidate)


class PromotionClient:
    def __init__(self) -> None:
        self.version = 1
        self.promoted: list[int] = []
        self.rejected: list[int] = []

    async def serving_version(self, _model_name: str) -> int:
        return self.version

    async def promote(
        self, _model_name: str, model_version: int, _comparison: dict[str, Any]
    ) -> None:
        self.version = model_version
        self.promoted.append(model_version)

    async def reject(
        self,
        _model_name: str,
        model_version: int,
        _reason: str,
        _comparison: dict[str, Any],
    ) -> None:
        self.rejected.append(model_version)


class Clock:
    def __init__(self) -> None:
        self.value = datetime(2025, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.value
