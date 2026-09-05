"""Prediction-error detection, retraining deduplication, and safe promotion."""

from __future__ import annotations

import asyncio
import math
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from smo.aimlfw.common.config import RuntimeThresholdConfig
from smo.aimlfw.common.constants import TARGET_KPIS
from smo.aimlfw.common.models import FeatureRecord, ParameterSet

MAX_DETECTION_SECONDS = 60.0
JOB_TIMEOUT = timedelta(hours=24)
MIN_TRIGGER_IMPROVEMENT_PP = 1.0


class PredictionClient(Protocol):
    async def predict(self, parameter_set: ParameterSet, time_step: int) -> dict[str, Any]: ...


class TrainingClient(Protocol):
    async def create_job(
        self,
        feature_group: str,
        model_name: str,
        seed: int,
        metadata: dict[str, Any],
    ) -> str: ...

    async def get_job(self, job_id: str) -> dict[str, Any]: ...


class ModelEvaluator(Protocol):
    async def errors(
        self,
        model_name: str,
        model_version: int,
        records: list[FeatureRecord],
    ) -> dict[str, float]: ...


class PromotionClient(Protocol):
    async def serving_version(self, model_name: str) -> int: ...

    async def promote(
        self,
        model_name: str,
        model_version: int,
        error_comparison: dict[str, dict[str, float]],
    ) -> None: ...

    async def reject(
        self,
        model_name: str,
        model_version: int,
        reason: str,
        error_comparison: dict[str, dict[str, float]],
    ) -> None: ...


@dataclass(frozen=True)
class PredictionErrorRecord:
    target_kpi: str
    prediction_error_percent: float
    model_name: str
    model_version: int
    calculated_at: datetime
    source_dir: str


@dataclass(frozen=True)
class RetrainingEvent:
    event_type: str
    occurred_at: datetime
    target_kpi: str | None = None
    job_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveRetrainingJob:
    job_id: str
    target_kpi: str
    model_name: str
    trigger_error_percent: float
    created_at: datetime
    evaluation_records: list[FeatureRecord]


def prediction_error_percent(predicted: float, actual: float) -> float:
    """Return ``abs(predicted-actual)/abs(actual)*100`` rounded to one decimal."""
    if not all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in (predicted, actual)):
        raise ValueError("predicted and actual values must be numbers")
    if not math.isfinite(predicted) or not math.isfinite(actual) or actual == 0:
        raise ValueError("predicted and actual must be finite and actual must be non-zero")
    return round(abs(predicted - actual) / abs(actual) * 100.0, 1)


class RetrainingController:
    """Poll Feature_Store records and coordinate retraining without changing serving early."""

    def __init__(
        self,
        prediction_client: PredictionClient,
        training_client: TrainingClient,
        evaluator: ModelEvaluator,
        promotion_client: PromotionClient,
        *,
        feature_group: str = "default",
        model_name: str = "gnn",
        seed: int = 0,
        retraining_threshold: Any = None,
        poll_seconds: float = MAX_DETECTION_SECONDS,
        now: Any = None,
    ) -> None:
        threshold_payload = (
            {}
            if retraining_threshold is None
            else {"retraining_threshold": retraining_threshold}
        )
        resolved = RuntimeThresholdConfig.model_validate(threshold_payload)
        self.retraining_threshold = resolved.retraining_threshold
        self.threshold_warnings = list(resolved.warnings)
        self.prediction_client = prediction_client
        self.training_client = training_client
        self.evaluator = evaluator
        self.promotion_client = promotion_client
        self.feature_group = feature_group
        self.model_name = model_name
        self.seed = seed
        self.poll_seconds = max(0.1, min(float(poll_seconds), MAX_DETECTION_SECONDS))
        self._now = now or (lambda: datetime.now(UTC))
        self.processed_source_dirs: set[str] = set()
        self.active_jobs: dict[str, ActiveRetrainingJob] = {}
        self.prediction_errors: list[PredictionErrorRecord] = []
        self.events: list[RetrainingEvent] = []
        for warning in self.threshold_warnings:
            self._event("threshold_defaulted", details={"warning": warning})

    def _event(
        self,
        event_type: str,
        *,
        target_kpi: str | None = None,
        job_id: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        self.events.append(
            RetrainingEvent(event_type, self._now(), target_kpi, job_id, details or {})
        )

    async def process_result_records(self, source_dir: str, records: list[FeatureRecord]) -> None:
        """Evaluate one newly observed result directory and create per-KPI jobs."""
        usable = [record for record in records if record.data_quality != "insufficient"]
        if not usable:
            self._event("result_skipped", details={"source_dir": source_dir, "reason": "no_usable_records"})
            self.processed_source_dirs.add(source_dir)
            return
        trigger_errors: dict[str, float] = {}
        trigger_versions: dict[str, tuple[str, int]] = {}
        for record in usable:
            parameter_set = ParameterSet(cells={record.cell_id: record.control_parameters})
            try:
                prediction = await self.prediction_client.predict(parameter_set, record.time_step)
            except Exception as exc:  # noqa: BLE001 - prediction failure is localized and recorded.
                self._event(
                    "prediction_skipped",
                    details={"source_dir": source_dir, "cell_id": record.cell_id, "reason": str(exc)},
                )
                continue
            model_name = str(prediction["model_name"])
            model_version = int(prediction["model_version"])
            cell_values = prediction.get("cell_kpi", {}).get(record.cell_id, {})
            target_values = prediction.get("target_kpi", {})
            for target_kpi in TARGET_KPIS:
                actual = record.features.get(target_kpi)
                predicted = cell_values.get(target_kpi, target_values.get(target_kpi))
                if actual is None or predicted is None or actual == 0:
                    self._event(
                        "prediction_error_skipped",
                        target_kpi=target_kpi,
                        details={"source_dir": source_dir, "reason": "actual_or_prediction_missing_or_zero"},
                    )
                    continue
                error = prediction_error_percent(float(predicted), float(actual))
                error_record = PredictionErrorRecord(
                    target_kpi,
                    error,
                    model_name,
                    model_version,
                    self._now(),
                    source_dir,
                )
                self.prediction_errors.append(error_record)
                if error > self.retraining_threshold and error >= trigger_errors.get(target_kpi, -1.0):
                    trigger_errors[target_kpi] = error
                    trigger_versions[target_kpi] = (model_name, model_version)

        for target_kpi, error in trigger_errors.items():
            if target_kpi in self.active_jobs:
                self._event(
                    "duplicate_trigger_suppressed",
                    target_kpi=target_kpi,
                    job_id=self.active_jobs[target_kpi].job_id,
                    details={"prediction_error_percent": error},
                )
                continue
            used_model_name, used_version = trigger_versions[target_kpi]
            metadata = {
                "target_kpi": target_kpi,
                "trigger_prediction_error_percent": error,
                "trigger_model_name": used_model_name,
                "trigger_model_version": used_version,
                "triggered_at": self._now().isoformat(),
                "source_dir": source_dir,
            }
            try:
                job_id = await self.training_client.create_job(
                    self.feature_group,
                    self.model_name,
                    self.seed,
                    metadata,
                )
            except Exception as exc:  # noqa: BLE001
                self._event(
                    "retraining_trigger_failed",
                    target_kpi=target_kpi,
                    details={"reason": str(exc), **metadata},
                )
                continue
            self.active_jobs[target_kpi] = ActiveRetrainingJob(
                job_id,
                target_kpi,
                self.model_name,
                error,
                self._now(),
                usable,
            )
            self._event("retraining_triggered", target_kpi=target_kpi, job_id=job_id, details=metadata)
        self.processed_source_dirs.add(source_dir)

    async def scan_once(self, records: list[FeatureRecord]) -> None:
        """Process every newly seen ``source_dir`` exactly once."""
        grouped: dict[str, list[FeatureRecord]] = {}
        for record in records:
            if record.source_dir not in self.processed_source_dirs:
                grouped.setdefault(record.source_dir, []).append(record)
        for source_dir in sorted(grouped):
            await self.process_result_records(source_dir, grouped[source_dir])
        await self.check_jobs()

    async def check_jobs(self) -> None:
        """Finalize failed/timed-out jobs or compare and promote completed candidates."""
        for target_kpi, active in list(self.active_jobs.items()):
            if self._now() - active.created_at > JOB_TIMEOUT:
                self._event(
                    "retraining_failed",
                    target_kpi=target_kpi,
                    job_id=active.job_id,
                    details={"reason": "job_timeout_24h"},
                )
                del self.active_jobs[target_kpi]
                continue
            try:
                job = await self.training_client.get_job(active.job_id)
            except Exception as exc:  # noqa: BLE001
                self._event(
                    "job_status_unavailable",
                    target_kpi=target_kpi,
                    job_id=active.job_id,
                    details={"reason": str(exc)},
                )
                continue
            status = job.get("status")
            if status in {"pending", "running"}:
                continue
            if status != "completed":
                self._event(
                    "retraining_failed",
                    target_kpi=target_kpi,
                    job_id=active.job_id,
                    details={"reason": job.get("failure_reason") or f"job_status={status}"},
                )
                del self.active_jobs[target_kpi]
                continue
            summary = job.get("training_summary") or {}
            candidate_version = summary.get("model_version")
            if not isinstance(candidate_version, int):
                self._event(
                    "retraining_failed",
                    target_kpi=target_kpi,
                    job_id=active.job_id,
                    details={"reason": "completed_job_has_no_model_version"},
                )
                del self.active_jobs[target_kpi]
                continue
            try:
                current_version = await self.promotion_client.serving_version(active.model_name)
                current_errors, candidate_errors = await asyncio.gather(
                    self.evaluator.errors(active.model_name, current_version, active.evaluation_records),
                    self.evaluator.errors(active.model_name, candidate_version, active.evaluation_records),
                )
            except Exception as exc:  # noqa: BLE001
                self._event(
                    "promotion_evaluation_failed",
                    target_kpi=target_kpi,
                    job_id=active.job_id,
                    details={"reason": str(exc), "candidate_version": candidate_version},
                )
                del self.active_jobs[target_kpi]
                continue
            comparison = {
                kpi: {"current": current_errors[kpi], "candidate": candidate_errors[kpi]}
                for kpi in TARGET_KPIS
                if kpi in current_errors and kpi in candidate_errors
            }
            all_not_worse = len(comparison) == len(TARGET_KPIS) and all(
                values["candidate"] <= values["current"] for values in comparison.values()
            )
            trigger_improvement = (
                current_errors.get(target_kpi, -math.inf) - candidate_errors.get(target_kpi, math.inf)
            )
            if all_not_worse and trigger_improvement >= MIN_TRIGGER_IMPROVEMENT_PP:
                try:
                    await self.promotion_client.promote(active.model_name, candidate_version, comparison)
                except Exception as exc:  # noqa: BLE001 - serving client compensates before raising.
                    self._event(
                        "promotion_failed",
                        target_kpi=target_kpi,
                        job_id=active.job_id,
                        details={"reason": str(exc), "candidate_version": candidate_version},
                    )
                    del self.active_jobs[target_kpi]
                    continue
                self._event(
                    "model_promoted",
                    target_kpi=target_kpi,
                    job_id=active.job_id,
                    details={"model_version": candidate_version, "error_comparison": comparison},
                )
            else:
                reason = (
                    "candidate_has_worse_target_kpi"
                    if not all_not_worse
                    else "trigger_kpi_improvement_below_1pp"
                )
                try:
                    await self.promotion_client.reject(
                        active.model_name,
                        candidate_version,
                        reason,
                        comparison,
                    )
                except Exception as exc:  # noqa: BLE001 - rejection never changes serving state.
                    self._event(
                        "rejection_record_failed",
                        target_kpi=target_kpi,
                        job_id=active.job_id,
                        details={"reason": str(exc), "candidate_version": candidate_version},
                    )
                    del self.active_jobs[target_kpi]
                    continue
                self._event(
                    "model_rejected",
                    target_kpi=target_kpi,
                    job_id=active.job_id,
                    details={"reason": reason, "error_comparison": comparison},
                )
            del self.active_jobs[target_kpi]

    async def run(self, records_provider: Any, stop_event: asyncio.Event) -> None:
        """Poll at most every 60 seconds until ``stop_event`` is set."""
        while not stop_event.is_set():
            try:
                records = await records_provider()
                await self.scan_once(records)
            except Exception as exc:  # noqa: BLE001 - one poll failure must not stop future detection.
                self._event("poll_failed", details={"reason": str(exc)})
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=self.poll_seconds)
            except TimeoutError:
                pass


__all__ = [
    "ActiveRetrainingJob",
    "PredictionErrorRecord",
    "RetrainingController",
    "RetrainingEvent",
    "prediction_error_percent",
]
