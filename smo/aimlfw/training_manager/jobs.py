"""Training_Manager job orchestration: the 4-stage GNN training pipeline.

Implements design.md's Training_Manager component (Requirement 3) as an
in-process asyncio job runner:

    extract_features -> train_model -> save_artifact -> register_metrics

Each stage's start/end time and result (``succeeded``/``failed``) is
recorded on the job (Requirement 3.2). A failing stage aborts all
subsequent stages (Property 15) except ``register_metrics``, whose failure
after 3 retries (Requirement 3.7, Property 19) leaves the job ``completed``
with the failure reason recorded rather than marking the whole job
``failed`` — that is the one deliberate exception to "stage failure means
job failure" the requirements carve out.

``TrainingJob`` state is kept in memory and mirrored to a JSON snapshot
file per job (design.md storage mapping: "프로세스 내 메모리 +
training_manager/jobs/<job_id>.json 스냅샷") so status survives a process
restart (Requirement 3.3 fast status reads come from memory; the snapshot
is the durability fallback).
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import sys
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from smo.aimlfw.common.models import StageRecord, TrainingJob
from smo.aimlfw.feature_store import FeatureStore, FeatureStoreError
from smo.aimlfw.model_registry import ModelRegistry, ModelRegistryError
from smo.aimlfw.model_storage import ModelStorage

from .errors import TrainingManagerError

DEFAULT_MIN_TRAINING_SAMPLES = 200
DEFAULT_MAX_TRAINING_SECONDS = 3600.0
DEFAULT_METRICS_REGISTRATION_ATTEMPTS = 3
DEFAULT_TRAIN_SPLIT = 0.8
DEFAULT_MAX_EPOCHS = 50
DEFAULT_PATIENCE = 8
DEFAULT_LEARNING_RATE = 0.05

_STAGE_ORDER = ("extract_features", "train_model", "save_artifact", "register_metrics")
_TRAIN_GNN_SCRIPT = Path(__file__).resolve().parent / "train_gnn.py"


class _PipelineAborted(Exception):
    """Raised internally once a stage has already finalized the job as failed."""


class _MetricsRegistrationFailed(Exception):
    def __init__(self, reason: str):
        self.reason = reason
        super().__init__(reason)


@dataclass
class _JobState:
    """Mutable working state for one training job; ``snapshot()`` is the API view."""

    job_id: str
    feature_group: str
    model_name: str
    seed: int
    metadata: dict[str, Any] = field(default_factory=dict)
    status: str = "pending"
    current_stage: str | None = None
    stage_history: list[StageRecord] = field(default_factory=list)
    failure_type: str | None = None
    failure_reason: str | None = None
    metrics_registration_failure_reason: str | None = None
    training_summary: dict[str, Any] | None = None

    def snapshot(self) -> TrainingJob:
        return TrainingJob(
            job_id=self.job_id,
            feature_group=self.feature_group,
            model_name=self.model_name,
            seed=self.seed,
            status=self.status,  # type: ignore[arg-type]
            current_stage=self.current_stage,
            stage_history=list(self.stage_history),
            failure_type=self.failure_type,  # type: ignore[arg-type]
            failure_reason=self.failure_reason,
            metrics_registration_failure_reason=self.metrics_registration_failure_reason,
            metadata=dict(self.metadata),
            training_summary=None if self.training_summary is None else dict(self.training_summary),
        )


class JobManager:
    """Create, run, and report on GNN training jobs (design.md Training_Manager)."""

    def __init__(
        self,
        feature_store: FeatureStore,
        model_registry: ModelRegistry,
        model_storage: ModelStorage,
        *,
        jobs_dir: Path | str,
        min_training_samples: int = DEFAULT_MIN_TRAINING_SAMPLES,
        max_training_seconds: float = DEFAULT_MAX_TRAINING_SECONDS,
        metrics_registration_max_attempts: int = DEFAULT_METRICS_REGISTRATION_ATTEMPTS,
        train_split: float = DEFAULT_TRAIN_SPLIT,
        max_epochs: int = DEFAULT_MAX_EPOCHS,
        patience: int = DEFAULT_PATIENCE,
        learning_rate: float = DEFAULT_LEARNING_RATE,
        train_script: Path | str = _TRAIN_GNN_SCRIPT,
        python_executable: str = sys.executable,
    ) -> None:
        self.feature_store = feature_store
        self.model_registry = model_registry
        self.model_storage = model_storage
        self.jobs_dir = Path(jobs_dir)
        self.min_training_samples = min_training_samples
        self.max_training_seconds = max_training_seconds
        self.metrics_registration_max_attempts = metrics_registration_max_attempts
        self.train_split = train_split
        self.max_epochs = max_epochs
        self.patience = patience
        self.learning_rate = learning_rate
        self.train_script = Path(train_script)
        self.python_executable = python_executable
        self._jobs: dict[str, _JobState] = {}
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    # ---- job creation and lookup -------------------------------------------------

    def _count_valid_records(self, feature_group: str) -> int:
        try:
            records = self.feature_store.records(feature_group)
        except FeatureStoreError as exc:
            raise TrainingManagerError(
                "insufficient_training_data",
                "Feature_Group is unavailable in the Feature_Store",
                {
                    "feature_group": feature_group,
                    "sample_count": 0,
                    "minimum_required": self.min_training_samples,
                    "reason": exc.error_code,
                },
            ) from exc
        return sum(1 for record in records if record.data_quality != "insufficient")

    async def create_job(
        self,
        feature_group: str,
        model_name: str,
        seed: int,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Validate data sufficiency, register a ``pending`` job, and start it (3.1, 3.8)."""
        sample_count = self._count_valid_records(feature_group)
        if sample_count < self.min_training_samples:
            raise TrainingManagerError(
                "insufficient_training_data",
                "Feature_Group does not have enough valid Feature_Records to train",
                {
                    "feature_group": feature_group,
                    "sample_count": sample_count,
                    "minimum_required": self.min_training_samples,
                },
            )

        job_id = self._new_job_id()
        state = _JobState(
            job_id=job_id,
            feature_group=feature_group,
            model_name=model_name,
            seed=seed,
            metadata=dict(metadata or {}),
        )
        self._jobs[job_id] = state
        self._persist(state)
        asyncio.create_task(self._run_pipeline(job_id))
        return job_id

    def _new_job_id(self) -> str:
        job_id = f"job-{uuid.uuid4().hex}"
        while job_id in self._jobs or (self.jobs_dir / f"{job_id}.json").exists():
            job_id = f"job-{uuid.uuid4().hex}"
        return job_id

    def get_job(self, job_id: str) -> TrainingJob:
        """Return the current status snapshot (3.3), reloading from disk if needed."""
        state = self._jobs.get(job_id)
        if state is not None:
            return state.snapshot()
        snapshot_path = self.jobs_dir / f"{job_id}.json"
        if snapshot_path.is_file():
            return TrainingJob.model_validate_json(snapshot_path.read_text(encoding="utf-8"))
        raise TrainingManagerError("job_not_found", "No training job exists for the given id", {"job_id": job_id})

    # ---- persistence --------------------------------------------------------------

    def _persist(self, state: _JobState) -> None:
        path = self.jobs_dir / f"{state.job_id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(state.snapshot().model_dump_json(), encoding="utf-8")
        temporary.replace(path)

    # ---- stage bookkeeping ---------------------------------------------------------

    def _begin_stage(self, job_id: str, stage: str) -> None:
        state = self._jobs[job_id]
        state.status = "running"
        state.current_stage = stage
        state.stage_history.append(StageRecord(stage=stage, started_at=datetime.now(UTC)))  # type: ignore[arg-type]
        self._persist(state)

    def _end_stage(self, job_id: str, *, result: str, reason: str | None = None) -> None:
        state = self._jobs[job_id]
        last = state.stage_history[-1]
        state.stage_history[-1] = last.model_copy(
            update={"ended_at": datetime.now(UTC), "result": result, "reason": reason}
        )
        self._persist(state)

    def _finalize_failure(self, job_id: str, failure_type: str, reason: str) -> None:
        state = self._jobs[job_id]
        state.status = "failed"
        state.current_stage = None
        state.failure_type = failure_type
        state.failure_reason = reason
        self._persist(state)

    def _finalize_success(self, job_id: str, *, metrics_registration_failure_reason: str | None = None) -> None:
        state = self._jobs[job_id]
        state.status = "completed"
        state.current_stage = None
        state.metrics_registration_failure_reason = metrics_registration_failure_reason
        self._persist(state)

    def _set_training_summary(self, job_id: str, summary: dict[str, Any]) -> None:
        state = self._jobs[job_id]
        state.training_summary = summary
        self._persist(state)

    async def _run_stage(self, job_id: str, stage: str, work: Any, failure_type: str) -> Any:
        self._begin_stage(job_id, stage)
        try:
            result = await work()
        except Exception as exc:  # noqa: BLE001 - stage boundary: classify and record
            self._end_stage(job_id, result="failed", reason=str(exc))
            self._finalize_failure(job_id, failure_type, str(exc))
            raise _PipelineAborted from exc
        self._end_stage(job_id, result="succeeded")
        return result

    # ---- pipeline stages ------------------------------------------------------------

    async def _extract_features(self, job_id: str) -> list[dict[str, Any]]:
        state = self._jobs[job_id]
        records = self.feature_store.records(state.feature_group)
        valid = [record for record in records if record.data_quality != "insufficient"]
        if not valid:
            raise ValueError(f"no valid Feature_Records remain for feature_group={state.feature_group!r}")
        return [record.model_dump(mode="json") for record in valid]

    async def _train_model(self, job_id: str, records: list[dict[str, Any]], work_dir: Path) -> dict[str, Any]:
        state = self._jobs[job_id]
        records_path = work_dir / "records.json"
        artifact_path = work_dir / "model.pt"
        metrics_path = work_dir / "metrics.json"
        records_path.write_text(json.dumps(records), encoding="utf-8")

        repository_root = Path(__file__).resolve().parents[3]
        env = {**os.environ, "PYTHONPATH": str(repository_root)}
        process = await asyncio.create_subprocess_exec(
            self.python_executable,
            str(self.train_script),
            "--records-file", str(records_path),
            "--artifact-output", str(artifact_path),
            "--metrics-output", str(metrics_path),
            "--seed", str(state.seed),
            "--train-split", str(self.train_split),
            "--max-epochs", str(self.max_epochs),
            "--patience", str(self.patience),
            "--learning-rate", str(self.learning_rate),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            _, stderr = await asyncio.wait_for(process.communicate(), timeout=self.max_training_seconds)
        except TimeoutError:
            process.kill()
            await process.wait()
            raise TimeoutError(
                f"training exceeded max_training_seconds={self.max_training_seconds}"
            ) from None

        if process.returncode != 0:
            reason = stderr.decode(errors="replace")[:2000]
            raise RuntimeError(f"train_gnn.py exited with code {process.returncode}: {reason}")

        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        return {"artifact_path": artifact_path, "metrics_payload": metrics_payload}

    async def _next_version(self, model_name: str) -> int:
        try:
            latest = await asyncio.to_thread(self.model_registry.latest, model_name)
        except ModelRegistryError as exc:
            if exc.error_code == "model_not_found":
                return 1
            raise
        return latest.version + 1

    async def _save_artifact(self, job_id: str, train_result: dict[str, Any]) -> dict[str, Any]:
        state = self._jobs[job_id]
        version = await self._next_version(state.model_name)
        artifact_uri = await asyncio.to_thread(
            self.model_storage.save_artifact, state.model_name, version, train_result["artifact_path"]
        )
        return {"artifact_uri": artifact_uri, "version": version}

    async def _register_metrics_with_retry(
        self, job_id: str, metrics_payload: dict[str, Any], save_result: dict[str, Any]
    ) -> None:
        """Retry Model_Registry.register up to ``metrics_registration_max_attempts`` times (3.7)."""
        state = self._jobs[job_id]
        metrics = metrics_payload["metrics"]
        last_reason: str | None = None
        for _ in range(self.metrics_registration_max_attempts):
            try:
                await asyncio.to_thread(
                    self.model_registry.register,
                    state.model_name,
                    state.feature_group,
                    metrics,
                    save_result["artifact_uri"],
                    metrics_payload["train_split"],
                    metrics_payload["validation_split"],
                    metrics_payload["train_samples"],
                    metrics_payload["validation_samples"],
                    metrics_payload["seed"],
                )
                return
            except Exception as exc:  # noqa: BLE001 - retry boundary: classify and record
                last_reason = str(exc)
        raise _MetricsRegistrationFailed(last_reason or "unknown metrics registration failure")

    async def _run_pipeline(self, job_id: str) -> None:
        work_dir = self.jobs_dir / "work" / job_id
        work_dir.mkdir(parents=True, exist_ok=True)
        try:
            records = await self._run_stage(
                job_id,
                "extract_features",
                lambda: self._extract_features(job_id),
                "data_error",
            )
            train_result = await self._run_stage(
                job_id, "train_model", lambda: self._train_model(job_id, records, work_dir), "training_error"
            )
            save_result = await self._run_stage(
                job_id, "save_artifact", lambda: self._save_artifact(job_id, train_result), "storage_error"
            )
        except _PipelineAborted:
            return
        finally:
            pass

        metrics_payload = train_result["metrics_payload"]
        self._begin_stage(job_id, "register_metrics")
        try:
            await self._register_metrics_with_retry(job_id, metrics_payload, save_result)
        except _MetricsRegistrationFailed as failure:
            self._end_stage(job_id, result="failed", reason=failure.reason)
            self._set_training_summary(job_id, self._training_summary(metrics_payload, save_result))
            self._finalize_success(job_id, metrics_registration_failure_reason=failure.reason)
        else:
            self._end_stage(job_id, result="succeeded")
            self._set_training_summary(job_id, self._training_summary(metrics_payload, save_result))
            self._finalize_success(job_id)
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)

    @staticmethod
    def _training_summary(metrics_payload: dict[str, Any], save_result: dict[str, Any]) -> dict[str, Any]:
        # Requirement 3.4: the training data split ratio, sample counts, and seed
        # are forwarded to Model_Registry.register (see
        # _register_metrics_with_retry) and persisted on ModelVersionRecord. This
        # summary is additionally kept on the TrainingJob itself so it stays
        # retrievable via GET /jobs/{id} even if metrics registration fails after
        # all retries.
        return {
            "train_split": metrics_payload["train_split"],
            "validation_split": metrics_payload["validation_split"],
            "train_samples": metrics_payload["train_samples"],
            "validation_samples": metrics_payload["validation_samples"],
            "seed": metrics_payload["seed"],
            "metrics": metrics_payload["metrics"],
            "model_version": save_result["version"],
            "artifact_uri": save_result["artifact_uri"],
        }


__all__ = ["JobManager"]
