"""Property 15 test module (own module to avoid collisions with parallel Property tasks).

Task 3.4: 학습 파이프라인의 4단계(``extract_features`` -> ``train_model`` ->
``save_artifact`` -> ``register_metrics``) 중 임의의 한 단계가 실패하도록 구성된
경우, Training_Manager가 해당 단계까지만 실행하고 후속 단계를 실행하지 않으며,
각 실행된 단계의 시작/종료 시각과 결과(``succeeded``/``failed``)를 기록하는지
검증한다.

Targets ``smo.aimlfw.training_manager.jobs.JobManager._run_pipeline`` (Task 3.3),
as specified in requirements.md:

Requirement 3.2:
    THE Training_Manager SHALL 학습 파이프라인을 피처 추출 단계, 모델 학습 단계,
    모델 저장 단계, 모델 지표 저장 단계의 순서로 실행하고 각 단계의 시작 시각,
    종료 시각, 단계 결과(``succeeded`` 또는 ``failed``)를 작업 메타데이터에
    기록하며, 어느 단계가 ``failed`` 이면 후속 단계를 실행하지 않는다.

design.md Property 15: 파이프라인 단계 순서와 조기 중단
    *For any* 학습 파이프라인의 4단계(``extract_features``, ``train_model``,
    ``save_artifact``, ``register_metrics``) 중 임의의 한 단계가 실패하도록
    구성된 경우, Training_Manager는 해당 단계까지만 실행하고 후속 단계를 실행하지
    않으며, 각 실행된 단계의 시작/종료 시각과 결과를 기록한다.
    **Validates: Requirements 3.2**

Note on ``register_metrics``: per jobs.py's own module docstring, a failure in the
*last* stage (after its internal retries are exhausted) is a deliberate exception to
"stage failure means job failure" -- the job stays ``completed`` with the failure
reason recorded, rather than ``failed``. That exception does not affect this
property: ``register_metrics`` is already the last stage, so "no subsequent stage
runs" holds trivially, and it is still true that its start/end/result are recorded
and that stages 1..3 ran to completion beforehand in order.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.training_manager.jobs import (
    JobManager,
    _JobState,
    _MetricsRegistrationFailed,
)
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

_STAGE_ORDER = ("extract_features", "train_model", "save_artifact", "register_metrics")
_FAILURE_TYPE_BY_STAGE = {
    "extract_features": "data_error",
    "train_model": "training_error",
    "save_artifact": "storage_error",
}

# Vary which stage fails, including "no stage fails" (successful pipeline run).
_FAILING_STAGE = st.one_of(st.none(), st.sampled_from(_STAGE_ORDER))

_METRICS_PAYLOAD: dict[str, Any] = {
    "metrics": {"delay_p95_ms": 1.0},
    "train_split": 0.8,
    "validation_split": 0.2,
    "train_samples": 200,
    "validation_samples": 50,
    "seed": 1,
}


def _install_stage_fakes(manager: JobManager, failing_stage: str | None) -> None:
    """Replace the four pipeline stage hooks with fakes that succeed unless their
    own name matches ``failing_stage``, in which case they raise."""

    async def fake_extract_features(_job_id: str) -> list[dict[str, Any]]:
        if failing_stage == "extract_features":
            raise ValueError("boom-extract_features")
        return []

    async def fake_train_model(_job_id: str, _records: list[dict[str, Any]], work_dir: Path) -> dict[str, Any]:
        if failing_stage == "train_model":
            raise ValueError("boom-train_model")
        return {"artifact_path": work_dir / "model.pt", "metrics_payload": dict(_METRICS_PAYLOAD)}

    async def fake_save_artifact(_job_id: str, _train_result: dict[str, Any]) -> dict[str, Any]:
        if failing_stage == "save_artifact":
            raise ValueError("boom-save_artifact")
        return {"artifact_uri": "models/model/1/model.pt", "version": 1}

    async def fake_register_metrics_with_retry(
        _job_id: str, _metrics_payload: dict[str, Any], _save_result: dict[str, Any]
    ) -> None:
        if failing_stage == "register_metrics":
            raise _MetricsRegistrationFailed("boom-register_metrics")

    manager._extract_features = fake_extract_features  # type: ignore[assignment]
    manager._train_model = fake_train_model  # type: ignore[assignment]
    manager._save_artifact = fake_save_artifact  # type: ignore[assignment]
    manager._register_metrics_with_retry = fake_register_metrics_with_retry  # type: ignore[assignment]


# **Property 15: 파이프라인 단계 순서와 조기 중단**
# **Validates: Requirements 3.2**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(failing_stage=_FAILING_STAGE)
def test_pipeline_stage_order_and_early_abort(failing_stage: str | None) -> None:
    with TemporaryDirectory() as raw_root:
        manager = JobManager(
            feature_store=None,  # type: ignore[arg-type]
            model_registry=None,  # type: ignore[arg-type]
            model_storage=None,  # type: ignore[arg-type]
            jobs_dir=Path(raw_root) / "jobs",
        )
        _install_stage_fakes(manager, failing_stage)

        job_id = "job-property-15"
        state = _JobState(job_id=job_id, feature_group="fg", model_name="model", seed=1)
        manager._jobs[job_id] = state

        asyncio.run(manager._run_pipeline(job_id))

        job = manager.get_job(job_id)

        if failing_stage is None:
            expected_ran_stages = _STAGE_ORDER
        else:
            failure_index = _STAGE_ORDER.index(failing_stage)
            expected_ran_stages = _STAGE_ORDER[: failure_index + 1]

        # (a) Stages execute in the fixed order, and (b) no stage runs after the
        # first failure: the recorded stage_history is exactly the prefix of the
        # canonical order up to (and including, if present) the failing stage.
        assert [record.stage for record in job.stage_history] == list(expected_ran_stages)

        for record in job.stage_history[:-1] if failing_stage is not None else job.stage_history:
            assert record.result == "succeeded"
            assert record.reason is None
            assert record.ended_at is not None
            assert record.ended_at >= record.started_at

        last_record = job.stage_history[-1]
        assert last_record.ended_at is not None
        assert last_record.ended_at >= last_record.started_at

        if failing_stage is None:
            assert last_record.result == "succeeded"
            assert job.status == "completed"
            assert job.failure_type is None
            assert job.metrics_registration_failure_reason is None
        elif failing_stage == "register_metrics":
            # Deliberate exception: the last stage's own failure (post-retry) keeps
            # the job "completed" rather than "failed", but it is still recorded as
            # a failed stage and no stage runs after it (there is none).
            assert last_record.result == "failed"
            assert last_record.reason == "boom-register_metrics"
            assert job.status == "completed"
            assert job.failure_type is None
            assert job.metrics_registration_failure_reason == "boom-register_metrics"
        else:
            assert last_record.result == "failed"
            assert last_record.reason == f"boom-{failing_stage}"
            assert job.status == "failed"
            assert job.failure_type == _FAILURE_TYPE_BY_STAGE[failing_stage]
            assert job.failure_reason == f"boom-{failing_stage}"
            assert job.current_stage is None
