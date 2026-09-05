"""Property 19 test module (own module to avoid collisions with parallel Property tasks).

Task 3.8: 지표 기록 재시도와 최종 상태.

Targets the Training_Manager pipeline's ``register_metrics`` stage, as
specified in requirements.md 3.7 and design.md's Property 19, implemented on
``smo.aimlfw.training_manager.jobs.JobManager._register_metrics_with_retry``
(task 3.3):

    IF 학습은 성공했으나 Model_Registry 지표 기록이 실패하면, THEN THE
    Training_Manager SHALL 최대 3회까지 기록을 재시도하고 3회 모두 실패한
    경우 해당 작업의 상태를 `completed` 로 유지하며 지표 기록 실패 사유를
    작업 메타데이터에 기록한다.

Design tag (design.md, "Training_Manager (Requirement 3)"):

    #### Property 19: 지표 기록 재시도와 최종 상태
    *For any* Model_Registry 지표 기록이 최대 3회까지 실패하도록 구성된
    경우, Training_Manager는 정확히 3회까지 재시도하고 모두 실패하면 작업
    상태를 `completed`로 유지하며 실패 사유를 기록한다.
    **Validates: Requirements 3.7**

Oracle grounding: ``jobs.py``'s ``_register_metrics_with_retry`` loops up to
``metrics_registration_max_attempts`` (default 3, Requirement 3.7) times,
calling ``model_registry.register(...)`` on a worker thread and returning as
soon as one attempt does not raise; if every attempt raises it wraps the last
exception's message in ``_MetricsRegistrationFailed``. ``_run_pipeline``
catches that exception, records the failure reason as
``metrics_registration_failure_reason`` and finalizes the job as
``completed`` (the one deliberate exception to "stage failure means job
failure" documented at the top of ``jobs.py``) rather than ``failed``.

This test drives the real ``JobManager._run_pipeline`` state machine. The
three earlier stages (``extract_features``, ``train_model``,
``save_artifact``) are replaced with fast fakes that always succeed and
produce a valid completion-metadata payload (Requirement 3.4 fields), since
they are orthogonal to this property. ``model_registry`` is replaced with a
call-counting fake whose ``register`` either raises on every call or raises
on every call up to (and not including) a Hypothesis-chosen attempt number,
then succeeds -- this directly controls how many of the
``metrics_registration_max_attempts`` retries are consumed without needing a
real MongoDB-backed ``ModelRegistry``.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.training_manager.jobs import JobManager, _JobState
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

_MODEL_NAME = "ran-gnn-property-19"
_FEATURE_GROUP = "default"


class _RetryCountingRegistry:
    """Call-counting stand-in for ``ModelRegistry`` that only implements ``register``.

    ``succeed_at`` is the 1-based attempt number on which ``register``
    should stop raising; ``None`` means every attempt raises (all retries
    exhausted).
    """

    def __init__(self, succeed_at: int | None, failure_reason: str) -> None:
        self.succeed_at = succeed_at
        self.failure_reason = failure_reason
        self.call_count = 0

    def register(self, *args: Any, **kwargs: Any) -> None:
        self.call_count += 1
        if self.succeed_at is not None and self.call_count >= self.succeed_at:
            return None
        raise RuntimeError(self.failure_reason)


def _install_trivial_earlier_stage_fakes(manager: JobManager, metrics_payload: dict[str, Any]) -> None:
    """Make extract_features/train_model/save_artifact succeed trivially so the
    pipeline reaches ``register_metrics`` deterministically and fast."""

    async def fake_extract_features(_job_id: str) -> list[dict[str, Any]]:
        return []

    async def fake_train_model(_job_id: str, _records: list[dict[str, Any]], work_dir: Path) -> dict[str, Any]:
        return {"artifact_path": work_dir / "model.pt", "metrics_payload": metrics_payload}

    async def fake_save_artifact(_job_id: str, _train_result: dict[str, Any]) -> dict[str, Any]:
        return {"artifact_uri": "unused-artifact-uri", "version": 1}

    manager._extract_features = fake_extract_features  # type: ignore[assignment]
    manager._train_model = fake_train_model  # type: ignore[assignment]
    manager._save_artifact = fake_save_artifact  # type: ignore[assignment]


_METRICS_PAYLOAD: dict[str, Any] = {
    "train_split": 0.8,
    "validation_split": 0.2,
    "train_samples": 200,
    "validation_samples": 50,
    "seed": 7,
    "metrics": {"cell_goodput_mbps": 3.21},
}

# ``succeed_at``: None exercises "fails on every one of the max attempts";
# an integer 1..3 exercises "succeeds on attempt N and stops retrying".
_RETRY_CASE = st.tuples(
    st.one_of(st.none(), st.integers(min_value=1, max_value=3)),
    st.text(min_size=1, max_size=64).filter(lambda text: text.strip() != ""),
)


# **Property 19: 지표 기록 재시도와 최종 상태**
# **Validates: Requirements 3.7**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(case=_RETRY_CASE)
def test_metrics_registration_retries_and_final_job_status(case: tuple[int | None, str]) -> None:
    succeed_at, reason = case

    with TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        registry = _RetryCountingRegistry(succeed_at, reason)
        manager = JobManager(
            feature_store=None,  # type: ignore[arg-type]
            model_registry=registry,  # type: ignore[arg-type]
            model_storage=None,  # type: ignore[arg-type]
            jobs_dir=root / "jobs",
        )
        max_attempts = manager.metrics_registration_max_attempts
        assert max_attempts == 3  # Requirement 3.7 fixes the retry ceiling at 3.
        _install_trivial_earlier_stage_fakes(manager, _METRICS_PAYLOAD)

        job_id = "job-property-19"
        state = _JobState(job_id=job_id, feature_group=_FEATURE_GROUP, model_name=_MODEL_NAME, seed=1)
        manager._jobs[job_id] = state

        asyncio.run(manager._run_pipeline(job_id))

        job = manager.get_job(job_id)

        # Requirement 3.7: the job's status must remain "completed" regardless
        # of whether metrics registration ultimately succeeded or exhausted
        # all retries -- never "failed".
        assert job.status == "completed"
        assert job.current_stage is None

        register_stage = job.stage_history[-1]
        assert register_stage.stage == "register_metrics"

        if succeed_at is None:
            # Every one of the up-to-3 attempts failed: register() must have
            # been called exactly metrics_registration_max_attempts times,
            # and the failure reason must be recorded on the job.
            assert registry.call_count == max_attempts
            assert job.metrics_registration_failure_reason == reason
            assert register_stage.result == "failed"
            assert register_stage.reason == reason
        else:
            # register() succeeded on attempt `succeed_at` (<= 3): retrying
            # must stop immediately, so call_count == succeed_at exactly, and
            # no failure reason is recorded.
            assert registry.call_count == succeed_at
            assert job.metrics_registration_failure_reason is None
            assert register_stage.result == "succeeded"
            assert register_stage.reason is None

        # The training summary (Requirement 3.4 completion metadata) is
        # retained on the job regardless of registration outcome.
        assert manager._jobs[job_id].training_summary is not None
        assert manager._jobs[job_id].training_summary["seed"] == _METRICS_PAYLOAD["seed"]
