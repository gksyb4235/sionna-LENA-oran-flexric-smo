"""Property 21 test module (own module to avoid collisions with parallel Property tasks).

Task 3.10: 최대 실행 시간 초과 처리.

Targets ``smo.aimlfw.training_manager.jobs.JobManager._train_model`` (Task 3.3),
as specified in requirements.md:

Requirement 3.9:
    IF 하나의 학습 작업의 실행 시간이 최대 허용 실행 시간(기본값 3600초)을
    초과하면, THEN THE Training_Manager SHALL 해당 작업을 중단하고 상태를
    `failed` 로 설정하며 시간 초과를 실패 사유로 기록한다.

Design tag (design.md "Training_Manager (Requirement 3)"):

    #### Property 21: 최대 실행 시간 초과 처리
    *For any* 학습 작업의 실행 시간이 설정된 `max_training_seconds` 를 초과하면,
    Training_Manager는 해당 작업을 중단하고 상태를 `failed` 로 설정하며 시간
    초과를 실패 사유로 기록한다.
    **Validates: Requirements 3.9**

Oracle grounding: ``jobs.py``'s ``_train_model`` wraps
``process.communicate()`` in ``asyncio.wait_for(..., timeout=self.max_training_seconds)``.
On ``asyncio.TimeoutError`` it calls ``process.kill()``, awaits ``process.wait()``,
and raises ``TimeoutError(f"training exceeded max_training_seconds={self.max_training_seconds}")``.
That exception propagates through ``_run_stage(job_id, "train_model", ..., "training_error")``,
which records the stage as ``failed`` with ``reason=str(exc)`` and finalizes the
job as ``status="failed"``, ``failure_type="training_error"``.

This test drives the real ``JobManager._run_pipeline`` / ``_train_model`` code
path (including the real ``asyncio.create_subprocess_exec`` + ``process.kill()``
machinery) against a tiny standalone Python script that ignores all CLI args
and sleeps far longer than any ``max_training_seconds`` value under test, so
every Hypothesis example deterministically times out without needing to wait
out real multi-second/hour timeouts. Only ``max_training_seconds`` itself
varies across examples, exercising different timeout margins relative to the
slow script's fixed sleep duration. Only ``_extract_features`` is replaced
with a trivial fake (to avoid depending on a real Feature_Store), matching the
pattern used by the Property 15/17 tests for the same pipeline.
"""

from __future__ import annotations

import asyncio
import textwrap
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.training_manager.jobs import JobManager, _JobState
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

# The slow "training" stand-in always takes far longer than any
# `max_training_seconds` value this test exercises (all well under one
# second), so every example deterministically times out; the only thing
# Hypothesis varies is *how tight* that timeout margin is.
_SLOW_SCRIPT_SLEEP_SECONDS = 2.0
_MARKER_CHECK_BUFFER_SECONDS = 0.5

_MAX_TRAINING_SECONDS = st.floats(
    min_value=0.005,
    max_value=0.2,
    allow_nan=False,
    allow_infinity=False,
)


def _write_slow_script(path: Path, marker_path: Path) -> None:
    """Write a standalone script that ignores all CLI args, sleeps far longer
    than any timeout under test, then writes ``marker_path`` -- so if the
    marker exists after the job manager returns, the subprocess was *not*
    actually killed in time."""
    path.write_text(
        textwrap.dedent(
            f"""
            import time
            from pathlib import Path

            time.sleep({_SLOW_SCRIPT_SLEEP_SECONDS})
            Path({str(marker_path)!r}).write_text("done")
            """
        ),
        encoding="utf-8",
    )


def _install_extract_features_fake(manager: JobManager) -> None:
    async def fake_extract_features(_job_id: str) -> list[dict[str, Any]]:
        return []

    manager._extract_features = fake_extract_features  # type: ignore[assignment]


# **Property 21: 최대 실행 시간 초과 처리**
# **Validates: Requirements 3.9**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(max_training_seconds=_MAX_TRAINING_SECONDS)
def test_training_job_aborts_and_kills_subprocess_on_timeout(max_training_seconds: float) -> None:
    with TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        slow_script = root / "slow_train.py"
        marker_path = root / "marker.txt"
        _write_slow_script(slow_script, marker_path)

        manager = JobManager(
            feature_store=None,  # type: ignore[arg-type]
            model_registry=None,  # type: ignore[arg-type]
            model_storage=None,  # type: ignore[arg-type]
            jobs_dir=root / "jobs",
            max_training_seconds=max_training_seconds,
            train_script=slow_script,
        )
        _install_extract_features_fake(manager)

        job_id = "job-property-21"
        state = _JobState(job_id=job_id, feature_group="fg", model_name="model", seed=1)
        manager._jobs[job_id] = state

        started = time.monotonic()
        asyncio.run(manager._run_pipeline(job_id))
        elapsed = time.monotonic() - started

        job = manager.get_job(job_id)

        # (Requirement 3.9) the job is aborted and marked failed, with a
        # timeout-referencing failure reason recorded.
        assert job.status == "failed"
        assert job.current_stage is None
        assert job.failure_type == "training_error"
        assert job.failure_reason is not None
        assert "max_training_seconds" in job.failure_reason

        # Only extract_features (faked, succeeds) and train_model (times out)
        # ran; no stage after train_model was executed.
        assert [record.stage for record in job.stage_history] == ["extract_features", "train_model"]
        train_record = job.stage_history[-1]
        assert train_record.result == "failed"
        assert train_record.reason == job.failure_reason
        assert train_record.ended_at is not None

        # The abort itself must happen close to max_training_seconds, not
        # after the full slow-script sleep -- i.e. asyncio.wait_for actually
        # fired instead of silently waiting out the subprocess.
        assert elapsed < _SLOW_SCRIPT_SLEEP_SECONDS

        # The subprocess must actually be killed, not merely abandoned: give
        # it a bounded buffer well short of the slow script's full sleep and
        # confirm it never got to write the completion marker.
        time.sleep(_MARKER_CHECK_BUFFER_SECONDS)
        assert not marker_path.exists()
