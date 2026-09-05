"""Property 17 test module (own module to avoid collisions with parallel Property tasks).

Task 3.6: 학습 실패 메타데이터.

Targets the Training_Manager pipeline's failure path, as specified in
requirements.md 3.5 and design.md's Property 17, implemented on
``smo.aimlfw.training_manager.jobs.JobManager`` (task 3.3):

    IF 학습 작업이 실패하면, THEN THE Training_Manager SHALL 해당 작업의 상태를
    `failed` 로 설정하고 실패한 단계 이름과 실패 유형(`data_error`, `training_error`,
    `storage_error` 중 하나)을 작업 메타데이터에 기록하며, 해당 작업의 모델 산출물을
    Model_Registry 에 등록하지 않는다.

Design tag (design.md, "Training_Manager (Requirement 3)"):

    #### Property 17: 학습 실패 메타데이터
    *For any* 파이프라인 단계에서 실패가 발생한 학습 작업, Training_Manager는
    상태를 `failed`로 설정하고 실패한 단계 이름과 실패 유형(`data_error`,
    `training_error`, `storage_error` 중 하나)을 기록하며 모델을 Model_Registry에
    등록하지 않는다.
    **Validates: Requirements 3.5**

Oracle grounding: ``jobs.py`` classifies each of the three early pipeline
stages -- ``extract_features``, ``train_model``, ``save_artifact`` -- to a
fixed failure type via ``JobManager._run_pipeline``'s ``_run_stage`` calls
(``"data_error"``, ``"training_error"``, ``"storage_error"`` respectively;
see the ``_FAILURE_TYPE_BY_STAGE`` mapping mirrored below and the literal
strings passed at each ``_run_stage(...)`` call site in ``jobs.py``). A
failure in ``register_metrics`` is a deliberate documented exception (job
stays ``completed``, not ``failed``) and is therefore excluded from this
property, which only concerns Requirement 3.5's ``failed`` job outcome.

This test drives the real ``JobManager._run_pipeline`` state machine against
a real ``ModelRegistry``/``ModelStorage`` pair (task 3.1/3.2) so that "the
job's model artifact is not registered in Model_Registry" is checked against
production registry code, not re-derived. Only the three pipeline stage
hooks are replaced with fast fakes (one of which is forced to fail with a
Hypothesis-controlled reason string) to avoid depending on real Feature_Store
contents or a real PyTorch training subprocess, which are orthogonal to this
property.
"""

from __future__ import annotations

import asyncio
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pymongo.errors import DuplicateKeyError

from smo.aimlfw.model_registry import ModelRegistry, ModelRegistryError
from smo.aimlfw.model_storage import ModelStorage
from smo.aimlfw.training_manager.jobs import JobManager, _JobState
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

_STAGE_ORDER = ("extract_features", "train_model", "save_artifact", "register_metrics")
# Only the three stages that actually mark the job "failed" per jobs.py's own
# documented exception for register_metrics (Requirement 3.5 concerns the
# "failed" outcome only).
_FAILING_STAGES = ("extract_features", "train_model", "save_artifact")
_FAILURE_TYPE_BY_STAGE = {
    "extract_features": "data_error",
    "train_model": "training_error",
    "save_artifact": "storage_error",
}

_MODEL_NAME = "ran-gnn-property-17"
_FEATURE_GROUP = "default"


class _FakeCursor:
    """Minimal ``pymongo`` cursor stand-in supporting ``sort``/``limit``/iteration."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents

    def sort(self, key: str, direction: int | None = None) -> "_FakeCursor":
        self.documents.sort(key=lambda item: item[key], reverse=direction == -1)
        return self

    def limit(self, count: int) -> "_FakeCursor":
        self.documents = self.documents[:count]
        return self

    def __iter__(self):
        return iter(deepcopy(self.documents))


class _FakeCollection:
    """Minimal in-memory stand-in for the MongoDB collection ``ModelRegistry`` expects."""

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.lock = Lock()

    def create_index(self, keys: Any, **kwargs: Any) -> str:
        return kwargs.get("name", "index")

    @staticmethod
    def _matches(document: dict[str, Any], query: dict[str, Any]) -> bool:
        return all(document.get(key) == value for key, value in query.items())

    def find_one(self, query: dict[str, Any], *args: Any, **kwargs: Any):
        matching = [item for item in self.documents if self._matches(item, query)]
        sort = kwargs.get("sort")
        if sort:
            key, direction = sort[0]
            matching.sort(key=lambda item: item[key], reverse=direction == -1)
        return deepcopy(matching[0]) if matching else None

    def find(self, query: dict[str, Any]) -> _FakeCursor:
        return _FakeCursor([deepcopy(item) for item in self.documents if self._matches(item, query)])

    def update_one(self, query: dict[str, Any], update: dict[str, Any], **kwargs: Any):
        with self.lock:
            if not any(self._matches(item, query) for item in self.documents):
                document = deepcopy(query)
                document.update(deepcopy(update.get("$setOnInsert", {})))
                self.documents.append(document)

    def insert_one(self, document: dict[str, Any]):
        with self.lock:
            duplicate = any(
                item.get("document_type") == "version"
                and item.get("model_name") == document.get("model_name")
                and item.get("version") == document.get("version")
                for item in self.documents
            )
            if duplicate:
                raise DuplicateKeyError("duplicate model version")
            self.documents.append(deepcopy(document))


def _install_stage_fakes(manager: JobManager, failing_stage: str, reason: str) -> None:
    """Make every stage up to and including ``failing_stage`` succeed trivially,
    except ``failing_stage`` itself, which raises ``reason`` as its error message.

    ``register_metrics`` is never exercised by this property (see module
    docstring), so it is left untouched -- the pipeline always aborts at one
    of the three earlier stages before reaching it.
    """

    async def fake_extract_features(_job_id: str) -> list[dict[str, Any]]:
        if failing_stage == "extract_features":
            raise ValueError(reason)
        return []

    async def fake_train_model(_job_id: str, _records: list[dict[str, Any]], work_dir: Path) -> dict[str, Any]:
        if failing_stage == "train_model":
            raise ValueError(reason)
        return {"artifact_path": work_dir / "model.pt", "metrics_payload": {}}

    async def fake_save_artifact(_job_id: str, _train_result: dict[str, Any]) -> dict[str, Any]:
        if failing_stage == "save_artifact":
            raise ValueError(reason)
        return {"artifact_uri": "unused", "version": 1}

    manager._extract_features = fake_extract_features  # type: ignore[assignment]
    manager._train_model = fake_train_model  # type: ignore[assignment]
    manager._save_artifact = fake_save_artifact  # type: ignore[assignment]


_FAILURE_CASE = st.tuples(
    st.sampled_from(_FAILING_STAGES),
    st.text(min_size=1, max_size=64).filter(lambda text: text.strip() != ""),
)


# **Property 17: 학습 실패 메타데이터**
# **Validates: Requirements 3.5**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(case=_FAILURE_CASE)
def test_failed_training_job_records_failure_metadata_and_skips_registration(case: tuple[str, str]) -> None:
    failing_stage, reason = case

    with TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        model_storage = ModelStorage(root / "models")
        registry = ModelRegistry(_FakeCollection(), model_storage)
        manager = JobManager(
            feature_store=None,  # type: ignore[arg-type]
            model_registry=registry,
            model_storage=model_storage,
            jobs_dir=root / "jobs",
        )
        _install_stage_fakes(manager, failing_stage, reason)

        job_id = "job-property-17"
        state = _JobState(job_id=job_id, feature_group=_FEATURE_GROUP, model_name=_MODEL_NAME, seed=1)
        manager._jobs[job_id] = state

        asyncio.run(manager._run_pipeline(job_id))

        job = manager.get_job(job_id)

        # The job's overall status must be "failed" (Requirement 3.5).
        assert job.status == "failed"
        assert job.current_stage is None

        # The failed stage name and its failure type classification must be
        # recorded in the job metadata, matching jobs.py's fixed stage ->
        # failure_type mapping.
        assert job.failure_type == _FAILURE_TYPE_BY_STAGE[failing_stage]
        assert job.failure_reason == reason

        failure_index = _STAGE_ORDER.index(failing_stage)
        expected_ran_stages = _STAGE_ORDER[: failure_index + 1]
        assert [record.stage for record in job.stage_history] == list(expected_ran_stages)

        failed_record = job.stage_history[-1]
        assert failed_record.stage == failing_stage
        assert failed_record.result == "failed"
        assert failed_record.reason == reason
        assert failed_record.ended_at is not None

        # No stage runs after the failing one.
        assert len(job.stage_history) == failure_index + 1

        # The job's model artifact must not be registered in Model_Registry:
        # this model_name was never registered before this run, so any query
        # against the real registry must still report "model_not_found".
        with pytest.raises(ModelRegistryError) as exc_info:
            registry.latest(_MODEL_NAME)
        assert exc_info.value.error_code == "model_not_found"

        with pytest.raises(ModelRegistryError) as exc_info:
            registry.list_versions(_MODEL_NAME)
        assert exc_info.value.error_code == "model_not_found"
