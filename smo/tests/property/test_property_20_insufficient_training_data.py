"""Property 20 test module (own module to avoid collisions with parallel Property tasks).

Task 3.9: 학습 데이터 부족 오류.

Targets the Training_Manager job-creation validation path,
``smo.aimlfw.training_manager.jobs.JobManager.create_job`` (task 3.3), as
specified in requirements.md Requirement 3.8:

    IF 요청된 Feature_Group 이름이 Feature_Store 에 없거나 해당 Feature_Group 의
    유효 Feature_Record 수가 최소 학습 표본 수(기본값 200) 미만이면, THEN THE
    Training_Manager SHALL 학습 작업을 생성하지 않고 insufficient_training_data
    오류 코드와 부족한 표본 수를 담은 오류 응답을 호출자에게 반환한다.

Design tag (design.md, "Training_Manager (Requirement 3)"):

    #### Property 20: 학습 데이터 부족 오류
    *For any* 존재하지 않는 Feature_Group 이름 또는 유효 Feature_Record 수가
    최소 학습 표본 수 미만인 Feature_Group에 대한 학습 작업 생성 요청,
    Training_Manager는 작업을 생성하지 않고 insufficient_training_data 오류와
    부족한 표본 수를 반환한다.
    **Validates: Requirements 3.8**

Oracle grounding: this test drives the real ``JobManager.create_job`` /
``JobManager._count_valid_records`` (``smo/aimlfw/training_manager/jobs.py``)
against a real ``FeatureStore`` (task 2.x) seeded with real, schema-valid
``FeatureRecord``s (task 1.2's ``strategies.feature_records`` composite), so
"valid Feature_Record count" is derived from production Feature_Store code,
not re-derived by the test. Only ``ModelRegistry``/``ModelStorage`` are
irrelevant to job *creation* validation and are constructed against a real
temporary directory / in-memory Mongo-like collection so ``JobManager`` can
be built without a live MongoDB.

Three cases are covered:
  (a) a Feature_Group name that has no CSV file in the Feature_Store at all
      (`sample_count=0`) is rejected with `insufficient_training_data` and no
      job is created;
  (b) a Feature_Group whose valid (non-`insufficient` data_quality) record
      count is below the configured minimum (a Hypothesis-controlled
      constructor param, not necessarily the 200 default) is rejected the
      same way, with the actual/missing sample count in the error details;
  (c) a Feature_Group meeting the minimum succeeds in creating a job.

The two success-path cases ((b)'s ``expected_valid >= minimum`` branch and
(c)) install trivial fakes for the ``extract_features``/``train_model``/
``save_artifact`` pipeline stages on the ``JobManager`` instance, mirroring
Property 19's ``_install_trivial_earlier_stage_fakes`` pattern, so the
background pipeline task spawned by ``create_job`` finishes deterministically
and fast instead of spawning a real ``train_gnn.py`` subprocess.
"""

from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.common.constants import TARGET_KPIS
from smo.aimlfw.common.models import FeatureRecord
from smo.aimlfw.feature_store import FeatureStore
from smo.aimlfw.model_registry import ModelRegistry
from smo.aimlfw.model_storage import ModelStorage
from smo.aimlfw.training_manager.errors import TrainingManagerError
from smo.aimlfw.training_manager.jobs import JobManager
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS, feature_records

_FEATURE_GROUP = "default"
_MODEL_NAME = "ran-gnn-property-20"


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

    def update_one(self, query: dict[str, Any], update: dict[str, Any], **kwargs: Any) -> None:
        with self.lock:
            if not any(self._matches(item, query) for item in self.documents):
                document = deepcopy(query)
                document.update(deepcopy(update.get("$setOnInsert", {})))
                self.documents.append(document)

    def insert_one(self, document: dict[str, Any]) -> None:
        with self.lock:
            self.documents.append(deepcopy(document))


def _build_manager(root: Path, *, min_training_samples: int) -> JobManager:
    model_storage = ModelStorage(root / "models")
    registry = ModelRegistry(_FakeCollection(), model_storage)
    return JobManager(
        FeatureStore(root / "feature_store"),
        registry,
        model_storage,
        jobs_dir=root / "jobs",
        min_training_samples=min_training_samples,
    )


def _as_group(record: FeatureRecord, feature_group: str) -> FeatureRecord:
    return FeatureRecord.model_validate({**record.model_dump(mode="python"), "feature_group": feature_group})


_METRICS_PAYLOAD: dict[str, Any] = {
    "train_split": 0.8,
    "validation_split": 0.2,
    "train_samples": 200,
    "validation_samples": 50,
    "seed": 7,
    "metrics": {"cell_goodput_mbps": 3.21},
}


def _install_trivial_earlier_stage_fakes(manager: JobManager) -> None:
    """Replace extract_features/train_model/save_artifact with trivial fast fakes
    (same pattern as Property 19's ``_install_trivial_earlier_stage_fakes``).

    This property is only about ``create_job``'s validation of the training
    sample count, not the training pipeline itself. Without this, the real
    ``_train_model`` would spawn a real ``train_gnn.py`` subprocess via the
    background task started by ``create_job`` (``asyncio.create_task``), and
    ``asyncio.run(manager.create_job(...))`` returning while that subprocess
    is still running causes asyncio to try to cancel/clean up the still-live
    task, which hangs waiting on the real subprocess -- multiplied by
    Hypothesis running many examples, this makes the test never finish in a
    reasonable time.
    """

    async def fake_extract_features(_job_id: str) -> list[dict[str, Any]]:
        return []

    async def fake_train_model(_job_id: str, _records: list[dict[str, Any]], work_dir: Path) -> dict[str, Any]:
        return {"artifact_path": work_dir / "model.pt", "metrics_payload": _METRICS_PAYLOAD}

    async def fake_save_artifact(_job_id: str, _train_result: dict[str, Any]) -> dict[str, Any]:
        return {"artifact_uri": "unused-artifact-uri", "version": 1}

    manager._extract_features = fake_extract_features  # type: ignore[assignment]
    manager._train_model = fake_train_model  # type: ignore[assignment]
    manager._save_artifact = fake_save_artifact  # type: ignore[assignment]


@st.composite
def _record_pool(draw: Any) -> list[FeatureRecord]:
    """Generate a small pool of distinct (time_step, cell_id) Feature_Records."""
    count = draw(st.integers(min_value=1, max_value=8))
    keys = draw(
        st.lists(
            st.tuples(st.integers(min_value=0, max_value=4), st.sampled_from(("gNB_5G", "gNB_4G_1", "gNB_4G_2"))),
            min_size=count,
            max_size=count,
            unique=True,
        )
    )
    records: list[FeatureRecord] = []
    for time_step, cell_id in keys:
        record = draw(feature_records())
        record = FeatureRecord.model_validate(
            {**record.model_dump(mode="python"), "time_step": time_step, "cell_id": cell_id}
        )
        records.append(record)
    return records


# **Property 20: 학습 데이터 부족 오류**
# **Validates: Requirements 3.8**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(feature_group=st.text(min_size=1, max_size=32).filter(str.strip))
def test_nonexistent_feature_group_is_rejected_with_insufficient_training_data(feature_group: str) -> None:
    with TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        manager = _build_manager(root, min_training_samples=200)

        with pytest.raises(TrainingManagerError) as error:
            import asyncio

            asyncio.run(manager.create_job(feature_group, _MODEL_NAME, 0))

        assert error.value.error_code == "insufficient_training_data"
        assert error.value.details["feature_group"] == feature_group
        assert error.value.details["sample_count"] == 0
        assert error.value.details["minimum_required"] == 200
        # No job was created: the manager's in-memory job table stays empty and no
        # per-job snapshot file exists on disk.
        assert manager._jobs == {}
        assert list((root / "jobs").glob("*.json")) == []


# **Property 20: 학습 데이터 부족 오류**
# **Validates: Requirements 3.8**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    minimum=st.integers(min_value=1, max_value=20),
    records=_record_pool(),
)
def test_feature_group_below_minimum_is_rejected_with_actual_and_missing_counts(
    minimum: int, records: list[FeatureRecord]
) -> None:
    with TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        store = FeatureStore(root / "feature_store")
        grouped = [_as_group(record, _FEATURE_GROUP) for record in records]
        store.upsert(grouped)
        expected_valid = sum(1 for record in grouped if record.data_quality != "insufficient")

        manager = JobManager(
            store,
            ModelRegistry(_FakeCollection(), ModelStorage(root / "models")),
            ModelStorage(root / "models"),
            jobs_dir=root / "jobs",
            min_training_samples=minimum,
        )
        # Installed unconditionally (harmless for the "else" branch below, since
        # create_job raises synchronously before ever scheduling the background
        # pipeline there): avoids spawning the real train_gnn.py subprocess on
        # the success path, which would otherwise hang asyncio.run()'s cleanup
        # of the still-running background task when this test function returns.
        _install_trivial_earlier_stage_fakes(manager)

        import asyncio

        if expected_valid >= minimum:
            job_id = asyncio.run(manager.create_job(_FEATURE_GROUP, _MODEL_NAME, 0))
            assert job_id in manager._jobs
        else:
            with pytest.raises(TrainingManagerError) as error:
                asyncio.run(manager.create_job(_FEATURE_GROUP, _MODEL_NAME, 0))

            assert error.value.error_code == "insufficient_training_data"
            assert error.value.details["feature_group"] == _FEATURE_GROUP
            assert error.value.details["sample_count"] == expected_valid
            assert error.value.details["minimum_required"] == minimum
            assert manager._jobs == {}
            assert list((root / "jobs").glob("*.json")) == []


# **Property 20: 학습 데이터 부족 오류**
# **Validates: Requirements 3.8**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(extra=st.integers(min_value=0, max_value=5))
def test_feature_group_meeting_minimum_creates_a_job(extra: int) -> None:
    """A Feature_Group whose valid record count meets (or exceeds) the configured
    minimum must succeed in creating a job (the complementary case to
    Requirement 3.8's rejection path)."""
    minimum = 3
    with TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        store = FeatureStore(root / "feature_store")
        cells = ("gNB_5G", "gNB_4G_1", "gNB_4G_2")
        # Time_Step is constrained to 0..4 (Requirement 1.1), so distinct
        # (time_step, cell_id) keys are drawn from the cartesian product of the
        # 5 valid time steps and the 3 cells (15 combinations), which comfortably
        # covers minimum + extra (3..8) records without repeating a key.
        keys = list(product(range(5), cells))[: minimum + extra]
        records = [
            FeatureRecord(
                feature_group=_FEATURE_GROUP,
                time_step=time_step,
                cell_id=cell_id,
                control_parameters={
                    "tx_power_dbm": 43.0,
                    "ret_tilt_deg": 5.0,
                    "cio_bias_db": 0.5,
                    "hysteresis_db": 2.5,
                    "ttt_ms": 160,
                },
                features={kpi: 10.0 for kpi in TARGET_KPIS},
                sample_counts={kpi: {"valid": 1, "excluded": 0} for kpi in TARGET_KPIS},
                data_quality="complete",
                source_dir="/tmp/smo-property-20-test",
                seed=0,
            )
            for time_step, cell_id in keys
        ]
        store.upsert(records)

        manager = JobManager(
            store,
            ModelRegistry(_FakeCollection(), ModelStorage(root / "models")),
            ModelStorage(root / "models"),
            jobs_dir=root / "jobs",
            min_training_samples=minimum,
        )
        # Avoid spawning the real train_gnn.py subprocess: this test always hits
        # the success path, so create_job schedules a real background pipeline
        # task via asyncio.create_task, and asyncio.run() returning before that
        # task (and its subprocess) finishes causes a hang during task cleanup.
        _install_trivial_earlier_stage_fakes(manager)

        import asyncio

        job_id = asyncio.run(manager.create_job(_FEATURE_GROUP, _MODEL_NAME, 0))
        assert job_id in manager._jobs
        assert (root / "jobs" / f"{job_id}.json").is_file()
