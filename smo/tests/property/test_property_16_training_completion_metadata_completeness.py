"""Property 16 test module (own module to avoid collisions with parallel Property tasks).

Task 3.5: 학습 완료 메타데이터 완전성.

Targets the Training_Manager pipeline's completion path, as specified in
requirements.md 3.4 and design.md's Property 16, implemented across
``smo.aimlfw.training_manager.jobs.JobManager`` (task 3.3) and
``smo.aimlfw.model_registry.registry.ModelRegistry.register`` (task 3.2):

    WHEN 학습 작업이 완료되면, THE Training_Manager SHALL 학습 데이터 분할 비율(기본값
    학습 0.8, 검증 0.2), 학습 표본 수, 검증 표본 수, 난수 시드, Feature_Group 이름,
    Target_KPI 별 평균 절대 백분율 오차(단위 percent, 소수점 둘째 자리까지)를
    Model_Registry 에 기록한다.

Design tag (design.md, "Training_Manager (Requirement 3)"):

    #### Property 16: 학습 완료 메타데이터 완전성
    *For any* 성공적으로 완료된 학습 작업, Model_Registry에 기록되는 메타데이터는
    데이터 분할 비율, 학습/검증 표본 수, 난수 시드, Feature_Group 이름, Target_KPI별
    평가 지표(소수점 둘째 자리)를 모두 포함한다.
    **Validates: Requirements 3.4**

Oracle grounding: this test drives the real ``JobManager._run_pipeline`` state
machine (``smo/aimlfw/training_manager/jobs.py``) end-to-end -- real
``extract_features``, real ``save_artifact`` (``ModelStorage``), and the real
``register_metrics`` retry loop that calls
``smo.aimlfw.model_registry.registry.ModelRegistry.register`` -- so every
assertion below is checked against production orchestration code, not a
reimplementation of it. Only the expensive ``train_model`` stage (which
shells out to ``train_gnn.py`` and trains a real PyTorch model) is replaced
with a fast stage that returns a Hypothesis-controlled ``metrics_payload``
shaped exactly like ``train_gnn.py``'s real output contract (see
``run_training``'s ``payload`` in train_gnn.py: ``train_split``,
``validation_split``, ``train_samples``, ``validation_samples``, ``seed``,
``metrics``), keeping the rest of the pipeline (``jobs.py``) and the
Model_Registry/ModelVersionRecord schema (``smo/aimlfw/model_registry/registry.py``,
``smo/aimlfw/common/models.py``) exactly as shipped -- this is the "schemas.py"
half of the oracle the task calls out (Training_Manager's ``schemas.py`` only
wraps ``TrainingJob``/``CreateJobRequest``, so the actual completion-metadata
contract lives on ``ModelVersionRecord`` via ``model_registry.register()``,
which this test calls unmodified).
"""

from __future__ import annotations

import asyncio
import tempfile
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pymongo.errors import DuplicateKeyError

from smo.aimlfw.common.constants import MAX_SEED, TARGET_KPIS
from smo.aimlfw.common.models import CellParameters, FeatureRecord, SampleCount
from smo.aimlfw.feature_store import FeatureStore
from smo.aimlfw.model_registry import ModelRegistry
from smo.aimlfw.model_storage import ModelStorage
from smo.aimlfw.training_manager.jobs import JobManager
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

_CELLS = ("gNB_5G", "gNB_4G_1", "gNB_4G_2")
_MIN_TRAINING_SAMPLES = 5


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


class _FastJobManager(JobManager):
    """A JobManager whose ``train_model`` stage is a fast stand-in.

    Every other stage (``extract_features``, ``save_artifact``,
    ``register_metrics``) runs the real ``jobs.py`` code untouched, so this
    class only removes the real PyTorch subprocess (task 3.3's
    ``train_gnn.py``), which is orthogonal to Property 16 (completion
    metadata completeness) and would make a 100-example Hypothesis run
    prohibitively slow.
    """

    def __init__(self, *args: Any, metrics_payload: dict[str, Any], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._stub_metrics_payload = metrics_payload

    async def _train_model(self, job_id: str, records: list[dict[str, Any]], work_dir: Path) -> dict[str, Any]:
        artifact_path = work_dir / "model.pt"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_bytes(b"fake-artifact-for-property-16")
        return {"artifact_path": artifact_path, "metrics_payload": self._stub_metrics_payload}


def _feature_record(time_step: int, cell_id: str, index: int) -> FeatureRecord:
    return FeatureRecord(
        feature_group="default",
        time_step=time_step,
        cell_id=cell_id,
        control_parameters=CellParameters(
            tx_power_dbm=43.0, ret_tilt_deg=5.0, cio_bias_db=0.5, hysteresis_db=2.5, ttt_ms=160
        ),
        features={kpi: 10.0 + index * 0.01 for kpi in TARGET_KPIS},
        sample_counts={kpi: SampleCount(valid=1, excluded=0) for kpi in TARGET_KPIS},
        data_quality="complete",
        source_dir="/tmp/smo-property-16-test",
        seed=0,
    )


def _seed_feature_store(root: Path, feature_group: str) -> FeatureStore:
    store = FeatureStore(root / "feature_store")
    records = [
        FeatureRecord.model_validate(
            {**_feature_record(time_step, cell_id, index).model_dump(mode="python"), "feature_group": feature_group}
        )
        for time_step in range(5)
        for cell_id in _CELLS
        for index in range(3)
    ]
    store.upsert(records)
    return store


@st.composite
def _completion_metadata(draw: Any) -> dict[str, Any]:
    """Generate a ``train_gnn.py``-shaped metrics payload (Requirement 3.4 fields)."""
    train_split = draw(st.sampled_from((0.5, 0.6, 0.7, 0.8, 0.9)))
    train_samples = draw(st.integers(min_value=1, max_value=10_000))
    validation_samples = draw(st.integers(min_value=0, max_value=10_000))
    seed = draw(st.integers(min_value=0, max_value=MAX_SEED))
    kpi_subset = draw(
        st.lists(st.sampled_from(TARGET_KPIS), min_size=1, max_size=len(TARGET_KPIS), unique=True)
    )
    metrics = {
        kpi: round(draw(st.floats(min_value=0.0, max_value=1_000.0, allow_nan=False, allow_infinity=False)), 2)
        for kpi in kpi_subset
    }
    return {
        "train_split": train_split,
        "validation_split": round(1.0 - train_split, 6),
        "train_samples": train_samples,
        "validation_samples": validation_samples,
        "seed": seed,
        "metrics": metrics,
    }


async def _run_job_to_completion(manager: _FastJobManager, feature_group: str, model_name: str, seed: int) -> None:
    job_id = await manager.create_job(feature_group, model_name, seed)
    for _ in range(500):
        job = manager.get_job(job_id)
        if job.status in ("completed", "failed"):
            assert job.status == "completed", (job.status, job.failure_type, job.failure_reason)
            return
        await asyncio.sleep(0.01)
    raise AssertionError("training job did not finish in time")


# **Property 16: 학습 완료 메타데이터 완전성**
# **Validates: Requirements 3.4**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(metadata=_completion_metadata())
def test_completed_job_registers_full_requirement_3_4_metadata(metadata: dict[str, Any]) -> None:
    feature_group = "default"
    model_name = "ran-gnn"
    job_seed = metadata["seed"]

    with tempfile.TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        store = _seed_feature_store(root, feature_group)
        model_storage = ModelStorage(root / "models")
        registry = ModelRegistry(_FakeCollection(), model_storage)
        manager = _FastJobManager(
            store,
            registry,
            model_storage,
            jobs_dir=root / "jobs",
            min_training_samples=_MIN_TRAINING_SAMPLES,
            metrics_payload=metadata,
        )

        asyncio.run(_run_job_to_completion(manager, feature_group, model_name, job_seed))

        # Requirement 3.4: once the job completes, Model_Registry must hold a
        # record for this model carrying every one of: data split ratio,
        # train/validation sample counts, random seed, Feature_Group name, and
        # per-Target_KPI MAPE (2 decimals) -- all together, non-null, correctly
        # typed. ModelVersionRecord (smo/aimlfw/common/models.py) is the actual
        # persisted shape ``registry.register()`` (called from jobs.py's
        # ``_register_metrics_with_retry``) returns/stores.
        registered = registry.latest(model_name)

        required_fields = {
            "train_split": metadata["train_split"],
            "validation_split": metadata["validation_split"],
            "train_samples": metadata["train_samples"],
            "validation_samples": metadata["validation_samples"],
            "seed": metadata["seed"],
            "feature_group": feature_group,
            "metrics": metadata["metrics"],
        }
        registered_dump = registered.model_dump(mode="python")
        missing = [name for name in required_fields if name not in registered_dump or registered_dump[name] is None]
        assert not missing, (
            "Model_Registry record is missing Requirement 3.4 completion metadata fields: "
            f"{missing}; registered record was {registered_dump!r}"
        )

        assert registered_dump["feature_group"] == feature_group
        assert registered_dump["train_split"] == required_fields["train_split"]
        assert registered_dump["validation_split"] == required_fields["validation_split"]
        assert registered_dump["train_samples"] == required_fields["train_samples"]
        assert registered_dump["validation_samples"] == required_fields["validation_samples"]
        assert registered_dump["seed"] == required_fields["seed"]
        for kpi, value in metadata["metrics"].items():
            assert registered_dump["metrics"][kpi] == value
