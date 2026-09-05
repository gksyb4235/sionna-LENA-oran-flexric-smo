"""Task 3.18: Training_Manager + Model_Registry (MongoDB-backed) integration test.

Exercises the real ``smo.aimlfw.training_manager.JobManager`` (task 3.3) driving
the real ``smo.aimlfw.model_registry.ModelRegistry`` (task 3.2, including the
Requirement 3.4 completion-metadata fields added by task 3.5's fix) and the
real ``smo.aimlfw.model_storage.ModelStorage`` (task 3.1) end-to-end through a
full ``extract_features -> train_model -> save_artifact -> register_metrics``
pipeline run.

No local MongoDB is reachable in this environment (verified by probing
``localhost:27017`` before writing this test), so ``ModelRegistry`` is wired
to the same minimal in-memory ``_FakeCollection``/``_FakeCursor`` stand-in
already used by the Property 16/17/18/20/22/23/24/25/26/27/28 test modules
(``smo/tests/property/test_property_23_version_list_sort_and_cap.py`` etc.) —
that stand-in implements exactly the ``create_index``/``find_one``/``find``/
``update_one``/``insert_one`` surface ``ModelRegistry`` calls, so this is a
faithful integration test of the registry's real logic even without a live
Mongo server. If a real local MongoDB is ever available, ``pymongo.MongoClient``
could be substituted directly since ``ModelRegistry`` only depends on the
``CollectionLike`` protocol.

Covers:
  * Job creation completes within 5 seconds (Requirement 3.1).
  * Job status query completes within 1 second (Requirement 3.3).
  * A completed job's stage_history contains all 4 stages
    (extract_features, train_model, save_artifact, register_metrics) in
    order with ``succeeded`` results (Requirement 3.2).
  * The trained artifact is actually registered in Model_Registry:
    ``registry.get()``/``registry.latest()`` return a real record whose
    ``artifact_uri`` exists in Model_Storage (Requirements 3.4, 4.1, 4.3, 4.6).
  * Running the same Feature_Group + same seed twice produces reproducible
    results: per-KPI MAPE within 1.0 percentage point and identical
    train_samples (Requirement 3.6). This is a concrete integration-level
    check reusing the same run_training() double-invocation approach as
    Property 18's test (task 3.7,
    ``smo/tests/property/test_property_18_identical_seed_reproducibility.py``),
    but driven through the full JobManager pipeline rather than calling
    ``run_training`` directly.

**Validates: Requirements 3.1-3.9, 4.1-4.9**
"""

from __future__ import annotations

import time
from copy import deepcopy
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from smo.aimlfw.common.constants import TARGET_KPIS
from smo.aimlfw.common.models import CellParameters, FeatureRecord, SampleCount, TrainingJob
from smo.aimlfw.feature_store import FeatureStore
from smo.aimlfw.model_registry import ModelRegistry
from smo.aimlfw.model_storage import ModelStorage
from smo.aimlfw.training_manager import JobManager, create_app

_CELLS = ("gNB_5G", "gNB_4G_1", "gNB_4G_2")
_FEATURE_GROUP = "default"
_JOB_CREATION_TIME_LIMIT_SECONDS = 5.0
_JOB_STATUS_QUERY_TIME_LIMIT_SECONDS = 1.0
_JOB_COMPLETION_TIMEOUT_SECONDS = 90.0


# --------------------------------------------------------------------------
# In-memory MongoDB collection stand-in (same pattern as the Property
# 16/17/18/20/22/23/24/25/26/27/28 test modules under smo/tests/property/).
# --------------------------------------------------------------------------


class _FakeCursor:
    """Minimal ``pymongo`` cursor stand-in supporting ``sort``/``limit``/iteration."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents

    def sort(self, key: str, direction: int | None = None) -> "_FakeCursor":
        self.documents = sorted(self.documents, key=lambda item: item[key], reverse=direction == -1)
        return self

    def limit(self, count: int) -> "_FakeCursor":
        self.documents = self.documents[:count]
        return self

    def __iter__(self):
        return iter(deepcopy(self.documents))


class _FakeCollection:
    """Minimal in-memory stand-in for the PyMongo collection ``ModelRegistry`` uses."""

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []

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
        if not any(self._matches(item, query) for item in self.documents):
            document = deepcopy(query)
            document.update(deepcopy(update.get("$setOnInsert", {})))
            self.documents.append(document)

    def insert_one(self, document: dict[str, Any]) -> None:
        self.documents.append(deepcopy(document))


# --------------------------------------------------------------------------
# Fixture helpers
# --------------------------------------------------------------------------


def _feature_record(time_step: int, cell_id: str, index: int) -> FeatureRecord:
    return FeatureRecord(
        feature_group=_FEATURE_GROUP,
        time_step=time_step,
        cell_id=cell_id,
        control_parameters=CellParameters(
            tx_power_dbm=43.0, ret_tilt_deg=5.0, cio_bias_db=0.5, hysteresis_db=2.5, ttt_ms=160
        ),
        features={
            "cell_goodput_mbps": 10.0 + time_step + index * 0.01,
            "avg_ue_goodput_mbps": 5.0 + index * 0.02,
            "ue_goodput_p5_mbps": 1.0,
            "sinr_p50_db": 8.0,
            "prb_utilization_pct": 50.0,
            "delay_p95_ms": 20.0,
            "ho_failure_count": 1.0,
            "pingpong_count": 0.0,
            "rlf_count": 0.0,
            "interval_energy_j": 100.0,
        },
        sample_counts={name: SampleCount(valid=1, excluded=0) for name in TARGET_KPIS},
        data_quality="complete",
        source_dir="/tmp/smo-task-3-18-integration",
        seed=0,
    )


def _seed_feature_store(root: Path, records_per_cell_step: int = 20) -> FeatureStore:
    """Populate a Feature_Store with enough records to satisfy min_training_samples."""
    store = FeatureStore(root)
    records = [
        _feature_record(time_step, cell_id, index)
        for time_step in range(5)
        for cell_id in _CELLS
        for index in range(records_per_cell_step)
    ]
    store.upsert(records)
    return store


def _build_job_manager(
    tmp_path: Path,
    label: str,
    feature_store: FeatureStore,
    *,
    min_training_samples: int = 10,
) -> tuple[JobManager, ModelRegistry, ModelStorage]:
    model_storage = ModelStorage(tmp_path / f"models-{label}")
    registry = ModelRegistry(_FakeCollection(), model_storage)
    manager = JobManager(
        feature_store,
        registry,
        model_storage,
        jobs_dir=tmp_path / f"jobs-{label}",
        min_training_samples=min_training_samples,
        max_training_seconds=60,
        max_epochs=3,
        patience=2,
    )
    return manager, registry, model_storage


def _wait_for_completion(manager: JobManager, job_id: str, timeout_s: float) -> TrainingJob:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        snapshot = manager.get_job(job_id)
        if snapshot.status in ("completed", "failed"):
            return snapshot
        time.sleep(0.1)
    raise AssertionError(f"training job {job_id} did not finish within {timeout_s}s")


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------


class TestTrainingJobModelRegistryIntegration:
    """Requirements 3.1-3.9, 4.1-4.9 exercised against the real component stack."""

    def test_job_creation_completes_within_5_seconds(self, tmp_path: Path) -> None:
        """Requirement 3.1: POST /jobs returns a job_id within 5 seconds."""
        store = _seed_feature_store(tmp_path / "feature_store")
        manager, _registry, _storage = _build_job_manager(tmp_path, "creation", store)

        with TestClient(create_app(manager)) as client:
            started = time.monotonic()
            response = client.post(
                "/jobs", json={"feature_group": _FEATURE_GROUP, "model_name": "ran-gnn-creation", "seed": 1}
            )
            elapsed = time.monotonic() - started

            assert response.status_code == 200
            job_id = response.json()["job_id"]
            assert job_id
            assert elapsed <= _JOB_CREATION_TIME_LIMIT_SECONDS, (
                f"job creation took {elapsed:.3f}s, exceeding the "
                f"{_JOB_CREATION_TIME_LIMIT_SECONDS:.0f}s SLA (Requirement 3.1)"
            )

            # Drain the background pipeline so no dangling task outlives the test.
            _wait_for_completion(manager, job_id, _JOB_COMPLETION_TIMEOUT_SECONDS)

    def test_job_status_query_completes_within_1_second(self, tmp_path: Path) -> None:
        """Requirement 3.3: GET /jobs/{id} responds within 1 second while running and after completion."""
        store = _seed_feature_store(tmp_path / "feature_store")
        manager, _registry, _storage = _build_job_manager(tmp_path, "status", store)

        with TestClient(create_app(manager)) as client:
            job_id = client.post(
                "/jobs", json={"feature_group": _FEATURE_GROUP, "model_name": "ran-gnn-status", "seed": 2}
            ).json()["job_id"]

            # Poll a few times while the job is still in flight, timing each query.
            for _ in range(5):
                started = time.monotonic()
                status_response = client.get(f"/jobs/{job_id}")
                elapsed = time.monotonic() - started
                assert status_response.status_code == 200
                assert elapsed <= _JOB_STATUS_QUERY_TIME_LIMIT_SECONDS, (
                    f"status query took {elapsed:.3f}s, exceeding the "
                    f"{_JOB_STATUS_QUERY_TIME_LIMIT_SECONDS:.0f}s SLA (Requirement 3.3)"
                )
                if status_response.json()["status"] in ("completed", "failed"):
                    break
                time.sleep(0.2)

            final = _wait_for_completion(manager, job_id, _JOB_COMPLETION_TIMEOUT_SECONDS)
            assert final.status == "completed", final

            # One more timed query after completion.
            started = time.monotonic()
            final_response = client.get(f"/jobs/{job_id}")
            elapsed = time.monotonic() - started
            assert final_response.status_code == 200
            assert elapsed <= _JOB_STATUS_QUERY_TIME_LIMIT_SECONDS

    def test_completed_job_stage_history_has_four_succeeded_stages_in_order(self, tmp_path: Path) -> None:
        """Requirement 3.2: stage_history records all 4 stages, in order, each succeeded."""
        store = _seed_feature_store(tmp_path / "feature_store")
        manager, _registry, _storage = _build_job_manager(tmp_path, "stages", store)

        with TestClient(create_app(manager)) as client:
            job_id = client.post(
                "/jobs", json={"feature_group": _FEATURE_GROUP, "model_name": "ran-gnn-stages", "seed": 3}
            ).json()["job_id"]
            final = _wait_for_completion(manager, job_id, _JOB_COMPLETION_TIMEOUT_SECONDS)

        assert final.status == "completed", final
        stage_names = [stage.stage for stage in final.stage_history]
        assert stage_names == ["extract_features", "train_model", "save_artifact", "register_metrics"]
        assert all(stage.result == "succeeded" for stage in final.stage_history)
        # Requirement 3.2: every stage records a start and end timestamp, and stages
        # run in non-decreasing chronological order.
        for stage in final.stage_history:
            assert stage.started_at is not None
            assert stage.ended_at is not None
            assert stage.ended_at >= stage.started_at
        for earlier, later in zip(final.stage_history, final.stage_history[1:]):
            assert later.started_at >= earlier.started_at

    def test_completed_job_artifact_is_registered_and_retrievable_from_model_registry(
        self, tmp_path: Path
    ) -> None:
        """Requirements 3.4, 4.1, 4.3, 4.6: the trained artifact is a real, retrievable registry entry."""
        store = _seed_feature_store(tmp_path / "feature_store")
        manager, registry, model_storage = _build_job_manager(tmp_path, "registry", store)
        model_name = "ran-gnn-registry"

        with TestClient(create_app(manager)) as client:
            job_id = client.post(
                "/jobs", json={"feature_group": _FEATURE_GROUP, "model_name": model_name, "seed": 4}
            ).json()["job_id"]
            final = _wait_for_completion(manager, job_id, _JOB_COMPLETION_TIMEOUT_SECONDS)

        assert final.status == "completed", final
        assert final.metrics_registration_failure_reason is None

        # Requirement 4.1/4.3: latest() returns a real version-1 record for a
        # freshly-registered model name, and get() returns the identical record.
        latest_record = registry.latest(model_name)
        assert latest_record.version == 1
        assert latest_record.feature_group == _FEATURE_GROUP
        assert latest_record.model_name == model_name

        fetched_record = registry.get(model_name, latest_record.version)
        assert fetched_record == latest_record

        # Requirement 3.4: training split ratio, sample counts, and seed were
        # forwarded to Model_Registry.register and persisted on the record.
        assert fetched_record.seed == 4
        assert fetched_record.train_samples is not None and fetched_record.train_samples > 0
        assert fetched_record.validation_samples is not None and fetched_record.validation_samples >= 0
        assert fetched_record.train_split is not None
        assert fetched_record.validation_split is not None

        # Requirement 4.6: the registered artifact_uri is a real, existing file
        # in Model_Storage (not a placeholder/dangling path).
        assert model_storage.artifact_exists(fetched_record.artifact_uri)
        assert Path(fetched_record.artifact_uri).is_file()

        # Requirement 4.1: metrics cover Target_KPIs with non-negative MAPE values.
        assert fetched_record.metrics
        assert set(fetched_record.metrics).issubset(set(TARGET_KPIS))
        assert all(value >= 0 for value in fetched_record.metrics.values())

    def test_identical_feature_group_and_seed_produce_reproducible_results(self, tmp_path: Path) -> None:
        """Requirement 3.6: same Feature_Group + same seed run twice -> reproducible metrics."""
        seed = 7
        model_name = "ran-gnn-repro"

        # Two fully independent stacks (separate Model_Registry/Model_Storage/
        # jobs_dir) but built from an identical Feature_Store snapshot, so any
        # divergence in results can only come from training non-determinism.
        store_a = _seed_feature_store(tmp_path / "feature_store_a")
        store_b = _seed_feature_store(tmp_path / "feature_store_b")
        manager_a, registry_a, _storage_a = _build_job_manager(tmp_path, "repro-a", store_a)
        manager_b, registry_b, _storage_b = _build_job_manager(tmp_path, "repro-b", store_b)

        with TestClient(create_app(manager_a)) as client_a:
            job_id_a = client_a.post(
                "/jobs", json={"feature_group": _FEATURE_GROUP, "model_name": model_name, "seed": seed}
            ).json()["job_id"]
            final_a = _wait_for_completion(manager_a, job_id_a, _JOB_COMPLETION_TIMEOUT_SECONDS)

        with TestClient(create_app(manager_b)) as client_b:
            job_id_b = client_b.post(
                "/jobs", json={"feature_group": _FEATURE_GROUP, "model_name": model_name, "seed": seed}
            ).json()["job_id"]
            final_b = _wait_for_completion(manager_b, job_id_b, _JOB_COMPLETION_TIMEOUT_SECONDS)

        assert final_a.status == "completed", final_a
        assert final_b.status == "completed", final_b

        record_a = registry_a.latest(model_name)
        record_b = registry_b.latest(model_name)

        # Requirement 3.6: identical training sample counts.
        assert record_a.train_samples == record_b.train_samples
        assert record_a.validation_samples == record_b.validation_samples

        # Requirement 3.6: per-Target_KPI MAPE absolute difference within 1.0
        # percentage point (same tolerance and approach as Property 18's test).
        assert record_a.metrics.keys() == record_b.metrics.keys()
        for kpi, value_a in record_a.metrics.items():
            value_b = record_b.metrics[kpi]
            assert abs(value_a - value_b) <= 1.0, (
                f"MAPE for {kpi} diverged by more than 1.0pp across identical-seed "
                f"runs: {value_a} vs {value_b}"
            )
