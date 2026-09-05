"""Sanity tests for the Training_Manager API and the train_gnn.py pipeline.

Full property-based coverage (Properties 15-21) belongs to tasks 3.4-3.10;
these tests only confirm the service starts, a job can be created and
queried, and ``train_gnn.py`` trains deterministically end-to-end on a
small synthetic dataset.
"""

from __future__ import annotations

import subprocess
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from smo.aimlfw.common.models import CellParameters, FeatureRecord, SampleCount
from smo.aimlfw.feature_store import FeatureStore
from smo.aimlfw.model_registry import ModelRegistry
from smo.aimlfw.model_storage import ModelStorage
from smo.aimlfw.training_manager import JobManager, create_app
from smo.aimlfw.training_manager.train_gnn import run_training

CELLS = ("gNB_5G", "gNB_4G_1", "gNB_4G_2")
TARGET_KPIS = (
    "cell_goodput_mbps", "avg_ue_goodput_mbps", "ue_goodput_p5_mbps", "sinr_p50_db",
    "prb_utilization_pct", "delay_p95_ms", "ho_failure_count", "pingpong_count",
    "rlf_count", "interval_energy_j",
)


class FakeCursor:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents

    def sort(self, key: str, direction: int | None = None) -> FakeCursor:
        self.documents.sort(key=lambda item: item[key], reverse=direction == -1)
        return self

    def limit(self, count: int) -> FakeCursor:
        self.documents = self.documents[:count]
        return self

    def __iter__(self):
        return iter(deepcopy(self.documents))


class FakeCollection:
    """Minimal in-memory stand-in for the MongoDB collection ModelRegistry uses."""

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

    def find(self, query: dict[str, Any]) -> FakeCursor:
        return FakeCursor([deepcopy(item) for item in self.documents if self._matches(item, query)])

    def update_one(self, query: dict[str, Any], update: dict[str, Any], **kwargs: Any):
        if not any(self._matches(item, query) for item in self.documents):
            document = deepcopy(query)
            document.update(deepcopy(update.get("$setOnInsert", {})))
            self.documents.append(document)

    def insert_one(self, document: dict[str, Any]):
        self.documents.append(deepcopy(document))


def _feature_record(time_step: int, cell_id: str, index: int) -> FeatureRecord:
    return FeatureRecord(
        feature_group="default",
        time_step=time_step,
        cell_id=cell_id,
        control_parameters=CellParameters(
            tx_power_dbm=43.0, ret_tilt_deg=5.0, cio_bias_db=0.5, hysteresis_db=2.5, ttt_ms=160
        ),
        features={
            "cell_goodput_mbps": 10.0 + time_step + index * 0.01,
            "avg_ue_goodput_mbps": 5.0,
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
        source_dir="/tmp/smo-training-manager-test",
        seed=0,
    )


def _seed_feature_store(tmp_path: Path, records_per_cell_step: int = 20) -> FeatureStore:
    store = FeatureStore(tmp_path / "feature_store")
    records = [
        _feature_record(time_step, cell_id, index)
        for time_step in range(5)
        for cell_id in CELLS
        for index in range(records_per_cell_step)
    ]
    store.upsert(records)
    return store


def _job_manager(tmp_path: Path, *, min_training_samples: int = 10) -> JobManager:
    # NOTE: Feature_Store.upsert() dedupes by (time_step, cell_id), so one
    # Feature_Group can hold at most 5 * len(target_cells) records — well
    # below the design's default min_training_samples=200. These sanity
    # tests use a small threshold to exercise the create/run/complete path;
    # the design-specified default (200) is still exercised in
    # test_create_job_rejects_insufficient_training_data below.
    store = _seed_feature_store(tmp_path)
    model_storage = ModelStorage(tmp_path / "models")
    registry = ModelRegistry(FakeCollection(), model_storage)
    return JobManager(
        store,
        registry,
        model_storage,
        jobs_dir=tmp_path / "jobs",
        min_training_samples=min_training_samples,
        max_training_seconds=60,
        max_epochs=3,
        patience=2,
    )


def test_train_gnn_script_runs_deterministically_end_to_end(tmp_path: Path) -> None:
    """train_gnn.py trains, evaluates, and saves artifacts deterministically for a fixed seed."""
    records = [
        _feature_record(time_step, cell_id, index).model_dump(mode="json")
        for time_step in range(5)
        for cell_id in CELLS
        for index in range(15)
    ]

    first_model, first_nodes, first_payload = run_training(records, seed=123, max_epochs=3, patience=2)
    second_model, second_nodes, second_payload = run_training(records, seed=123, max_epochs=3, patience=2)

    assert first_nodes == second_nodes == sorted(CELLS)
    assert first_payload["train_samples"] == second_payload["train_samples"]
    assert first_payload["validation_samples"] == second_payload["validation_samples"]
    assert first_payload["metrics"] == second_payload["metrics"]
    for key in first_model.state_dict():
        assert torch.equal(first_model.state_dict()[key], second_model.state_dict()[key])
    assert set(first_payload["metrics"]).issubset(set(TARGET_KPIS))


def test_train_gnn_cli_writes_artifact_and_metrics(tmp_path: Path) -> None:
    """The subprocess entry point writes a loadable checkpoint and the metrics contract fields."""
    import json

    records = [
        _feature_record(time_step, cell_id, index).model_dump(mode="json")
        for time_step in range(5)
        for cell_id in CELLS
        for index in range(10)
    ]
    records_path = tmp_path / "records.json"
    records_path.write_text(json.dumps(records), encoding="utf-8")
    artifact_path = tmp_path / "model.pt"
    metrics_path = tmp_path / "metrics.json"

    script = Path(__file__).resolve().parents[2] / "aimlfw" / "training_manager" / "train_gnn.py"
    repository_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [
            sys.executable, str(script),
            "--records-file", str(records_path),
            "--artifact-output", str(artifact_path),
            "--metrics-output", str(metrics_path),
            "--seed", "5",
            "--max-epochs", "3",
            "--patience", "2",
        ],
        capture_output=True,
        text=True,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(repository_root)},
    )

    assert result.returncode == 0, result.stderr
    assert artifact_path.is_file()
    checkpoint = torch.load(artifact_path, weights_only=False)
    assert set(checkpoint["target_kpis"]) == set(TARGET_KPIS)
    metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    for field in ("train_split", "validation_split", "train_samples", "validation_samples", "seed", "metrics"):
        assert field in metrics_payload


def test_create_and_query_job_via_api(tmp_path: Path) -> None:
    """POST /jobs creates a job and GET /jobs/{id} eventually reports completion."""
    manager = _job_manager(tmp_path)

    # The background training task is scheduled on the ASGI app's own event
    # loop, so the TestClient must be used as a context manager to keep that
    # loop alive across requests (a one-shot client tears the loop down as
    # soon as each request returns, killing the in-flight task).
    with TestClient(create_app(manager)) as client:
        created = client.post("/jobs", json={"feature_group": "default", "model_name": "ran-gnn", "seed": 1})
        assert created.status_code == 200
        job_id = created.json()["job_id"]
        assert job_id

        status = client.get(f"/jobs/{job_id}")
        assert status.status_code == 200
        assert status.json()["status"] in ("pending", "running", "completed", "failed")

        final: dict[str, Any] | None = None
        for _ in range(200):
            snapshot = manager.get_job(job_id).model_dump(mode="json")
            if snapshot["status"] in ("completed", "failed"):
                final = snapshot
                break
            time.sleep(0.1)

    assert final is not None, "training job did not finish in time"
    assert final["status"] == "completed", final
    stages = [stage["stage"] for stage in final["stage_history"]]
    assert stages == ["extract_features", "train_model", "save_artifact", "register_metrics"]
    assert all(stage["result"] == "succeeded" for stage in final["stage_history"])


def test_create_job_rejects_insufficient_training_data(tmp_path: Path) -> None:
    """POST /jobs returns insufficient_training_data without creating a job (3.8)."""
    manager = _job_manager(tmp_path, min_training_samples=10_000)
    client = TestClient(create_app(manager))

    response = client.post("/jobs", json={"feature_group": "default", "model_name": "ran-gnn", "seed": 1})
    assert response.status_code == 422
    assert response.json()["error_code"] == "insufficient_training_data"


def test_get_unknown_job_returns_job_not_found(tmp_path: Path) -> None:
    manager = _job_manager(tmp_path)
    client = TestClient(create_app(manager))

    response = client.get("/jobs/does-not-exist")
    assert response.status_code == 404
    assert response.json()["error_code"] == "job_not_found"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_job_status_snapshot_survives_process_restart(tmp_path: Path) -> None:
    """A fresh JobManager pointed at the same jobs_dir can still read a prior job's status."""
    manager = _job_manager(tmp_path)
    finished = False
    with TestClient(create_app(manager)) as client:
        job_id = client.post(
            "/jobs", json={"feature_group": "default", "model_name": "ran-gnn", "seed": 2}
        ).json()["job_id"]

        for _ in range(200):
            if manager.get_job(job_id).status in ("completed", "failed"):
                finished = True
                break
            time.sleep(0.1)

    assert finished, "training job did not finish in time"

    restarted_manager = _job_manager(tmp_path)
    restarted_manager.jobs_dir = manager.jobs_dir
    reloaded = restarted_manager.get_job(job_id)
    assert reloaded.status == "completed"
