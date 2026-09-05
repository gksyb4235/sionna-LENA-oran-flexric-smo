"""Task 5.10: 추론 서비스 로딩·성능·실패 통합 테스트.

Exercises the real ``smo.aimlfw.inference_service`` stack (task 5.1) end to
end against a real registered artifact: a real ``CellGraphGnn`` trained via
``train_gnn.run_training`` (Training_Manager, task 3.3), saved through the
real ``ModelStorage`` (task 3.1), and registered through the real
``ModelRegistry`` (task 3.2). MongoDB is not reachable in this environment
(the same situation ``test_training_job_model_registry_integration.py``
documents), so ``ModelRegistry`` is wired to the same minimal in-memory
``_FakeCollection``/``_FakeCursor`` stand-in used there and by the Property
16/17/18/20/22-28 test modules -- that stand-in implements exactly the
``create_index``/``find_one``/``find``/``update_one``/``insert_one`` surface
``ModelRegistry`` calls, so this remains a faithful integration test of the
registry's real logic.

This test focuses on the SLA/lifecycle contract of Requirement 5 rather than
re-deriving every property already covered by Properties 29-36
(``smo/tests/property/test_property_29_*.py`` through
``test_property_36_*.py``):

  * Requirement 5.1: starting the real ``InferenceEngine`` against the real
    registered artifact reaches ``model_load_status == "loaded"`` with the
    correct model name/version, and does so quickly (not merely "under 60s",
    which would be vacuous for a tiny model).
  * Requirement 5.6: a single ``Batch_Prediction_Request`` with exactly 64
    Parameter_Set entries returns 200 with 64 predictions within 5.0 seconds
    (``@pytest.mark.performance``, per this codebase's convention for
    single-shot SLA checks -- see ``conftest.py``'s marker registration).
  * Requirement 5.11: a load that exceeds its configured timeout leaves
    ``/status`` at ``not_loaded`` and ``POST /predict/batch`` returns
    ``model_not_loaded`` with no predictions. The 60-second-class timeout is
    exercised via the same tiny-timeout-plus-slow-registry technique as
    ``test_property_35_model_not_loaded_state_transition.py`` (durations well
    under one second), and the production default of 60.0 seconds is
    separately confirmed as a static regression guard on the "60초" figure.
  * Requirement 5.10: whenever ``POST /predict/batch`` returns a non-200
    response in any scenario exercised here (not_loaded, batch_too_large,
    parameter_out_of_range), the JSON body never contains a ``predictions``
    key.

**Validates: Requirements 5.1, 5.6, 5.10, 5.11**
"""

from __future__ import annotations

import asyncio
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient

from smo.aimlfw.common.constants import TARGET_KPIS, TTT_ALLOWED_MS
from smo.aimlfw.inference_service.engine import DEFAULT_LOAD_TIMEOUT_SECONDS, InferenceEngine
from smo.aimlfw.inference_service.schemas import MAX_BATCH_SIZE
from smo.aimlfw.inference_service.server import create_app
from smo.aimlfw.model_registry import ModelRegistry
from smo.aimlfw.model_storage import ModelStorage
from smo.aimlfw.training_manager.train_gnn import run_training

_MODEL_NAME = "gnn-task-5-10"
_FEATURE_GROUP = "default"
_SEED = 11
_CELLS = ("gNB_5G", "gNB_4G_1", "gNB_4G_2")
_BATCH_SLA_SECONDS = 5.0
_LOAD_SLA_SECONDS = 2.0  # well under the 60s bound; guards against artificial stalling.

# All sleeps/timeouts stay well under one second, following test_property_21/35's
# technique of exercising the real asyncio.wait_for timeout *mechanism* rather
# than waiting out a real 60-second load.
_SLOW_REGISTRY_SLEEP_SECONDS = 0.3
_TINY_LOAD_TIMEOUT_SECONDS = 0.02


# --------------------------------------------------------------------------
# In-memory MongoDB collection stand-in (same pattern as
# test_training_job_model_registry_integration.py and the Property
# 16/17/18/20/22-28 test modules under smo/tests/property/).
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
# Real-artifact fixture helpers: train_gnn.run_training -> ModelStorage ->
# ModelRegistry, exactly the pipeline Training_Manager drives in production
# (see jobs.py's _train_model/_save_artifact/_register_metrics_with_retry).
# --------------------------------------------------------------------------


def _training_records() -> list[dict[str, Any]]:
    """Build a small, real Feature_Record-shaped input for run_training."""
    records: list[dict[str, Any]] = []
    for time_step in range(5):
        for index, cell_id in enumerate(_CELLS):
            records.append(
                {
                    "time_step": time_step,
                    "cell_id": cell_id,
                    "control_parameters": {
                        "tx_power_dbm": 40.0,
                        "ret_tilt_deg": 5.0,
                        "cio_bias_db": 0.5,
                        "hysteresis_db": 2.0,
                        "ttt_ms": 160,
                    },
                    "features": {kpi: 10.0 + time_step + index * 0.1 for kpi in TARGET_KPIS},
                }
            )
    return records


def _register_real_artifact(tmp_path: Path) -> ModelRegistry:
    """Train a real tiny CellGraphGnn and register it through real Model_Storage/Registry."""
    model, _node_order, payload = run_training(
        _training_records(), seed=_SEED, max_epochs=3, patience=2, learning_rate=0.05
    )
    artifact_path = tmp_path / "artifact.pt"
    torch.save({"state_dict": model.state_dict()}, artifact_path)

    model_storage = ModelStorage(tmp_path / "models")
    registry = ModelRegistry(_FakeCollection(), model_storage)
    artifact_uri = model_storage.save_artifact(_MODEL_NAME, 1, artifact_path)
    registry.register(
        _MODEL_NAME,
        _FEATURE_GROUP,
        payload["metrics"],
        artifact_uri,
        train_split=payload["train_split"],
        validation_split=payload["validation_split"],
        train_samples=payload["train_samples"],
        validation_samples=payload["validation_samples"],
        seed=_SEED,
    )
    return registry


def _valid_cell_parameters(index: int) -> dict[str, Any]:
    """Deterministically vary one range/step-valid CellParametersInput payload."""
    return {
        "tx_power_dbm": float(30 + index % 17),
        "ret_tilt_deg": float(index % 16),
        "cio_bias_db": -6.0 + 0.5 * (index % 25),
        "hysteresis_db": 0.5 * (index % 21),
        "ttt_ms": TTT_ALLOWED_MS[index % len(TTT_ALLOWED_MS)],
    }


def _parameter_set_payload(index: int) -> dict[str, Any]:
    return {"cells": {"cell_0": _valid_cell_parameters(index)}}


@pytest.fixture
def loaded_engine(tmp_path: Path) -> InferenceEngine:
    """A real InferenceEngine, loaded against a real registered artifact."""
    registry = _register_real_artifact(tmp_path)
    engine = InferenceEngine(registry, _MODEL_NAME)  # production default load_timeout_seconds
    asyncio.run(engine.load())
    assert engine.model_load_status == "loaded", "fixture setup must reach loaded before tests run"
    return engine


# --------------------------------------------------------------------------
# Requirement 5.1: real registered artifact load
# --------------------------------------------------------------------------


def test_engine_loads_real_registered_artifact_and_status_reports_loaded(tmp_path: Path) -> None:
    """Requirement 5.1: loading a real registered artifact reaches 'loaded' quickly."""
    registry = _register_real_artifact(tmp_path)
    engine = InferenceEngine(registry, _MODEL_NAME)  # production default load_timeout_seconds

    started = time.monotonic()
    asyncio.run(engine.load())
    elapsed = time.monotonic() - started

    assert engine.model_load_status == "loaded"
    assert engine.loaded_model_name == _MODEL_NAME
    assert engine.loaded_model_version == 1
    assert elapsed < _LOAD_SLA_SECONDS, (
        f"loading a tiny registered artifact took {elapsed:.3f}s -- the mechanism "
        "should not artificially stall well below the 60s bound (Requirement 5.1)"
    )

    with TestClient(create_app(engine)) as client:
        response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {
        "model_load_status": "loaded",
        "model_name": _MODEL_NAME,
        "model_version": 1,
    }


# --------------------------------------------------------------------------
# Requirement 5.6: 64-request 5-second SLA
# --------------------------------------------------------------------------


@pytest.mark.performance
def test_64_parameter_set_batch_completes_within_5_second_sla(loaded_engine: InferenceEngine) -> None:
    """Requirement 5.6: a 64-Parameter_Set batch returns 200 with 64 predictions within 5s."""
    request_body = {
        "parameter_sets": [_parameter_set_payload(index) for index in range(MAX_BATCH_SIZE)],
        "time_step": None,
    }
    assert len(request_body["parameter_sets"]) == 64

    with TestClient(create_app(loaded_engine)) as client:
        started = time.monotonic()
        response = client.post("/predict/batch", json=request_body)
        elapsed = time.monotonic() - started

    assert response.status_code == 200, response.text
    body = response.json()
    assert len(body["predictions"]) == 64
    assert [prediction["index"] for prediction in body["predictions"]] == list(range(64))
    assert elapsed <= _BATCH_SLA_SECONDS, (
        f"64-Parameter_Set batch took {elapsed:.3f}s, exceeding the "
        f"{_BATCH_SLA_SECONDS:.0f}s SLA (Requirement 5.6)"
    )


# --------------------------------------------------------------------------
# Requirement 5.11: 60-second-class load failure -> not_loaded + model_not_loaded
# --------------------------------------------------------------------------


class _FakeRecord:
    """Minimal stand-in exposing only the attributes ``_load_sync`` reads."""

    def __init__(self, artifact_uri: str, version: int) -> None:
        self.artifact_uri = artifact_uri
        self.version = version


class _SlowRegistry:
    """A stand-in ``ModelRegistry`` whose lookup blocks synchronously.

    ``InferenceEngine._load_sync`` runs on a worker thread via
    ``asyncio.to_thread``, so a plain ``time.sleep`` here genuinely blocks
    that thread without blocking the event loop -- ``asyncio.wait_for`` still
    fires on schedule (same technique as test_property_35).
    """

    def __init__(self, artifact_uri: str, version: int = 1) -> None:
        self.artifact_uri = artifact_uri
        self.version = version

    def latest(self, _model_name: str) -> _FakeRecord:
        time.sleep(_SLOW_REGISTRY_SLEEP_SECONDS)
        return _FakeRecord(self.artifact_uri, self.version)

    def get(self, _model_name: str, _version: int) -> _FakeRecord:
        return self.latest(_model_name)


def test_load_exceeding_timeout_leaves_not_loaded_and_predict_batch_rejects(tmp_path: Path) -> None:
    """Requirement 5.11 mechanism: a load exceeding its configured timeout leaves
    /status at not_loaded, and /predict/batch returns model_not_loaded with no
    predictions -- exercised via a tiny timeout so the test stays well under a
    second while still driving the real asyncio.wait_for timeout path."""
    registry = _register_real_artifact(tmp_path)
    latest = registry.latest(_MODEL_NAME)
    slow_registry = _SlowRegistry(latest.artifact_uri, latest.version)
    engine = InferenceEngine(slow_registry, _MODEL_NAME, load_timeout_seconds=_TINY_LOAD_TIMEOUT_SECONDS)

    async def _load_and_time() -> float:
        # Measure elapsed time *inside* the event loop, before asyncio.run's
        # shutdown phase (which waits for the still-sleeping executor thread
        # backing the abandoned _load_sync call) can inflate the measurement
        # (same technique as test_property_35_model_not_loaded_state_transition.py).
        started = time.monotonic()
        await engine.load()
        return time.monotonic() - started

    elapsed = asyncio.run(_load_and_time())

    assert elapsed < _SLOW_REGISTRY_SLEEP_SECONDS, "the timeout mechanism must fire, not the full slow sleep"
    assert engine.model_load_status == "not_loaded"

    with TestClient(create_app(engine)) as client:
        status_response = client.get("/status")
        assert status_response.json() == {
            "model_load_status": "not_loaded",
            "model_name": None,
            "model_version": None,
        }

        predict_response = client.post(
            "/predict/batch",
            json={"parameter_sets": [_parameter_set_payload(0)], "time_step": None},
        )
    assert predict_response.status_code == 503
    body = predict_response.json()
    assert body["error_code"] == "model_not_loaded"
    assert "predictions" not in body


def test_production_default_load_timeout_is_60_seconds(tmp_path: Path) -> None:
    """Static regression guard on the "60초" figure itself (Requirement 5.1/5.11),
    distinct from the timeout mechanism test above -- no real or simulated wait."""
    assert DEFAULT_LOAD_TIMEOUT_SECONDS == 60.0

    registry = _register_real_artifact(tmp_path)
    engine = InferenceEngine(registry, _MODEL_NAME)  # load_timeout_seconds not overridden
    assert engine.load_timeout_seconds == 60.0


# --------------------------------------------------------------------------
# Requirement 5.10: absence of partial results on failure
# --------------------------------------------------------------------------


def test_not_loaded_failure_response_has_no_predictions_key(tmp_path: Path) -> None:
    """Requirement 5.10: a model_not_loaded (503) response never contains 'predictions'."""
    registry = _register_real_artifact(tmp_path)
    latest = registry.latest(_MODEL_NAME)
    slow_registry = _SlowRegistry(latest.artifact_uri, latest.version)
    engine = InferenceEngine(slow_registry, _MODEL_NAME, load_timeout_seconds=_TINY_LOAD_TIMEOUT_SECONDS)
    asyncio.run(engine.load())
    assert engine.model_load_status == "not_loaded"

    with TestClient(create_app(engine)) as client:
        response = client.post(
            "/predict/batch",
            json={"parameter_sets": [_parameter_set_payload(0)], "time_step": None},
        )
    assert response.status_code != 200
    assert "predictions" not in response.json()


def test_batch_too_large_failure_response_has_no_predictions_key(loaded_engine: InferenceEngine) -> None:
    """Requirement 5.10: a batch_too_large (422) response never contains 'predictions'."""
    request_body = {
        "parameter_sets": [_parameter_set_payload(index) for index in range(MAX_BATCH_SIZE + 1)],
        "time_step": None,
    }
    with TestClient(create_app(loaded_engine)) as client:
        response = client.post("/predict/batch", json=request_body)
    assert response.status_code != 200
    body = response.json()
    assert body["error_code"] == "batch_too_large"
    assert "predictions" not in body


def test_parameter_out_of_range_failure_response_has_no_predictions_key(loaded_engine: InferenceEngine) -> None:
    """Requirement 5.10: a parameter_out_of_range (422) response never contains 'predictions'."""
    invalid_payload = _parameter_set_payload(0)
    invalid_payload["cells"]["cell_0"]["tx_power_dbm"] = 100.0  # outside the 30..46 allowed range

    with TestClient(create_app(loaded_engine)) as client:
        response = client.post(
            "/predict/batch",
            json={"parameter_sets": [invalid_payload], "time_step": None},
        )
    assert response.status_code != 200
    body = response.json()
    assert body["error_code"] == "parameter_out_of_range"
    assert "predictions" not in body
