"""Property 35 test module (own module to avoid collisions with parallel Property tasks).

Task 5.8: 모델 미로드 상태 전이.

Targets ``smo.aimlfw.inference_service.engine.InferenceEngine`` (Task 5.1),
as specified in requirements.md:

Requirement 5.11:
    IF Inference_Service가 모델 산출물을 로드하지 못했거나 로드가 60초를 초과하면,
    THEN THE Inference_Service SHALL 상태 조회 응답에서 모델 로드 상태를
    `not_loaded` 로 반환하고, 이 시점 이후 도착하는 모든 Batch_Prediction_Request에
    대해 예측을 수행하지 않고 `model_not_loaded` 오류를 반환한다.

Design tag (design.md "Inference_Service (Requirement 5)"):

    #### Property 35: 모델 미로드 상태 전이
    *For any* 모델 산출물 로드가 실패했거나 60초를 초과한 상태, 상태 조회 응답의
    모델 로드 상태는 `not_loaded`이고, 이후 도착하는 모든 Batch_Prediction_Request는
    예측을 수행하지 않고 `model_not_loaded` 오류를 반환한다.
    **Validates: Requirements 5.11**

Oracle grounding: ``engine.py``'s ``InferenceEngine.load`` wraps
``asyncio.to_thread(self._load_sync)`` in ``asyncio.wait_for(..., timeout=self.load_timeout_seconds)``
and catches *any* exception (registry lookup failure, corrupt/missing artifact,
or a ``asyncio.TimeoutError``), leaving ``model_load_status == "not_loaded"``
rather than raising. ``predict_batch`` checks ``model_load_status != "loaded"``
*first*, before batch size/time_step/parameter validation, and raises
``InferenceServiceError("model_not_loaded", ...)`` -- Requirement 5.11 states
this must hold for *every* Batch_Prediction_Request while the model is
unavailable, regardless of what else might be wrong with the request.
``server.py`` exposes this as ``GET /status`` (``StatusResponse``) and
``POST /predict/batch`` (503 via the ``InferenceServiceError`` handler).

This test drives three distinct ways ``load()`` can fail to reach ``loaded``
(registry lookup raising, corrupt/missing artifact, and a real timeout via a
tiny configurable ``load_timeout_seconds`` plus a slow synchronous registry
call -- following the ``test_property_21`` technique of keeping every
duration well under one second while still exercising the real
``asyncio.wait_for`` timeout mechanism), plus a positive-control case with a
real artifact to guard against the not_loaded assertions being vacuously true.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, TARGET_KPIS
from smo.aimlfw.inference_service.engine import InferenceEngine
from smo.aimlfw.inference_service.errors import InferenceServiceError
from smo.aimlfw.inference_service.schemas import ParameterSetInput
from smo.aimlfw.inference_service.server import create_app
from smo.aimlfw.training_manager.train_gnn import CellGraphGnn
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS, valid_parameter_sets

_IN_DIM = len(CONTROL_PARAMETER_NAMES) + 1
_OUT_DIM = len(TARGET_KPIS)
_MODEL_NAME = "gnn"

# All sleeps/timeouts stay well under one second, following test_property_21's
# technique of exercising the real timeout *mechanism* rather than waiting out
# a real 60-second load.
_SLOW_REGISTRY_SLEEP_SECONDS = 0.3
_TINY_LOAD_TIMEOUTS = st.floats(min_value=0.01, max_value=0.05, allow_nan=False, allow_infinity=False)


class _FakeRecord:
    """Minimal stand-in exposing only the attributes ``_load_sync`` reads."""

    def __init__(self, artifact_uri: str, version: int) -> None:
        self.artifact_uri = artifact_uri
        self.version = version


class _RaisingRegistry:
    """A stand-in ``ModelRegistry`` whose lookup always raises (case 1a)."""

    def latest(self, _model_name: str) -> _FakeRecord:
        raise RuntimeError("model registry lookup failed")

    def get(self, _model_name: str, _version: int) -> _FakeRecord:
        raise RuntimeError("model registry lookup failed")


class _CorruptArtifactRegistry:
    """A stand-in ``ModelRegistry`` that resolves but points at a bad artifact (case 1b)."""

    def __init__(self, artifact_uri: str, version: int = 1) -> None:
        self.artifact_uri = artifact_uri
        self.version = version

    def latest(self, _model_name: str) -> _FakeRecord:
        return _FakeRecord(artifact_uri=self.artifact_uri, version=self.version)

    def get(self, _model_name: str, _version: int) -> _FakeRecord:
        return self.latest(_model_name)


class _SlowRegistry:
    """A stand-in ``ModelRegistry`` whose lookup blocks synchronously (case 2).

    ``InferenceEngine._load_sync`` runs on a worker thread via
    ``asyncio.to_thread``, so a plain ``time.sleep`` here genuinely blocks
    that thread without blocking the event loop -- ``asyncio.wait_for`` still
    fires on schedule.
    """

    def __init__(self, artifact_uri: str, version: int = 1) -> None:
        self.artifact_uri = artifact_uri
        self.version = version

    def latest(self, _model_name: str) -> _FakeRecord:
        time.sleep(_SLOW_REGISTRY_SLEEP_SECONDS)
        return _FakeRecord(artifact_uri=self.artifact_uri, version=self.version)

    def get(self, _model_name: str, _version: int) -> _FakeRecord:
        return self.latest(_model_name)


class _WorkingRegistry:
    """A stand-in ``ModelRegistry`` that resolves to a real, loadable artifact (case 3)."""

    def __init__(self, artifact_uri: str, version: int = 1) -> None:
        self.artifact_uri = artifact_uri
        self.version = version

    def latest(self, _model_name: str) -> _FakeRecord:
        return _FakeRecord(artifact_uri=self.artifact_uri, version=self.version)

    def get(self, _model_name: str, _version: int) -> _FakeRecord:
        return self.latest(_model_name)


def _write_real_artifact(root: Path) -> str:
    """Write a real, loadable ``CellGraphGnn`` checkpoint and return its path."""
    model = CellGraphGnn(_IN_DIM, _OUT_DIM)
    artifact_path = root / "model.pt"
    torch.save({"state_dict": model.state_dict()}, artifact_path)
    return str(artifact_path)


def _assert_not_loaded_status(engine: InferenceEngine) -> None:
    assert engine.model_load_status == "not_loaded"
    with TestClient(create_app(engine)) as client:
        response = client.get("/status")
    assert response.status_code == 200
    assert response.json() == {
        "model_load_status": "not_loaded",
        "model_name": None,
        "model_version": None,
    }


def _assert_predict_batch_rejects_with_model_not_loaded(
    engine: InferenceEngine, parameter_sets: list[ParameterSetInput]
) -> None:
    # Direct engine call: no prediction performed, no side effects.
    with pytest.raises(InferenceServiceError) as engine_error:
        engine.predict_batch(parameter_sets, None)
    assert engine_error.value.error_code == "model_not_loaded"
    assert "predictions" not in engine_error.value.details
    assert engine.model_load_status == "not_loaded"

    # App boundary: same error code, no partial/predictions payload, 503 status.
    with TestClient(create_app(engine)) as client:
        response = client.post(
            "/predict/batch",
            json={"parameter_sets": [ps.model_dump(mode="json") for ps in parameter_sets], "time_step": None},
        )
    assert response.status_code == 503
    body = response.json()
    assert body["error_code"] == "model_not_loaded"
    assert "predictions" not in body


def _to_input(parameter_set: Any) -> ParameterSetInput:
    return ParameterSetInput.model_validate(parameter_set.model_dump(mode="python"))


# **Property 35: 모델 미로드 상태 전이**
# **Validates: Requirements 5.11**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    parameter_sets=st.lists(valid_parameter_sets(), min_size=2, max_size=3),
)
def test_registry_lookup_failure_leaves_model_not_loaded(parameter_sets: list[Any]) -> None:
    """Case 1a: ``ModelRegistry.latest()`` raising must land in not_loaded, never loaded."""
    engine = InferenceEngine(_RaisingRegistry(), _MODEL_NAME, load_timeout_seconds=1.0)
    asyncio.run(engine.load())

    _assert_not_loaded_status(engine)
    inputs = [_to_input(ps) for ps in parameter_sets]
    for parameter_set in inputs:
        _assert_predict_batch_rejects_with_model_not_loaded(engine, [parameter_set])


# **Property 35: 모델 미로드 상태 전이**
# **Validates: Requirements 5.11**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    parameter_sets=st.lists(valid_parameter_sets(), min_size=2, max_size=3),
)
def test_corrupt_artifact_leaves_model_not_loaded(parameter_sets: list[Any]) -> None:
    """Case 1b: a resolvable record whose artifact_uri is missing/corrupt must not_loaded."""
    with TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        # A path that does not exist at all -- torch.load must fail inside _load_sync.
        missing_artifact = str(root / "does_not_exist.pt")
        engine = InferenceEngine(
            _CorruptArtifactRegistry(missing_artifact), _MODEL_NAME, load_timeout_seconds=1.0
        )
        asyncio.run(engine.load())

        _assert_not_loaded_status(engine)
        inputs = [_to_input(ps) for ps in parameter_sets]
        for parameter_set in inputs:
            _assert_predict_batch_rejects_with_model_not_loaded(engine, [parameter_set])


# **Property 35: 모델 미로드 상태 전이**
# **Validates: Requirements 5.11**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    load_timeout_seconds=_TINY_LOAD_TIMEOUTS,
    parameter_sets=st.lists(valid_parameter_sets(), min_size=2, max_size=3),
)
def test_load_timeout_leaves_model_not_loaded(load_timeout_seconds: float, parameter_sets: list[Any]) -> None:
    """Case 2: a load exceeding load_timeout_seconds must not_loaded, never loaded.

    ``_SlowRegistry.latest`` always sleeps far longer than any generated
    ``load_timeout_seconds`` (all well under one second), so every example
    deterministically exercises the ``asyncio.wait_for`` timeout path.
    """
    with TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        artifact_uri = _write_real_artifact(root)  # a valid artifact -- only the lookup is slow
        engine = InferenceEngine(
            _SlowRegistry(artifact_uri), _MODEL_NAME, load_timeout_seconds=load_timeout_seconds
        )

        async def _load_and_time() -> float:
            # Measure elapsed time *inside* the event loop, before asyncio.run's
            # shutdown phase (which waits for the still-sleeping executor thread
            # backing the abandoned _load_sync call) can inflate the measurement.
            started = time.monotonic()
            await engine.load()
            return time.monotonic() - started

        elapsed = asyncio.run(_load_and_time())

        # The timeout mechanism actually fired -- load() returned close to
        # load_timeout_seconds, not after the full slow-registry sleep.
        assert elapsed < _SLOW_REGISTRY_SLEEP_SECONDS

        _assert_not_loaded_status(engine)
        inputs = [_to_input(ps) for ps in parameter_sets]
        for parameter_set in inputs:
            _assert_predict_batch_rejects_with_model_not_loaded(engine, [parameter_set])


# **Property 35: 모델 미로드 상태 전이**
# **Validates: Requirements 5.11**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(parameter_sets=st.lists(valid_parameter_sets(), min_size=1, max_size=3))
def test_successful_load_reaches_loaded_and_predicts(parameter_sets: list[Any]) -> None:
    """Case 3 (positive control): a real artifact must reach loaded and predict normally.

    This guards against the not_loaded assertions in the other cases being
    vacuously true because of a broken test harness rather than genuine
    not_loaded behavior.
    """
    with TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        artifact_uri = _write_real_artifact(root)
        engine = InferenceEngine(_WorkingRegistry(artifact_uri), _MODEL_NAME, load_timeout_seconds=5.0)
        asyncio.run(engine.load())

        assert engine.model_load_status == "loaded"
        assert engine.loaded_model_name == _MODEL_NAME
        assert engine.loaded_model_version == 1

        with TestClient(create_app(engine)) as client:
            status_response = client.get("/status")
            assert status_response.json() == {
                "model_load_status": "loaded",
                "model_name": _MODEL_NAME,
                "model_version": 1,
            }

        inputs = [_to_input(ps) for ps in parameter_sets]
        predictions = engine.predict_batch(inputs, None)
        assert len(predictions) == len(inputs)
        for index, prediction in enumerate(predictions):
            assert prediction["index"] == index

        with TestClient(create_app(engine)) as client:
            predict_response = client.post(
                "/predict/batch",
                json={
                    "parameter_sets": [ps.model_dump(mode="json") for ps in inputs],
                    "time_step": None,
                },
            )
        assert predict_response.status_code == 200
        assert len(predict_response.json()["predictions"]) == len(inputs)
