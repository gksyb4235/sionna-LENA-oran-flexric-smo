"""Property 34 test module (own module to avoid collisions with parallel Property tasks).

Task 5.7: 오류 응답 완전성과 부분결과 배제.

Targets ``smo.aimlfw.inference_service.engine.InferenceEngine.predict_batch``
and ``smo.aimlfw.inference_service.server``'s ``POST /predict/batch`` route
(Task 5.1), as specified in requirements.md:

Requirement 5.10:
    IF 예측값이 생성된 후 오류가 발생하면, THEN THE Inference_Service SHALL
    오류 응답에 사용된 모델 이름과 버전 및 오류 코드를 포함하고 부분 예측
    결과를 포함하지 않는다.

Design tag (design.md "Inference_Service (Requirement 5)"):

    #### Property 34: 오류 응답 완전성과 부분결과 배제
    *For any* 예측값 생성 후 발생한 오류, 오류 응답은 사용된 모델 이름/버전과
    오류 코드를 포함하고 부분 예측 결과를 포함하지 않는다.
    **Validates: Requirements 5.10**

Oracle grounding: before this task's fix, ``engine.py``'s ``predict_batch``
ran its per-item prediction loop (``self._predict_one(...)`` for each
Parameter_Set) with no exception handling at all. Every *validation* error
(``model_not_loaded``, ``empty_batch``, ``batch_too_large``,
``invalid_time_step``, ``parameter_out_of_range``) is raised *before* that
loop even starts, so none of them actually exercise "an error after
predictions were generated" -- the scenario Requirement 5.10 describes. If
``_predict_one`` itself ever raised for item k after items 0..k-1 already
succeeded (e.g. a corrupt/incompatible loaded artifact whose forward pass
fails on a particular input), that raw exception propagated straight out of
``predict_batch`` as-is: not a well-formed ``InferenceServiceError``, and
FastAPI's default handling of an unhandled exception is a bare 500 with no
``error_code``/``model_name``/``model_version`` at all -- a genuine 5.10 gap.

The fix wraps the per-item loop in ``predict_batch`` so any exception raised
while producing a prediction is translated into
``InferenceServiceError("prediction_failed", ..., {"model_name": ...,
"model_version": ..., "parameter_set_index": ...})``, and adds a catch-all
FastAPI exception handler as a last-resort fallback so any *other* unhandled
exception during request handling still returns a structured
``error_code``/``model_name``/``model_version`` body instead of a bare 500.

This test drives the real ``predict_batch``/``_predict_one`` code path (and,
for the HTTP-level checks, the real FastAPI route) against a fault-injecting
stand-in for the loaded model whose ``__call__`` succeeds for the first
``fail_at_call_index`` calls (producing recognizable, distinct-valued
predictions) and then raises on the call at ``fail_at_call_index`` --
guaranteeing at least one Parameter_Set in the batch was already
successfully predicted before the fault fires, exactly the "error after
predictions were generated" scenario Requirement 5.10 is about. The model
registry/storage/``torch.load`` machinery is bypassed entirely (as in
Property 32), since this property is about the response contract on
mid-batch failure, not about model loading.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, TARGET_KPIS
from smo.aimlfw.common.models import CellParameters, ParameterSet
from smo.aimlfw.inference_service.engine import InferenceEngine
from smo.aimlfw.inference_service.errors import InferenceServiceError
from smo.aimlfw.inference_service.schemas import MAX_BATCH_SIZE
from smo.aimlfw.inference_service.server import create_app
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

_IN_DIM = len(CONTROL_PARAMETER_NAMES) + 1  # +1 for the normalized Time_Step input (mirrors engine.py)
_OUT_DIM = len(TARGET_KPIS)
_MODEL_NAME = "gnn-property-34"
_MODEL_VERSION = 7

# A single, fixed, range/step-valid one-cell Parameter_Set. Only the fault
# location within the batch varies across Hypothesis examples, not the
# Parameter_Set content (batch-content diversity is Property 29's concern).
_SAMPLE_PARAMETER_SET = ParameterSet(
    cells={
        "cell_0": CellParameters(
            tx_power_dbm=40.0,
            ret_tilt_deg=5.0,
            cio_bias_db=0.0,
            hysteresis_db=2.0,
            ttt_ms=100,
        )
    }
)

# A value baked into every successfully-predicted item's KPI output so the
# test can positively assert it never leaks into an error response, rather
# than only checking for the *absence* of a generically-named field.
_LEAK_MARKER_KPI_VALUE = 123456.0


class _FaultInjectingModel:
    """Stand-in for a loaded ``CellGraphGnn`` whose forward pass fails partway
    through a batch, after at least one earlier call already succeeded."""

    def __init__(self, out_dim: int, fail_at_call_index: int) -> None:
        self.out_dim = out_dim
        self.fail_at_call_index = fail_at_call_index
        self.calls = 0

    def __call__(self, node_inputs: torch.Tensor) -> torch.Tensor:
        call_index = self.calls
        self.calls += 1
        if call_index == self.fail_at_call_index:
            raise RuntimeError("simulated forward-pass failure")
        num_nodes = node_inputs.shape[0]
        return torch.full((num_nodes, self.out_dim), _LEAK_MARKER_KPI_VALUE, dtype=torch.float32)


class _AlwaysSucceedsModel:
    """Stand-in for a loaded model that never fails, used for the happy-path check."""

    def __init__(self, out_dim: int) -> None:
        self.out_dim = out_dim
        self.calls = 0

    def __call__(self, node_inputs: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        num_nodes = node_inputs.shape[0]
        return torch.full((num_nodes, self.out_dim), _LEAK_MARKER_KPI_VALUE, dtype=torch.float32)


def _build_loaded_engine(model: Any) -> InferenceEngine:
    """Build one ``InferenceEngine`` already marked ``loaded`` with ``model``
    installed directly, bypassing Model_Registry/Model_Storage/``torch.load``
    entirely (those paths are Property 29/35's concern, not this one's)."""
    engine = InferenceEngine(model_registry=None, model_name=_MODEL_NAME)  # type: ignore[arg-type]
    engine._model = model  # type: ignore[assignment]
    engine.model_load_status = "loaded"
    engine.loaded_model_name = _MODEL_NAME
    engine.loaded_model_version = _MODEL_VERSION
    return engine


def _assert_no_partial_results_leak(payload: Any) -> None:
    """Serialize ``payload`` and confirm neither a ``predictions``/``target_kpi``/
    ``cell_kpi`` field nor the leak-marker KPI value appears anywhere in it."""
    serialized = json.dumps(payload, default=str)
    assert "predictions" not in serialized
    assert "target_kpi" not in serialized
    assert "cell_kpi" not in serialized
    assert str(_LEAK_MARKER_KPI_VALUE) not in serialized


# **Property 34: 오류 응답 완전성과 부분결과 배제**
# **Validates: Requirements 5.10**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    batch_size=st.integers(min_value=2, max_value=8),
    fail_at_call_index=st.integers(min_value=1, max_value=7),
)
def test_mid_batch_failure_raises_prediction_failed_without_partial_results(
    batch_size: int, fail_at_call_index: int
) -> None:
    """Requirement 5.10: an error raised by the per-item forward pass after at
    least one earlier Parameter_Set in the same batch already succeeded must
    surface as a well-formed InferenceServiceError carrying model_name,
    model_version, and error_code -- and must not leak any already-computed
    predictions."""
    if fail_at_call_index >= batch_size:
        return  # keep "at least one earlier item already succeeded" satisfiable.

    model = _FaultInjectingModel(_OUT_DIM, fail_at_call_index)
    engine = _build_loaded_engine(model)
    parameter_sets = [_SAMPLE_PARAMETER_SET] * batch_size

    with pytest.raises(InferenceServiceError) as exc_info:
        engine.predict_batch(parameter_sets, None)

    error = exc_info.value
    # At least one item was predicted before the fault fired.
    assert model.calls == fail_at_call_index + 1
    assert fail_at_call_index >= 1

    assert error.error_code == "prediction_failed"
    assert error.details["model_name"] == engine.loaded_model_name == _MODEL_NAME
    assert error.details["model_version"] == engine.loaded_model_version == _MODEL_VERSION
    assert error.details["parameter_set_index"] == fail_at_call_index

    # No partial results (from the successfully-predicted earlier items) leak
    # into the error at all.
    assert not hasattr(error, "predictions")
    _assert_no_partial_results_leak(error.response.model_dump(mode="json"))


# **Property 34: 오류 응답 완전성과 부분결과 배제**
# **Validates: Requirements 5.10**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    batch_size=st.integers(min_value=2, max_value=8),
    fail_at_call_index=st.integers(min_value=1, max_value=7),
)
def test_http_response_for_mid_batch_failure_has_no_partial_results(
    batch_size: int, fail_at_call_index: int
) -> None:
    """Same scenario as above, but driven through the real FastAPI
    ``POST /predict/batch`` route so the HTTP error body itself is checked
    for model_name/model_version/error_code and the absence of any
    partial-results field."""
    if fail_at_call_index >= batch_size:
        return

    model = _FaultInjectingModel(_OUT_DIM, fail_at_call_index)
    engine = _build_loaded_engine(model)
    app = create_app(engine=engine)
    client = TestClient(app)

    request_body = {
        "parameter_sets": [
            {"cells": {"cell_0": {"tx_power_dbm": 40.0, "ret_tilt_deg": 5.0, "cio_bias_db": 0.0, "hysteresis_db": 2.0, "ttt_ms": 100}}}
            for _ in range(batch_size)
        ],
        "time_step": None,
    }
    response = client.post("/predict/batch", json=request_body)

    assert response.status_code == 422, response.text
    body = response.json()
    assert body["error_code"] == "prediction_failed"
    assert body["details"]["model_name"] == _MODEL_NAME
    assert body["details"]["model_version"] == _MODEL_VERSION
    assert body["details"]["parameter_set_index"] == fail_at_call_index
    _assert_no_partial_results_leak(body)


# **Property 34: 오류 응답 완전성과 부분결과 배제**
# **Validates: Requirements 5.10**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(batch_size=st.integers(min_value=1, max_value=MAX_BATCH_SIZE))
def test_all_succeeding_batch_still_returns_full_predictions(batch_size: int) -> None:
    """Sanity check: the mid-batch-failure fix must not affect the happy path --
    a batch with no fault injected still returns exactly one prediction per
    Parameter_Set, in order."""
    model = _AlwaysSucceedsModel(_OUT_DIM)
    engine = _build_loaded_engine(model)
    parameter_sets = [_SAMPLE_PARAMETER_SET] * batch_size

    predictions = engine.predict_batch(parameter_sets, None)

    assert len(predictions) == batch_size
    assert model.calls == batch_size
    for index, prediction in enumerate(predictions):
        assert prediction["index"] == index
        assert set(prediction["target_kpi"]) == set(TARGET_KPIS)
        for value in prediction["target_kpi"].values():
            assert value == _LEAK_MARKER_KPI_VALUE
