"""Property 32 test module (own module to avoid collisions with parallel Property tasks).

Task 5.5: 배치 크기 경계 오류.

Targets ``smo.aimlfw.inference_service.engine.InferenceEngine.predict_batch``
(Task 5.1), as specified in requirements.md via design.md's Inference_Service
section (Requirement 5).

Design tag (design.md "Inference_Service (Requirement 5)"):

    #### Property 32: 배치 크기 경계 오류
    *For any* Batch_Prediction_Request의 Parameter_Set 개수, 개수가 0이면
    `empty_batch` 오류를, 64를 초과하면 `batch_too_large` 오류를 반환하고
    예측을 수행하지 않으며, 1 이상 64 이하이면 정상 처리된다.
    **Validates: Requirements 5.6, 5.7**

Oracle grounding: ``engine.py``'s ``InferenceEngine.predict_batch`` checks, in
fixed order, model-loaded -> batch size -> Time_Step validity ->
Control_Parameter range/step. The batch-size check raises
``InferenceServiceError("empty_batch", ..., {"received": 0})`` when
``len(parameter_sets) == 0``, and
``InferenceServiceError("batch_too_large", ..., {"received": len(parameter_sets)})``
when ``len(parameter_sets) > MAX_BATCH_SIZE`` (``MAX_BATCH_SIZE = 64`` in
``schemas.py``). Both raise *before* any prediction is computed, so the
error itself carries no partial ``predictions`` field. Sizes strictly
between 0 and 64 (inclusive) reach the forward pass and return exactly one
``PredictionResult``-shaped dict per Parameter_Set.

Performance: constructing a real ``CellGraphGnn`` and running a forward pass
per Hypothesis example is the expensive part, not building the small
``ParameterSet`` payloads. Following the approach used for Property 30, the
``InferenceEngine`` (and its untrained but structurally valid model) is
built exactly once at module import time and marked ``loaded`` directly,
bypassing the real Model_Registry/Model_Storage/``torch.load`` path (which
Property 29/35 cover elsewhere). Each Hypothesis example only varies the
batch *size*; the single-cell ``ParameterSet`` payload is a fixed, cheaply
constructed value that is reused (not reconstructed or diversified) across
every repeated entry and every example, since Property 32 is purely about
batch-size boundaries, not batch-content diversity (Property 29's concern).
"""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, TARGET_KPIS
from smo.aimlfw.common.models import CellParameters, ParameterSet
from smo.aimlfw.inference_service.engine import InferenceEngine
from smo.aimlfw.inference_service.errors import InferenceServiceError
from smo.aimlfw.inference_service.schemas import MAX_BATCH_SIZE
from smo.aimlfw.training_manager.train_gnn import CellGraphGnn
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

_IN_DIM = len(CONTROL_PARAMETER_NAMES) + 1  # +1 for the normalized Time_Step input (mirrors engine.py)
_OUT_DIM = len(TARGET_KPIS)

# A single, fixed, range/step-valid one-cell Parameter_Set. Cheap to construct
# and reused (not rebuilt) across every batch entry and every Hypothesis
# example -- only the batch *size* varies in this module.
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


def _build_loaded_engine() -> InferenceEngine:
    """Build one ``InferenceEngine`` with a structurally valid model already
    marked ``loaded``, bypassing Model_Registry/Model_Storage/``torch.load``
    entirely (those paths are Property 29/35's concern, not this one's)."""
    engine = InferenceEngine(model_registry=None, model_name="gnn")  # type: ignore[arg-type]
    engine._model = CellGraphGnn(_IN_DIM, _OUT_DIM)
    engine.model_load_status = "loaded"
    engine.loaded_model_name = "gnn"
    engine.loaded_model_version = 1
    return engine


# Built once at import time -- shared, read-only across every example in this module.
_ENGINE = _build_loaded_engine()


# **Property 32: 배치 크기 경계 오류**
# **Validates: Requirements 5.6**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(time_step=st.one_of(st.none(), st.integers(min_value=0, max_value=4)))
def test_empty_batch_returns_empty_batch_error(time_step: int | None) -> None:
    with pytest.raises(InferenceServiceError) as exc_info:
        _ENGINE.predict_batch([], time_step)

    error = exc_info.value
    assert error.error_code == "empty_batch"
    assert error.details == {"received": 0}
    # The error is raised before any prediction is computed, so there is no
    # partial-results field to check -- the exception itself carries none.
    assert not hasattr(error, "predictions")


# **Property 32: 배치 크기 경계 오류**
# **Validates: Requirements 5.7**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(batch_size=st.integers(min_value=65, max_value=70))
def test_oversized_batch_returns_batch_too_large_error(batch_size: int) -> None:
    assert batch_size > MAX_BATCH_SIZE
    parameter_sets = [_SAMPLE_PARAMETER_SET] * batch_size

    with pytest.raises(InferenceServiceError) as exc_info:
        _ENGINE.predict_batch(parameter_sets, None)

    error = exc_info.value
    assert error.error_code == "batch_too_large"
    assert error.details == {"received": batch_size}
    assert not hasattr(error, "predictions")


# **Property 32: 배치 크기 경계 오류**
# **Validates: Requirements 5.6, 5.7**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(batch_size=st.sampled_from((1, 2, 3, 32, 63, 64)))
def test_boundary_batch_sizes_are_processed_normally(batch_size: int) -> None:
    assert 1 <= batch_size <= MAX_BATCH_SIZE
    parameter_sets = [_SAMPLE_PARAMETER_SET] * batch_size

    predictions = _ENGINE.predict_batch(parameter_sets, None)

    assert len(predictions) == batch_size
