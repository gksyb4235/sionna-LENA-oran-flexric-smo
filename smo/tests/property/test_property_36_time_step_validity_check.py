"""Property 36 test module (own module to avoid collisions with parallel Property tasks).

Task 5.9: Time_Step 유효성 검증.

Targets ``smo.aimlfw.inference_service.engine.InferenceEngine.predict_batch``
and the ``POST /predict/batch`` boundary in ``smo.aimlfw.inference_service.server``
(Task 5.1), as specified in requirements.md via design.md's Inference_Service
section (Requirement 5).

Design tag (design.md "Inference_Service (Requirement 5)"):

    #### Property 36: Time_Step 유효성 검증
    *For any* Time_Step 값이 정수가 아니거나 0 이상 4 이하 범위를 벗어나는 경우,
    Inference_Service는 예측을 수행하지 않고 수신된 값을 포함한
    `invalid_time_step` 오류를 반환한다.
    **Validates: Requirements 5.12**

Oracle grounding: ``engine.py``'s ``InferenceEngine.predict_batch`` resolves
``resolved_time_step = MIN_TIME_STEP if time_step is None else time_step``
(``None`` means "omitted", defaulting to 0 -- never an error) and then rejects
whenever ``not isinstance(resolved_time_step, int) or isinstance(resolved_time_step, bool)
or not MIN_TIME_STEP <= resolved_time_step <= MAX_TIME_STEP``, raising
``InferenceServiceError("invalid_time_step", ..., {"time_step": time_step})`` --
note the *original* received ``time_step`` is echoed in ``details``, not the
resolved/defaulted value. Python ``bool`` is a subclass of ``int``, so the
explicit ``isinstance(resolved_time_step, bool)`` guard is what keeps
``True``/``False`` out of the otherwise-passing ``isinstance(..., int)`` check.
A numerically-integral ``float`` such as ``2.0`` still fails
``isinstance(x, int)`` (Python does not consider ``float`` an ``int``
subclass), so it is rejected purely on type, matching the "정수가 아니거나"
("is not an integer") wording. This check runs *after* the model-loaded and
batch-size checks but *before* the Control_Parameter range/step check (see
``predict_batch``'s docstring for the fixed order), so no prediction is
computed once it fires.

``schemas.py``'s ``BatchPredictionRequest.time_step: Any = None`` is
deliberately untyped at the Pydantic layer specifically so malformed values
(strings, floats, lists, booleans, out-of-range integers) reach the engine
unmodified instead of being rejected earlier as a generic FastAPI 422 -- this
module's HTTP-boundary tests confirm that JSON deserialization actually
preserves the distinguishing Python type (e.g. a JSON string ``"2"`` arrives
as a Python ``str``, not coerced to ``int``; a JSON ``2.0`` arrives as a
Python ``float``) before asserting on the resulting error.

Performance: following the Property 32/35 pattern, the ``InferenceEngine``
(and its untrained but structurally valid model) is built once at module
import time and marked ``loaded`` directly, bypassing the real
Model_Registry/Model_Storage/``torch.load`` path -- Property 36 is purely
about Time_Step validation, not model loading.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, MAX_TIME_STEP, MIN_TIME_STEP, TARGET_KPIS
from smo.aimlfw.common.models import CellParameters, ParameterSet
from smo.aimlfw.inference_service.engine import InferenceEngine
from smo.aimlfw.inference_service.errors import InferenceServiceError
from smo.aimlfw.inference_service.server import create_app
from smo.aimlfw.training_manager.train_gnn import CellGraphGnn
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

_IN_DIM = len(CONTROL_PARAMETER_NAMES) + 1  # +1 for the normalized Time_Step input (mirrors engine.py)
_OUT_DIM = len(TARGET_KPIS)

# A single, fixed, range/step-valid one-cell Parameter_Set. Cheap to construct
# and reused across every example -- only Time_Step varies in this module.
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
_SAMPLE_PARAMETER_SETS = [_SAMPLE_PARAMETER_SET]

# The same Parameter_Set expressed as a JSON-serializable request body for the
# HTTP-boundary tests.
_SAMPLE_CELL_PAYLOAD = {
    "tx_power_dbm": 40.0,
    "ret_tilt_deg": 5.0,
    "cio_bias_db": 0.0,
    "hysteresis_db": 2.0,
    "ttt_ms": 100,
}
_SAMPLE_PARAMETER_SETS_PAYLOAD = [{"cells": {"cell_0": _SAMPLE_CELL_PAYLOAD}}]


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


def _assert_invalid_time_step(received_value: object) -> None:
    with pytest.raises(InferenceServiceError) as exc_info:
        _ENGINE.predict_batch(_SAMPLE_PARAMETER_SETS, received_value)

    error = exc_info.value
    assert error.error_code == "invalid_time_step"
    # The *original* received value must be echoed, not the resolved/defaulted one.
    assert error.details == {"time_step": received_value}
    # The error is raised before any prediction is computed, so there is no
    # partial-results field to check -- the exception itself carries none.
    assert not hasattr(error, "predictions")


# Deliberately includes integral-valued floats (0.0, 2.0, 4.0, ...): the
# requirement rejects on *type* ("정수가 아니거나"), so a float that happens to
# be numerically integral must still be rejected because ``isinstance(2.0, int)``
# is ``False`` in Python.
_NON_INTEGER_FLOATS = st.one_of(
    st.floats(min_value=-1_000_000.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    st.sampled_from((0.0, 1.0, 2.0, 3.0, 4.0, -1.0, 5.0, 1.5, 2.5, -0.5)),
)
_NON_INTEGER_STRINGS = st.one_of(
    st.text(max_size=20),
    st.sampled_from(("2", "0", "4", "abc", "")),
)
_NON_INTEGER_LISTS = st.lists(st.integers(), max_size=5)
_NON_INTEGER_TYPES = st.one_of(_NON_INTEGER_FLOATS, _NON_INTEGER_STRINGS, _NON_INTEGER_LISTS)

# Out-of-range integers: negative, and greater than MAX_TIME_STEP (including
# very large values -- hypothesis's default int strategy already samples
# arbitrarily large magnitudes on both sides of these bounds).
_OUT_OF_RANGE_INTEGERS = st.one_of(
    st.integers(max_value=MIN_TIME_STEP - 1),
    st.integers(min_value=MAX_TIME_STEP + 1),
)


# **Property 36: Time_Step 유효성 검증**
# **Validates: Requirements 5.12**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(time_step=_NON_INTEGER_TYPES)
def test_non_integer_type_time_step_is_rejected(time_step: object) -> None:
    """floats (incl. integral-valued), strings, and lists must all be rejected
    on type alone -- ``None`` (omitted, defaults to Time_Step 0) is
    intentionally excluded from this domain; it is covered separately as a
    *valid* case below."""
    _assert_invalid_time_step(time_step)


# **Property 36: Time_Step 유효성 검증**
# **Validates: Requirements 5.12**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(time_step=st.booleans())
def test_boolean_time_step_is_rejected_despite_being_an_int_subclass(time_step: bool) -> None:
    """Python's ``bool`` is a subclass of ``int``, so ``isinstance(True, int)``
    is ``True``. ``engine.py`` explicitly excludes ``bool`` via
    ``isinstance(resolved_time_step, bool)`` -- confirm both ``True`` and
    ``False`` are rejected rather than silently accepted as 1/0."""
    _assert_invalid_time_step(time_step)


# **Property 36: Time_Step 유효성 검증**
# **Validates: Requirements 5.12**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(time_step=_OUT_OF_RANGE_INTEGERS)
def test_out_of_range_integer_time_step_is_rejected(time_step: int) -> None:
    assert time_step < MIN_TIME_STEP or time_step > MAX_TIME_STEP
    _assert_invalid_time_step(time_step)


# **Property 36: Time_Step 유효성 검증**
# **Validates: Requirements 5.12**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(time_step=st.integers(min_value=MIN_TIME_STEP, max_value=MAX_TIME_STEP))
def test_valid_integer_time_step_does_not_raise(time_step: int) -> None:
    """Boundary/regression check: 0..4 (inclusive) integers must never raise
    ``invalid_time_step`` and must produce a normal prediction."""
    predictions = _ENGINE.predict_batch(_SAMPLE_PARAMETER_SETS, time_step)
    assert len(predictions) == len(_SAMPLE_PARAMETER_SETS)


# **Property 36: Time_Step 유효성 검증**
# **Validates: Requirements 5.12**
def test_omitted_time_step_defaults_and_does_not_raise() -> None:
    """``None`` (an omitted Time_Step) is a special case that must default to
    Time_Step 0, not be treated as an invalid value."""
    predictions = _ENGINE.predict_batch(_SAMPLE_PARAMETER_SETS, None)
    assert len(predictions) == len(_SAMPLE_PARAMETER_SETS)


@pytest.fixture(scope="module")
def client() -> TestClient:
    return TestClient(create_app(engine=_build_loaded_engine()))


# **Property 36: Time_Step 유효성 검증**
# **Validates: Requirements 5.12**
@pytest.mark.parametrize(
    "raw_time_step",
    [
        2.0,  # JSON `2.0` must decode to a Python float, not an int.
        "2",  # JSON string `"2"` must decode to a Python str, not an int.
        "abc",
        [1, 2],
        True,
        False,
        5,
        -1,
    ],
)
def test_http_boundary_rejects_invalid_time_step_with_original_value_and_type(
    client: TestClient, raw_time_step: object
) -> None:
    """Confirms the FastAPI/Pydantic ``Any`` layer passes these malformed
    values through to the engine with their JSON-native Python type intact
    (no silent coercion that would defeat the engine's ``isinstance`` checks),
    and that the HTTP boundary surfaces the resulting ``invalid_time_step``
    error with the exact received value and a 422 status."""
    payload = {"parameter_sets": _SAMPLE_PARAMETER_SETS_PAYLOAD, "time_step": raw_time_step}
    response = client.post("/predict/batch", json=payload)

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "invalid_time_step"
    assert body["details"] == {"time_step": raw_time_step}
    # bool is a JSON-native type distinct from int/float -- confirm httpx/FastAPI
    # actually preserved it as such rather than coercing to 1/0.
    if isinstance(raw_time_step, bool):
        assert isinstance(body["details"]["time_step"], bool)
    assert "predictions" not in body


# **Property 36: Time_Step 유효성 검증**
# **Validates: Requirements 5.12**
def test_http_boundary_accepts_valid_time_step_values(client: TestClient) -> None:
    """Boundary/regression check over the HTTP boundary: 0..4 and omitted
    Time_Step must succeed (200) rather than raising ``invalid_time_step``."""
    for time_step in (0, 1, 2, 3, 4):
        response = client.post(
            "/predict/batch",
            json={"parameter_sets": _SAMPLE_PARAMETER_SETS_PAYLOAD, "time_step": time_step},
        )
        assert response.status_code == 200
        assert response.json()["applied_time_step"] == time_step

    omitted_response = client.post(
        "/predict/batch", json={"parameter_sets": _SAMPLE_PARAMETER_SETS_PAYLOAD}
    )
    assert omitted_response.status_code == 200
    assert omitted_response.json()["applied_time_step"] == 0
