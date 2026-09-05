"""Property 37 test module (own module per the repository's Property-per-file convention).

Task 6.2: 성공 응답 pass-through 불변식.

Targets ``smo/agentic_ai/agents/kpi-advisor/mcp_server.py`` (task 6.1) directly
through its Python tool functions (``predict_batch``, ``marginal_effect``,
``current_model``), with a stub ``httpx`` transport standing in for
Inference_Service, exactly as ``test_mcp_server_contract.py`` does. Follows
``test_mcp_server_contract.py``'s manual ``sys.path`` setup convention since
``kpi-advisor`` is not installed as a package during tests, and follows
``smo/tests/property/test_property_29_batch_order_preservation_and_completeness.py``'s
convention of redefining small test-private helpers locally rather than
importing them from another test module.

Requirement 6.4:
    WHEN MCP 도구가 호출되고 Inference_Service 가 성공 응답을 반환하면, THE
    GNN_MCP_Server SHALL Inference_Service 응답의 필드 이름, 필드 값, 결과 순서를
    변경하지 않고 사용된 모델 이름과 버전을 포함한 상태로 도구 결과에 포함한다.

Design tag (design.md, "GNN_MCP_Server (Requirement 6)"):

    #### Property 37: 성공 응답 pass-through 불변식
    *For any* Inference_Service가 성공 응답을 반환하는 도구 호출, GNN_MCP_Server의
    도구 결과는 Inference_Service 응답의 필드 이름, 필드 값, 결과 순서를 그대로
    유지하고 사용된 모델 이름과 버전을 포함한다.
    **Validates: Requirements 6.4**

Unlike ``test_mcp_server_contract.py``'s single fixed-example spot checks for
``predict_batch``/``current_model``, this module varies batch size (1..64),
Target_KPI/cell_kpi float values, model name/version, and applied Time_Step
across many Hypothesis-generated fake Inference_Service response bodies, and
additionally covers ``marginal_effect``'s pass-through of ``model_name``/
``model_version`` from the underlying batch response it makes internally.
"""

from __future__ import annotations

import string
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from hypothesis import given
from hypothesis import strategies as st

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AGENT_DIR))

from smo.aimlfw.common.constants import (  # noqa: E402
    CONTROL_PARAMETER_NAMES,
    MAX_TIME_STEP,
    MIN_TIME_STEP,
    TARGET_KPIS,
)
from smo.aimlfw.common.models import ParameterSet  # noqa: E402
from smo.aimlfw.inference_service.schemas import MAX_BATCH_SIZE  # noqa: E402
from smo.tests.property.strategies import (  # noqa: E402
    PROPERTY_TEST_SETTINGS,
    cell_ids,
    valid_parameter_sets,
)

import mcp_server  # noqa: E402

_FINITE_FLOAT = st.floats(
    min_value=-1_000_000.0,
    max_value=1_000_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
_MODEL_NAME_STRATEGY = st.text(
    alphabet=string.ascii_letters + string.digits + "-_",
    min_size=1,
    max_size=32,
)


def _client_for(transport: httpx.BaseTransport) -> httpx.Client:
    return httpx.Client(transport=transport, base_url="http://test-inference-service")


class _StubTransport(httpx.BaseTransport):
    """Records every request and returns a pre-scripted response (local copy of the
    ``test_mcp_server_contract.py`` helper; see module docstring)."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return self.handler(request)


@pytest.fixture(autouse=True)
def _reset_client_factory():
    yield
    mcp_server.reset_client_factory()


def _default_cell_parameters() -> dict[str, Any]:
    """A baseline cell whose +1 step is valid for every Control_Parameter.

    Used by the ``marginal_effect`` test below so that, regardless of which
    Control_Parameter subset Hypothesis picks, every requested parameter is
    available for probing and the underlying batch request always has a
    deterministic, known size (see comment at the call site).
    """
    return {
        "tx_power_dbm": 43.0,
        "ret_tilt_deg": 5.0,
        "cio_bias_db": 0.5,
        "hysteresis_db": 2.5,
        "ttt_ms": 160,
    }


def _parameter_set_payload(parameter_set: ParameterSet) -> dict[str, Any]:
    """Convert a domain ``ParameterSet`` into the dict shape ``predict_batch`` expects."""
    return {"cells": parameter_set.model_dump(mode="json")["cells"]}


@st.composite
def _prediction_result(draw: Any, *, index: int, response_cell_ids: list[str]) -> dict[str, Any]:
    """Generate one fake ``PredictionResult`` entry (Inference_Service's own shape)."""
    return {
        "index": index,
        "target_kpi": {kpi: draw(_FINITE_FLOAT) for kpi in TARGET_KPIS},
        "cell_kpi": {
            cell_id: {kpi: draw(_FINITE_FLOAT) for kpi in TARGET_KPIS} for cell_id in response_cell_ids
        },
    }


@st.composite
def _batch_prediction_responses(draw: Any, *, batch_size: int) -> dict[str, Any]:
    """Generate a fake ``/predict/batch`` success response body with ``batch_size`` entries.

    ``response_cell_ids`` intentionally need not match whatever cell_ids the
    caller's ``parameter_sets`` used: Requirement 6.4/``predict_batch`` never
    inspects or re-shapes the response, so varying this independently still
    exercises the pass-through invariant faithfully.
    """
    response_cell_ids = draw(cell_ids(min_size=1, max_size=3))
    predictions = [
        draw(_prediction_result(index=index, response_cell_ids=response_cell_ids)) for index in range(batch_size)
    ]
    return {
        "model_name": draw(_MODEL_NAME_STRATEGY),
        "model_version": draw(st.integers(min_value=1, max_value=10_000)),
        "applied_time_step": draw(st.integers(min_value=MIN_TIME_STEP, max_value=MAX_TIME_STEP)),
        "predictions": predictions,
    }


@st.composite
def _status_responses(draw: Any) -> dict[str, Any]:
    """Generate a fake ``GET /status`` ``model_load_status: loaded`` response body."""
    return {
        "model_load_status": "loaded",
        "model_name": draw(_MODEL_NAME_STRATEGY),
        "model_version": draw(st.integers(min_value=1, max_value=10_000)),
    }


# **Property 37: 성공 응답 pass-through 불변식**
# **Validates: Requirements 6.4**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    parameter_sets=st.lists(
        valid_parameter_sets(min_cells=1, max_cells=2),
        min_size=1,
        max_size=MAX_BATCH_SIZE,
    ),
    time_step=st.one_of(st.none(), st.integers(min_value=MIN_TIME_STEP, max_value=MAX_TIME_STEP)),
    data=st.data(),
)
def test_predict_batch_proxies_success_response_verbatim_across_shapes(
    parameter_sets: list[ParameterSet], time_step: int | None, data: st.DataObject
) -> None:
    fake_response = data.draw(_batch_prediction_responses(batch_size=len(parameter_sets)))
    transport = _StubTransport(lambda request: httpx.Response(200, json=fake_response))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.predict_batch(
        parameter_sets=[_parameter_set_payload(parameter_set) for parameter_set in parameter_sets],
        time_step=time_step,
    )

    # Field names, field values, and result order all unchanged.
    assert result == fake_response
    assert [prediction["index"] for prediction in result["predictions"]] == [
        prediction["index"] for prediction in fake_response["predictions"]
    ]
    # Model name/version used are included in the tool result.
    assert result["model_name"] == fake_response["model_name"]
    assert result["model_version"] == fake_response["model_version"]
    assert len(transport.calls) == 1


# **Property 37: 성공 응답 pass-through 불변식**
# **Validates: Requirements 6.4**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(status_body=_status_responses())
def test_current_model_proxies_model_name_and_version_across_shapes(status_body: dict[str, Any]) -> None:
    transport = _StubTransport(lambda request: httpx.Response(200, json=status_body))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.current_model()

    assert result == {
        "model_name": status_body["model_name"],
        "model_version": status_body["model_version"],
    }


# **Property 37: 성공 응답 pass-through 불변식**
# **Validates: Requirements 6.4**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    control_parameters=st.lists(
        st.sampled_from(CONTROL_PARAMETER_NAMES), min_size=1, max_size=5, unique=True
    ),
    target_kpis=st.lists(st.sampled_from(TARGET_KPIS), min_size=1, max_size=len(TARGET_KPIS), unique=True),
    data=st.data(),
)
def test_marginal_effect_proxies_model_name_and_version_from_underlying_batch(
    control_parameters: list[str], target_kpis: list[str], data: st.DataObject
) -> None:
    # ``_default_cell_parameters`` guarantees a +1 step is in range for every
    # Control_Parameter, so every entry in ``control_parameters`` is
    # available and the underlying batch request is deterministically
    # baseline + one variant per requested Control_Parameter.
    batch_size = 1 + len(control_parameters)
    fake_response = data.draw(_batch_prediction_responses(batch_size=batch_size))
    transport = _StubTransport(lambda request: httpx.Response(200, json=fake_response))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.marginal_effect(
        baseline={"cells": {"gNB_5G": _default_cell_parameters()}},
        control_parameters=control_parameters,
        target_kpis=target_kpis,
    )

    # marginal_effects itself is derived (not a literal pass-through), but
    # model_name/model_version must exactly match the underlying
    # Inference_Service batch response Requirement 6.4 concerns.
    assert result["unavailable_control_parameters"] == []
    assert result["model_name"] == fake_response["model_name"]
    assert result["model_version"] == fake_response["model_version"]
    assert len(transport.calls) == 1
