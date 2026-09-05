"""Property 38 test module (own module per task 6.3, no collisions with 6.2/6.4/6.5).

Task 6.3: Property 38 속성 테스트 작성: 오류 응답 pass-through 불변식.

Targets ``smo.agentic_ai.agents.kpi-advisor.mcp_server``'s ``predict_batch``,
``current_model``, and ``marginal_effect`` tools (task 6.1's
``_passthrough_error`` helper), as specified in requirements.md:

Requirement 6.5:
    IF Inference_Service 가 오류를 반환하면, THEN THE GNN_MCP_Server SHALL
    동일한 오류 코드와 오류 메시지를 도구 결과로 반환하고 부분 예측값을
    도구 결과에 포함하지 않는다.

Design tag (design.md "GNN_MCP_Server (Requirement 6)"):

    #### Property 38: 오류 응답 pass-through 불변식
    *For any* Inference_Service가 반환하는 오류, GNN_MCP_Server는 동일한
    오류 코드와 메시지를 도구 결과로 반환하고 부분 예측값을 포함하지
    않는다.
    **Validates: Requirements 6.5**

``test_mcp_server_contract.py`` (task 6.1) already covers this with one fixed
example per tool (``test_predict_batch_proxies_error_verbatim_with_no_predictions``,
``test_marginal_effect_proxies_inference_service_error_verbatim``). This
module upgrades that spot check to a genuine Hypothesis property: varied
Inference_Service error bodies -- error_code drawn from the actual codes
``smo.aimlfw.inference_service.engine`` raises
(``model_not_loaded``, ``empty_batch``, ``batch_too_large``,
``invalid_time_step``, ``parameter_out_of_range``, ``prediction_failed``),
varied message text, and varied ``details`` shapes (empty dict, a
``violations`` list of dicts, or a flat scalar-valued dict) -- returned with
a non-200 HTTP status (422 or 503, matching
``inference_service.server._status_code``) via a stub ``httpx`` transport.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AGENT_DIR))

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, TARGET_KPIS  # noqa: E402

import mcp_server  # noqa: E402

PROPERTY_TEST_SETTINGS = settings(max_examples=100, deadline=None)


def _run(coro):
    return asyncio.run(coro)


def _cell_parameters(**overrides: Any) -> dict[str, Any]:
    base = {
        "tx_power_dbm": 43.0,
        "ret_tilt_deg": 5.0,
        "cio_bias_db": 0.5,
        "hysteresis_db": 2.5,
        "ttt_ms": 160,
    }
    base.update(overrides)
    return base


def _parameter_set(**overrides: Any) -> dict[str, Any]:
    return {"cells": {"gNB_5G": _cell_parameters(**overrides)}}


def _client_for(transport: httpx.BaseTransport) -> httpx.Client:
    return httpx.Client(transport=transport, base_url="http://test-inference-service")


class _StubTransport(httpx.BaseTransport):
    """Always returns the same pre-scripted response, recording every request."""

    def __init__(self, response: httpx.Response):
        self.response = response
        self.calls: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return self.response


@pytest.fixture(autouse=True)
def _reset_client_factory():
    yield
    mcp_server.reset_client_factory()


# Real Inference_Service error codes (smo/aimlfw/inference_service/engine.py's
# InferenceServiceError call sites), used as the Hypothesis example pool
# instead of arbitrary strings so this test reflects the actual contract.
_REAL_ERROR_CODES = (
    "parameter_out_of_range",
    "empty_batch",
    "batch_too_large",
    "model_not_loaded",
    "invalid_time_step",
    "prediction_failed",
)

_error_code_strategy = st.sampled_from(_REAL_ERROR_CODES)
_message_strategy = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), min_codepoint=32),
    min_size=1,
    max_size=120,
).filter(lambda text: text.strip() != "")
_status_code_strategy = st.sampled_from([422, 503])

_violation_strategy = st.fixed_dictionaries(
    {
        "parameter_set_index": st.integers(min_value=0, max_value=63),
        "cell_id": st.text(min_size=1, max_size=16, alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ_"),
        "parameter": st.sampled_from(CONTROL_PARAMETER_NAMES),
        "value": st.one_of(
            st.integers(min_value=-100, max_value=10_000), st.floats(allow_nan=False, allow_infinity=False)
        ),
    }
)

# Varied realistic ``details`` shapes: empty, a populated violations list
# (parameter_out_of_range's real shape), or a flat scalar-valued dict
# (empty_batch/batch_too_large/invalid_time_step/model_not_loaded/
# prediction_failed's real shapes).
_details_strategy = st.one_of(
    st.just({}),
    st.builds(lambda violations: {"violations": violations}, st.lists(_violation_strategy, min_size=0, max_size=5)),
    st.fixed_dictionaries({"received": st.integers(min_value=0, max_value=1000)}),
    st.fixed_dictionaries({"time_step": st.one_of(st.none(), st.integers(min_value=-10, max_value=10))}),
    st.fixed_dictionaries(
        {
            "model_name": st.text(min_size=1, max_size=20),
            "model_version": st.integers(min_value=1, max_value=1000),
        }
    ),
)


def _error_body_strategy():
    return st.builds(
        lambda error_code, message, details: {"error_code": error_code, "message": message, "details": details},
        _error_code_strategy,
        _message_strategy,
        _details_strategy,
    )


def _assert_passthrough(result: dict[str, Any], error_body: dict[str, Any]) -> None:
    assert result["error_code"] == error_body["error_code"]
    assert result["message"] == error_body["message"]
    assert result["details"] == error_body["details"]
    assert "predictions" not in result


# **Property 38: 오류 응답 pass-through 불변식**
# **Validates: Requirements 6.5**
@PROPERTY_TEST_SETTINGS
@given(status_code=_status_code_strategy, error_body=_error_body_strategy())
def test_predict_batch_passes_through_varied_inference_service_errors(
    status_code: int, error_body: dict[str, Any]
) -> None:
    transport = _StubTransport(httpx.Response(status_code, json=error_body))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.predict_batch(parameter_sets=[_parameter_set()])

    _assert_passthrough(result, error_body)


# **Property 38: 오류 응답 pass-through 불변식**
# **Validates: Requirements 6.5**
@PROPERTY_TEST_SETTINGS
@given(status_code=_status_code_strategy, error_body=_error_body_strategy())
def test_current_model_passes_through_varied_inference_service_errors(
    status_code: int, error_body: dict[str, Any]
) -> None:
    transport = _StubTransport(httpx.Response(status_code, json=error_body))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.current_model()

    _assert_passthrough(result, error_body)


# **Property 38: 오류 응답 pass-through 불변식**
# **Validates: Requirements 6.5**
@PROPERTY_TEST_SETTINGS
@given(status_code=_status_code_strategy, error_body=_error_body_strategy())
def test_marginal_effect_passes_through_varied_inference_service_errors(
    status_code: int, error_body: dict[str, Any]
) -> None:
    transport = _StubTransport(httpx.Response(status_code, json=error_body))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.marginal_effect(
        baseline=_parameter_set(),
        control_parameters=["tx_power_dbm"],
        target_kpis=[TARGET_KPIS[0]],
    )

    _assert_passthrough(result, error_body)
