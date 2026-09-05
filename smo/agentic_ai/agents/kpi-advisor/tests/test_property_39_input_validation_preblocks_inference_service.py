"""Property 39 test module (GNN_MCP_Server, Requirement 6.9, task 6.4).

Follows ``test_mcp_server_contract.py``'s manual ``sys.path`` setup convention
since ``kpi-advisor`` is not installed as a package during tests, and follows
``smo/tests/property``'s ``PROPERTY_TEST_SETTINGS``/design-tag convention for
Hypothesis-based property modules.

Design tag (design.md, "GNN_MCP_Server (Requirement 6)"):

    #### Property 39: 입력 검증 선차단
    *For any* 필수 인자가 누락되었거나 Parameter_Set 개수가 64를 초과하거나
    Time_Step 값이 0~4 범위를 벗어나는 도구 호출, GNN_MCP_Server는
    Inference_Service를 호출하지 않고 위반된 인자 이름을 담은 입력 검증 오류를
    반환한다.
    **Validates: Requirements 6.9**

This module upgrades the fixed-example coverage already present in
``test_mcp_server_contract.py`` (one oversized batch, one out-of-range
time_step, one too-many-control_parameters, one unknown-target_kpi) to
genuine Hypothesis property tests that sweep a wide range of invalid inputs
for both ``predict_batch`` and ``marginal_effect``, and adds explicit
coverage for the "필수 인자가 누락" (missing required argument) half of
Requirement 6.9, which is enforced by Python's own function-call mechanism
rather than by this module's ``_validation_error`` path.
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

from smo.aimlfw.common.constants import (  # noqa: E402
    CONTROL_PARAMETER_NAMES,
    MAX_TIME_STEP,
    MIN_TIME_STEP,
    TARGET_KPIS,
)
from smo.aimlfw.inference_service.schemas import MAX_BATCH_SIZE  # noqa: E402

import mcp_server  # noqa: E402

PROPERTY_TEST_SETTINGS = settings(max_examples=25, deadline=None)


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
    """Records every request; fails the test outright if Inference_Service is ever called."""

    def __init__(self) -> None:
        self.calls: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return httpx.Response(200, json={})


@pytest.fixture(autouse=True)
def _reset_client_factory():
    yield
    mcp_server.reset_client_factory()


@pytest.fixture()
def stub_transport():
    transport = _StubTransport()
    mcp_server.set_client_factory(lambda: _client_for(transport))
    return transport


# ---------------------------------------------------------------------------
# predict_batch: parameter_sets count out of [1, 64]
# ---------------------------------------------------------------------------


# **Property 39: 입력 검증 선차단 (predict_batch, parameter_sets count)**
# **Validates: Requirements 6.9**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(count=st.integers(min_value=MAX_BATCH_SIZE + 1, max_value=MAX_BATCH_SIZE + 20))
def test_predict_batch_rejects_oversized_batches_without_calling_inference_service(count: int) -> None:
    transport = _StubTransport()
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.predict_batch(parameter_sets=[_parameter_set()] * count)

    assert result["error_code"] == "input_validation_error"
    assert "parameter_sets" in result["details"]["violated_arguments"]
    assert transport.calls == []
    mcp_server.reset_client_factory()


# **Property 39: 입력 검증 선차단 (predict_batch, empty parameter_sets)**
# **Validates: Requirements 6.9**
def test_predict_batch_rejects_empty_batch_without_calling_inference_service(stub_transport) -> None:
    result = mcp_server.predict_batch(parameter_sets=[])

    assert result["error_code"] == "input_validation_error"
    assert "parameter_sets" in result["details"]["violated_arguments"]
    assert stub_transport.calls == []


# ---------------------------------------------------------------------------
# predict_batch: time_step out of range or wrong type
# ---------------------------------------------------------------------------


# **Property 39: 입력 검증 선차단 (predict_batch, out-of-range integer time_step)**
# **Validates: Requirements 6.9**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    time_step=st.integers(min_value=-1000, max_value=1000).filter(
        lambda value: not (MIN_TIME_STEP <= value <= MAX_TIME_STEP)
    )
)
def test_predict_batch_rejects_out_of_range_integer_time_step(time_step: int) -> None:
    transport = _StubTransport()
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.predict_batch(parameter_sets=[_parameter_set()], time_step=time_step)

    assert result["error_code"] == "input_validation_error"
    assert "time_step" in result["details"]["violated_arguments"]
    assert transport.calls == []
    mcp_server.reset_client_factory()


# **Property 39: 입력 검증 선차단 (predict_batch, non-integer time_step)**
# **Validates: Requirements 6.9**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    time_step=st.one_of(
        st.floats(allow_nan=False, allow_infinity=False),
        st.text(),
        st.booleans(),
    )
)
def test_predict_batch_rejects_non_integer_time_step(time_step: Any) -> None:
    transport = _StubTransport()
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.predict_batch(parameter_sets=[_parameter_set()], time_step=time_step)

    assert result["error_code"] == "input_validation_error"
    assert "time_step" in result["details"]["violated_arguments"]
    assert transport.calls == []
    mcp_server.reset_client_factory()


# ---------------------------------------------------------------------------
# marginal_effect: control_parameters
# ---------------------------------------------------------------------------


# **Property 39: 입력 검증 선차단 (marginal_effect, empty control_parameters)**
# **Validates: Requirements 6.9**
def test_marginal_effect_rejects_empty_control_parameters(stub_transport) -> None:
    result = mcp_server.marginal_effect(
        baseline=_parameter_set(),
        control_parameters=[],
        target_kpis=["cell_goodput_mbps"],
    )

    assert result["error_code"] == "input_validation_error"
    assert "control_parameters" in result["details"]["violated_arguments"]
    assert stub_transport.calls == []


# **Property 39: 입력 검증 선차단 (marginal_effect, more than 5 control_parameters)**
# **Validates: Requirements 6.9**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(extra_count=st.integers(min_value=1, max_value=15))
def test_marginal_effect_rejects_too_many_control_parameters(extra_count: int) -> None:
    transport = _StubTransport()
    mcp_server.set_client_factory(lambda: _client_for(transport))
    # CONTROL_PARAMETER_NAMES has 5 entries; padding with an out-of-vocabulary
    # placeholder keeps every generated list strictly larger than the 5-item cap.
    control_parameters = list(CONTROL_PARAMETER_NAMES) + [
        f"extra_parameter_{index}" for index in range(extra_count)
    ]

    result = mcp_server.marginal_effect(
        baseline=_parameter_set(),
        control_parameters=control_parameters,
        target_kpis=["cell_goodput_mbps"],
    )

    assert result["error_code"] == "input_validation_error"
    assert "control_parameters" in result["details"]["violated_arguments"]
    assert transport.calls == []
    mcp_server.reset_client_factory()


# **Property 39: 입력 검증 선차단 (marginal_effect, duplicate control_parameters)**
# **Validates: Requirements 6.9**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    name=st.sampled_from(CONTROL_PARAMETER_NAMES),
    repeats=st.integers(min_value=2, max_value=5),
)
def test_marginal_effect_rejects_duplicate_control_parameters(name: str, repeats: int) -> None:
    transport = _StubTransport()
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.marginal_effect(
        baseline=_parameter_set(),
        control_parameters=[name] * repeats,
        target_kpis=["cell_goodput_mbps"],
    )

    assert result["error_code"] == "input_validation_error"
    assert "control_parameters" in result["details"]["violated_arguments"]
    assert transport.calls == []
    mcp_server.reset_client_factory()


# **Property 39: 입력 검증 선차단 (marginal_effect, unknown control_parameters)**
# **Validates: Requirements 6.9**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    unknown_name=st.text(min_size=1, max_size=20).filter(
        lambda value: value not in CONTROL_PARAMETER_NAMES
    )
)
def test_marginal_effect_rejects_unknown_control_parameter_names(unknown_name: str) -> None:
    transport = _StubTransport()
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.marginal_effect(
        baseline=_parameter_set(),
        control_parameters=[unknown_name],
        target_kpis=["cell_goodput_mbps"],
    )

    assert result["error_code"] == "input_validation_error"
    assert "control_parameters" in result["details"]["violated_arguments"]
    assert transport.calls == []
    mcp_server.reset_client_factory()


# ---------------------------------------------------------------------------
# marginal_effect: target_kpis
# ---------------------------------------------------------------------------


# **Property 39: 입력 검증 선차단 (marginal_effect, empty target_kpis)**
# **Validates: Requirements 6.9**
def test_marginal_effect_rejects_empty_target_kpis(stub_transport) -> None:
    result = mcp_server.marginal_effect(
        baseline=_parameter_set(),
        control_parameters=["tx_power_dbm"],
        target_kpis=[],
    )

    assert result["error_code"] == "input_validation_error"
    assert "target_kpis" in result["details"]["violated_arguments"]
    assert stub_transport.calls == []


# **Property 39: 입력 검증 선차단 (marginal_effect, unknown target_kpis)**
# **Validates: Requirements 6.9**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    unknown_kpis=st.lists(
        st.text(min_size=1, max_size=20).filter(lambda value: value not in TARGET_KPIS),
        min_size=1,
        max_size=5,
    )
)
def test_marginal_effect_rejects_unknown_target_kpis(unknown_kpis: list[str]) -> None:
    transport = _StubTransport()
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.marginal_effect(
        baseline=_parameter_set(),
        control_parameters=["tx_power_dbm"],
        target_kpis=unknown_kpis,
    )

    assert result["error_code"] == "input_validation_error"
    assert "target_kpis" in result["details"]["violated_arguments"]
    assert transport.calls == []
    mcp_server.reset_client_factory()


# ---------------------------------------------------------------------------
# marginal_effect: malformed baseline
# ---------------------------------------------------------------------------


# **Property 39: 입력 검증 선차단 (marginal_effect, baseline missing "cells")**
# **Validates: Requirements 6.9**
def test_marginal_effect_rejects_baseline_missing_cells_key(stub_transport) -> None:
    result = mcp_server.marginal_effect(
        baseline={},
        control_parameters=["tx_power_dbm"],
        target_kpis=["cell_goodput_mbps"],
    )

    assert result["error_code"] == "input_validation_error"
    assert "baseline" in result["details"]["violated_arguments"]
    assert stub_transport.calls == []


# **Property 39: 입력 검증 선차단 (marginal_effect, baseline with empty "cells")**
# **Validates: Requirements 6.9**
def test_marginal_effect_rejects_baseline_with_empty_cells(stub_transport) -> None:
    result = mcp_server.marginal_effect(
        baseline={"cells": {}},
        control_parameters=["tx_power_dbm"],
        target_kpis=["cell_goodput_mbps"],
    )

    assert result["error_code"] == "input_validation_error"
    assert "baseline" in result["details"]["violated_arguments"]
    assert stub_transport.calls == []


# ---------------------------------------------------------------------------
# 필수 인자 누락 (missing required argument): rejected by Python's own
# function-call mechanism before this module's validation code -- and
# therefore before any Inference_Service call -- ever runs. Requirement 6.9
# covers this case distinctly from the range/count checks above.
# ---------------------------------------------------------------------------


def test_predict_batch_missing_required_argument_raises_before_any_inference_service_call(
    stub_transport,
) -> None:
    with pytest.raises(TypeError):
        mcp_server.predict_batch()  # type: ignore[call-arg]

    assert stub_transport.calls == []


def test_marginal_effect_missing_required_arguments_raises_before_any_inference_service_call(
    stub_transport,
) -> None:
    with pytest.raises(TypeError):
        mcp_server.marginal_effect(baseline=_parameter_set())  # type: ignore[call-arg]

    assert stub_transport.calls == []
