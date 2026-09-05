"""Property 40 test module (own module to avoid collisions with parallel Property tasks).

Task 6.5: 타임아웃 시 무변경 오류.

Targets ``smo.agentic_ai.agents.kpi-advisor.mcp_server``'s three read-only
MCP tools (``predict_batch``, ``current_model``, ``marginal_effect``) when
their single ``httpx`` call to Inference_Service times out or is otherwise
unreachable, as specified in requirements.md:

Requirement 6.10:
    IF Inference_Service 가 30 초 이내에 응답하지 않거나 연결할 수 없으면, THEN
    THE GNN_MCP_Server SHALL 재시도 없이 오류를 반환하고 호출자가 전달한 입력
    인자를 변경하지 않는다.

Design tag (design.md, "GNN_MCP_Server (Requirement 6)"):

    #### Property 40: 타임아웃 시 무변경 오류
    *For any* Inference_Service가 30초 이내에 응답하지 않거나 연결 불가한 상황,
    GNN_MCP_Server는 재시도 없이 오류를 반환하고 호출자가 전달한 입력 인자를
    변경하지 않는다.
    **Validates: Requirements 6.10**

This module upgrades the fixed-example ``_RaisingTransport`` coverage
already present in ``test_mcp_server_contract.py`` (task 6.1) into a real
Hypothesis property: it varies the input arguments (batch sizes,
Time_Step, baseline/control_parameters/target_kpis for ``marginal_effect``)
*and* the concrete ``httpx`` exception subtype raised by the transport
across examples, and it adds ``marginal_effect`` coverage the fixed-example
tests never exercised for this scenario. It also asserts the previously
untested "input arguments unchanged" half of the property by deep-copying
each generated argument before the call and comparing it to the same
object afterwards -- this specifically guards against the tool function
mutating its own input dicts/lists in place (e.g. while assembling the
outbound HTTP request body).

Oracle grounding: this test drives the real ``mcp_server.predict_batch``,
``mcp_server.current_model``, and ``mcp_server.marginal_effect`` tool
functions with a real ``httpx.Client`` bound to a stub ``httpx.BaseTransport``
that raises a timeout/connection exception (module under test, unmodified);
only the Inference_Service HTTP transport is faked.
"""

from __future__ import annotations

import sys
from copy import deepcopy
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
from smo.aimlfw.inference_service.schemas import MAX_BATCH_SIZE  # noqa: E402
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS  # noqa: E402

import mcp_server  # noqa: E402

# One exception per Requirement 6.10's two disjuncts ("does not respond within
# 30 seconds" and "unreachable"), plus a couple of concrete httpx subclasses so
# the property exercises more than a single exception type.
_UNAVAILABLE_EXCEPTIONS: tuple[httpx.HTTPError, ...] = (
    httpx.ReadTimeout("timed out reading response"),
    httpx.ConnectTimeout("timed out connecting"),
    httpx.ConnectError("connection refused"),
    httpx.RemoteProtocolError("server closed the connection"),
)


class _RaisingTransport(httpx.BaseTransport):
    """Always raises the given exception; records every request it sees."""

    def __init__(self, exc: Exception) -> None:
        self.exc = exc
        self.calls: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        raise self.exc


def _client_for(transport: httpx.BaseTransport) -> httpx.Client:
    return httpx.Client(transport=transport, base_url="http://test-inference-service")


@pytest.fixture(autouse=True)
def _reset_client_factory():
    yield
    mcp_server.reset_client_factory()


def _cell_parameters(seed: int) -> dict[str, Any]:
    """One deterministic, in-range set of Control_Parameter values (varies by seed)."""
    return {
        "tx_power_dbm": 30.0 + (seed % 17),
        "ret_tilt_deg": float(seed % 16),
        "cio_bias_db": -6.0 + (seed % 25) * 0.5,
        "hysteresis_db": (seed % 21) * 0.5,
        "ttt_ms": 160,
    }


def _parameter_set(seed: int, *, cell_count: int = 1) -> dict[str, Any]:
    return {
        "cells": {
            f"gNB_{cell_index}": _cell_parameters(seed + cell_index) for cell_index in range(cell_count)
        }
    }


# **Property 40: 타임아웃 시 무변경 오류**
# **Validates: Requirements 6.10**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    batch_size=st.integers(min_value=1, max_value=MAX_BATCH_SIZE),
    seed=st.integers(min_value=0, max_value=1_000_000),
    time_step=st.one_of(st.none(), st.integers(min_value=MIN_TIME_STEP, max_value=MAX_TIME_STEP)),
    exc=st.sampled_from(_UNAVAILABLE_EXCEPTIONS),
)
def test_predict_batch_timeout_or_unreachable_returns_inference_unavailable_without_retry_or_mutation(
    batch_size: int, seed: int, time_step: int | None, exc: httpx.HTTPError
) -> None:
    parameter_sets = [_parameter_set(seed + index) for index in range(batch_size)]
    parameter_sets_snapshot = deepcopy(parameter_sets)
    time_step_snapshot = deepcopy(time_step)

    transport = _RaisingTransport(exc)
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.predict_batch(parameter_sets=parameter_sets, time_step=time_step)

    assert result["error_code"] == "inference_unavailable"
    assert len(transport.calls) == 1  # exactly one call: no retry, regardless of exception subtype.
    assert parameter_sets == parameter_sets_snapshot  # input arguments left unchanged.
    assert time_step == time_step_snapshot


# **Property 40: 타임아웃 시 무변경 오류**
# **Validates: Requirements 6.10**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(seed=st.integers(min_value=0, max_value=1_000_000), exc=st.sampled_from(_UNAVAILABLE_EXCEPTIONS))
def test_current_model_timeout_or_unreachable_returns_inference_unavailable_without_retry(
    seed: int, exc: httpx.HTTPError
) -> None:
    del seed  # current_model() takes no arguments; kept for a stable, varied example corpus.
    transport = _RaisingTransport(exc)
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.current_model()

    assert result["error_code"] == "inference_unavailable"
    assert len(transport.calls) == 1


# **Property 40: 타임아웃 시 무변경 오류**
# **Validates: Requirements 6.10**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    seed=st.integers(min_value=0, max_value=1_000_000),
    cell_count=st.integers(min_value=1, max_value=4),
    control_parameter_count=st.integers(min_value=1, max_value=len(CONTROL_PARAMETER_NAMES)),
    target_kpi_count=st.integers(min_value=1, max_value=len(TARGET_KPIS)),
    exc=st.sampled_from(_UNAVAILABLE_EXCEPTIONS),
)
def test_marginal_effect_timeout_or_unreachable_returns_inference_unavailable_without_retry_or_mutation(
    seed: int, cell_count: int, control_parameter_count: int, target_kpi_count: int, exc: httpx.HTTPError
) -> None:
    baseline = _parameter_set(seed, cell_count=cell_count)
    control_parameters = list(CONTROL_PARAMETER_NAMES[:control_parameter_count])
    target_kpis = list(TARGET_KPIS[:target_kpi_count])

    baseline_snapshot = deepcopy(baseline)
    control_parameters_snapshot = deepcopy(control_parameters)
    target_kpis_snapshot = deepcopy(target_kpis)

    transport = _RaisingTransport(exc)
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.marginal_effect(
        baseline=baseline, control_parameters=control_parameters, target_kpis=target_kpis
    )

    assert result["error_code"] == "inference_unavailable"
    assert len(transport.calls) == 1  # exactly one batch call: no retry.
    assert baseline == baseline_snapshot  # input arguments left unchanged.
    assert control_parameters == control_parameters_snapshot
    assert target_kpis == target_kpis_snapshot
