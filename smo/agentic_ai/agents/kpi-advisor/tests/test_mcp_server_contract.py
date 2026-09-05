"""Contract tests for the GNN_MCP_Server (Requirement 6, task 6.1).

Follows ``test_contract.py``'s manual ``sys.path`` setup convention since
``kpi-advisor`` is not installed as a package during tests.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AGENT_DIR))

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, TARGET_KPIS  # noqa: E402

import mcp_server  # noqa: E402


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
    """Records every request and returns a pre-scripted response."""

    def __init__(self, handler):
        self.handler = handler
        self.calls: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        return self.handler(request)


class _RaisingTransport(httpx.BaseTransport):
    """Always raises to simulate a timeout/unreachable Inference_Service."""

    def __init__(self, exc: Exception):
        self.exc = exc
        self.calls: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        raise self.exc


@pytest.fixture(autouse=True)
def _reset_client_factory():
    yield
    mcp_server.reset_client_factory()


def test_list_tools_reports_exactly_three_read_only_tools_with_metadata() -> None:
    tools = _run(mcp_server.mcp.list_tools())
    names = {tool.name for tool in tools}
    assert names == {"predict_batch", "marginal_effect", "current_model"}

    by_name = {tool.name: tool for tool in tools}

    predict_schema = by_name["predict_batch"].inputSchema
    assert predict_schema["required"] == ["parameter_sets"]
    assert predict_schema["properties"]["parameter_sets"]["minItems"] == 1
    assert predict_schema["properties"]["parameter_sets"]["maxItems"] == 64
    assert "time_step" not in predict_schema["required"]
    time_step_schema = predict_schema["properties"]["time_step"]
    assert time_step_schema["minimum"] == 0
    assert time_step_schema["maximum"] == 4

    marginal_schema = by_name["marginal_effect"].inputSchema
    assert set(marginal_schema["required"]) == {"baseline", "control_parameters", "target_kpis"}
    assert marginal_schema["properties"]["control_parameters"]["minItems"] == 1
    assert marginal_schema["properties"]["control_parameters"]["maxItems"] == 5
    assert marginal_schema["properties"]["target_kpis"]["minItems"] == 1

    current_model_schema = by_name["current_model"].inputSchema
    assert current_model_schema["properties"] == {}
    assert current_model_schema.get("required", []) == []


def test_predict_batch_rejects_oversized_batch_without_calling_inference_service() -> None:
    transport = _StubTransport(lambda request: httpx.Response(200, json={}))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.predict_batch(parameter_sets=[_parameter_set()] * 65)

    assert result["error_code"] == "input_validation_error"
    assert "parameter_sets" in result["details"]["violated_arguments"]
    assert transport.calls == []


def test_predict_batch_rejects_out_of_range_time_step_without_calling_inference_service() -> None:
    transport = _StubTransport(lambda request: httpx.Response(200, json={}))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.predict_batch(parameter_sets=[_parameter_set()], time_step=9)

    assert result["error_code"] == "input_validation_error"
    assert "time_step" in result["details"]["violated_arguments"]
    assert transport.calls == []


def test_predict_batch_proxies_successful_response_verbatim() -> None:
    fake_response = {
        "model_name": "gnn",
        "model_version": 3,
        "applied_time_step": 2,
        "predictions": [
            {
                "index": 0,
                "target_kpi": {kpi: 1.0 for kpi in TARGET_KPIS},
                "cell_kpi": {"gNB_5G": {kpi: 1.0 for kpi in TARGET_KPIS}},
            }
        ],
    }
    transport = _StubTransport(lambda request: httpx.Response(200, json=fake_response))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.predict_batch(parameter_sets=[_parameter_set()], time_step=2)

    assert result == fake_response
    assert len(transport.calls) == 1


def test_predict_batch_proxies_error_verbatim_with_no_predictions() -> None:
    violation = {"parameter_set_index": 0, "cell_id": "gNB_5G", "parameter": "tx_power_dbm", "value": 99}
    fake_error = {
        "error_code": "parameter_out_of_range",
        "message": "One or more Control_Parameter values are outside the allowed range or step",
        "details": {"violations": [violation]},
    }
    transport = _StubTransport(lambda request: httpx.Response(422, json=fake_error))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.predict_batch(parameter_sets=[_parameter_set(tx_power_dbm=99)])

    assert result["error_code"] == "parameter_out_of_range"
    assert result["message"] == fake_error["message"]
    assert result["details"] == fake_error["details"]
    assert "predictions" not in result


def test_predict_batch_timeout_returns_inference_unavailable_without_retry() -> None:
    transport = _RaisingTransport(httpx.ReadTimeout("timed out"))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.predict_batch(parameter_sets=[_parameter_set()])

    assert result["error_code"] == "inference_unavailable"
    assert len(transport.calls) == 1  # called exactly once: no retry.


def test_predict_batch_unreachable_returns_inference_unavailable_without_retry() -> None:
    transport = _RaisingTransport(httpx.ConnectError("connection refused"))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.predict_batch(parameter_sets=[_parameter_set()])

    assert result["error_code"] == "inference_unavailable"
    assert len(transport.calls) == 1


def test_current_model_proxies_status_verbatim() -> None:
    status_body = {"model_load_status": "loaded", "model_name": "gnn", "model_version": 5}
    transport = _StubTransport(lambda request: httpx.Response(200, json=status_body))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.current_model()

    assert result == {"model_name": "gnn", "model_version": 5}


def test_current_model_reports_not_loaded_without_partial_state() -> None:
    transport = _StubTransport(lambda request: httpx.Response(200, json={"model_load_status": "not_loaded"}))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.current_model()

    assert result["error_code"] == "model_not_loaded"


def test_current_model_timeout_returns_inference_unavailable() -> None:
    transport = _RaisingTransport(httpx.ReadTimeout("timed out"))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.current_model()

    assert result["error_code"] == "inference_unavailable"
    assert len(transport.calls) == 1


def test_marginal_effect_rejects_too_many_control_parameters_without_calling_inference_service() -> None:
    transport = _StubTransport(lambda request: httpx.Response(200, json={}))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.marginal_effect(
        baseline=_parameter_set(),
        control_parameters=list(CONTROL_PARAMETER_NAMES) + ["tx_power_dbm"],
        target_kpis=["cell_goodput_mbps"],
    )

    assert result["error_code"] == "input_validation_error"
    assert "control_parameters" in result["details"]["violated_arguments"]
    assert transport.calls == []


def test_marginal_effect_rejects_unknown_target_kpi_without_calling_inference_service() -> None:
    transport = _StubTransport(lambda request: httpx.Response(200, json={}))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.marginal_effect(
        baseline=_parameter_set(),
        control_parameters=["tx_power_dbm"],
        target_kpis=["not_a_real_kpi"],
    )

    assert result["error_code"] == "input_validation_error"
    assert "target_kpis" in result["details"]["violated_arguments"]
    assert transport.calls == []


def test_marginal_effect_computes_clamped_percent_change_per_pair() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = request.read()
        import json

        payload = json.loads(body)
        results = []
        for index, parameter_set in enumerate(payload["parameter_sets"]):
            tx_power = parameter_set["cells"]["gNB_5G"]["tx_power_dbm"]
            # Baseline tx_power_dbm=43 -> target_kpi=10.0; +1 step (44) -> 11.0 (10% increase).
            value = 10.0 + (tx_power - 43.0)
            results.append(
                {
                    "index": index,
                    "target_kpi": {"cell_goodput_mbps": value},
                    "cell_kpi": {"gNB_5G": {"cell_goodput_mbps": value}},
                }
            )
        return httpx.Response(
            200, json={"model_name": "gnn", "model_version": 1, "applied_time_step": 0, "predictions": results}
        )

    transport = _StubTransport(handler)
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.marginal_effect(
        baseline=_parameter_set(),
        control_parameters=["tx_power_dbm"],
        target_kpis=["cell_goodput_mbps"],
    )

    assert result["model_name"] == "gnn"
    assert result["model_version"] == 1
    assert result["marginal_effects"]["tx_power_dbm"]["cell_goodput_mbps"] == pytest.approx(10.0)
    assert result["unavailable_control_parameters"] == []
    assert len(transport.calls) == 1  # baseline + 1 variant in a single batch request.


def test_marginal_effect_proxies_inference_service_error_verbatim() -> None:
    fake_error = {"error_code": "model_not_loaded", "message": "no model loaded", "details": {}}
    transport = _StubTransport(lambda request: httpx.Response(503, json=fake_error))
    mcp_server.set_client_factory(lambda: _client_for(transport))

    result = mcp_server.marginal_effect(
        baseline=_parameter_set(),
        control_parameters=["tx_power_dbm"],
        target_kpis=["cell_goodput_mbps"],
    )

    assert result["error_code"] == "model_not_loaded"
    assert result["message"] == fake_error["message"]
