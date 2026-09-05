from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import httpx
import pytest
from fastapi.routing import APIRoute
from pydantic import ValidationError

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AGENT_DIR))

from smo.aimlfw.common.models import CellParameters, ParameterSet  # noqa: E402

import server  # noqa: E402
from schemas import (  # noqa: E402
    HealthResponse,
    InvokeRequest,
    InvokeResponse,
    Recommendation,
    XAppParameterRequest,
    validate_response_for_intent,
)


def parameter_set() -> ParameterSet:
    return ParameterSet(
        cells={
            "gNB_5G": CellParameters(
                tx_power_dbm=43.0,
                ret_tilt_deg=5.0,
                cio_bias_db=0.5,
                hysteresis_db=2.5,
                ttt_ms=160,
            )
        }
    )


def valid_payload() -> dict[str, object]:
    return {
        "intent": "이 파라미터의 KPI 영향을 평가해 주세요.",
        "baseline_parameter_set": parameter_set().model_dump(),
        "time_step": 2,
    }


def response(rationale: str = "요청 조합은 현재 근거에서 허용 가능합니다.") -> InvokeResponse:
    return InvokeResponse(
        degradation_verdict="acceptable",
        recommendations=[Recommendation(variant_id="v1", parameter_set=parameter_set(), improvement_percent=1.2)],
        evidence_record_id="evidence-1",
        rationale_summary=rationale,
    )


def asgi_request(method: str, path: str, **kwargs: object) -> httpx.Response:
    """Exercise the ASGI boundary without TestClient's environment-specific thread portal."""

    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://advisor") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_invoke_request_accepts_exactly_one_valid_parameter_source() -> None:
    baseline = InvokeRequest.model_validate(valid_payload())
    assert baseline.time_step == 2
    partial = InvokeRequest(
        intent="평가",
        xapp_request=XAppParameterRequest(
            target_cells=["gNB_5G"],
            parameter_overrides={"gNB_5G": {"tx_power_dbm": 44.0}},
        ),
    )
    assert partial.xapp_request is not None

    with pytest.raises(ValidationError):
        InvokeRequest(intent="평가", time_step=5)
    with pytest.raises(ValidationError):
        InvokeRequest(
            intent="평가",
            baseline_parameter_set=parameter_set(),
            xapp_request=partial.xapp_request,
        )


def test_xapp_request_rejects_empty_or_non_target_overrides() -> None:
    with pytest.raises(ValidationError):
        XAppParameterRequest(target_cells=["gNB_5G"], parameter_overrides={})
    with pytest.raises(ValidationError):
        XAppParameterRequest(
            target_cells=["gNB_5G"],
            parameter_overrides={"other": {"tx_power_dbm": 44.0}},
        )


def test_response_contract_limits_recommendations_and_forbids_commands() -> None:
    recommendation = Recommendation(variant_id="v", parameter_set=parameter_set())
    with pytest.raises(ValidationError):
        InvokeResponse(
            degradation_verdict="acceptable",
            recommendations=[recommendation] * 11,
            evidence_record_id="e-1",
            rationale_summary="정상 근거",
        )
    with pytest.raises(ValidationError):
        InvokeResponse(
            degradation_verdict="unknown",
            recommendations=[],
            evidence_record_id="e-1",
            rationale_summary="kubectl get pods",
        )


def test_korean_intent_requires_korean_rationale() -> None:
    validate_response_for_intent(response(), "영향을 평가해 주세요")
    with pytest.raises(ValueError, match="Korean"):
        validate_response_for_intent(response("No supporting evidence."), "영향을 평가해 주세요")


def test_health_contract_and_only_two_external_routes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KPI_ADVISOR_MODEL_NAME", "ran-gnn")
    monkeypatch.setenv("KPI_ADVISOR_MODEL_VERSION", "3")
    result = asgi_request("GET", "/health")
    assert result.status_code == 200
    assert result.json() == {"status": "ready", "model_name": "ran-gnn", "model_version": 3}
    routes = {(route.path, next(iter(route.methods))) for route in server.app.routes if isinstance(route, APIRoute)}
    assert {path for path, _method in routes} == {"/health", "/invoke"}


def test_invalid_input_lists_fields_before_agent_execution(monkeypatch: pytest.MonkeyPatch) -> None:
    called = False

    async def should_not_run(_request: InvokeRequest) -> InvokeResponse:
        nonlocal called
        called = True
        return response()

    monkeypatch.setattr(server, "run_advisor", should_not_run)
    result = asgi_request("POST", "/invoke", json={"intent": "평가", "time_step": 9})
    assert result.status_code == 422
    assert result.json()["error_code"] == "invalid_request"
    assert "time_step" in result.json()["details"]["invalid_fields"]
    assert called is False


def test_invoke_returns_valid_atomic_korean_response(monkeypatch: pytest.MonkeyPatch) -> None:
    async def successful(_request: InvokeRequest) -> InvokeResponse:
        return response()

    monkeypatch.setattr(server, "run_advisor", successful)
    result = asgi_request("POST", "/invoke", json=valid_payload())
    assert result.status_code == 200
    assert result.json()["degradation_verdict"] == "acceptable"
    assert len(result.json()["recommendations"]) == 1
    assert result.json()["evidence_record_id"] == "evidence-1"


def test_invoke_enforces_timeout_without_partial_result(monkeypatch: pytest.MonkeyPatch) -> None:
    async def slow(_request: InvokeRequest) -> InvokeResponse:
        await asyncio.sleep(0.02)
        return response()

    monkeypatch.setattr(server, "run_advisor", slow)
    monkeypatch.setattr(server, "INVOKE_TIMEOUT_SECONDS", 0.001)
    result = asgi_request("POST", "/invoke", json=valid_payload())
    assert result.status_code == 504
    assert result.json()["error_code"] == "advisor_timeout"
    assert "recommendations" not in result.json()


def test_not_ready_health_requires_no_model_metadata() -> None:
    health = HealthResponse(status="not_ready")
    assert health.model_name is None
    with pytest.raises(ValidationError):
        HealthResponse(status="ready")
