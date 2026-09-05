"""Task 7.16: async ASGI integration checks without Starlette TestClient."""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

import httpx
from fastapi.routing import APIRoute

TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR), str(TEST_DIR)]

import test_agent_baseline_and_probe_plan as probe_cases  # noqa: E402

import server  # noqa: E402
from schemas import HealthResponse, InvokeResponse  # noqa: E402


def _request(method: str, path: str, **kwargs: object) -> httpx.Response:
    async def send() -> httpx.Response:
        transport = httpx.ASGITransport(app=server.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://advisor") as client:
            return await client.request(method, path, **kwargs)

    return asyncio.run(send())


def test_health_and_invoke_contract_slas_and_korean_response(monkeypatch) -> None:
    baseline = probe_cases._simple_plan().baseline

    async def advise(_request: object) -> InvokeResponse:
        return InvokeResponse(
            degradation_verdict="unknown",
            recommendations=[],
            evidence_record_id="evidence-integration",
            rationale_summary="신뢰 구간 정보가 없어 판단을 보류합니다.",
        )

    monkeypatch.setattr(
        server, "get_model_health", lambda: HealthResponse(status="ready", model_name="gnn", model_version=1)
    )
    monkeypatch.setattr(server, "run_advisor", advise)

    started = time.monotonic()
    health = _request("GET", "/health")
    assert time.monotonic() - started < 3.0
    assert health.status_code == 200 and health.json()["model_version"] == 1

    started = time.monotonic()
    response = _request(
        "POST",
        "/invoke",
        json={
            "intent": "KPI 영향을 분석해 주세요",
            "baseline_parameter_set": baseline.model_dump(mode="json"),
            "time_step": 0,
        },
    )
    assert time.monotonic() - started < 60.0
    assert response.status_code == 200
    assert response.json()["evidence_record_id"] == "evidence-integration"
    assert any("가" <= char <= "힣" for char in response.json()["rationale_summary"])
    routes = {
        (route.path, frozenset(route.methods or set())) for route in server.app.routes if isinstance(route, APIRoute)
    }
    assert routes == {("/health", frozenset({"GET"})), ("/invoke", frozenset({"POST"}))}


def test_invalid_input_and_timeout_are_atomic(monkeypatch) -> None:
    calls = 0

    async def slow(_request: object) -> InvokeResponse:
        nonlocal calls
        calls += 1
        await asyncio.sleep(0.02)
        raise AssertionError("timeout should cancel this coroutine")

    monkeypatch.setattr(server, "run_advisor", slow)
    invalid = _request("POST", "/invoke", json={"intent": "invalid"})
    assert invalid.status_code == 422
    assert invalid.json()["error_code"] == "invalid_request"
    assert calls == 0

    monkeypatch.setattr(server, "INVOKE_TIMEOUT_SECONDS", 0.001)
    baseline = probe_cases._simple_plan().baseline
    timed_out = _request(
        "POST",
        "/invoke",
        json={
            "intent": "evaluate",
            "baseline_parameter_set": baseline.model_dump(mode="json"),
            "time_step": 0,
        },
    )
    assert timed_out.status_code == 504
    assert timed_out.json() == {
        "error_code": "advisor_timeout",
        "message": "KPI Advisor did not respond within 60 seconds",
        "details": {},
    }
