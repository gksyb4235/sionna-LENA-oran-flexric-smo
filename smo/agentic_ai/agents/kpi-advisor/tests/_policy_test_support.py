"""Shared deterministic A1 test doubles and valid domain fixtures."""

from __future__ import annotations

import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR)]

from smo.aimlfw.common.models import (  # noqa: E402
    CellParameters,
    ParameterSet,
    TemporalPlan,
    TimeStepAssignment,
)

import policy_manager  # noqa: E402


def fixed_parameter_set(**overrides: float | int) -> ParameterSet:
    values: dict[str, float | int] = {
        "tx_power_dbm": 40.0,
        "ret_tilt_deg": 5.0,
        "cio_bias_db": 0.0,
        "hysteresis_db": 2.0,
        "ttt_ms": 160,
    }
    values.update(overrides)
    return ParameterSet(cells={cell: CellParameters.model_validate(values) for cell in policy_manager.FIXED_CELLS})


def temporal_plan() -> TemporalPlan:
    assignments = [
        TimeStepAssignment(
            time_step=step,
            xapp_objective="unspecified",
            parameter_set=fixed_parameter_set(tx_power_dbm=float(40 + step)),
            predicted_kpis={"cell_goodput_mbps": {cell: 10.0 for cell in policy_manager.FIXED_CELLS}},
            model_name="gnn",
            model_version=1,
        )
        for step in range(5)
    ]
    return TemporalPlan(assignments=assignments, aggregate_kpis={"cell_goodput_mbps": 50.0})


def approval() -> policy_manager.ApprovalRecord:
    return policy_manager.ApprovalRecord(approver_id="operator", approved_at=datetime.now(UTC))


class MockA1:
    def __init__(self, *, status_code: int = 200, status_available: bool = True) -> None:
        self.policy_types: dict[int, dict[str, Any]] = {}
        self.instances: dict[int, dict[str, dict[str, Any]]] = {}
        self.requests: list[tuple[str, str]] = []
        self.status_code = status_code
        self.status_available = status_available

    def handler(self, request: httpx.Request) -> httpx.Response:
        method, path = request.method, request.url.path
        self.requests.append((method, path))
        parts = path.strip("/").split("/")
        if len(parts) == 3 and parts[:2] == ["a1-p", "policytypes"]:
            policy_type_id = int(parts[2])
            if method == "GET":
                body = self.policy_types.get(policy_type_id)
                return httpx.Response(200, json=body) if body else httpx.Response(404, json={"detail": "missing"})
            if method == "PUT":
                if self.status_code >= 400:
                    return httpx.Response(self.status_code, json={"detail": "rejected"})
                body = __import__("json").loads(request.content)
                self.policy_types[policy_type_id] = body
                self.instances.setdefault(policy_type_id, {})
                return httpx.Response(200, json=body)
        if len(parts) >= 5 and parts[:2] == ["a1-p", "policytypes"] and parts[3] == "policies":
            policy_type_id, instance_id = int(parts[2]), parts[4]
            if method == "PUT" and len(parts) == 5:
                if self.status_code >= 400:
                    return httpx.Response(self.status_code, json={"detail": "rejected"})
                body = __import__("json").loads(request.content)
                self.instances.setdefault(policy_type_id, {})[instance_id] = body
                return httpx.Response(200, json={"data": body})
            if method == "GET" and len(parts) == 6 and parts[5] == "status":
                if not self.status_available:
                    return httpx.Response(500, json={"detail": "status unavailable"})
                return httpx.Response(200, json={"data": self.instances[policy_type_id][instance_id]})
        return httpx.Response(404, json={"detail": "unhandled"})

    def client_factory(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self.handler), base_url="http://a1")


def install_mock(mock: MockA1) -> None:
    policy_manager.set_client_factory(mock.client_factory)
    policy_manager.clear_publish_attempts()


def uninstall_mock() -> None:
    policy_manager.reset_client_factory()
    policy_manager.clear_publish_attempts()
