"""Property 83: invalid advisor input is blocked before the HTTP adapter."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import patch

from hypothesis import given, settings
from hypothesis import strategies as st

PLANNING_DIR = Path(__file__).resolve().parents[1]
KPI_DIR = PLANNING_DIR.parent / "kpi-advisor"
AGENTIC_ROOT = PLANNING_DIR.parents[1]
REPO_ROOT = PLANNING_DIR.parents[3]
sys.path[:0] = [str(REPO_ROOT), str(AGENTIC_ROOT), str(PLANNING_DIR), str(KPI_DIR)]

fake_langgraph = types.ModuleType("langgraph")
fake_graph = types.ModuleType("langgraph.graph")
fake_graph.END = "__end__"
fake_graph.START = "__start__"
fake_graph.StateGraph = object
fake_langgraph.graph = fake_graph
sys.modules.setdefault("langgraph", fake_langgraph)
sys.modules.setdefault("langgraph.graph", fake_graph)

from runtime_registration import communication_contract  # noqa: E402

from tools import subgraph_runtime  # noqa: E402


# **Property 83: 입력 스키마 검증 선차단**
# **Validates: Requirement 12.11**
@settings(max_examples=100, deadline=None)
@given(step=st.one_of(st.integers(max_value=-1), st.integers(min_value=5, max_value=1000)))
def test_invalid_time_step_never_reaches_http_adapter(step: int) -> None:
    calls = 0

    def invoke(*_args: object, **_kwargs: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return {}

    contract = communication_contract()
    with patch.object(subgraph_runtime, "invoke_kpi_advisor_agent", invoke):
        result = subgraph_runtime._invoke_kpi_advisor_node(
            original_intent="evaluate",
            subtask_id="subtask-1",
            subtask="predict",
            langgraph_spec={
                "evaluation_request": {
                    "baseline_parameter_set": {
                        "cells": {
                            "gNB_5G": {
                                "tx_power_dbm": 40.0,
                                "ret_tilt_deg": 5.0,
                                "cio_bias_db": 0.0,
                                "hysteresis_db": 2.0,
                                "ttt_ms": 160,
                            }
                        }
                    },
                    "time_step": step,
                }
            },
            node={
                "id": "kpi-advisor-agent",
                "name": "KPI Advisor Agent",
                "_runtime_communication_contract": contract,
            },
            run_id=None,
        )
    assert calls == 0
    assert result["status"] == "advisor_unavailable"
