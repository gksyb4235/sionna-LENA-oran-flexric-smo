"""Property 82: advisor failures become a continuation-safe status."""

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


def _node() -> dict[str, object]:
    return {
        "id": "kpi-advisor-agent",
        "name": "KPI Advisor Agent",
        "_runtime_communication_contract": communication_contract(),
    }


def _spec(time_step: int = 0) -> dict[str, object]:
    return {
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
            "time_step": time_step,
        }
    }


# **Property 82: advisor_unavailable 허용 진행**
# **Validates: Requirement 12.10**
@settings(max_examples=100, deadline=None)
@given(reason=st.text(min_size=1, max_size=40))
def test_advisor_failure_is_localized_as_continuable(reason: str) -> None:
    def fail(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError(reason)

    with patch.object(subgraph_runtime, "invoke_kpi_advisor_agent", fail):
        result = subgraph_runtime._invoke_kpi_advisor_node(
            original_intent="evaluate RAN impact",
            subtask_id="subtask-1",
            subtask="predict KPI",
            langgraph_spec=_spec(),
            node=_node(),
            run_id=None,
        )
    assert result["status"] == "advisor_unavailable"
    assert subgraph_runtime._adapter_completion(result) == "advisor_unavailable"
