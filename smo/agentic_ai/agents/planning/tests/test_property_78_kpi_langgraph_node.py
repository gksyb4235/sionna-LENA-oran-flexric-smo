"""Property 78: RAN impact plans include the registered KPI Advisor node."""

from __future__ import annotations

import sys
import types
from pathlib import Path

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

from runtime_registration import planner_document  # noqa: E402

import agent  # noqa: E402
from tools.knowledge_registry import KnowledgeRegistryResult  # noqa: E402


# **Property 78: langgraph_spec 노드 포함**
# **Validates: Requirement 12.5**
@settings(max_examples=100, deadline=None)
@given(marker=st.sampled_from(["RAN parameter impact", "KPI prediction", "파라미터 영향", "GNN"]))
def test_registered_advisor_node_contains_baseline_and_time_step(marker: str) -> None:
    baseline = {
        "cells": {
            "gNB_5G": {
                "tx_power_dbm": 40.0,
                "ret_tilt_deg": 5.0,
                "cio_bias_db": 0.0,
                "hysteresis_db": 2.0,
                "ttt_ms": 160,
            }
        }
    }
    seed = {
        "nodes": [],
        "edges": [],
        "evaluation_request": {"baseline_parameter_set": baseline, "time_step": 3},
    }
    knowledge = KnowledgeRegistryResult(used=True, agents=[planner_document()], tools=[])
    spec = agent.ensure_kpi_advisor_node(seed, knowledge, intent=marker, subtask_id="subtask-1", subtask=marker)
    assert spec["entrypoint"] == "kpi-advisor-agent"
    assert any(node["id"] == "kpi-advisor-agent" for node in spec["nodes"])
    assert spec["evaluation_request"] == seed["evaluation_request"]
