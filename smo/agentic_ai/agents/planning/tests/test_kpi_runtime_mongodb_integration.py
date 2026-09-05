"""Task 11.9: real MongoDB manifest registration and Planning candidate selection."""

from __future__ import annotations

import os
import sys
import types
import uuid
from pathlib import Path

import pytest
from pymongo import MongoClient

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

from runtime_registration import register_kpi_advisor  # noqa: E402

import agent  # noqa: E402
from tools.knowledge_registry import KnowledgeRegistryResult, planner_manifest_is_valid  # noqa: E402


@pytest.mark.skipif(not os.getenv("TEST_MONGODB_URI"), reason="TEST_MONGODB_URI is required")
def test_real_knowledge_db_roundtrip_exposes_planning_candidate() -> None:
    client = MongoClient(os.environ["TEST_MONGODB_URI"], serverSelectionTimeoutMS=2000)
    database_name = f"knowledge_contract_{uuid.uuid4().hex}"
    try:
        collection = client[database_name]["agents"]
        registered = register_kpi_advisor(collection)
        loaded = collection.find_one({"agent_id": "kpi-advisor-agent"})
        assert loaded is not None and loaded["_id"] == registered["_id"]
        assert planner_manifest_is_valid(loaded)

        knowledge = KnowledgeRegistryResult(used=True, agents=[loaded], tools=[])
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
        spec = agent.ensure_kpi_advisor_node(
            {
                "nodes": [],
                "edges": [],
                "evaluation_request": {"baseline_parameter_set": baseline, "time_step": 0},
            },
            knowledge,
            intent="RAN parameter impact",
            subtask_id="subtask-1",
            subtask="KPI prediction",
        )
        assert spec["entrypoint"] == "kpi-advisor-agent"
    finally:
        client.drop_database(database_name)
        client.close()
