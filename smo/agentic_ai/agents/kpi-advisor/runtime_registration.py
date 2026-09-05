"""Knowledge DB registration for the KPI Advisor planning runtime."""

from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from typing import Any, Protocol

AGENT_ID = "kpi-advisor-agent"
DEFAULT_DATABASE = "knowledge"
DEFAULT_COLLECTION = "agents"
DEFAULT_TIMEOUT_MS = 1500


class CollectionLike(Protocol):
    def create_index(self, keys: Any, **kwargs: Any) -> Any: ...

    def update_one(self, filter: dict[str, Any], update: dict[str, Any], **kwargs: Any) -> Any: ...

    def find_one(self, filter: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any] | None: ...


def _cell_parameters_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["tx_power_dbm", "ret_tilt_deg", "cio_bias_db", "hysteresis_db", "ttt_ms"],
        "properties": {
            "tx_power_dbm": {"type": "number", "minimum": 30},
            "ret_tilt_deg": {"type": "number", "minimum": 0},
            "cio_bias_db": {"type": "number", "minimum": -6},
            "hysteresis_db": {"type": "number", "minimum": 0},
            "ttt_ms": {"type": "integer", "minimum": 0},
        },
        "additionalProperties": False,
    }


def _parameter_set_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "required": ["cells"],
        "properties": {
            "cells": {
                "type": "object",
                "additionalProperties": _cell_parameters_schema(),
            }
        },
        "additionalProperties": False,
    }


def communication_contract() -> dict[str, Any]:
    """Return the authoritative Planning/runtime schema for the advisor."""
    parameter_set = _parameter_set_schema()
    evaluation_request = {
        "type": "object",
        "required": ["baseline_parameter_set", "time_step"],
        "properties": {
            "baseline_parameter_set": parameter_set,
            "time_step": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
        },
        "additionalProperties": False,
    }
    return {
        "version": "kpi-advisor-v1",
        "schema_dialect": "json-schema-subset-v1",
        "planning": {"evaluation_request_schema": evaluation_request},
        "request_schema": {
            "type": "object",
            "required": ["intent", "baseline_parameter_set", "time_step"],
            "properties": {
                "intent": {"type": "string", "minLength": 1},
                "baseline_parameter_set": parameter_set,
                "time_step": {"type": "integer", "enum": [0, 1, 2, 3, 4]},
            },
            "additionalProperties": False,
        },
        "response_schema": {
            "type": "object",
            "required": [
                "degradation_verdict",
                "recommendations",
                "evidence_record_id",
                "rationale_summary",
            ],
            "properties": {
                "degradation_verdict": {"type": "string", "enum": ["acceptable", "degrading", "unknown"]},
                "recommendations": {"type": "array", "items": {"type": "object"}},
                "evidence_record_id": {"type": "string", "minLength": 1},
                "rationale_summary": {"type": "string", "minLength": 1},
            },
            "additionalProperties": False,
        },
    }


def _contract_ref(contract: dict[str, Any]) -> dict[str, str]:
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return {
        "agent_id": AGENT_ID,
        "version": str(contract["version"]),
        "hash": f"sha256:{hashlib.sha256(canonical.encode('utf-8')).hexdigest()}",
    }


def planner_document() -> dict[str, Any]:
    """Build the single Knowledge DB agent document used by Planning Agent."""
    contract = communication_contract()
    return {
        "id": AGENT_ID,
        "agent_id": AGENT_ID,
        "type": "agent",
        "role": "kpi_advisor",
        "name": "KPI Advisor Agent",
        "domain": "O-RAN RAN parameter impact prediction",
        "description": "Predicts KPI effects of cell parameter sets and returns evidence-backed recommendations.",
        "can_do": [
            "predict RAN parameter KPI impact",
            "detect indirect KPI conflicts",
            "build temporal parameter plans",
        ],
        "cannot_do": ["change RAN state directly", "issue shell or kubectl commands"],
        "capabilities": ["gnn-batch-probing", "indirect-conflict-detection", "temporal-planning"],
        "tags": ["RAN", "KPI", "GNN", "parameter", "prediction", "conflict", "temporal"],
        "keywords": ["RAN parameter impact", "baseline parameter set", "time step", "KPI prediction"],
        "inputs": ["intent", "baseline_parameter_set", "time_step"],
        "outputs": ["degradation_verdict", "recommendations", "evidence_record_id", "rationale_summary"],
        "runtime": {
            "transport": "http",
            "url_env": "KPI_ADVISOR_AGENT_URL",
            "default_url": "http://127.0.0.1:8110",
        },
        "status": "active",
        "communication_contract": contract,
        "planner_manifest": {
            "contract_version": contract["version"],
            "purpose": "Use for RAN parameter impact prediction with an explicit baseline and Time_Step.",
            "capabilities": ["gnn-batch-probing", "indirect-conflict-detection", "temporal-planning"],
            "constraints": ["Read-only advisory execution", "Requires baseline_parameter_set and time_step"],
            "planning_contract": {
                "required_inputs": ["baseline_parameter_set", "time_step"],
                "result": "evidence-backed-kpi-advice",
            },
            "planning_schema_path": "planning.evaluation_request_schema",
            "contract_ref": _contract_ref(contract),
        },
    }


def register_kpi_advisor(collection: CollectionLike) -> dict[str, Any]:
    """Atomically upsert and read back the KPI Advisor planner manifest."""
    document = planner_document()
    now = datetime.now(UTC)
    collection.create_index("agent_id", unique=True)
    collection.update_one(
        {"agent_id": AGENT_ID},
        {"$set": {**document, "updated_at": now}, "$setOnInsert": {"created_at": now}},
        upsert=True,
    )
    registered = collection.find_one({"agent_id": AGENT_ID})
    if not isinstance(registered, dict):
        raise RuntimeError("KPI Advisor planner_manifest registration could not be read back")
    return registered


def register_from_environment() -> dict[str, Any] | None:
    """Register with configured MongoDB; an unset URI intentionally means no candidate exposure."""
    uri = os.getenv("KNOWLEDGE_MONGODB_URI", "").strip()
    if not uri:
        return None
    from pymongo import MongoClient

    timeout_ms = int(os.getenv("KNOWLEDGE_TIMEOUT_MS", str(DEFAULT_TIMEOUT_MS)))
    client = MongoClient(uri, serverSelectionTimeoutMS=timeout_ms)
    try:
        client.admin.command("ping")
        database = client[os.getenv("KNOWLEDGE_DATABASE", DEFAULT_DATABASE)]
        collection = database[os.getenv("KNOWLEDGE_AGENTS_COLLECTION", DEFAULT_COLLECTION)]
        return register_kpi_advisor(collection)
    finally:
        client.close()


__all__ = [
    "AGENT_ID",
    "communication_contract",
    "planner_document",
    "register_from_environment",
    "register_kpi_advisor",
]
