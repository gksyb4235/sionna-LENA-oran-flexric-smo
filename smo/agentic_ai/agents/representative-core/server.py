"""FastAPI server for the Representative Core Agent."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from agent import approve_report, run_representative_core

app = FastAPI(title="Representative Core Agent", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "representative-core", "default_namespace": "free5gc-v4"}


@app.post("/invoke")
def invoke(payload: dict[str, Any]) -> dict[str, Any]:
    intent = payload.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        raise HTTPException(status_code=400, detail="'intent' must be a non-empty string.")
    try:
        return run_representative_core(intent).to_dict()
    except Exception as exc:  # noqa: BLE001 - API should return useful operational errors.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/approve")
def approve(payload: dict[str, Any]) -> dict[str, Any]:
    report = payload.get("report")
    approved = payload.get("approved")
    approved_by = payload.get("approved_by")
    reason = payload.get("reason")
    if not isinstance(report, dict):
        raise HTTPException(status_code=400, detail="'report' must be a Representative Core report object.")
    if not isinstance(approved, bool):
        raise HTTPException(status_code=400, detail="'approved' must be a boolean.")
    if approved_by is not None and not isinstance(approved_by, str):
        raise HTTPException(status_code=400, detail="'approved_by' must be a string when provided.")
    if reason is not None and not isinstance(reason, str):
        raise HTTPException(status_code=400, detail="'reason' must be a string when provided.")
    try:
        return approve_report(report, approved=approved, approved_by=approved_by, reason=reason).to_dict()
    except Exception as exc:  # noqa: BLE001 - API should return useful operational errors.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
