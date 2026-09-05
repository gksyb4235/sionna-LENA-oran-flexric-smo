"""FastAPI server for the Planning Agent."""

from __future__ import annotations

import hmac
import os
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException

from agent import apply_monitoring_feedback, approve_core_report_from_payload, load_env_file, plan_from_decomposition, resume_planning_run

app = FastAPI(title="Planning Agent", version="0.1.0")


def _require_monitoring_token(authorization: str | None) -> None:
    load_env_file()
    token = os.getenv("MONITORING_AGENT_TOKEN", "").strip()
    if not token:
        raise HTTPException(status_code=503, detail="MONITORING_AGENT_TOKEN is not configured.")
    if not authorization or not hmac.compare_digest(authorization, f"Bearer {token}"):
        raise HTTPException(status_code=401, detail="Invalid Monitoring Agent token.")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "agent": "planning",
        "runtime": "orchestrator_loop_v1",
    }


@app.post("/invoke")
def invoke(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        return plan_from_decomposition(payload).to_dict()
    except Exception as exc:  # noqa: BLE001 - API should return a useful validation error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/resume")
def resume(payload: dict[str, Any]) -> dict[str, Any]:
    run_id = payload.get("run_id")
    last_agent_result = payload.get("last_agent_result")
    if not isinstance(run_id, str) or not run_id.strip():
        raise HTTPException(status_code=400, detail="'run_id' must be a non-empty string.")
    if last_agent_result is not None:
        raise HTTPException(status_code=400, detail="'last_agent_result' is not accepted; use /approve or /feedback.")
    try:
        return resume_planning_run(run_id, last_agent_result=last_agent_result).to_dict()
    except Exception as exc:  # noqa: BLE001 - API should return a useful validation/runtime error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/feedback")
def feedback(payload: dict[str, Any], authorization: Annotated[str | None, Header()] = None) -> dict[str, Any]:
    _require_monitoring_token(authorization)
    try:
        return apply_monitoring_feedback(payload).to_dict()
    except Exception as exc:  # noqa: BLE001 - API should return a useful validation/runtime error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/approve")
def approve(payload: dict[str, Any]) -> dict[str, Any]:
    report = payload.get("report")
    approved = payload.get("approved")
    approved_by = payload.get("approved_by")
    reason = payload.get("reason")
    run_id = payload.get("run_id") or payload.get("planning_run_id")
    if not isinstance(report, dict):
        raise HTTPException(status_code=400, detail="'report' must be a Representative Core report object.")
    if not isinstance(run_id, str) or not run_id.strip():
        raise HTTPException(status_code=400, detail="'run_id' must be a non-empty string.")
    if not isinstance(approved, bool):
        raise HTTPException(status_code=400, detail="'approved' must be a boolean.")
    if approved_by is not None and not isinstance(approved_by, str):
        raise HTTPException(status_code=400, detail="'approved_by' must be a string when provided.")
    if reason is not None and not isinstance(reason, str):
        raise HTTPException(status_code=400, detail="'reason' must be a string when provided.")
    approval_payload: dict[str, Any] = {"report": report, "run_id": run_id}
    try:
        return approve_core_report_from_payload(
            approval_payload,
            approved=approved,
            approved_by=approved_by,
            reason=reason,
        )
    except Exception as exc:  # noqa: BLE001 - API should return a useful validation/runtime error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
