"""FastAPI server for the Probe Agent."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from agent import run_monitoring

app = FastAPI(title="Probe Agent", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "monitoring", "metric": "cpu_usage_last_1m_avg"}


@app.post("/invoke")
def invoke(payload: dict[str, Any]) -> dict[str, Any]:
    intent = payload.get("intent")
    nf = payload.get("nf")
    evaluation_request = payload.get("evaluation_request")
    if not isinstance(intent, str) or not intent.strip():
        raise HTTPException(status_code=400, detail="'intent' must be a non-empty string.")
    if nf is not None and not isinstance(nf, str):
        raise HTTPException(status_code=400, detail="'nf' must be a string when provided.")
    if evaluation_request is not None and not isinstance(evaluation_request, dict):
        raise HTTPException(status_code=400, detail="'evaluation_request' must be an object when provided.")
    try:
        return run_monitoring(intent, nf=nf, evaluation_request=evaluation_request).to_dict()
    except Exception as exc:  # noqa: BLE001 - API should return useful operational errors.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
