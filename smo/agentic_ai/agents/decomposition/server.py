"""FastAPI server for the Decomposition Agent."""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException

from agent import run_decomposition, run_decomposition_then_planning

app = FastAPI(title="Decomposition Agent", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "agent": "decomposition",
    }


@app.post("/invoke")
def invoke(payload: dict[str, Any]) -> dict[str, Any]:
    intent = payload.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        raise HTTPException(status_code=400, detail="'intent' must be a non-empty string.")
    try:
        return run_decomposition(intent).to_dict()
    except Exception as exc:  # noqa: BLE001 - API should return useful validation/runtime errors.
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/invoke-and-plan")
def invoke_and_plan(payload: dict[str, Any]) -> dict[str, Any]:
    intent = payload.get("intent")
    if not isinstance(intent, str) or not intent.strip():
        raise HTTPException(status_code=400, detail="'intent' must be a non-empty string.")
    try:
        return run_decomposition_then_planning(intent)
    except Exception as exc:  # noqa: BLE001 - API should return useful validation/runtime errors.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
