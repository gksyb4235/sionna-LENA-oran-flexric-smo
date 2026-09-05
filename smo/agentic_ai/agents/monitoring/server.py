"""FastAPI server for the Monitoring Agent."""

from __future__ import annotations

import hmac
import os
from typing import Annotated, Any

from fastapi import FastAPI, Header, HTTPException

from agent import load_env_file, run_monitoring

app = FastAPI(title="Monitoring Agent", version="0.1.0")


def _require_token(authorization: str | None) -> None:
    load_env_file()
    token = os.getenv("MONITORING_AGENT_TOKEN", "").strip()
    if not token:
        raise HTTPException(status_code=503, detail="MONITORING_AGENT_TOKEN is not configured.")
    if not authorization or not hmac.compare_digest(authorization, f"Bearer {token}"):
        raise HTTPException(status_code=401, detail="Invalid Monitoring Agent token.")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "agent": "monitoring", "probe": "delegated"}


@app.post("/invoke")
def invoke(payload: dict[str, Any], authorization: Annotated[str | None, Header()] = None) -> dict[str, Any]:
    _require_token(authorization)
    try:
        return run_monitoring(payload).to_dict()
    except Exception as exc:  # noqa: BLE001 - API should return a useful validation error.
        raise HTTPException(status_code=400, detail=str(exc)) from exc
