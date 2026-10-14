"""Loopback HTTP routes for real FlexRIC/E2 deployment.

The Non-RT endpoint is a local A1 policy adapter, not a complete O-RAN SC RIC.
Application success comes from ns-3 readbacks, independently of xApp exit status.
"""

import json
from fastapi import BackgroundTasks, FastAPI
from pydantic import BaseModel, ConfigDict, Field
from .dispatch import submit
from .protocol import identifier
from .state import RUNS, live_run, reconcile

app = FastAPI(title="AGORA RIC/ns-3 control gateway")


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    live_run_id: str = Field(pattern="^[0-9a-f]{32}$")
    plan: dict


@app.exception_handler(ValueError)
async def invalid(_, exc):
    from fastapi.responses import JSONResponse

    return JSONResponse(status_code=409, content={"detail": str(exc)})


@app.get("/health")
def health():
    return {"ok": True, "mode": "real_flexric_e2", "automatic_offline_queue_consumer": False}


@app.get("/api/runs")
def runs():
    result = []
    for path in sorted(RUNS.glob("*/status.json"), key=lambda p: p.stat().st_mtime, reverse=True)[
        :40
    ]:
        try:
            result.append(live_run(path.parent.name))
        except (ValueError, OSError, json.JSONDecodeError):
            continue
    return {"runs": result}


@app.post("/api/near-rt/deployments", status_code=202)
def near_rt(body: Submission, background: BackgroundTasks):
    return submit(body, background, "near_rt")


@app.put("/a1-p/policytypes/401/policies/{ident}", status_code=202)
def non_rt(ident: str, body: Submission, background: BackgroundTasks):
    if identifier(ident) != body.plan.get("deployment_id"):
        raise ValueError("Policy id mismatch")
    return submit(body, background, "non_rt")


@app.get("/a1-p/policytypes/401/policies/{ident}/status")
def policy_status(ident: str):
    return reconcile(ident)


@app.get("/api/deployments/{ident}")
def deployment(ident: str):
    return reconcile(ident)


def main():
    import argparse, uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18082)
    uvicorn.run(app, host="127.0.0.1", port=parser.parse_args().port)


if __name__ == "__main__":
    main()
