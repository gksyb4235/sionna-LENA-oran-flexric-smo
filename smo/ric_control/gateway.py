"""Loopback HTTP routes for real FlexRIC/E2 deployment.

rApp execution requires a matching policy in A1_Mediator_standalone.
Application success comes from ns-3 readbacks, independently of xApp exit status.
"""

import json
from fastapi import BackgroundTasks, FastAPI, HTTPException
import os
import httpx
from .messages import Submission, RAppSubmission, submission_digest
from .dispatch import submit
from .state import RUNS, live_run, reconcile

app = FastAPI(title="AGORA RIC/ns-3 control gateway")


A1_BASE = os.environ.get("AGORA_A1_MEDIATOR", "http://127.0.0.1:9000").rstrip("/")


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
    raise HTTPException(410, "Submit rApp policies to A1_Mediator_standalone on port 9000")


@app.post("/api/rapp/deployments", status_code=202)
async def from_rapp(body: RAppSubmission, background: BackgroundTasks):
    if body.policy_id != body.plan.get("deployment_id"):
        raise ValueError("A1 policy identity mismatch")
    try:
        async with httpx.AsyncClient(timeout=10, trust_env=False) as client:
            response = await client.get(
                A1_BASE + "/a1-p/policytypes/401/policies/" + body.policy_id
            )
        response.raise_for_status()
        policy = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(503, "Standalone A1 policy verification failed") from exc
    expected = body.model_dump(include={"live_run_id", "plan"})
    if (
        policy.get("submission") != expected
        or policy.get("policy_sha256") != body.policy_sha256
        or body.policy_sha256 != submission_digest(body)
    ):
        raise ValueError("Forwarded rApp plan differs from the stored A1 policy")
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
