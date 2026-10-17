"""Local-only offline API and streaming agent workbench."""

import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
import os
import re
from typing import Literal
from types import SimpleNamespace
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, ConfigDict
from . import __version__
from . import live_control
from .runtime import Runtime
from .estimator import Estimator
from .intent_policy import catalog, compile_spec
from .orchestrator import Session
from .operator_review import ensure_review
from .applications import application_for
from .deployment_queue import enqueue, get_receipt, list_receipts, QueueConflict
from .provider import public_configuration
from .settings import ROOT, STATE, read_settings, write_settings
from .search_limits import MAX_BATCH_SIZE, MAX_ALTERNATIVE_ROUNDS, MAX_CANDIDATES

runtime = None
sessions = {}


@asynccontextmanager
async def lifespan(app):
    global runtime
    runtime = await asyncio.to_thread(Runtime)
    yield
    tasks = [s.task for s in sessions.values() if s.task and not s.task.done()]
    for task in tasks:
        task.cancel()
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)


app = FastAPI(title="AGORA offline Temporal GNN workbench", version=__version__, lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")


@app.exception_handler(ValueError)
async def bad_request(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.get("/")
async def index():
    static = ROOT / "static"
    html = (static / "index.html").read_text(encoding="utf-8")
    for name in (
        "app.js",
        "styles.css",
        "policy-editor.js",
        "policy-editor.css",
        "operator-review.js",
        "operator-review.css",
    ):
        if f'"/static/{name}"' not in html:
            continue
        version = hashlib.sha256((static / name).read_bytes()).hexdigest()[:16]
        html = html.replace(f'"/static/{name}"', f'"/static/{name}?v={version}"')
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/api/health")
async def health():
    return {
        "app_id": "agora-offline-final-gnn",
        "version": __version__,
        "ready": runtime is not None,
        "pid": os.getpid(),
    }


@app.get("/api/status")
async def status():
    if runtime is None:
        raise HTTPException(503, "Model loading")
    return {
        "mode": "offline",
        "model": runtime.metadata(),
        "provider": public_configuration(read_settings()),
        "limits": {
            "horizons_s": [90, 180],
            "delays_s": [0, 30, 60, 90],
            "issue_time_s": [30, 720],
            "max_batch_size": MAX_BATCH_SIZE,
            "max_alternative_rounds": MAX_ALTERNATIVE_ROUNDS,
            "max_candidates": MAX_CANDIDATES,
        },
        "legacy_overview_url": "http://127.0.0.1:8773",
        "legacy_agent_dashboard_url": "http://127.0.0.1:18080",
        "active_sessions": sum(s.status in ("queued", "running") for s in sessions.values()),
    }


@app.get("/api/runs")
async def runs(split: str | None = None):
    return {"runs": runtime.list_runs(split)}


@app.get("/api/snapshot")
async def snapshot(run_id: str, t: int = 180):
    return await asyncio.to_thread(runtime.snapshot, run_id, t)


@app.get("/api/settings")
async def settings():
    result = read_settings()
    result["provider"] = public_configuration(result)
    return result


@app.put("/api/settings")
async def update_settings(body: dict):
    result = write_settings(body)
    result["provider"] = public_configuration(result)
    return result


@app.get("/api/intent-policy")
async def intent_policy(horizon_s: int = 180):
    return catalog(horizon_s)


class SessionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    prompt: str = Field(min_length=3, max_length=16000)
    run_id: str = Field(min_length=1, max_length=80)
    issue_time_s: int = Field(ge=30, le=720)
    mode: Literal["llm", "deterministic"] = "llm"
    horizon_s: Literal[90, 180] = 180
    intent_override: dict | None = None
    intent_spec: dict | None = None


@app.post("/api/sessions")
async def create_session(body: SessionInput):
    if body.intent_spec is not None:
        if body.intent_override is not None:
            raise HTTPException(422, "Use the policy editor or legacy JSON override, not both")
        compile_spec(body.intent_spec, body.horizon_s)
    if sum(s.status in ("running", "queued") for s in sessions.values()) >= 2:
        raise HTTPException(429, "At most two sessions may run concurrently")
    await asyncio.to_thread(runtime.snapshot, body.run_id, body.issue_time_s)
    settings = read_settings()
    if body.mode == "llm" and not public_configuration(settings)["configured"]:
        raise HTTPException(
            422,
            "No OpenAI key is configured in the existing server environment; use deterministic mode or configure server .env",
        )
    session = Session(body.model_dump(), runtime, settings)
    sessions[session.id] = session
    session.task = asyncio.create_task(session.run())
    return {"session_id": session.id}


def get_session(ident):
    if ident not in sessions:
        if not re.fullmatch(r"[0-9a-f]{32}", ident):
            raise HTTPException(404, "Unknown session")
        path = STATE / "sessions" / ident / "session.json"
        if not path.is_file():
            raise HTTPException(404, "Unknown session")
        saved = json.loads(path.read_text(encoding="utf-8"))
        sessions[ident] = SimpleNamespace(
            id=ident,
            status=saved["status"],
            result=saved["result"],
            events=saved["events"],
            state=lambda: saved,
            changed=asyncio.Event(),
            task=None,
        )
    return sessions[ident]


@app.get("/api/sessions/{ident}")
async def session_state(ident: str):
    session = get_session(ident)
    ensure_review(session.result, session.events)
    view = dict(session.state())
    view["deployment"] = await asyncio.to_thread(get_receipt, STATE, ident)
    return view


class DeploymentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    candidate_id: str = Field(min_length=1, max_length=80)
    target_ric: Literal["near_rt", "non_rt", "split"]
    review_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    accept_constraint_violations: bool = False


class LiveDeploymentInput(DeploymentInput):
    live_run_id: str = Field(pattern=r"^[0-9a-f]{32}$")


@app.get("/api/live/runs")
async def live_runs():
    return await live_control.request("GET", "/api/runs")


@app.post("/api/sessions/{ident}/deploy-live")
async def live_deployment(ident: str, body: LiveDeploymentInput):
    session = get_session(ident)
    if body.target_ric == "split":
        raise HTTPException(422, "실제 적용은 Near-RT 또는 Non-RT 경로를 선택하세요.")
    try:
        result = ensure_review(session.result, session.events)
        if session.status != "completed" or not result:
            raise HTTPException(409, "Complete the evaluated application intent first")
        application = application_for(result.get("intent"))
        if application is None:
            raise HTTPException(422, "Live deployment needs one of the five application intents")
        candidate = next(
            (
                row
                for row in result["operator_review"]["candidates"]
                if row["candidate_id"] == body.candidate_id
            ),
            None,
        )
        if candidate is None:
            raise HTTPException(422, "Unknown reviewed candidate")
        application.prepare(
            {
                "target_ric": body.target_ric,
                "candidate": candidate,
                "current_parameters": result["execution_plan"]["current_parameters"],
            }
        )
        receipt = await asyncio.to_thread(
            enqueue, STATE, session, **body.model_dump(exclude={"live_run_id"})
        )
        return await live_control.deploy(receipt, body.live_run_id)
    except QueueConflict as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/sessions/{ident}/deployment-live")
async def live_deployment_status(ident: str):
    get_session(ident)
    receipt = await asyncio.to_thread(get_receipt, STATE, ident)
    if receipt is None:
        raise HTTPException(404, "등록된 실행안이 없습니다.")
    return await live_control.deployment_status(receipt)


@app.get("/api/deployments")
async def deployments(ric: Literal["near_rt", "non_rt"] | None = None):
    return {
        "mode": "offline_queue",
        "deployments": await asyncio.to_thread(list_receipts, STATE, ric),
        "forwarded": False,
        "applied": False,
    }


@app.post("/api/sessions/{ident}/deploy")
async def queue_deployment(ident: str, body: DeploymentInput):
    session = get_session(ident)
    try:
        return await asyncio.to_thread(enqueue, STATE, session, **body.model_dump())
    except QueueConflict as exc:
        raise HTTPException(409, str(exc))


@app.get("/api/sessions/{ident}/deployment")
async def deployment_state(ident: str):
    get_session(ident)
    receipt = await asyncio.to_thread(get_receipt, STATE, ident)
    if receipt is None:
        raise HTTPException(404, "등록된 배포 대기 항목이 없습니다.")
    return JSONResponse(
        receipt,
        headers={"Content-Disposition": f'attachment; filename="agora-ric-queue-{ident}.json"'},
    )


@app.post("/api/sessions/{ident}/cancel")
async def cancel(ident: str):
    session = get_session(ident)
    if session.status in ("running", "queued"):
        was_queued = session.status == "queued"
        session.cancelled = True
        session.task.cancel()
        if was_queued:
            session.status = "cancelled"
            await session.emit(
                "done", "system", "cancelled", {"status": "cancelled", "session_id": ident}
            )
            session.persist()
    return {"session_id": ident, "status": session.status, "cancellation_requested": True}


@app.get("/api/sessions/{ident}/events")
async def events(ident: str, request: Request, after: int = 0):
    session = get_session(ident)
    try:
        cursor = max(after, int(request.headers.get("last-event-id", "0")))
    except ValueError:
        raise HTTPException(422, "Invalid Last-Event-ID")

    async def stream():
        nonlocal cursor
        while True:
            session.changed.clear()
            for event in session.events[cursor:]:
                cursor = event["seq"]
                yield f"id: {cursor}\ndata: {json.dumps(event, ensure_ascii=False, allow_nan=False)}\n\n"
            if session.status in ("completed", "failed", "cancelled"):
                break
            if await request.is_disconnected():
                break
            try:
                await asyncio.wait_for(session.changed.wait(), timeout=10)
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/sessions/{ident}/export")
async def export(ident: str):
    session = get_session(ident)
    if session.result is None:
        raise HTTPException(409, "No completed decision is available for export")
    ensure_review(session.result, session.events)
    return JSONResponse(
        content={
            "result": session.result,
            "events": session.events,
            "deployment": await asyncio.to_thread(get_receipt, STATE, ident),
        },
        headers={"Content-Disposition": f'attachment; filename="agora-{ident}.json"'},
    )


@app.post("/api/estimate/batch")
async def estimate(body: dict):
    if not isinstance(body.get("candidates"), list) or not 1 <= len(body["candidates"]) <= MAX_BATCH_SIZE:
        raise HTTPException(422, f"Supply1..{MAX_BATCH_SIZE} candidates")
    required = ("run_id", "issue_time_s", "intent", "requested_plan", "candidates")
    if any(k not in body for k in required):
        raise HTTPException(422, "Missing one of: " + ", ".join(required))
    estimator = Estimator(runtime, read_settings()["policy"])
    return await asyncio.to_thread(
        estimator.evaluate, *[body[k] for k in required], 2, bool(body.get("explain", False))
    )


@app.post("/api/predict/batch")
async def predict(body: dict):
    if not isinstance(body.get("plans"), list) or not 1 <= len(body["plans"]) <= MAX_BATCH_SIZE:
        raise HTTPException(422, f"Supply1..{MAX_BATCH_SIZE} plans")
    if "run_id" not in body or "issue_time_s" not in body:
        raise HTTPException(422, "run_id and issue_time_s are required")
    return {
        "results": await asyncio.to_thread(
            runtime.predict_batch, body["run_id"], body["issue_time_s"], body["plans"]
        )
    }


def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    uvicorn.run("agora_offline.server:app", host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
