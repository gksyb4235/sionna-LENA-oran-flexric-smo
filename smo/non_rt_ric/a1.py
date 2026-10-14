"""Durable A1 policies, shared rApp handlers, and downstream execution status."""

from contextlib import contextmanager
import json
import os
from pathlib import Path
import sqlite3
import httpx
from fastapi import APIRouter, HTTPException
from smo.agentic_ai.agora_offline.applications import applications, prepare_application
from smo.ric_control.messages import Submission, submission_digest
from smo.ric_control.protocol import identifier
from smo.ric_control.state import now

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path(os.environ.get("AGORA_RUNTIME_DIR", ROOT / "experiments/agora_runtime"))
GATEWAY = os.environ.get("AGORA_RIC_GATEWAY", "http://127.0.0.1:18082").rstrip("/")
router = APIRouter()
PREFIX = "/a1-p/policytypes/401"


@contextmanager
def database():
    RUNTIME.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(RUNTIME / "a1-policies.sqlite3", timeout=15)
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS policies "
            "(id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        connection.commit()
        with connection:
            yield connection
    finally:
        connection.close()


def read_policy(ident):
    identifier(ident)
    with database() as db:
        row = db.execute("SELECT payload FROM policies WHERE id=?", (ident,)).fetchone()
    if row is None:
        raise HTTPException(404, "Unknown AGORA A1 policy")
    return json.loads(row[0])


def save_policy(value):
    with database() as db:
        db.execute(
            "UPDATE policies SET payload=? WHERE id=?",
            (json.dumps(value, allow_nan=False), value["policy_id"]),
        )


async def gateway(method, path, body=None):
    try:
        async with httpx.AsyncClient(timeout=15, trust_env=False) as client:
            response = await client.request(method, GATEWAY + path, json=body)
        payload = response.json()
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(503, "Execution gateway unavailable; policy remains stored") from exc
    if response.is_error:
        raise HTTPException(response.status_code, payload.get("detail", "Execution rejected"))
    return payload


@router.get("/api/non-rt/health")
def health():
    return {
        "ok": True,
        "component": "agora_non_rt_ric",
        "host": "non-rt-ric",
        "policy_type": 401,
    }


@router.get("/api/non-rt/apps")
def app_catalog():
    return {
        "apps": [app.describe() for app in applications().values()],
        "rapps": [app.describe() for app in applications().values() if app.role == "rApp"],
    }


@router.get(PREFIX)
def policy_type():
    return {
        "policy_type_id": 401,
        "name": "AGORA reviewed rApp intervention",
        "create_schema": Submission.model_json_schema(),
        "application_ids": ["energy_saving", "coverage"],
        "implementation": "local A1 profile; not an O-RAN conformance claim",
    }


@router.put(PREFIX)
@router.delete(PREFIX)
def fixed_policy_type():
    raise HTTPException(409, "AGORA policy type 401 is application-managed")


@router.get(PREFIX + "/policies")
def policy_ids():
    with database() as db:
        return [row[0] for row in db.execute("SELECT id FROM policies ORDER BY rowid DESC")]


@router.get(PREFIX + "/policies/{ident}")
def policy_instance(ident: str):
    return read_policy(ident)


@router.put(PREFIX + "/policies/{ident}", status_code=202)
async def receive_policy(ident: str, body: Submission):
    try:
        if identifier(ident) != body.plan.get("deployment_id"):
            raise ValueError("Policy id does not match the reviewed deployment")
        if body.plan.get("schema_version") != "agora-ric-queue-v1":
            raise ValueError("Expected an operator-reviewed deployment")
        app, prepared = prepare_application(body.plan, role="rApp")
        fingerprint = submission_digest(body)
    except (ValueError, KeyError, TypeError) as exc:
        raise HTTPException(422, str(exc)) from exc
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("SELECT fingerprint FROM policies WHERE id=?", (ident,)).fetchone()
        if existing and existing[0] != fingerprint:
            raise HTTPException(409, "Policy id already binds a different plan or simulator")
        if not existing:
            saved = {
                "policy_id": ident,
                "policy_type_id": 401,
                "policy_sha256": fingerprint,
                "submission": body.model_dump(),
                "application": app.describe(),
                "policy_stored": True,
                "created_at": now(),
                "status": "policy_stored",
                "rapp_invocation": {
                    "application_id": app.app_id,
                    "status": "prepared",
                    "prepared_at": now(),
                },
                "path": ["Non_RT_RIC", app.name, "execution_xApp", "FlexRIC_Near_RT", "E2", "ns3"],
            }
            db.execute(
                "INSERT INTO policies VALUES (?,?,?)",
                (ident, fingerprint, json.dumps(saved, allow_nan=False)),
            )
    saved = read_policy(ident)
    if saved.get("forwarded"):
        return await policy_status(ident)
    try:
        downstream = await gateway(
            "POST",
            "/api/rapp/deployments",
            {
                "plan": prepared,
                "live_run_id": body.live_run_id,
                "policy_id": ident,
                "policy_sha256": fingerprint,
            },
        )
    except HTTPException as exc:
        saved.update(status="dispatch_failed", transport_note=str(exc.detail))
        save_policy(saved)
        raise
    saved.update(forwarded=True, status="execution_accepted", downstream=downstream)
    saved["rapp_invocation"].update(status="forwarded", forwarded_at=now())
    save_policy(saved)
    return {
        **downstream,
        "policy_stored": True,
        "application": saved["application"],
        "rapp_invocation": saved["rapp_invocation"],
        "path": saved["path"],
        "a1_policy_id": ident,
    }


@router.get(PREFIX + "/policies/{ident}/status")
async def policy_status(ident: str):
    saved = read_policy(ident)
    try:
        downstream = await gateway("GET", "/api/deployments/" + ident)
    except HTTPException as exc:
        # Storage alone never implies scheduling or application.
        return {
            "deployment_id": ident,
            "candidate_id": saved["submission"]["plan"]["candidate_id"],
            "live_run_id": saved["submission"]["live_run_id"],
            "target_ric": "non_rt",
            "status": saved["status"],
            "policy_stored": True,
            "forwarded": False,
            "applied": False,
            "restored": False,
            "receipts": [],
            "application": saved["application"],
            "path": saved["path"],
            "transport_note": str(exc.detail),
            "execution_status_available": False,
            "binding": {
                "start_s": saved["submission"]["plan"]["candidate"]["interval_start_s"],
                "end_s": saved["submission"]["plan"]["candidate"]["interval_end_s"],
            },
        }
    return {
        **downstream,
        "policy_stored": True,
        "application": saved["application"],
        "rapp_invocation": saved["rapp_invocation"],
        "path": saved["path"],
        "a1_policy_id": ident,
        "execution_status_available": True,
    }


@router.delete(PREFIX + "/policies/{ident}")
def delete_policy(ident: str):
    read_policy(ident)
    raise HTTPException(
        409, "An accepted timed intervention cannot be cancelled by deleting its audit record"
    )
