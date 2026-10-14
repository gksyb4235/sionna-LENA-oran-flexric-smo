"""Validate reviewed plans, reserve durable IDs, and send controls through E2."""

import json
import os
import subprocess
import threading
from .protocol import encode, identifier
from .messages import submission_digest
from smo.agentic_ai.agora_offline.applications import prepare_application
from .state import ROOT, RUNTIME, database, live_run, now, read_saved, reconcile, update

# FlexRIC xApps are short-lived clients. Serialize their connection lifecycle.
LOCK = threading.Lock()


def transmit(ident):
    """Serialize xApp processes; never infer application from a zero exit code."""
    with LOCK:
        saved = read_saved(ident)
        try:
            binding = encode(saved["plan"], live_run(saved["live_run_id"]))
            exe = ROOT / "near-rt-ric/build/examples/xApp/c/ctrl/xapp_agora_ctrl"
            if not exe.is_file():
                raise ValueError("Build xapp_agora_ctrl first")
            log = RUNTIME / "logs" / f"{ident}.xapp.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            env = dict(
                os.environ,
                AGORA_E2_PAYLOAD=binding["payload"],
                AGORA_E2_NODE_ID=str(binding["e2_node_id"]),
            )
            saved["status"] = "transmitting"
            saved["transmit_started_at"] = now()
            update(saved)
            with log.open("w") as output:
                result = subprocess.run(
                    [
                        str(exe),
                        "-c",
                        str(ROOT / "smo/ric_control/config/flexric.conf"),
                        "-p",
                        "/usr/local/lib/flexric/",
                    ],
                    cwd=RUNTIME,
                    env=env,
                    stdout=output,
                    stderr=subprocess.STDOUT,
                    timeout=25,
                )
            text = log.read_text(errors="replace")
            saved["xapp_exit_code"] = result.returncode
            saved["forwarded"] = result.returncode == 0 and "AGORA_E2_SUBMITTED" in text
            saved["status"] = "submitted" if saved["forwarded"] else "transport_failed"
            saved["transport_note"] = (
                "Awaiting ns-3 receipt" if saved["forwarded"] else text[-3000:]
            )
        except (ValueError, OSError, subprocess.TimeoutExpired) as exc:
            saved["status"] = "transport_failed"
            saved["transport_note"] = str(exc)
        saved["transport_finished_at"] = now()
        update(saved)


def submit(body, background, route):
    plan = body.plan
    if plan.get("schema_version") != "agora-ric-queue-v1":
        raise ValueError("Expected an operator-reviewed deployment plan")
    if plan.get("target_ric") != route:
        raise ValueError("Selected route does not match the endpoint")
    ident = identifier(plan["deployment_id"])
    application, plan = prepare_application(plan, "rApp" if route == "non_rt" else "xApp")
    digest = submission_digest(body)
    with database() as db:
        db.execute("BEGIN IMMEDIATE")
        existing = db.execute("SELECT fingerprint FROM deployments WHERE id=?", (ident,)).fetchone()
        if existing:
            if existing[0] != digest:
                raise ValueError(
                    "Deployment id already binds a different reviewed plan or simulation"
                )
        else:
            binding = encode(plan, live_run(body.live_run_id))
            saved = {
                "deployment_id": ident,
                "live_run_id": body.live_run_id,
                "session_id": plan["session_id"],
                "candidate_id": plan["candidate_id"],
                "target_ric": route,
                "status": "accepted",
                "plan": plan,
                "application": application.describe(),
                "a1_policy_id": ident if route == "non_rt" else None,
                "binding": binding,
                "created_at": now(),
                "policy_stored": route == "non_rt",
                "forwarded": False,
                "applied": False,
                "restored": False,
                "ns3_application_confirmed": False,
                "ns3_performance_validated": False,
                "receipts": [],
                "path": [
                    "Non_RT_RIC",
                    application.name,
                    "execution_xApp",
                    "FlexRIC_Near_RT",
                    "E2",
                    "ns3",
                ]
                if route == "non_rt"
                else [application.name, "execution_xApp", "FlexRIC_Near_RT", "E2", "ns3"],
            }
            db.execute(
                "INSERT INTO deployments VALUES (?,?,?)",
                (ident, digest, json.dumps(saved, allow_nan=False)),
            )
    if not existing:
        background.add_task(transmit, ident)
    return reconcile(ident)
