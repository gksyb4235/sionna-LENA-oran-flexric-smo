"""Persistent deployment records and independent ns-3 application readbacks."""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import time
from .protocol import identifier

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = Path(os.environ.get("AGORA_RUNTIME_DIR", ROOT / "experiment/khu/results/agora_runtime"))
RUNS = RUNTIME / "runs"


@contextmanager
def database():
    """Commit or roll back each transaction, then release its SQLite connection."""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(RUNTIME / "control.sqlite3", timeout=15)
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS deployments "
            "(id TEXT PRIMARY KEY, fingerprint TEXT NOT NULL, payload TEXT NOT NULL)"
        )
        connection.commit()
        with connection:
            yield connection
    finally:
        connection.close()


def alive(pid):
    try:
        args = Path(f"/proc/{int(pid)}/cmdline").read_bytes().replace(b"\0", b" ").decode()
        return "ns3.48-" in args and str(ROOT / "build/scratch") in args
    except (OSError, ValueError):
        return False


def live_run(run_id):
    identifier(run_id)
    path = RUNS / run_id / "status.json"
    if not path.exists():
        raise ValueError("Unknown ns-3 run")
    status = json.loads(path.read_text())
    if status["run_id"] != run_id:
        raise ValueError("Run manifest identity mismatch")
    status["process_alive"] = alive(status["pid"])
    status["heartbeat_age_s"] = max(0, time.time() - status["wall_time_s"])
    return status


def now():
    return datetime.now(timezone.utc).isoformat()


def read_saved(ident):
    identifier(ident)
    with database() as db:
        row = db.execute("SELECT payload FROM deployments WHERE id=?", (ident,)).fetchone()
    if not row:
        raise ValueError("Unknown live deployment")
    return json.loads(row[0])


def update(saved):
    with database() as db:
        db.execute(
            "UPDATE deployments SET payload=? WHERE id=?",
            (json.dumps(saved, allow_nan=False), saved["deployment_id"]),
        )


def reconcile(ident):
    saved = read_saved(ident)
    path = RUNS / saved["live_run_id"] / "receipts.jsonl"
    receipts = []
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # Last line can be in flight.
            if event.get("command_id") == ident and event.get("run_id") == saved["live_run_id"]:
                receipts.append(event)
    substantive = [r for r in receipts if r["status"] != "duplicate_ignored"]
    if substantive:
        saved["status"] = substantive[-1]["status"]
        saved["forwarded"] = True
        saved["applied"] = any(r["status"] == "applied" for r in substantive)
        saved["restored"] = any(r["status"] == "restored" for r in substantive)
        saved["receipts"] = receipts
        saved["ns3_application_confirmed"] = saved["applied"]
    try:
        saved["live"] = live_run(saved["live_run_id"])
    except ValueError:
        saved["live"] = None
    if saved["live"] and not saved["live"]["process_alive"]:
        if saved["status"] in ("accepted", "transmitting", "submitted", "scheduled"):
            saved["status"] = "unconfirmed_simulation_stopped"
        elif saved["status"] == "applied":
            saved["status"] = "restore_unconfirmed_simulation_stopped"
    return saved
