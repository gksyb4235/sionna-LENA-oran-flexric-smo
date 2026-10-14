"""Durable offline RIC outbox. No network client or automatic consumer."""

from copy import deepcopy
from datetime import datetime, timezone
import json
import sqlite3
import uuid
from .operator_review import ensure_review


class QueueConflict(ValueError):
    pass


def _connect(state):
    state.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(state / "deployments.sqlite3", timeout=10)
    connection.execute(
        "CREATE TABLE IF NOT EXISTS deployments (session_id TEXT PRIMARY KEY, deployment_id TEXT UNIQUE NOT NULL, payload TEXT NOT NULL)"
    )
    connection.commit()
    return connection


def get_receipt(state, session_id):
    if not (state / "deployments.sqlite3").exists():
        return None
    connection = _connect(state)
    try:
        row = connection.execute(
            "SELECT payload FROM deployments WHERE session_id=?", (session_id,)
        ).fetchone()
        return json.loads(row[0]) if row else None
    finally:
        connection.close()


def list_receipts(state, ric=None):
    if not (state / "deployments.sqlite3").exists():
        return []
    connection = _connect(state)
    try:
        rows = [
            json.loads(row[0])
            for row in connection.execute(
                "SELECT payload FROM deployments ORDER BY rowid DESC LIMIT 200"
            )
        ]
        return [
            row
            for row in rows
            if ric is None or any(route["ric"] == ric for route in row["routes"])
        ]
    finally:
        connection.close()


def enqueue(state, session, candidate_id, target_ric, review_sha256, accept_constraint_violations):
    if session.status != "completed" or not session.result:
        raise QueueConflict("평가가 완료된 세션에서만 후보를 등록할 수 있습니다.")
    result = ensure_review(session.result, session.events)
    review = result["operator_review"]
    if review_sha256 != review["review_sha256"]:
        raise QueueConflict("검토 결과가 변경되었습니다. 화면을 새로고침해 다시 선택하세요.")
    candidate = next(
        (row for row in review["candidates"] if row["candidate_id"] == candidate_id), None
    )
    if candidate is None:
        raise ValueError("최종 검토 후보에 없는 후보 ID입니다.")
    if not candidate["deployment_eligible"]:
        raise ValueError("근거 부족으로 위반 정도를 평가할 수 없는 후보는 등록할 수 없습니다.")
    if not candidate["feasible"] and not accept_constraint_violations:
        raise ValueError(
            "이 후보는 제약을 위반합니다. 위반 내역을 확인한 운영자의 선택이 필요합니다."
        )
    if target_ric not in ("near_rt", "non_rt", "split"):
        raise ValueError("Unknown RIC target")
    plan = result["execution_plan"]
    current = plan["current_parameters"]
    changes = candidate["changes"]
    routes = []
    for ric in ("non_rt", "near_rt"):
        selected = {
            key: value
            for key, value in changes.items()
            if target_ric == ric
            or target_ric == "split"
            and ((key.split("_")[0] in ("txp", "ret")) == (ric == "non_rt"))
        }
        if not selected:
            continue
        restore = {key: current[key] for key in selected}
        routes.append(
            {
                "ric": ric,
                "status": "queued_offline",
                "changes": selected,
                "schedule": [
                    {
                        "time_s": candidate["interval_start_s"],
                        "action": "set_parameters",
                        "changes": selected,
                    },
                    {
                        "time_s": candidate["interval_end_s"],
                        "action": "restore_pre_request_parameters",
                        "changes": restore,
                    },
                ],
                "forwarded": False,
                "applied": False,
            }
        )
    if not routes:
        raise ValueError("No changed controls to queue")
    receipt = {
        "schema_version": "agora-ric-queue-v1",
        "deployment_id": uuid.uuid4().hex,
        "session_id": session.id,
        "candidate_id": candidate_id,
        "queued_at": datetime.now(timezone.utc).isoformat(),
        "status": "queued_offline",
        "target_ric": target_ric,
        "routes": routes,
        "review_sha256": review_sha256,
        "operator_decision": {
            "source": "local_operator_ui_or_api",
            "constraint_override": not candidate["feasible"],
            "accept_constraint_violations": accept_constraint_violations,
        },
        "run_id": plan["run_id"],
        "issue_time_s": plan["issue_time_s"],
        "clock": "replay_simulation_seconds_not_wall_clock",
        "current_parameters": deepcopy(current),
        "candidate": deepcopy(candidate),
        "intent": deepcopy(result["intent"]),
        "model": deepcopy(plan["model"]),
        "search_coverage": deepcopy(result.get("search_coverage")),
        "forwarded": False,
        "applied": False,
        "ns3_validated": False,
        "execution_note": "Saved to an offline outbox only. A future RIC/ns3 adapter must bind the replay clock, verify current controls and supported domains, and report application separately. No automatic consumer is installed.",
    }
    connection = _connect(state)
    try:
        with connection:
            connection.execute(
                "INSERT OR IGNORE INTO deployments VALUES (?,?,?)",
                (
                    session.id,
                    receipt["deployment_id"],
                    json.dumps(receipt, ensure_ascii=False, allow_nan=False),
                ),
            )
            saved = json.loads(
                connection.execute(
                    "SELECT payload FROM deployments WHERE session_id=?", (session.id,)
                ).fetchone()[0]
            )
            if (saved["candidate_id"], saved["target_ric"], saved["review_sha256"]) != (
                candidate_id,
                target_ric,
                review_sha256,
            ):
                raise QueueConflict(
                    "이 세션은 이미 다른 후보 또는 RIC으로 등록되었습니다. 기존 등록 내역을 확인하세요."
                )
            return saved
    finally:
        connection.close()
