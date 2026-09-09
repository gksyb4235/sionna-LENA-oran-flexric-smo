"""Provision the Decomposition Knowledge DB and its read-only runtime user."""

from __future__ import annotations

import json
import os
import secrets
from pathlib import Path
from urllib.parse import quote_plus

from pymongo import MongoClient

JAECHAN_ROOT = Path("/home/ubuntu/jaechan")
CREDENTIALS_PATH = JAECHAN_ROOT / "mongodb/credentials.json"
DECOMPOSITION_ENV_PATH = JAECHAN_ROOT / "agents/decomposition/.env"
DATABASE = "decomposition"
COLLECTION = "decomposition_patterns"
READER_USERNAME = "decomposition_reader"

MAINTAIN_EXPECTED_STATE_PATTERN = {
    "pattern_id": "maintain_expected_state-v1",
    "status": "active",
    "intent_class": "maintain_expected_state",
    "summary": (
        "A request to keep or maintain X in expected condition Y for N minutes means: "
        "first converge X to Y when needed, then Monitoring Agent uses its observation tools "
        "throughout N minutes and sends fresh mismatch evidence to Planning Agent when Y is not satisfied."
    ),
    "keywords": [
        "유지",
        "유지해줘",
        "유지시켜줘",
        "동안 유지",
        "모니터링",
        "maintain",
        "keep",
        "minutes",
        "monitor",
    ],
    "required_semantics": [
        "preserve_target_x",
        "preserve_expected_condition_y",
        "preserve_requested_n_minute_duration",
        "monitoring_agent_uses_observation_tools_for_full_window",
        "confirmed_mismatch_notifies_planning_agent_with_fresh_evidence",
        "no_direct_remediation_subtask",
    ],
    "decomposition_rules": [
        "If the request includes an initial state change, keep that change as the first subtask.",
        "Create one subsequent monitoring subtask that preserves X, Y, and the exact N-minute duration.",
        "The monitoring subtask states that Monitoring Agent uses its observation tools throughout the requested window.",
        "When Y is not satisfied, Monitoring Agent sends fresh mismatch evidence to Planning Agent.",
        "Do not create a direct or unconditional remediation subtask; Planning Agent decides approval-gated recovery after notification.",
        "Observation-unavailable or unknown evidence is not a mismatch.",
    ],
    "validation": {
        "preserve_numbers": True,
        "preserve_uppercase_tokens": True,
        "required_concepts": {
            "monitoring_agent": ["모니터링 에이전트", "monitoring agent"],
            "observation_tools": ["관측 도구", "모니터링 도구", "observation tools", "monitoring tools"],
            "duration": ["분", "minute", "minutes"],
            "mismatch": ["만족하지", "불일치", "이탈", "not satisfied", "mismatch"],
            "planning_agent": ["플래닝 에이전트", "planning agent"],
            "notification": ["알린", "통보", "notify", "notification"],
        },
    },
    "examples": [
        {
            "intent": "NF를 2개로 만들고 5분 동안 유지시켜줘",
            "expected_subtasks": [
                "NF를 2개로 구성한다.",
                "모니터링 에이전트가 자신의 관측 도구를 사용해 5분 동안 NF가 2개인지 모니터링하고, 조건이 만족되지 않으면 새로운 관측 증거와 함께 플래닝 에이전트에 알린다.",
            ],
        },
        {
            "intent": "Keep NF at 3 replicas for 7 minutes",
            "expected_subtasks": [
                "Configure NF with 3 replicas.",
                "Monitoring Agent uses its observation tools for 7 minutes to monitor whether NF remains at 3 replicas and notifies Planning Agent with fresh evidence if the condition is not satisfied.",
            ],
        },
    ],
    "version": "1.0",
}


def _root_client(credentials: dict[str, str]) -> MongoClient:
    username = quote_plus(credentials["root_username"])
    password = quote_plus(credentials["root_password"])
    return MongoClient(f"mongodb://{username}:{password}@127.0.0.1:27017/admin?authSource=admin")


def _write_json_atomic(path: Path, value: dict[str, str]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, path)


def _update_env(path: Path, values: dict[str, str]) -> None:
    removed = {"MONGODB_URI", "MONGODB_DATABASE", "MONGODB_COLLECTION", "MONGODB_TIMEOUT_MS"}
    kept: list[str] = []
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            stripped = raw.strip()
            key = stripped.split("=", 1)[0].strip() if "=" in stripped else ""
            if key in values or key in removed:
                continue
            kept.append(raw)
    if kept and kept[-1].strip():
        kept.append("")
    kept.extend(f"{key}={value}" for key, value in values.items())
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    os.chmod(path, 0o600)


def main() -> None:
    credentials = json.loads(CREDENTIALS_PATH.read_text(encoding="utf-8"))
    reader_password = credentials.get("decomposition_reader_password") or secrets.token_urlsafe(32)
    client = _root_client(credentials)
    client.admin.command("ping")

    database = client[DATABASE]
    existing_users = database.command("usersInfo", READER_USERNAME).get("users", [])
    command = "updateUser" if existing_users else "createUser"
    database.command(
        command,
        READER_USERNAME,
        pwd=reader_password,
        roles=[{"role": "read", "db": DATABASE}],
    )

    collection = database[COLLECTION]
    collection.create_index("pattern_id", unique=True, name="decomposition_pattern_id_unique")
    collection.create_index(
        [("intent_class", "text"), ("summary", "text"), ("keywords", "text"), ("examples.intent", "text")],
        name="decomposition_pattern_text",
        default_language="none",
    )
    result = collection.update_one(
        {"pattern_id": MAINTAIN_EXPECTED_STATE_PATTERN["pattern_id"]},
        {"$set": MAINTAIN_EXPECTED_STATE_PATTERN},
        upsert=True,
    )

    credentials["decomposition_reader_username"] = READER_USERNAME
    credentials["decomposition_reader_password"] = reader_password
    _write_json_atomic(CREDENTIALS_PATH, credentials)

    runtime_uri = (
        f"mongodb://{quote_plus(READER_USERNAME)}:{quote_plus(reader_password)}"
        f"@127.0.0.1:27017/{DATABASE}?authSource={DATABASE}"
    )
    _update_env(
        DECOMPOSITION_ENV_PATH,
        {
            "DECOMPOSITION_KNOWLEDGE_MONGODB_URI": runtime_uri,
            "DECOMPOSITION_KNOWLEDGE_DATABASE": DATABASE,
            "DECOMPOSITION_KNOWLEDGE_COLLECTION": COLLECTION,
            "DECOMPOSITION_KNOWLEDGE_TIMEOUT_MS": "1500",
        },
    )
    client.close()
    print(json.dumps({
        "database": DATABASE,
        "collection": COLLECTION,
        "pattern_id": MAINTAIN_EXPECTED_STATE_PATTERN["pattern_id"],
        "matched_count": result.matched_count,
        "modified_count": result.modified_count,
        "upserted": result.upserted_id is not None,
        "runtime_user_role": "read",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
