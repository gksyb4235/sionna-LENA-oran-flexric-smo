"""Read-only verification for the Decomposition Knowledge DB rollout."""

from __future__ import annotations

import json
import os
from pathlib import Path

from pymongo import MongoClient

ROOT = Path("/home/ubuntu/jaechan")
ENV_PATH = ROOT / "agents/decomposition/.env"
PATTERN_ID = "maintain_expected_state-v1"


def _load_env() -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def main() -> None:
    env = _load_env()
    client = MongoClient(env["DECOMPOSITION_KNOWLEDGE_MONGODB_URI"], serverSelectionTimeoutMS=1500)
    client.admin.command("ping")
    collection = client[env["DECOMPOSITION_KNOWLEDGE_DATABASE"]][env["DECOMPOSITION_KNOWLEDGE_COLLECTION"]]
    document = collection.find_one({"pattern_id": PATTERN_ID, "status": "active"}, {"_id": 0})
    assert isinstance(document, dict), "active maintain pattern is missing"
    assert document.get("intent_class") == "maintain_expected_state"
    assert document.get("validation", {}).get("preserve_numbers") is True
    assert "monitoring_agent_uses_observation_tools_for_full_window" in document.get("required_semantics", [])
    assert "confirmed_mismatch_notifies_planning_agent_with_fresh_evidence" in document.get("required_semantics", [])

    indexes = collection.index_information()
    assert indexes["decomposition_pattern_id_unique"].get("unique") is True
    assert "decomposition_pattern_text" in indexes

    def matched(intent: str) -> list[str]:
        return [item["pattern_id"] for item in collection.find(
            {"status": "active", "$text": {"$search": intent}},
            {"_id": 0, "pattern_id": 1},
        )]

    assert PATTERN_ID in matched("AMF 2개로 만들고 5분동안 유지시켜줘")
    assert PATTERN_ID in matched("Keep SMF at 3 replicas for 7 minutes")
    assert PATTERN_ID not in matched("AMF CPU 사용량을 한 번 조회해줘")
    client.close()
    print(json.dumps({
        "status": "ok",
        "pattern_id": PATTERN_ID,
        "runtime_access": "read",
        "text_match_cases": 3,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
