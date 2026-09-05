import os
from datetime import datetime, timezone
from pathlib import Path

from pymongo import MongoClient


for raw_line in Path("/home/ubuntu/jaechan/agents/planning/.env").read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if line and not line.startswith("#") and "=" in line:
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))

uri = os.environ.get("KNOWLEDGE_MONGODB_URI")
if not uri:
    raise RuntimeError("KNOWLEDGE_MONGODB_URI is not configured")

client = MongoClient(uri, serverSelectionTimeoutMS=5000)
collection = client[os.environ.get("KNOWLEDGE_DATABASE", "knowledge")][os.environ.get("KNOWLEDGE_AGENTS_COLLECTION", "agents")]
now = datetime.now(timezone.utc)
document = {
    "id": "monitoring-agent",
    "agent_id": "monitoring-agent",
    "type": "agent",
    "role": "outcome_monitor",
    "name": "Monitoring Agent",
    "domain": "RAN Core network operations",
    "description": "Selectively verifies Planning expectations over time with fresh Probe evidence and gates conditional recovery.",
    "can_do": [
        "verify expected RAN and Core outcomes",
        "monitor expected state for a requested duration",
        "use duration and sampling interval supplied by Planning Agent",
        "compare fresh Probe evidence with machine-readable thresholds",
        "allow Core remediation only after a confirmed mismatch",
    ],
    "cannot_do": [
        "mutate Kubernetes, Core, RAN, or Grafana directly",
        "replace Probe Agent for simple metric reads",
        "run automatically after every Planning completion",
        "choose a default interval or maximum observation duration",
        "treat missing observations as a mismatch",
    ],
    "capabilities": [
        "expected-outcome-verification",
        "timed-monitoring",
        "post-action-validation",
        "conditional-remediation-gate",
        "probe-delegation",
    ],
    "tags": ["monitoring", "verify", "maintain", "duration", "conditional-recovery", "RAN", "Core", "NF", "AMF", "SMF", "pod", "replica"],
    "keywords": ["expected state", "expected outcome", "network health", "pod count", "replica count", "keep", "maintain", "for five minutes", "if mismatched", "recover"],
    "inputs": ["planning_report", "subtask_id", "evaluation_request.duration_seconds", "evaluation_request.sample_interval_seconds"],
    "outputs": ["healthy", "observed", "replan_required", "observation_unavailable"],
    "runtime": {"transport": "cli", "entrypoint": "/home/ubuntu/jaechan/agents/monitoring/agent.py"},
    "status": "active",
    "updated_at": now,
}
result = collection.update_one(
    {"agent_id": "monitoring-agent"},
    {"$set": document, "$setOnInsert": {"created_at": now}},
    upsert=True,
)
registered = collection.find_one(
    {"agent_id": "monitoring-agent"},
    {"_id": 0, "agent_id": 1, "name": 1, "role": 1, "status": 1},
)
print({
    "matched_count": result.matched_count,
    "modified_count": result.modified_count,
    "upserted": result.upserted_id is not None,
    "registered": registered,
})
