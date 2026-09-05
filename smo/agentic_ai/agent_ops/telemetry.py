"""Small, process-safe JSONL telemetry store used by agents and dashboard."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

_SAFE_ID = re.compile(r"[^A-Za-z0-9_.-]+")
_ROOT = Path(
    os.getenv(
        "AGENT_TELEMETRY_ROOT",
        str(Path(__file__).resolve().parents[1] / ".state" / "telemetry"),
    )
)


def safe_run_id(run_id: str) -> str:
    """Return a filename-safe, non-empty run identifier."""
    safe = _SAFE_ID.sub("-", str(run_id).strip()).strip(".-")
    if not safe:
        raise ValueError("run_id must contain at least one safe character")
    return safe[:160]


def default_run_id(prefix: str) -> str:
    return f"{safe_run_id(prefix)}-{uuid.uuid4().hex}"


def new_call_id() -> str:
    return f"call-{uuid.uuid4().hex}"


def monotonic_ms() -> int:
    return round(time.monotonic() * 1000)


def _event_path(run_id: str) -> Path:
    return _ROOT / f"{safe_run_id(run_id)}.jsonl"


def log_event(*, run_id: str, agent: str, event_type: str, **fields: Any) -> dict[str, Any]:
    """Append one complete JSON line using a single OS write."""
    event = {
        "timestamp": datetime.now(UTC).isoformat(),
        "run_id": safe_run_id(run_id),
        "agent": str(agent),
        "event_type": str(event_type),
        **fields,
    }
    _ROOT.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(event, ensure_ascii=False, default=str) + "\n").encode()
    descriptor = os.open(_event_path(run_id), os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, payload)
    finally:
        os.close(descriptor)
    return event


def read_events(run_id: str) -> list[dict[str, Any]]:
    path = _event_path(run_id)
    if not path.is_file():
        return []
    events: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict):
            events.append(event)
    return events


def list_event_run_ids(limit: int = 100) -> list[str]:
    if not _ROOT.is_dir():
        return []
    paths = sorted(_ROOT.glob("*.jsonl"), key=lambda path: path.stat().st_mtime, reverse=True)
    return [path.stem for path in paths[: max(0, limit)]]


__all__ = [
    "default_run_id",
    "list_event_run_ids",
    "log_event",
    "monotonic_ms",
    "new_call_id",
    "read_events",
    "safe_run_id",
]
