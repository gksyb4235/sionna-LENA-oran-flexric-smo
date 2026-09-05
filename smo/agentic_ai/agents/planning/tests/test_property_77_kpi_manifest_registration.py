"""Property 77: KPI Advisor manifest registration is readable and planner-valid."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

PLANNING_DIR = Path(__file__).resolve().parents[1]
KPI_DIR = PLANNING_DIR.parent / "kpi-advisor"
AGENTIC_ROOT = PLANNING_DIR.parents[1]
REPO_ROOT = PLANNING_DIR.parents[3]
sys.path[:0] = [str(REPO_ROOT), str(AGENTIC_ROOT), str(PLANNING_DIR), str(KPI_DIR)]

from runtime_registration import AGENT_ID, register_kpi_advisor  # noqa: E402

from tools.knowledge_registry import planner_manifest_is_valid  # noqa: E402


class _Collection:
    def __init__(self) -> None:
        self.document: dict[str, Any] | None = None

    def create_index(self, *_args: Any, **_kwargs: Any) -> str:
        return "agent_id_unique"

    def update_one(self, _filter: dict[str, Any], update: dict[str, Any], **_kwargs: Any) -> None:
        current = self.document or {}
        self.document = {**current, **update.get("$setOnInsert", {}), **update["$set"]}

    def find_one(self, _filter: dict[str, Any], *_args: Any, **_kwargs: Any) -> dict[str, Any] | None:
        return self.document


# **Property 77: planner_manifest 등록과 후보 노출**
# **Validates: Requirements 12.4, 12.9**
@settings(max_examples=100, deadline=None)
@given(repeats=st.integers(min_value=1, max_value=5))
def test_registration_exposes_one_valid_active_candidate(repeats: int) -> None:
    collection = _Collection()
    result = None
    for _ in range(repeats):
        result = register_kpi_advisor(collection)
    assert result is not None
    assert result["agent_id"] == AGENT_ID
    assert result["status"] == "active"
    assert planner_manifest_is_valid(result)
