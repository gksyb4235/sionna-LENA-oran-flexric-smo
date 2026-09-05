"""Task 8.18 integration and deterministic single-shot performance checks."""

from __future__ import annotations

import sys
import time
from pathlib import Path

TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR), str(TEST_DIR)]

import test_temporal_scheduler_smoke as temporal_cases  # noqa: E402
from smo.aimlfw.common.models import MarginalEffectRecord  # noqa: E402

from conflict_analyzer import analyze  # noqa: E402


def test_analyze_200_pairs_within_two_seconds() -> None:
    effects = []
    for index in range(200):
        effects.extend(
            [
                MarginalEffectRecord(
                    cell_id=f"cell-{index}", control_parameter="tx_power_dbm", target_kpi="kpi", value_percent=10.0
                ),
                MarginalEffectRecord(
                    cell_id=f"cell-{index}", control_parameter="ret_tilt_deg", target_kpi="kpi", value_percent=-10.0
                ),
            ]
        )
    started = time.monotonic()
    result = analyze(effects, 5.0)
    elapsed_ms = (time.monotonic() - started) * 1000
    assert len(result.indirect_conflicts) == 200
    assert elapsed_ms < 2000, f"conflict analysis took {elapsed_ms:.1f}ms"


def test_temporal_full_batch_fallback_and_failure_contracts() -> None:
    temporal_cases.test_build_plan_objective_driven_picks_best_primary_kpi_candidate()
    temporal_cases.test_build_plan_derives_candidate_from_feature_store_when_none_given()
    temporal_cases.test_build_plan_raises_prediction_unavailable_on_tool_error()
