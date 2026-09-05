"""Property 61: temporal candidate evaluation uses ordered batches of at most 64."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR), str(TEST_DIR)]

import test_temporal_scheduler_smoke as cases  # noqa: E402


# **Property 61: 평가 배치 분할**
# **Validates: Requirement 10.3**
@settings(max_examples=100, deadline=None)
@given(per_objective=st.integers(min_value=22, max_value=64))
def test_temporal_batches_are_bounded_and_complete(per_objective: int) -> None:
    candidate = cases._parameter_set()
    counts: dict[int, list[int]] = defaultdict(list)

    def predict(parameter_sets: list[dict[str, Any]], time_step: int | None = None) -> dict[str, Any]:
        assert time_step is not None
        counts[time_step].append(len(parameter_sets))
        return cases._make_predict_batch(lambda _cells, _step: cases._default_kpis())(parameter_sets, time_step)

    objectives = ["energy_saving", "throughput_maximization", "mobility_robustness"]
    cases._build(
        objectives=objectives,
        candidates_by_objective={name: [candidate] * per_objective for name in objectives},
        predict_batch=predict,
    )
    for step in range(5):
        assert sum(counts[step]) == per_objective * 3
        assert all(1 <= size <= 64 for size in counts[step])
