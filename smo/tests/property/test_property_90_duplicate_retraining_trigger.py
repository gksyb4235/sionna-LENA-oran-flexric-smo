"""Property 90: a KPI with an active job cannot trigger a duplicate job."""

import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st

from smo.aimlfw.retraining_controller import RetrainingController

from ._retraining_support import (
    Evaluator,
    PredictionClient,
    PromotionClient,
    TrainingClient,
    record,
)


# **Property 90: 중복 트리거 방지**
# **Validates: Requirement 13.10**
@settings(max_examples=100, deadline=None)
@given(extra_runs=st.integers(min_value=1, max_value=5))
def test_active_kpi_job_suppresses_every_duplicate(extra_runs: int) -> None:
    training = TrainingClient()
    controller = RetrainingController(
        PredictionClient({"cell_goodput_mbps": 120.0}),
        training,
        Evaluator(),
        PromotionClient(),
    )
    asyncio.run(
        controller.process_result_records(
            "/results/run-0", [record(source_dir="/results/run-0")]
        )
    )
    for index in range(extra_runs):
        source = f"/results/run-{index + 1}"
        asyncio.run(
            controller.process_result_records(source, [record(source_dir=source)])
        )
    assert len(training.created) == 1
    duplicates = [
        event
        for event in controller.events
        if event.event_type == "duplicate_trigger_suppressed"
    ]
    assert len(duplicates) == extra_runs
