"""Property 91: failed or 24-hour jobs keep the old serving model."""

import asyncio
from datetime import timedelta

from hypothesis import given, settings
from hypothesis import strategies as st

from smo.aimlfw.retraining_controller import RetrainingController

from ._retraining_support import (
    Clock,
    Evaluator,
    PredictionClient,
    PromotionClient,
    TrainingClient,
    record,
)


# **Property 91: 실패/시간초과 처리**
# **Validates: Requirement 13.11**
@settings(max_examples=100, deadline=None)
@given(timeout=st.booleans())
def test_failure_or_timeout_removes_job_without_promotion(timeout: bool) -> None:
    clock = Clock()
    training = TrainingClient(status="running" if timeout else "failed")
    promotion = PromotionClient()
    controller = RetrainingController(
        PredictionClient({"cell_goodput_mbps": 120.0}),
        training,
        Evaluator(),
        promotion,
        now=clock,
    )
    asyncio.run(controller.process_result_records("/results/run-1", [record()]))
    if timeout:
        clock.value += timedelta(hours=24, seconds=1)
    asyncio.run(controller.check_jobs())
    assert controller.active_jobs == {}
    assert promotion.version == 1 and promotion.promoted == []
    assert any(event.event_type == "retraining_failed" for event in controller.events)
