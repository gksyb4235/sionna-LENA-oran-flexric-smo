"""Property 85: an above-threshold KPI error creates one KPI-specific job."""

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


# **Property 85: 재학습 트리거**
# **Validates: Requirement 13.3**
@settings(max_examples=100, deadline=None)
@given(error_percent=st.integers(min_value=16, max_value=100))
def test_above_threshold_error_creates_retraining_job(error_percent: int) -> None:
    training = TrainingClient()
    controller = RetrainingController(
        PredictionClient({"cell_goodput_mbps": 100.0 + error_percent}),
        training,
        Evaluator(),
        PromotionClient(),
    )
    asyncio.run(controller.process_result_records("/results/run-1", [record()]))
    assert len(training.created) == 1
    assert training.created[0]["metadata"]["target_kpi"] == "cell_goodput_mbps"
    assert "cell_goodput_mbps" in controller.active_jobs
