"""Property 88: pending/running retraining never changes the serving version."""

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


# **Property 88: 재학습 중 서빙 버전 불변**
# **Validates: Requirement 13.8**
@settings(max_examples=100, deadline=None)
@given(status=st.sampled_from(["pending", "running"]))
def test_serving_version_stays_old_while_job_is_active(status: str) -> None:
    training = TrainingClient(status=status)
    promotion = PromotionClient()
    controller = RetrainingController(
        PredictionClient({"cell_goodput_mbps": 120.0}), training, Evaluator(), promotion
    )
    asyncio.run(controller.process_result_records("/results/run-1", [record()]))
    asyncio.run(controller.check_jobs())
    assert promotion.version == 1
    assert promotion.promoted == []
    assert "cell_goodput_mbps" in controller.active_jobs
