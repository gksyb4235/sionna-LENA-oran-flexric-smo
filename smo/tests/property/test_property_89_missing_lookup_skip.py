"""Property 89: failed or incomplete prediction lookup is skipped and recorded."""

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


# **Property 89: 조회 실패 시 건너뛰기**
# **Validates: Requirement 13.9**
@settings(max_examples=100, deadline=None)
@given(fails=st.booleans())
def test_unavailable_or_missing_prediction_does_not_trigger(fails: bool) -> None:
    prediction = (
        PredictionClient(failure=RuntimeError("unavailable"))
        if fails
        else PredictionClient({})
    )
    training = TrainingClient()
    controller = RetrainingController(
        prediction, training, Evaluator(), PromotionClient()
    )
    asyncio.run(controller.process_result_records("/results/run-1", [record()]))
    assert training.created == []
    assert controller.active_jobs == {}
    assert any(
        event.event_type in {"prediction_skipped", "prediction_error_skipped"}
        for event in controller.events
    )
