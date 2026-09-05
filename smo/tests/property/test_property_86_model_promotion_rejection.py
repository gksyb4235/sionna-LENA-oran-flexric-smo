"""Property 86: candidates promote only when all KPIs are no worse and trigger KPI improves by 1pp."""

import asyncio

from hypothesis import given, settings
from hypothesis import strategies as st

from smo.aimlfw.common.constants import TARGET_KPIS
from smo.aimlfw.retraining_controller import RetrainingController

from ._retraining_support import (
    Evaluator,
    PredictionClient,
    PromotionClient,
    TrainingClient,
    record,
)


# **Property 86: 모델 승격/거부 조건**
# **Validates: Requirements 13.4, 13.5, 13.6**
@settings(max_examples=100, deadline=None)
@given(should_promote=st.booleans())
def test_candidate_promotion_and_rejection_rule(should_promote: bool) -> None:
    current = {name: 10.0 for name in TARGET_KPIS}
    candidate = {name: 9.0 for name in TARGET_KPIS}
    if not should_promote:
        candidate["avg_ue_goodput_mbps"] = 10.1
    training = TrainingClient(status="running")
    promotion = PromotionClient()
    controller = RetrainingController(
        PredictionClient({"cell_goodput_mbps": 120.0}),
        training,
        Evaluator(current, candidate),
        promotion,
    )
    asyncio.run(controller.process_result_records("/results/run-1", [record()]))
    training.status = "completed"
    asyncio.run(controller.check_jobs())
    assert promotion.version == (2 if should_promote else 1)
    assert promotion.promoted == ([2] if should_promote else [])
    assert promotion.rejected == ([] if should_promote else [2])
