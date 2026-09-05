"""Task 13.10: new-result detection, deduplication, promotion, rejection, and timeout flow."""

import asyncio

from smo.aimlfw.common.constants import TARGET_KPIS
from smo.aimlfw.retraining_controller import RetrainingController
from smo.tests.property._retraining_support import (
    Evaluator,
    PredictionClient,
    PromotionClient,
    TrainingClient,
    record,
)


def test_new_result_is_detected_once_and_promoted_on_same_evaluation_records() -> None:
    training = TrainingClient(status="running")
    promotion = PromotionClient()
    evaluator = Evaluator(
        {name: 10.0 for name in TARGET_KPIS},
        {name: 9.0 for name in TARGET_KPIS},
    )
    controller = RetrainingController(
        PredictionClient({"cell_goodput_mbps": 120.0}),
        training,
        evaluator,
        promotion,
        poll_seconds=60.0,
    )
    records = [record()]
    asyncio.run(controller.scan_once(records))
    asyncio.run(controller.scan_once(records))
    assert len(training.created) == 1
    assert controller.poll_seconds <= 60.0

    training.status = "completed"
    asyncio.run(controller.check_jobs())
    assert promotion.version == 2
    assert [version for version, _records in evaluator.calls] == [1, 2]
    assert evaluator.calls[0][1] is evaluator.calls[1][1]


def test_quality_regression_is_rejected_without_serving_change() -> None:
    candidate = {name: 9.0 for name in TARGET_KPIS}
    candidate["delay_p95_ms"] = 11.0
    training = TrainingClient(status="running")
    promotion = PromotionClient()
    controller = RetrainingController(
        PredictionClient({"cell_goodput_mbps": 120.0}),
        training,
        Evaluator({name: 10.0 for name in TARGET_KPIS}, candidate),
        promotion,
    )
    asyncio.run(controller.process_result_records("/results/run-1", [record()]))
    training.status = "completed"
    asyncio.run(controller.check_jobs())
    assert promotion.version == 1
    assert promotion.rejected == [2]
