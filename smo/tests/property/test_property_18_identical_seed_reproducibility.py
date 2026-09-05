"""Property 18: 동일 시드 재현성.

Grounded in ``train_gnn.py`` (Training_Manager, Requirement 3.6): the module's
own docstring states that every source of randomness in ``run_training`` (the
train/validation split via ``random.Random(seed)`` and PyTorch's weight
initialization / training-loop randomness via ``torch.manual_seed(seed)`` plus
``torch.use_deterministic_algorithms(True)``) is derived solely from
``--seed``, and running on CPU only avoids nondeterministic kernels — so two
runs with the same Feature_Group snapshot and seed should reproduce sample
counts exactly and Target_KPI MAPE within the 1.0 percentage-point tolerance
Requirement 3.6 allows (bit-identical per the module's own claim, since no GPU
nondeterminism is involved). This test formalizes that manual claim as a
Hypothesis property, varying the seed and the input records while keeping
epoch counts small for runtime.
"""

from __future__ import annotations

import copy
from typing import Any

import torch
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.common.constants import TARGET_KPIS
from smo.aimlfw.training_manager.train_gnn import run_training
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS, cell_ids, valid_cell_parameters

# Keep epoch/patience counts small so 100 Hypothesis examples (each running the
# training function twice) stay fast; determinism does not depend on how many
# epochs are run.
_MAX_EPOCHS_FOR_TEST = 5
_PATIENCE_FOR_TEST = 2
_TIME_STEPS_FOR_TEST = 3

_TARGET_VALUE = st.floats(
    min_value=-1_000.0,
    max_value=1_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)


@st.composite
def _training_records(draw: Any) -> list[dict[str, Any]]:
    """Generate a small, fully-populated Feature_Record-shaped input for train_gnn.run_training."""
    identifiers = draw(cell_ids(min_size=2, max_size=4))
    records: list[dict[str, Any]] = []
    for time_step in range(_TIME_STEPS_FOR_TEST):
        for cell_id in identifiers:
            control_parameters = draw(valid_cell_parameters()).model_dump()
            features = {kpi: draw(_TARGET_VALUE) for kpi in TARGET_KPIS}
            records.append(
                {
                    "time_step": time_step,
                    "cell_id": cell_id,
                    "control_parameters": control_parameters,
                    "features": features,
                }
            )
    return records


# **Property 18: 동일 시드 재현성**
# **Validates: Requirements 3.6**
@PROPERTY_TEST_SETTINGS
@given(
    records=_training_records(),
    seed=st.integers(min_value=0, max_value=4_294_967_295),
)
def test_property_18_identical_seed_reproducibility(records: list[dict[str, Any]], seed: int) -> None:
    model_a, node_order_a, payload_a = run_training(
        copy.deepcopy(records),
        seed=seed,
        max_epochs=_MAX_EPOCHS_FOR_TEST,
        patience=_PATIENCE_FOR_TEST,
    )
    model_b, node_order_b, payload_b = run_training(
        copy.deepcopy(records),
        seed=seed,
        max_epochs=_MAX_EPOCHS_FOR_TEST,
        patience=_PATIENCE_FOR_TEST,
    )

    # 학습 표본 수는 동일해야 한다 (Requirement 3.6).
    assert payload_a["train_samples"] == payload_b["train_samples"]
    assert payload_a["validation_samples"] == payload_b["validation_samples"]
    assert node_order_a == node_order_b

    # 각 Target_KPI 별 평가 지표 절대 차이는 1.0 percentage point 이내여야 한다
    # (Requirement 3.6). train_gnn.py는 CPU 전용 결정론적 실행을 근거로 이 값이
    # 실제로는 완전히 동일(bit-identical)하다고 주장하므로 그 오차 한도로 검증한다.
    assert payload_a["metrics"].keys() == payload_b["metrics"].keys()
    for kpi, value_a in payload_a["metrics"].items():
        value_b = payload_b["metrics"][kpi]
        assert abs(value_a - value_b) <= 1.0

    # train_gnn.py의 결정론 주장(동일 시드 -> 동일 가중치)을 모델 state_dict
    # 완전 일치로 직접 검증한다.
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    assert state_a.keys() == state_b.keys()
    for key in state_a:
        assert torch.equal(state_a[key], state_b[key])
