from __future__ import annotations

import math

import pytest
from hypothesis import given
from pydantic import ValidationError

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, TARGET_KPIS
from smo.aimlfw.common.models import (
    FeatureRecord,
    ParameterSet,
    ProbePlan,
    TemporalPlan,
    ThresholdKpi,
)
from smo.tests.property.strategies import (
    PROPERTY_TEST_SETTINGS,
    csv_values,
    feature_records,
    invalid_parameter_sets,
    probe_plans,
    temporal_plans,
    threshold_maps,
    valid_parameter_sets,
)


# **Property support: 유효/무효 ParameterSet 공통 생성기 계약**
# **Validates: Requirements 1.1~15.7**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(parameter_set=valid_parameter_sets())
def test_valid_parameter_set_strategy_obeys_all_cell_contracts(parameter_set: ParameterSet) -> None:
    assert 1 <= len(parameter_set.cells) <= 16
    assert all(set(parameters.model_dump()) == set(CONTROL_PARAMETER_NAMES) for parameters in parameter_set.cells.values())


@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(payload=invalid_parameter_sets())
def test_invalid_parameter_set_strategy_always_produces_rejected_payload(payload: dict) -> None:
    with pytest.raises(ValidationError):
        ParameterSet.model_validate(payload)


# **Property support: FeatureRecord 및 CSV 값 공통 생성기 계약**
# **Validates: Requirements 1.1~2.8**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(record=feature_records(), csv_value=csv_values())
def test_feature_record_and_csv_strategies_generate_usable_values(
    record: FeatureRecord,
    csv_value: str,
) -> None:
    assert set(record.features) == set(record.sample_counts)
    assert isinstance(csv_value, str)
    for value in record.features.values():
        assert value is None or math.isfinite(value)


# **Property support: Threshold, ProbePlan, TemporalPlan 공통 생성기 계약**
# **Validates: Requirements 5.1~15.7**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(thresholds=threshold_maps(), plan=probe_plans())
def test_threshold_and_probe_plan_strategies_preserve_domain_invariants(
    thresholds: dict[str, ThresholdKpi],
    plan: ProbePlan,
) -> None:
    assert set(thresholds) == set(TARGET_KPIS)
    for variant in plan.variants:
        changed = []
        for cell_id in plan.target_cells:
            baseline_values = plan.baseline.cells[cell_id].model_dump()
            variant_values = variant.parameter_set.cells[cell_id].model_dump()
            changed.extend(
                (cell_id, name)
                for name in CONTROL_PARAMETER_NAMES
                if baseline_values[name] != variant_values[name]
            )
        assert changed == [(variant.cell_id, variant.parameter_name)]


@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(plan=temporal_plans())
def test_temporal_plan_strategy_generates_exact_order_and_transitions(plan: TemporalPlan) -> None:
    assert [assignment.time_step for assignment in plan.assignments] == list(range(5))
    assert plan.transition_time_steps == [
        index
        for index in range(1, 5)
        if plan.assignments[index - 1].xapp_objective != plan.assignments[index].xapp_objective
    ]


def test_common_fixtures_are_canonical_and_consistent(
    framework_config,
    default_parameter_set: ParameterSet,
    default_feature_record: FeatureRecord,
) -> None:
    assert set(default_parameter_set.cells) == set(framework_config.feature_group.target_cells)
    assert tuple(default_feature_record.features) == tuple(framework_config.feature_group.features)
    assert default_feature_record.data_quality == "complete"
