"""Property 2 test module (own module to avoid collisions with parallel Property tasks).

Task 2.4: 임의 실수 목록의 sum/mean 결과와 소수점 6자리 반올림을 검증한다.
"""

from __future__ import annotations

import math

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.data_extractor.aggregation import AggregationRule, aggregate_values
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

# Same magnitude bounds as the shared finite-float generator in strategies.py,
# kept local because Property 2 operates on raw numeric samples rather than
# any domain model.
_SAMPLE_VALUE = st.floats(
    min_value=-1_000_000.0,
    max_value=1_000_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
)
_SAMPLE_POPULATION = st.lists(_SAMPLE_VALUE, min_size=1, max_size=50)
_AGGREGATION_RULE = st.sampled_from(("sum", "mean"))


def _expected_value(values: list[float], rule: AggregationRule) -> float:
    """Independently compute the rule-applied, 6-decimal-rounded reference value."""
    total = sum(values)
    raw = total if rule == "sum" else total / len(values)
    return round(raw, 6)


# **Property 2: 집계 규칙과 반올림**
# **Validates: Requirements 1.2**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(values=_SAMPLE_POPULATION, rule=_AGGREGATION_RULE)
def test_aggregate_values_matches_rule_and_six_decimal_rounding(
    values: list[float],
    rule: AggregationRule,
) -> None:
    actual = aggregate_values(values, rule)
    expected = _expected_value(values, rule)

    # The aggregated value must equal the specified rule's result rounded to
    # 6 decimal places. A small numeric tolerance absorbs floating-point
    # summation-order differences between the implementation and this
    # independently written oracle without weakening the rule/rounding check.
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-6)

    # The result must be a rounding to (at most) 6 decimal places: scaling by
    # 1e6 and rounding to the nearest integer must reproduce the same value.
    assert math.isclose(actual, round(actual * 1_000_000) / 1_000_000, rel_tol=0.0, abs_tol=1e-9)


@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(values=_SAMPLE_POPULATION)
def test_aggregate_values_rejects_unsupported_rule(values: list[float]) -> None:
    with pytest.raises(ValueError):
        aggregate_values(values, "median")  # type: ignore[arg-type]


def test_aggregate_values_rejects_empty_population() -> None:
    with pytest.raises(ValueError):
        aggregate_values([], "sum")
