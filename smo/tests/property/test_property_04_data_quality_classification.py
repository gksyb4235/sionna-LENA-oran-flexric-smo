"""Property 4 test module (own module to avoid collisions with parallel Property tasks).

Task 2.6: complete/partial/insufficient 조건의 상호 배타성과 정확한 분류를 검증한다.

Targets the Data_Extractor's data_quality classification logic, as specified in
requirements.md 1.4/1.11 and implemented by
``smo.aimlfw.data_extractor.pipeline._data_quality_for`` (Task 2.2):

    - 모든 요구 피처의 유효 표본 수가 1 이상이고 제외 표본 수 합이 0 -> "complete"
    - 모든 요구 피처의 유효 표본 수가 1 이상이고 제외 표본 수 합이 1 이상 -> "partial"
    - 어떤 요구 피처의 유효 표본 수가 0 -> "insufficient"

These three conditions are mutually exclusive and exhaustive over the valid
sample-count space, and exactly one of them is assigned per (Time_Step, cell_id)
group.
"""

from __future__ import annotations

from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

pipeline = pytest.importorskip(
    "smo.aimlfw.data_extractor.pipeline",
    reason=(
        "Task 2.2 (Data Extractor) has not implemented smo.aimlfw.data_extractor.pipeline yet. "
        "Property 4 targets pipeline._data_quality_for(counts: dict[str, SampleCount]) -> str."
    ),
)

from smo.aimlfw.common.models import SampleCount  # noqa: E402  (after importorskip)

_FEATURE_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyz_",
    min_size=1,
    max_size=16,
)
_SAMPLE_COUNT = st.builds(
    SampleCount,
    valid=st.integers(min_value=0, max_value=5),
    excluded=st.integers(min_value=0, max_value=5),
)


@st.composite
def _feature_sample_counts(draw: Any) -> dict[str, SampleCount]:
    """Generate a non-empty mapping of required feature name -> SampleCount(valid, excluded)."""
    names = draw(st.lists(_FEATURE_NAME, min_size=1, max_size=10, unique=True))
    counts = draw(st.lists(_SAMPLE_COUNT, min_size=len(names), max_size=len(names)))
    return dict(zip(names, counts, strict=True))


def _expected_quality(counts: dict[str, SampleCount]) -> str:
    """Independent oracle mirroring Requirement 1.4/1.11 verbatim."""
    if any(count.valid == 0 for count in counts.values()):
        return "insufficient"
    if sum(count.excluded for count in counts.values()) >= 1:
        return "partial"
    return "complete"


# **Property 4: data_quality 판정 규칙**
# **Validates: Requirements 1.4, 1.11**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(counts=_feature_sample_counts())
def test_data_quality_matches_expected_classification(counts: dict[str, SampleCount]) -> None:
    actual = pipeline._data_quality_for(counts)
    expected = _expected_quality(counts)

    assert actual == expected
    assert actual in ("complete", "partial", "insufficient")


# **Property 4: data_quality 판정 규칙**
# **Validates: Requirements 1.4, 1.11**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(counts=_feature_sample_counts())
def test_data_quality_conditions_are_mutually_exclusive(counts: dict[str, SampleCount]) -> None:
    is_insufficient = any(count.valid == 0 for count in counts.values())
    is_partial_condition = (
        all(count.valid >= 1 for count in counts.values())
        and sum(count.excluded for count in counts.values()) >= 1
    )
    is_complete_condition = (
        all(count.valid >= 1 for count in counts.values())
        and sum(count.excluded for count in counts.values()) == 0
    )

    # Exactly one of the three named conditions holds for any valid sample-count map.
    assert sum([is_insufficient, is_partial_condition, is_complete_condition]) == 1

    actual = pipeline._data_quality_for(counts)
    if is_insufficient:
        assert actual == "insufficient"
    elif is_partial_condition:
        assert actual == "partial"
    else:
        assert actual == "complete"


# **Property 4: data_quality 판정 규칙**
# **Validates: Requirements 1.4, 1.11**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    counts=_feature_sample_counts(),
    zero_valid_feature=st.integers(min_value=0),
)
def test_any_zero_valid_sample_forces_insufficient(
    counts: dict[str, SampleCount],
    zero_valid_feature: int,
) -> None:
    """Forcing one feature's valid count to 0 must always yield 'insufficient',
    regardless of every other feature's valid/excluded counts (1.4)."""
    names = list(counts)
    target = names[zero_valid_feature % len(names)]
    counts = dict(counts)
    counts[target] = SampleCount(valid=0, excluded=counts[target].excluded)

    assert pipeline._data_quality_for(counts) == "insufficient"
