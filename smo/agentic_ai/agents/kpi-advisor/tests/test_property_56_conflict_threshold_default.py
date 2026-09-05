"""Property 56: conflict thresholds validate or default with a warning."""

import sys
from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR)]

from conflict_analyzer import analyze  # noqa: E402


# **Property 56: Conflict_Threshold 유효성 검증과 기본값**
# **Validates: Requirements 9.5, 9.6, 9.9**
@settings(max_examples=100, deadline=None)
@given(
    value=st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
        st.floats(min_value=100.1, max_value=1e9, allow_nan=False, allow_infinity=False),
    )
)
def test_invalid_threshold_defaults_with_warning(value: float) -> None:
    result = analyze([], value)
    assert result.conflict_threshold == 5.0
    assert result.threshold_warning


@settings(max_examples=100, deadline=None)
@given(value=st.floats(min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False))
def test_valid_threshold_is_preserved(value: float) -> None:
    result = analyze([], value)
    assert result.conflict_threshold == value
    assert result.threshold_warning is None


def test_omitted_threshold_defaults_without_warning() -> None:
    result = analyze([], None)
    assert result.conflict_threshold == 5.0
    assert result.threshold_warning is None
