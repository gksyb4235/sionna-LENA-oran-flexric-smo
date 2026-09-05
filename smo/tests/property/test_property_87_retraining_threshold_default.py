"""Property 87: retraining thresholds validate or default with a recorded warning."""

from hypothesis import given, settings
from hypothesis import strategies as st

from smo.aimlfw.common.constants import DEFAULT_RETRAINING_THRESHOLD
from smo.aimlfw.retraining_controller import RetrainingController


# **Property 87: Retraining_Threshold 검증과 기본값**
# **Validates: Requirement 13.7**
@settings(max_examples=100, deadline=None)
@given(
    value=st.one_of(
        st.floats(max_value=0.0, allow_nan=False, allow_infinity=False),
        st.floats(
            min_value=100.1, max_value=1e9, allow_nan=False, allow_infinity=False
        ),
    )
)
def test_invalid_retraining_threshold_defaults_and_warns(value: float) -> None:
    controller = RetrainingController(
        object(), object(), object(), object(), retraining_threshold=value
    )
    assert controller.retraining_threshold == DEFAULT_RETRAINING_THRESHOLD
    assert controller.threshold_warnings
    assert controller.events[0].event_type == "threshold_defaulted"


@settings(max_examples=100, deadline=None)
@given(
    value=st.floats(
        min_value=0.1, max_value=100.0, allow_nan=False, allow_infinity=False
    )
)
def test_valid_retraining_threshold_is_preserved(value: float) -> None:
    controller = RetrainingController(
        object(), object(), object(), object(), retraining_threshold=value
    )
    assert controller.retraining_threshold == value
