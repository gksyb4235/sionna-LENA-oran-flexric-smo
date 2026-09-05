"""Property 84: prediction error follows the absolute percentage formula."""

from hypothesis import given, settings
from hypothesis import strategies as st

from smo.aimlfw.retraining_controller import prediction_error_percent


# **Property 84: Prediction_Error 계산**
# **Validates: Requirement 13.2**
@settings(max_examples=100, deadline=None)
@given(
    actual=st.one_of(
        st.floats(
            min_value=-1e6, max_value=-0.1, allow_nan=False, allow_infinity=False
        ),
        st.floats(min_value=0.1, max_value=1e6, allow_nan=False, allow_infinity=False),
    ),
    predicted=st.floats(
        min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
    ),
)
def test_prediction_error_matches_formula(actual: float, predicted: float) -> None:
    expected = round(abs(predicted - actual) / abs(actual) * 100.0, 1)
    assert prediction_error_percent(predicted, actual) == expected
