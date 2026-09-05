"""Property 75: invalid policy parameters are rejected before A1 publication."""

import pytest
from _policy_test_support import MockA1, fixed_parameter_set, install_mock, uninstall_mock
from hypothesis import given, settings
from hypothesis import strategies as st
from smo.aimlfw.common.models import CellParameters, ParameterSet

import policy_manager


# **Property 75: 파라미터 검증 실패 거부**
# **Validates: Requirement 11.14**
@settings(max_examples=100, deadline=None)
@given(invalid_power=st.one_of(st.integers(max_value=29), st.integers(min_value=47, max_value=10000)))
def test_out_of_range_parameter_is_rejected_before_a1(invalid_power: int) -> None:
    valid = fixed_parameter_set()
    invalid_cell = CellParameters.model_construct(
        tx_power_dbm=float(invalid_power), ret_tilt_deg=5.0, cio_bias_db=0.0, hysteresis_db=2.0, ttt_ms=160
    )
    parameter_set = ParameterSet.model_construct(cells={**valid.cells, "gNB_5G": invalid_cell})
    mock = MockA1()
    install_mock(mock)
    try:
        with pytest.raises(policy_manager.PolicyManagerError) as excinfo:
            policy_manager.publish_parameter_set(parameter_set, list(policy_manager.FIXED_CELLS))
        assert excinfo.value.error_code == "parameter_out_of_range"
        assert mock.requests == []
    finally:
        uninstall_mock()
