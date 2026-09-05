"""Property 70: all 15 A1 integer fields round-trip without parameter loss."""

from _policy_test_support import fixed_parameter_set
from hypothesis import given, settings
from hypothesis import strategies as st

import policy_manager


# **Property 70: Policy_Instance 직렬화 왕복**
# **Validates: Requirements 11.5~11.8**
@settings(max_examples=100, deadline=None)
@given(
    tx=st.integers(min_value=30, max_value=46),
    tilt=st.integers(min_value=0, max_value=15),
    cio=st.integers(min_value=-12, max_value=12).map(lambda value: value / 2),
    hys=st.integers(min_value=0, max_value=20).map(lambda value: value / 2),
    ttt=st.sampled_from([0, 40, 64, 80, 100, 128, 160, 256, 320, 480, 512, 1024, 1280, 2560, 5120]),
)
def test_serialized_instance_is_exact_integer_roundtrip(tx: int, tilt: int, cio: float, hys: float, ttt: int) -> None:
    parameter_set = fixed_parameter_set(
        tx_power_dbm=float(tx), ret_tilt_deg=float(tilt), cio_bias_db=cio, hysteresis_db=hys, ttt_ms=ttt
    )
    serialized = policy_manager.serialize_instance(parameter_set)
    assert len(serialized) == 15
    assert all(type(value) is int for value in serialized.values())
    assert policy_manager.parse_instance(serialized) == parameter_set
