"""Property 51: invalid KPI thresholds are rejected and valid configuration loads."""

from __future__ import annotations

from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError
from smo.aimlfw.common.config import load_kpi_thresholds
from smo.aimlfw.common.models import ThresholdKpi


# **Property 51: 임계값 설정 검증**
# **Validates: Requirements 8.1, 8.10**
@settings(max_examples=100, deadline=None)
@given(
    lower=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    gap=st.floats(min_value=0.1, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_lower_bound_above_upper_bound_is_rejected(lower: float, gap: float) -> None:
    upper = lower - gap
    with pytest.raises(ValidationError):
        ThresholdKpi(
            target_kpi="cell_goodput_mbps",
            lower_bound=lower,
            upper_bound=upper,
            improve_direction="higher_is_better",
        )


def test_repository_threshold_configuration_is_complete() -> None:
    config_path = Path(__file__).resolve().parents[4] / "aimlfw" / "config" / "kpi_thresholds.json"
    assert len(load_kpi_thresholds(config_path).thresholds) == 10
