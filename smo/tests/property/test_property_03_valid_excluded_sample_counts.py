"""Property 3 test module (own module to avoid collisions with parallel Property tasks).

Task 2.5: 정상·nan·빈 값·비실수 혼합에서 카운트 보존과 유효 표본 집계를 검증한다.

Targets the Data_Extractor pipeline's per-(Time_Step, cell_id, feature) valid/excluded
sample counting (Requirement 1.3): values that are the ``nan`` string, empty/blank
strings, or otherwise non-numeric strings must be excluded from the aggregation
population, while every other value must be counted valid, with
``valid_count + excluded_count`` always equal to the number of input samples and the
aggregated feature value depending only on the valid samples.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.data_extractor import extract
from smo.aimlfw.feature_store import FeatureStore
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

_TARGET_FEATURE = "cell_goodput_mbps"  # aggregation rule "mean" per config/feature_groups/default.json
_TIME_S = "1.5"  # always maps to Time_Step 0 (180*0 < 1.5 <= 180*1)
_TARGET_CELL = "gNB_5G"

_CSV_HEADER = (
    "time_s,interval_s,cell_id,cell,band,tx_power_dbm,ret_tilt_deg,cio_bias_db,ttt_ms,hysteresis_db,"
    "cell_goodput_mbps,avg_ue_goodput_mbps,ue_goodput_p5_mbps,sinr_p50_db,prb_utilization_pct,"
    "delay_p95_ms,ho_failure_count,pingpong_count,rlf_count,interval_energy_j\n"
)

_COMMAND_SH = (
    "#!/usr/bin/env bash\n"
    "set -euo pipefail\n"
    "'./ns3' '--runTag=seed7_prop3' '--cellTxPowerDbm=gNB_5G:43,gNB_4G_1:43,gNB_4G_2:43' "
    "'--cellRetTiltDeg=gNB_5G:5,gNB_4G_1:5,gNB_4G_2:5' '--cellCioDb=gNB_5G:0,gNB_4G_1:0,gNB_4G_2:0' "
    "'--cellHysteresisDb=gNB_5G:2.5,gNB_4G_1:2.5,gNB_4G_2:2.5' '--cellTttMs=gNB_5G:160,gNB_4G_1:160,gNB_4G_2:160'\n"
)

# Local generators for one CSV cell value, independent of strategies.py per the task's
# own-module note; these mirror Requirement 1.3's three exclusion categories plus
# ordinary valid numeric samples.
_VALID_VALUE = st.floats(
    min_value=-1_000_000.0,
    max_value=1_000_000.0,
    allow_nan=False,
    allow_infinity=False,
    width=64,
).map(repr)
_NAN_VALUE = st.sampled_from(("nan", "NaN", "NAN", "-nan", "+nan"))
_BLANK_VALUE = st.sampled_from(("", " ", "   ", "\t"))
_NON_NUMERIC_VALUE = st.sampled_from(("not-a-number", "abc", "N/A", "12.3.4", "true", "--"))

_CSV_CELL_VALUE = st.one_of(_VALID_VALUE, _NAN_VALUE, _BLANK_VALUE, _NON_NUMERIC_VALUE)
_MIXED_SAMPLE_POPULATION = st.lists(_CSV_CELL_VALUE, min_size=1, max_size=40)


def _oracle_classify(raw: str) -> float | None:
    """Independent reference for Requirement 1.3's valid/excluded classification.

    Excludes exactly: the ``nan`` string (any case/sign), empty/whitespace-only
    strings, and strings that do not parse as a real number. Everything else is a
    valid numeric sample.
    """
    stripped = raw.strip()
    if not stripped:
        return None
    if stripped.lower() in ("nan", "-nan", "+nan"):
        return None
    try:
        value = float(stripped)
    except ValueError:
        return None
    if value != value:  # defensive: float() can itself produce nan for some inputs
        return None
    return value


def _row(feature_value: str) -> str:
    return (
        f"{_TIME_S},1,1,{_TARGET_CELL},NR,43,5,0,160,2.5,{feature_value},"
        "1.0,1.0,10.0,50.0,5.0,0,0,0,10.0\n"
    )


def _write_result_dir(root: Path, feature_values: list[str]) -> Path:
    result_dir = root / "run"
    result_dir.mkdir()
    rows = "".join(_row(value) for value in feature_values)
    (result_dir / "cell_kpi.csv").write_text(_CSV_HEADER + rows, encoding="utf-8")
    (result_dir / "command.sh").write_text(_COMMAND_SH, encoding="utf-8")
    return result_dir


# **Property 3: 유효/제외 표본 카운트 보존**
# **Validates: Requirements 1.3**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(feature_values=_MIXED_SAMPLE_POPULATION)
def test_valid_and_excluded_counts_preserve_total_and_gate_aggregation(
    tmp_path_factory: pytest.TempPathFactory,
    feature_values: list[str],
) -> None:
    root = tmp_path_factory.mktemp("property-3")
    result_dir = _write_result_dir(root, feature_values)
    store = FeatureStore(root / "store")

    extract(result_dir, "default", store)

    record = store.records("default", cell_id=_TARGET_CELL, time_step=0)[0]
    counts = record.sample_counts[_TARGET_FEATURE]

    expected_valid_values = [
        value for value in (_oracle_classify(raw) for raw in feature_values) if value is not None
    ]
    expected_valid_count = len(expected_valid_values)
    expected_excluded_count = len(feature_values) - expected_valid_count

    # Count preservation: every input sample is classified as exactly one of
    # valid or excluded, and no sample is lost or double-counted.
    assert counts.valid == expected_valid_count
    assert counts.excluded == expected_excluded_count
    assert counts.valid + counts.excluded == len(feature_values)

    # Aggregation must use only the valid samples: with valid > 0 the recorded
    # feature value must equal the mean of exactly the valid samples (the
    # configured rule for cell_goodput_mbps), regardless of how many samples
    # were excluded; with valid == 0 the feature must be undefined.
    if expected_valid_count == 0:
        assert record.features[_TARGET_FEATURE] is None
        assert record.data_quality == "insufficient"
    else:
        expected_mean = round(sum(expected_valid_values) / expected_valid_count, 6)
        assert math.isclose(record.features[_TARGET_FEATURE], expected_mean, rel_tol=1e-6, abs_tol=1e-6)
