"""Property 6 test module (own module to avoid collisions with parallel Property tasks).

Task 2.8: 범위 밖/비실수 행만 제외되고 제외 수가 정확한지 검증한다.

design.md:

    #### Property 6: time_s 범위 필터링
    *For any* time_s 값 목록(1~900 범위 내/밖 값의 임의 혼합), Data_Extractor는 범위를
    벗어나거나 실수로 해석되지 않는 행만 제외하고, 제외된 행 수는 실제 조건 위반 행 수와
    일치하며 나머지 행은 정상 처리된다.
    **Validates: Requirements 1.7**

requirements.md 1.7:

    IF time_s 값이 1 미만이거나 900 을 초과하거나 실수로 해석되지 않으면, THEN THE
    Data_Extractor SHALL 해당 행을 집계에서 제외하고 제외된 행 수를 처리 결과에
    반환하며 나머지 행의 처리를 계속한다.

Targets ``smo.aimlfw.data_extractor.pipeline._time_s_in_range`` (the pure per-row
predicate) directly, and drives the full ``extract()`` pipeline end-to-end to
confirm the returned ``ExtractOutcome.excluded_rows`` count matches exactly the
number of rows whose ``time_s`` is outside 1..900 (inclusive) or non-numeric.
"""

from __future__ import annotations

import uuid
from decimal import Decimal
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.data_extractor import extract
from smo.aimlfw.data_extractor.pipeline import _time_s_in_range
from smo.aimlfw.feature_store import FeatureStore
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

_TARGET_CELL = "gNB_5G"
_TARGET_CELLS = ("gNB_5G", "gNB_4G_1", "gNB_4G_2")

_CSV_HEADER = (
    "time_s,interval_s,cell_id,cell,band,tx_power_dbm,ret_tilt_deg,cio_bias_db,ttt_ms,hysteresis_db,"
    "cell_goodput_mbps,avg_ue_goodput_mbps,ue_goodput_p5_mbps,sinr_p50_db,prb_utilization_pct,"
    "delay_p95_ms,ho_failure_count,pingpong_count,rlf_count,interval_energy_j\n"
)

_COMMAND_SH = (
    "#!/usr/bin/env bash\n"
    "set -euo pipefail\n"
    "'./ns3' '--runTag=seed7_prop6' '--cellTxPowerDbm=gNB_5G:43,gNB_4G_1:43,gNB_4G_2:43' "
    "'--cellRetTiltDeg=gNB_5G:5,gNB_4G_1:5,gNB_4G_2:5' '--cellCioDb=gNB_5G:0,gNB_4G_1:0,gNB_4G_2:0' "
    "'--cellHysteresisDb=gNB_5G:2.5,gNB_4G_1:2.5,gNB_4G_2:2.5' '--cellTttMs=gNB_5G:160,gNB_4G_1:160,gNB_4G_2:160'\n"
)

# Local generators, independent of strategies.py per the task's own-module note.
# Requirement 1.7's three exclusion categories (below range, above range,
# non-numeric) plus ordinary in-range numeric values.
_IN_RANGE_TIME_S = st.decimals(
    min_value=Decimal("1"),
    max_value=Decimal("900"),
    places=6,
    allow_nan=False,
    allow_infinity=False,
).map(str)
_BELOW_RANGE_TIME_S = st.one_of(
    st.decimals(min_value=Decimal("-1000"), max_value=Decimal("0.999999"), places=6, allow_nan=False, allow_infinity=False).map(str),
    st.just("0"),
)
_ABOVE_RANGE_TIME_S = st.decimals(
    min_value=Decimal("900.000001"), max_value=Decimal("10000"), places=6, allow_nan=False, allow_infinity=False
).map(str)
_NON_NUMERIC_TIME_S = st.sampled_from(
    (
        "",
        " ",
        "abc",
        "not-a-number",
        "1.2.3",
        "N/A",
        "--",
        "\t",
        # NaN is parseable by Decimal but is not a real number (requirements.md 1.7:
        # "실수로 해석되지 않으면"), matching how _parse_numeric already excludes it
        # for ordinary feature values.
        "nan",
        "NaN",
        "-nan",
        "+nan",
    )
)

_OUT_OF_RANGE_OR_NON_NUMERIC = st.one_of(_BELOW_RANGE_TIME_S, _ABOVE_RANGE_TIME_S, _NON_NUMERIC_TIME_S)
_ANY_TIME_S = st.one_of(_IN_RANGE_TIME_S, _OUT_OF_RANGE_OR_NON_NUMERIC)


def _oracle_excluded(raw: str) -> bool:
    """Independent reference for Requirement 1.7's row-exclusion rule.

    Excludes exactly when time_s does not parse as a real number, or parses but
    is < 1 or > 900. NaN parses under ``Decimal`` but is neither < 1 nor > 900
    when compared naively, so it is treated as excluded via an explicit
    finiteness check (a NaN/Infinity time_s is not a valid real number).
    """
    stripped = raw.strip()
    if not stripped:
        return True
    try:
        value = Decimal(stripped)
    except Exception:
        return True
    if not value.is_finite():
        return True
    return not (Decimal("1") <= value <= Decimal("900"))


def _row(time_s: str) -> str:
    return f"{time_s},1,1,{_TARGET_CELL},NR,43,5,0,160,2.5,1.0,1.0,1.0,10.0,50.0,5.0,0,0,0,10.0\n"


def _write_result_dir(root: Path, time_s_values: list[str]) -> Path:
    result_dir = root / "run"
    result_dir.mkdir()
    rows = "".join(_row(value) for value in time_s_values)
    (result_dir / "cell_kpi.csv").write_text(_CSV_HEADER + rows, encoding="utf-8")
    (result_dir / "command.sh").write_text(_COMMAND_SH, encoding="utf-8")
    return result_dir


# --- Direct unit-level property on the pure per-row predicate ---


# **Property 6: time_s 범위 필터링**
# **Validates: Requirements 1.7**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(time_s=_ANY_TIME_S)
def test_time_s_in_range_matches_oracle_exclusion_rule(time_s: str) -> None:
    result = _time_s_in_range(time_s)
    excluded = result is None

    assert excluded == _oracle_excluded(time_s)
    if not excluded:
        assert Decimal("1") <= result <= Decimal("900")


# **Property 6: time_s 범위 필터링**
# **Validates: Requirements 1.7**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(time_s=_IN_RANGE_TIME_S)
def test_in_range_time_s_is_never_excluded(time_s: str) -> None:
    assert _time_s_in_range(time_s) is not None


# **Property 6: time_s 범위 필터링**
# **Validates: Requirements 1.7**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(time_s=_OUT_OF_RANGE_OR_NON_NUMERIC)
def test_out_of_range_or_non_numeric_time_s_is_always_excluded(time_s: str) -> None:
    assert _time_s_in_range(time_s) is None


# --- End-to-end: excluded_rows returned by extract() matches the actual count ---


@st.composite
def _mixed_time_s_rows(draw: st.DrawFn) -> list[str]:
    return draw(st.lists(_ANY_TIME_S, min_size=1, max_size=40))


# **Property 6: time_s 범위 필터링**
# **Validates: Requirements 1.7**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(time_s_values=_mixed_time_s_rows())
def test_extract_excluded_rows_count_matches_actual_violations(
    tmp_path_factory: pytest.TempPathFactory,
    time_s_values: list[str],
) -> None:
    root = tmp_path_factory.mktemp(f"property-6-{uuid.uuid4().hex}")
    result_dir = _write_result_dir(root, time_s_values)
    store = FeatureStore(root / "store")

    outcome = extract(result_dir, "default", store)

    expected_excluded = sum(1 for value in time_s_values if _oracle_excluded(value))
    expected_kept = len(time_s_values) - expected_excluded

    assert outcome.excluded_rows == expected_excluded

    # The remaining (non-excluded) rows must still be processed: they land in one
    # of the target cell's Feature_Records without raising, and every valid
    # time_s produces a record for a Time_Step consistent with the boundary rule.
    records = store.records("default", cell_id=_TARGET_CELL)
    assert len(records) == 5  # one Feature_Record per Time_Step 0..4, regardless of row count

    if expected_kept > 0:
        kept_time_steps = set()
        for value in time_s_values:
            if not _oracle_excluded(value):
                decimal_value = Decimal(value.strip())
                for step in range(5):
                    if Decimal(180 * step) < decimal_value <= Decimal(180 * (step + 1)):
                        kept_time_steps.add(step)
                        break
        touched_records = [record for record in records if record.time_step in kept_time_steps]
        # Every Time_Step actually touched by a kept row must have at least one
        # valid sample recorded for some feature (i.e. was not silently dropped).
        for record in touched_records:
            total_valid = sum(count.valid for count in record.sample_counts.values())
            assert total_valid >= 1
