"""Property 1 test module (independent, per tasks.md parallel-collision notes).

Targets the Data Extractor's pure Time_Step mapping, as specified verbatim in
design.md's Data_Extractor pipeline step 3::

    Time_Step = floor((time_s - 1e-9) / 180)

which design.md states is equivalent to the ``180*k < time_s <= 180*(k+1)`` rule,
with ``time_s`` parsed as a string via ``Decimal`` to avoid float boundary errors.

Task 2.2 (Data Extractor) implements ``smo.aimlfw.data_extractor.pipeline`` and
exposes:

    def time_step_for_time_s(time_s: str) -> int
        \"\"\"Map a valid time_s string (1 <= time_s <= 900) to Time_Step 0..4
        such that 180*k < time_s <= 180*(k+1).\"\"\"

    def extract(result_dir, feature_group, feature_store, ...) -> ExtractOutcome
        \"\"\"Run the full CSV -> Feature_Record -> Feature_Store pipeline.\"\"\"

This module targets ``time_step_for_time_s`` directly (pure mapping) and also
drives ``extract()`` end-to-end against a real ``smo.aimlfw.feature_store.FeatureStore``
to confirm that the (Time_Step, cell_id) key uniqueness guarantee holds through the
full pipeline, not just in a hand-simulated grouping.
"""

from __future__ import annotations

import uuid
from decimal import Decimal

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS, cell_ids

pipeline = pytest.importorskip(
    "smo.aimlfw.data_extractor.pipeline",
    reason=(
        "Task 2.2 (Data Extractor) has not implemented smo.aimlfw.data_extractor.pipeline yet. "
        "Property 1 targets pipeline.time_step_for_time_s(time_s: str) -> int, matching design.md's "
        "'Time_Step = floor((time_s - 1e-9) / 180)' formula (180*k < time_s <= 180*(k+1))."
    ),
)
feature_store_module = pytest.importorskip(
    "smo.aimlfw.feature_store",
    reason="Task 2.1 (Feature Store) has not implemented smo.aimlfw.feature_store yet.",
)
FeatureStore = feature_store_module.FeatureStore

# Valid time_s domain per Requirement 1.1: 1 <= time_s <= 900, six-decimal precision
# to mirror the CSV's textual representation before Decimal parsing.
_VALID_TIME_S = st.decimals(
    min_value=Decimal("1"),
    max_value=Decimal("900"),
    places=6,
    allow_nan=False,
    allow_infinity=False,
)


def _expected_time_step(time_s: Decimal) -> int:
    """Reference oracle for the 180-second boundary rule (design.md pipeline step 3)."""
    for k in range(5):
        if Decimal(180 * k) < time_s <= Decimal(180 * (k + 1)):
            return k
    raise AssertionError(f"time_s={time_s} has no Time_Step in 0..4")


@st.composite
def _valid_csv_rows(draw: st.DrawFn) -> list[tuple[str, str]]:
    """Generate (time_s, cell_id) rows with valid time_s and a small cell_id pool
    so that repeated (Time_Step, cell_id) keys are exercised."""
    cells = draw(cell_ids(min_size=1, max_size=4))
    pairs = draw(
        st.lists(
            st.tuples(_VALID_TIME_S, st.sampled_from(cells)),
            min_size=1,
            max_size=50,
        )
    )
    return [(str(time_s), cell_id) for time_s, cell_id in pairs]


# **Property 1: Feature_Record 키 유일성과 Time_Step 매핑**
# **Validates: Requirements 1.1, 1.6**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(time_s=_VALID_TIME_S)
def test_time_step_mapping_satisfies_the_180_second_boundary_rule(time_s: Decimal) -> None:
    time_step = pipeline.time_step_for_time_s(str(time_s))

    assert isinstance(time_step, int)
    assert 0 <= time_step <= 4
    assert Decimal(180 * time_step) < time_s <= Decimal(180 * (time_step + 1))
    assert time_step == _expected_time_step(time_s)


# **Property 1: Feature_Record 키 유일성과 Time_Step 매핑**
# **Validates: Requirements 1.1, 1.6**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(rows=_valid_csv_rows())
def test_feature_record_keys_are_unique_per_time_step_and_cell(rows: list[tuple[str, str]]) -> None:
    groups: dict[tuple[int, str], list[str]] = {}
    for time_s, cell_id in rows:
        key = (pipeline.time_step_for_time_s(time_s), cell_id)
        groups.setdefault(key, []).append(time_s)

    # Exactly one group (=> exactly one Feature_Record) per distinct (Time_Step, cell_id) key.
    distinct_keys = {(pipeline.time_step_for_time_s(time_s), cell_id) for time_s, cell_id in rows}
    assert set(groups) == distinct_keys

    # No row is lost or duplicated across groups.
    assert sum(len(members) for members in groups.values()) == len(rows)

    # Every member of a (Time_Step, cell_id) group actually satisfies that Time_Step's boundary rule.
    for (time_step, _cell_id), members in groups.items():
        assert 0 <= time_step <= 4
        for time_s in members:
            decimal_time_s = Decimal(time_s)
            assert Decimal(180 * time_step) < decimal_time_s <= Decimal(180 * (time_step + 1))


# --- End-to-end: drive the real extract() pipeline into a real FeatureStore ---
#
# The two tests above exercise time_step_for_time_s() directly and a hand-simulated
# grouping of (time_s, cell_id) rows. This section instead builds an actual
# cell_kpi.csv + command.sh result directory and calls
# smo.aimlfw.data_extractor.pipeline.extract() against a real
# smo.aimlfw.feature_store.FeatureStore, confirming the (Time_Step, cell_id)
# key-uniqueness guarantee (Requirement 1.1) holds through the full pipeline,
# not just in an oracle simulation.

_CSV_HEADER = (
    "time_s,interval_s,cell_id,cell,band,tx_power_dbm,ret_tilt_deg,cio_bias_db,ttt_ms,hysteresis_db,"
    "cell_goodput_mbps,avg_ue_goodput_mbps,ue_goodput_p5_mbps,sinr_p50_db,prb_utilization_pct,"
    "delay_p95_ms,ho_failure_count,pingpong_count,rlf_count,interval_energy_j\n"
)
_COMMAND_SH = (
    "#!/usr/bin/env bash\n"
    "set -euo pipefail\n"
    "'./ns3' '--runTag=seed7_property1' '--cellTxPowerDbm=gNB_5G:43,gNB_4G_1:43,gNB_4G_2:43' "
    "'--cellRetTiltDeg=gNB_5G:5,gNB_4G_1:5,gNB_4G_2:5' '--cellCioDb=gNB_5G:0,gNB_4G_1:0,gNB_4G_2:0' "
    "'--cellHysteresisDb=gNB_5G:2.5,gNB_4G_1:2.5,gNB_4G_2:2.5' '--cellTttMs=gNB_5G:160,gNB_4G_1:160,gNB_4G_2:160'\n"
)
# The "default" Feature_Group (smo/aimlfw/config/feature_groups/default.json) fixes
# the target cell set; only rows for these cells produce Feature_Records.
_TARGET_CELLS = ("gNB_5G", "gNB_4G_1", "gNB_4G_2")


def _csv_row(time_s: str, cell: str) -> str:
    return f"{time_s},1,1,{cell},NR,43,5,0,160,2.5,1.0,1.0,1.0,10.0,50.0,5.0,0,0,0,10.0\n"


@st.composite
def _target_cell_time_s_rows(draw: st.DrawFn) -> list[tuple[str, str]]:
    """Generate (time_s, cell_id) rows restricted to the default Feature_Group's
    target cells, so every row actually produces/updates a Feature_Record."""
    pairs = draw(
        st.lists(
            st.tuples(_VALID_TIME_S, st.sampled_from(_TARGET_CELLS)),
            min_size=1,
            max_size=40,
        )
    )
    return [(str(time_s), cell_id) for time_s, cell_id in pairs]


@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(rows=_target_cell_time_s_rows())
def test_extract_pipeline_produces_unique_time_step_cell_keys_in_feature_store(
    tmp_path_factory: pytest.TempPathFactory, rows: list[tuple[str, str]]
) -> None:
    result_dir = tmp_path_factory.mktemp(f"p1-{uuid.uuid4().hex}")
    (result_dir / "cell_kpi.csv").write_text(
        _CSV_HEADER + "".join(_csv_row(time_s, cell_id) for time_s, cell_id in rows),
        encoding="utf-8",
    )
    (result_dir / "command.sh").write_text(_COMMAND_SH, encoding="utf-8")
    store = FeatureStore(tmp_path_factory.mktemp(f"p1-store-{uuid.uuid4().hex}"))

    pipeline.extract(result_dir, "default", store)

    records = store.records("default")

    # Exactly one Feature_Record per (Time_Step, cell_id) key: the store always holds
    # every target cell x all 5 Time_Steps regardless of which rows were present.
    keys = [(record.time_step, record.cell_id) for record in records]
    assert len(keys) == len(set(keys))
    assert set(keys) == {(step, cell_id) for step in range(5) for cell_id in _TARGET_CELLS}

    # Every input row's mapped Time_Step matches the boundary rule and lands in a
    # Feature_Record that actually exists for that (Time_Step, cell_id) key.
    for time_s, cell_id in rows:
        time_step = pipeline.time_step_for_time_s(time_s)
        assert 0 <= time_step <= 4
        assert Decimal(180 * time_step) < Decimal(time_s) <= Decimal(180 * (time_step + 1))
        assert (time_step, cell_id) in set(keys)
