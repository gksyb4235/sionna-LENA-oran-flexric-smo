"""Property 7 test module (own module to avoid collisions with parallel Property tasks).

Task 2.9: 누락 목록, 레코드 미생성 및 Feature Store 무변경을 검증한다.

Targets the Data_Extractor's column/aggregation-rule validation, as specified in
requirements.md 1.8 and implemented by
``smo.aimlfw.data_extractor.pipeline._validate_columns`` and ``extract`` (Task 2.2):

    - IF CSV 헤더에 Feature_Group 이 요구하는 열 이름이 없거나 Feature_Group 에 어떤
      요구 피처의 집계 규칙이 정의되어 있지 않으면, THEN THE Data_Extractor SHALL
      누락된 열 이름 목록과 집계 규칙이 정의되지 않은 피처 이름 목록을 담은 오류를
      반환하고, Feature_Record 를 하나도 생성하지 않으며 Feature_Store 를 변경하지
      않는다.

``FeatureGroupConfig.validate_contract`` (smo/aimlfw/common/config.py) already
guarantees ``aggregations`` exactly matches ``features`` at config-load time, so
an "undefined aggregation rule" Feature_Group can never reach ``_validate_columns``
at runtime; that half of Requirement 1.8 is covered by
``details["undefined_aggregation_features"] == []`` staying stable. The reachable,
property-testable half is the missing-CSV-header-column case, which this module
drives both directly against ``_validate_columns`` and end-to-end through
``extract()`` (with a Feature_Store snapshot taken before/after the failing call).
"""

from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES
from smo.aimlfw.data_extractor.errors import DataExtractorError
from smo.aimlfw.data_extractor.pipeline import CELL_COLUMN, TIME_S_COLUMN, _validate_columns, extract
from smo.aimlfw.feature_store import FeatureStore
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

# The "default" Feature_Group (smo/aimlfw/config/feature_groups/default.json) fixes
# target_cells and features/target_kpis to these values; Property 7 does not exercise
# Feature_Group selection itself, so the fixed default config keeps the generator
# focused on the missing-column invariant.
_TARGET_CELLS = ("gNB_5G", "gNB_4G_1", "gNB_4G_2")
_FEATURE_NAMES = (
    "cell_goodput_mbps", "avg_ue_goodput_mbps", "ue_goodput_p5_mbps", "sinr_p50_db",
    "prb_utilization_pct", "delay_p95_ms", "ho_failure_count", "pingpong_count",
    "rlf_count", "interval_energy_j",
)
_NON_REQUIRED_COLUMNS = ("interval_s", "cell_id", "band")
_FULL_CSV_HEADER = (TIME_S_COLUMN, *_NON_REQUIRED_COLUMNS, CELL_COLUMN, *CONTROL_PARAMETER_NAMES, *_FEATURE_NAMES)
_REQUIRED_COLUMNS = (TIME_S_COLUMN, CELL_COLUMN, *CONTROL_PARAMETER_NAMES, *_FEATURE_NAMES)

# Fixed, in-range Control_Parameter values used for the valid baseline CSV rows;
# their exact values are irrelevant to Property 7, only their validity matters.
_BASELINE_CONTROL_VALUES = {
    "tx_power_dbm": 43,
    "ret_tilt_deg": 5,
    "cio_bias_db": 0.0,
    "hysteresis_db": 2.0,
    "ttt_ms": 160,
}


def _dropped_columns() -> st.SearchStrategy[list[str]]:
    """Generate a non-empty subset of the columns `_validate_columns` requires."""
    return st.lists(
        st.sampled_from(_REQUIRED_COLUMNS),
        min_size=1,
        max_size=len(_REQUIRED_COLUMNS),
        unique=True,
    )


def _arbitrary_feature_names() -> st.SearchStrategy[list[str]]:
    return st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=16),
        min_size=1,
        max_size=8,
        unique=True,
    )


def _baseline_csv_text() -> str:
    """Render a fully valid cell_kpi.csv: 3 target cells x 5 Time_Steps."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_FULL_CSV_HEADER)
    for cell_id in _TARGET_CELLS:
        for time_s in (180, 360, 540, 720, 900):
            row = {
                TIME_S_COLUMN: time_s,
                "interval_s": 180,
                "cell_id": 1,
                CELL_COLUMN: cell_id,
                "band": "NR",
                **_BASELINE_CONTROL_VALUES,
                **{name: 1.0 for name in _FEATURE_NAMES},
            }
            writer.writerow([row[column] for column in _FULL_CSV_HEADER])
    return buffer.getvalue()


def _broken_csv_text(dropped_columns: list[str]) -> str:
    """Render a header-only cell_kpi.csv missing `dropped_columns` (no data rows needed:
    `_validate_columns` raises before any data row is read)."""
    header = [column for column in _FULL_CSV_HEADER if column not in dropped_columns]
    buffer = io.StringIO()
    csv.writer(buffer).writerow(header)
    return buffer.getvalue()


def _snapshot(store: FeatureStore) -> list[dict[str, Any]]:
    return [record.model_dump(mode="json") for record in store.records("default")]


# **Property 7: 필수 열/집계 규칙 누락 시 무변경 오류**
# **Validates: Requirements 1.8**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(feature_names=_arbitrary_feature_names(), dropped=st.data())
def test_validate_columns_reports_missing_column_list(
    feature_names: list[str],
    dropped: st.DataObject,
) -> None:
    """Direct unit-level property against `_validate_columns`: any non-empty subset
    of the required columns removed from the header must be reported, verbatim and
    completely, as `missing_columns`, with `undefined_aggregation_features` empty
    (Feature_Group config already guarantees every feature has an aggregation rule)."""
    required = (TIME_S_COLUMN, CELL_COLUMN, *CONTROL_PARAMETER_NAMES, *feature_names)
    dropped_columns = dropped.draw(
        st.lists(st.sampled_from(required), min_size=1, max_size=len(required), unique=True)
    )
    header = ["extra_col", *[column for column in required if column not in dropped_columns]]

    with pytest.raises(DataExtractorError) as excinfo:
        _validate_columns(header, feature_names)

    error = excinfo.value
    assert error.error_code == "missing_columns"
    assert error.details["missing_columns"] == sorted(dropped_columns)
    assert error.details["undefined_aggregation_features"] == []


# **Property 7: 필수 열/집계 규칙 누락 시 무변경 오류**
# **Validates: Requirements 1.8**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(dropped_columns=_dropped_columns())
def test_extract_raises_and_leaves_feature_store_unchanged(dropped_columns: list[str]) -> None:
    """End-to-end property against `extract()`: a cell_kpi.csv missing one or more
    columns required by the Feature_Group must raise `DataExtractorError` listing the
    missing columns, write zero FeatureRecords, and leave the Feature_Store's existing
    contents (populated or empty) completely unchanged."""
    with tempfile.TemporaryDirectory() as baseline_dir, tempfile.TemporaryDirectory() as broken_dir:
        baseline_path = Path(baseline_dir)
        broken_path = Path(broken_dir)
        (baseline_path / "cell_kpi.csv").write_text(_baseline_csv_text(), encoding="utf-8")
        (broken_path / "cell_kpi.csv").write_text(_broken_csv_text(dropped_columns), encoding="utf-8")

        # A fresh, never-written store: failure must leave it with zero records.
        with tempfile.TemporaryDirectory() as fresh_store_dir:
            fresh_store = FeatureStore(Path(fresh_store_dir))
            with pytest.raises(DataExtractorError) as fresh_excinfo:
                extract(broken_path, "default", fresh_store)
            fresh_error = fresh_excinfo.value
            assert fresh_error.error_code == "missing_columns"
            assert fresh_error.details["missing_columns"] == sorted(dropped_columns)
            assert fresh_error.details["undefined_aggregation_features"] == []
            assert _snapshot(fresh_store) == []

        # A pre-populated store: failure must leave its existing records untouched
        # (not merely empty), i.e. no partial write and no unrelated mutation.
        with tempfile.TemporaryDirectory() as populated_store_dir:
            populated_store = FeatureStore(Path(populated_store_dir))
            baseline_outcome = extract(baseline_path, "default", populated_store)
            assert baseline_outcome.records_written == len(_TARGET_CELLS) * 5
            before = _snapshot(populated_store)
            assert before  # baseline extraction must have actually populated the store

            with pytest.raises(DataExtractorError) as populated_excinfo:
                extract(broken_path, "default", populated_store)
            populated_error = populated_excinfo.value
            assert populated_error.error_code == "missing_columns"
            assert populated_error.details["missing_columns"] == sorted(dropped_columns)

            after = _snapshot(populated_store)
            assert after == before
