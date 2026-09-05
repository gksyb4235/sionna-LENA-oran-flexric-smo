"""Property 14 test module (own module to avoid collisions with parallel Property tasks).

Task 2.16: 임의 위치 손상에서 최초 인덱스와 무부분결과를 검증한다.

Targets the Feature_Store's CSV parsing, as specified in requirements.md 2.7 and
implemented by ``smo.aimlfw.feature_store.store.FeatureStore.parse`` (Task 2.1):

    IF 파싱 대상 파일이 잘린 레코드 또는 헤더에 선언된 필드 개수와 다른 필드 개수를 갖는
    레코드를 포함하면, THEN THE Feature_Store SHALL 최초 위반 레코드 인덱스를 포함한
    오류를 반환하고 Feature_Record 를 하나도 반환하지 않는다.

design.md Property 14 broadens "손상" beyond field-count mismatches to also cover a
malformed numeric value within an otherwise well-formed row (the ``corrupted_record``
error code ``FeatureStore._number``/``_parse_row`` raise for that case); both failure
shapes share the same observable contract this test verifies: the raised error's
``record_index`` identifies exactly the first corrupted row (0-based, counting data
rows after the header), and no Feature_Record is returned when the row is corrupted
at an arbitrary position among otherwise-valid rows.

A canonical, well-formed CSV file is produced via ``FeatureStore.serialize`` (Task 2.1)
itself, then exactly one of its data rows -- at a Hypothesis-drawn position -- is
corrupted in place (short field count, long field count, or a malformed numeric
Control_Parameter value) before being re-written and handed to ``FeatureStore.parse``.
"""

from __future__ import annotations

import csv
import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.common.config import load_feature_group
from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES
from smo.aimlfw.common.models import FeatureRecord
from smo.aimlfw.feature_store import FeatureStore, FeatureStoreError
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS, feature_records

_FEATURE_GROUP = "default"
_FEATURE_NAMES = load_feature_group(_FEATURE_GROUP).features

# Bounded well under the 100,000-record contractual maximum: this property concerns
# the *position* of a single corrupted row among otherwise-valid rows, not scale
# (covered separately by Task 2.17's performance test).
_MAX_GENERATED_RECORDS = 15

# Garbage values chosen so that neither ``int()`` nor ``float()`` accepts them --
# unlike "nan"/"inf", which parse successfully and fail a later finiteness check
# instead, exercising a different (but equally "corrupted_record") code path.
_GARBAGE_VALUES = ("abc", "N/A", "###", "12abc", "not_a_number")

_CORRUPTION_MODES = ("field_count_short", "field_count_long", "malformed_numeric")


def _as_default_group(record: FeatureRecord) -> FeatureRecord:
    """Force a generated record onto the fixed Feature_Group this test validates."""
    return record.model_copy(update={"feature_group": _FEATURE_GROUP})


_records_for_group = feature_records(feature_names=_FEATURE_NAMES).map(_as_default_group)


@st.composite
def _corruption_case(draw: Any) -> tuple[list[FeatureRecord], int, str, tuple[str, str] | None]:
    """Draw a valid record set, a corruption position within it, and a corruption mode."""
    records = draw(st.lists(_records_for_group, min_size=1, max_size=_MAX_GENERATED_RECORDS))
    index = draw(st.integers(min_value=0, max_value=len(records) - 1))
    mode = draw(st.sampled_from(_CORRUPTION_MODES))
    numeric_corruption = None
    if mode == "malformed_numeric":
        # Control_Parameter columns are always populated (never the empty-string
        # "missing feature" marker), so corrupting one is guaranteed effective.
        column_name = draw(st.sampled_from(CONTROL_PARAMETER_NAMES))
        garbage = draw(st.sampled_from(_GARBAGE_VALUES))
        numeric_corruption = (column_name, garbage)
    return records, index, mode, numeric_corruption


def _read_csv_rows(path: Path) -> tuple[list[str], list[list[str]]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        rows = list(csv.reader(input_file))
    return rows[0], rows[1:]


def _write_csv_rows(path: Path, header: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(rows)


def _apply_corruption(
    header: list[str],
    row: list[str],
    mode: str,
    numeric_corruption: tuple[str, str] | None,
) -> tuple[list[str], str]:
    """Corrupt one already-valid data row and return it with the expected error code."""
    if mode == "field_count_short":
        return row[:-1], "truncated_record"
    if mode == "field_count_long":
        return [*row, "999"], "truncated_record"
    column_name, garbage = numeric_corruption  # mode == "malformed_numeric"
    corrupted = list(row)
    corrupted[header.index(column_name)] = garbage
    return corrupted, "corrupted_record"


# **Property 14: 손상된 레코드 파싱 오류**
# **Validates: Requirements 2.7**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(case=_corruption_case())
def test_corrupted_record_at_arbitrary_position_raises_with_first_index_and_no_partial_result(
    case: tuple[list[FeatureRecord], int, str, tuple[str, str] | None],
) -> None:
    records, index, mode, numeric_corruption = case

    with tempfile.TemporaryDirectory() as raw_root:
        store = FeatureStore(Path(raw_root) / "store")
        valid_path = Path(raw_root) / "valid.csv"
        store.serialize(_FEATURE_GROUP, records, valid_path)

        header, rows = _read_csv_rows(valid_path)
        assert len(rows) == len(records)  # sanity: one data row per record, none dropped

        rows[index], expected_error_code = _apply_corruption(header, rows[index], mode, numeric_corruption)

        corrupted_path = Path(raw_root) / "corrupted.csv"
        _write_csv_rows(corrupted_path, header, rows)

        with pytest.raises(FeatureStoreError) as excinfo:
            # No partial result: parse() must raise rather than return any records,
            # even though every row up to `index` is well-formed.
            store.parse(corrupted_path, feature_group=_FEATURE_GROUP)

        error = excinfo.value
        assert error.error_code == expected_error_code
        # The error identifies exactly the first violating row, 0-based among data
        # rows -- every row before `index` was left untouched and thus valid.
        assert error.details["record_index"] == index
