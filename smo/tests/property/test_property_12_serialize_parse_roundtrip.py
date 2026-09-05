"""Property 12 test module (own module to avoid collisions with parallel Property tasks).

Task 2.14: 빈 집합을 포함해 필드·개수·수치 오차·문자열 동일성을 검증한다.

Targets the Feature_Store's CSV serialize -> parse round trip, as specified in
requirements.md 2.2, 2.3, 2.8 and implemented by
``smo.aimlfw.feature_store.store.FeatureStore.serialize``/``parse`` (Task 2.1):

    2.2: WHEN 직렬화된 파일 경로를 담은 파싱 요청이 도착하면, THE Feature_Store SHALL
    해당 파일의 모든 데이터 레코드를 Feature_Record 집합으로 복원하고 복원된 레코드
    개수를 반환한다.

    2.3: WHEN 1개 이상 100,000개 이하의 유효한 Feature_Record 집합이 직렬화된 후
    파싱되면, THE Feature_Store SHALL 원본과 동일한 필드 이름 집합, 동일한 레코드
    개수, 각 수치 필드 값이 원본 대비 절대 오차 1e-9 이내인 값, 각 문자열 필드 값이
    원본과 완전히 일치하는 값을 갖는 집합을 반환한다.

    2.8: WHEN Feature_Record 개수가 0 인 집합이 직렬화된 후 파싱되면, THE Feature_Store
    SHALL 오류 없이 레코드 개수가 0 인 Feature_Record 집합을 반환한다.

To keep the Hypothesis run fast, this test bounds the generated collection size well
below the 100,000-record contractual maximum (which is exercised, without round-trip
fidelity concerns, by Task 2.17's dedicated performance test) while still exercising
the empty-set case (2.8) explicitly via ``@example`` alongside Hypothesis-drawn
non-empty collections (2.2, 2.3).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import example, given
from hypothesis import strategies as st

from smo.aimlfw.common.config import load_feature_group
from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES
from smo.aimlfw.common.models import FeatureRecord
from smo.aimlfw.feature_store import FeatureStore
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS, feature_records

_FEATURE_GROUP = "default"
_FEATURE_NAMES = load_feature_group(_FEATURE_GROUP).features
_NUMERIC_TOLERANCE = 1e-9

# Bounded well under the 100,000-record contractual maximum for fast, focused
# verification of the round-trip fidelity invariant itself (see module docstring).
_MAX_GENERATED_RECORDS = 300


def _as_default_group(record: FeatureRecord) -> FeatureRecord:
    """Force a generated record onto the fixed Feature_Group this test validates."""
    return record.model_copy(update={"feature_group": _FEATURE_GROUP})


_records_for_group = feature_records(feature_names=_FEATURE_NAMES).map(_as_default_group)

# A well-formed Feature_Record *set* has a unique (time_step, cell_id) key per record
# (Requirement 1.1); serialize()/parse() do not deduplicate (that is upsert()'s job), so
# generating duplicate keys here would make key-based round-trip comparison ambiguous
# rather than exercising Requirement 2.3 itself.
_unique_records_for_group = st.lists(
    _records_for_group,
    min_size=0,
    max_size=_MAX_GENERATED_RECORDS,
    unique_by=lambda record: (record.time_step, record.cell_id),
)


def _assert_numeric_close(original: float | int, restored: float | int, field_name: str) -> None:
    assert abs(float(original) - float(restored)) <= _NUMERIC_TOLERANCE, (
        f"{field_name}: {original!r} vs {restored!r} exceeds {_NUMERIC_TOLERANCE} tolerance"
    )


def _assert_record_round_trip(original: FeatureRecord, restored: FeatureRecord) -> None:
    """Compare one (original, restored) pair field-by-field per Requirement 2.3."""
    original_payload: dict[str, Any] = original.model_dump(mode="json")
    restored_payload: dict[str, Any] = restored.model_dump(mode="json")

    # Same top-level field name set (identical schema on both sides).
    assert set(original_payload) == set(restored_payload)

    # String fields: exact equality, no tolerance.
    for field_name in ("feature_group", "cell_id", "data_quality", "source_dir"):
        assert original_payload[field_name] == restored_payload[field_name], field_name

    # Integer identity fields: exact equality.
    assert original.time_step == restored.time_step
    assert original.seed == restored.seed

    # control_parameters: same field name set, numeric tolerance per value.
    assert set(original_payload["control_parameters"]) == set(restored_payload["control_parameters"])
    for name in CONTROL_PARAMETER_NAMES:
        _assert_numeric_close(
            getattr(original.control_parameters, name),
            getattr(restored.control_parameters, name),
            f"control_parameters.{name}",
        )

    # features: same feature-name set as the original record (Requirement 2.3's
    # "동일한 필드 이름 집합"), None preserved exactly, numeric values within tolerance.
    assert set(original.features) == set(restored.features)
    for name, original_value in original.features.items():
        restored_value = restored.features[name]
        if original_value is None:
            assert restored_value is None, name
        else:
            assert restored_value is not None, name
            _assert_numeric_close(original_value, restored_value, f"features.{name}")

    # sample_counts: same feature-name set, exact integer equality (no tolerance for counts).
    assert set(original.sample_counts) == set(restored.sample_counts)
    for name, count in original.sample_counts.items():
        restored_count = restored.sample_counts[name]
        assert count.valid == restored_count.valid, f"sample_counts.{name}.valid"
        assert count.excluded == restored_count.excluded, f"sample_counts.{name}.excluded"


# **Property 12: 직렬화-파싱 라운드트립**
# **Validates: Requirements 2.2, 2.3, 2.8**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(records=_unique_records_for_group)
@example(records=[])
def test_serialize_then_parse_round_trips_fields_count_and_values(
    records: list[FeatureRecord],
) -> None:
    with tempfile.TemporaryDirectory() as raw_root:
        store = FeatureStore(Path(raw_root) / "store")
        target = Path(raw_root) / "export.csv"

        written = store.serialize(_FEATURE_GROUP, records, target)
        assert written == len(records)

        restored_records = store.parse(target, feature_group=_FEATURE_GROUP)

        # 2.2 / 2.8: parsing restores exactly as many records as were written,
        # including zero for an empty input set, and never raises for that case.
        assert len(restored_records) == len(records)

        # 2.3: match restored records back to their originals by (time_step, cell_id)
        # -- the key Feature_Store sorts and de-duplicates on -- then compare every
        # field for name-set, numeric-tolerance, and string-exactness equivalence.
        restored_by_key = {(record.time_step, record.cell_id): record for record in restored_records}
        assert set(restored_by_key) == {(record.time_step, record.cell_id) for record in records}
        for original in records:
            restored = restored_by_key[(original.time_step, original.cell_id)]
            _assert_record_round_trip(original, restored)
