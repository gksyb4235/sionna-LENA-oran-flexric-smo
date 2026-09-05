"""Property 11 test module (own module to avoid collisions with parallel Property tasks).

Task 2.13: 최대 100,000개 레코드의 정렬 순서와 기록 개수를 검증한다.

Targets the Feature_Store's CSV serialization, as specified in requirements.md 2.1
and implemented by ``smo.aimlfw.feature_store.store.FeatureStore.serialize`` (Task 2.1):

    Feature_Record 집합과 대상 파일 경로를 담은 직렬화 요청이 도착하면, THE Feature_Store
    SHALL 최대 100,000개의 Feature_Record 를 (Time_Step, cell_id) 오름차순으로 단일 파일에
    기록하고 기록된 레코드 개수를 반환한다.

To keep the Hypothesis run fast, this test bounds the generated collection size well
below the 100,000-record contractual maximum (which is exercised, without sorting/count
concerns, by Task 2.17's dedicated performance test) while still exercising duplicate
keys, ties, and varied Time_Step/cell_id combinations.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.common.config import load_feature_group
from smo.aimlfw.common.models import FeatureRecord
from smo.aimlfw.feature_store import FeatureStore
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS, feature_records

_FEATURE_GROUP = "default"
_FEATURE_NAMES = load_feature_group(_FEATURE_GROUP).features

# Bounded well under the 100,000-record contractual maximum for fast, focused
# verification of the sort/count invariant itself (see module docstring).
_MAX_GENERATED_RECORDS = 300


def _as_default_group(record: FeatureRecord) -> FeatureRecord:
    """Force a generated record onto the fixed Feature_Group this test validates."""
    return record.model_copy(update={"feature_group": _FEATURE_GROUP})


_records_for_group = feature_records(feature_names=_FEATURE_NAMES).map(_as_default_group)


# **Property 11: 직렬화 순서와 개수**
# **Validates: Requirements 2.1**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    records=st.lists(_records_for_group, min_size=1, max_size=_MAX_GENERATED_RECORDS),
)
def test_serialize_orders_by_time_step_and_cell_id_and_preserves_record_count(
    records: list[FeatureRecord],
) -> None:
    expected_keys = sorted((record.time_step, record.cell_id) for record in records)

    with tempfile.TemporaryDirectory() as raw_root:
        store = FeatureStore(Path(raw_root) / "store")
        target = Path(raw_root) / "export.csv"

        written = store.serialize(_FEATURE_GROUP, records, target)

        # Returned count matches the input record count exactly.
        assert written == len(records)

        parsed = store.parse(target, feature_group=_FEATURE_GROUP)
        assert len(parsed) == len(records)

        # Serialized order is the (Time_Step, cell_id) ascending order of the input set.
        actual_keys = [(record.time_step, record.cell_id) for record in parsed]
        assert actual_keys == expected_keys
        assert actual_keys == sorted(actual_keys)
