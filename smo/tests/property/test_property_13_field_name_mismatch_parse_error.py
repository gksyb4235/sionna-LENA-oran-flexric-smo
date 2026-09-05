"""Property 13 test module (own module to avoid collisions with parallel Property tasks).

Task 2.15: 누락/미정의 필드 상세와 무부분결과를 검증한다.

Targets the Feature_Store's CSV header validation, as specified in
requirements.md 2.4 and implemented by
``smo.aimlfw.feature_store.store.FeatureStore.parse`` / ``_validate_header``
(Task 2.1):

    IF 파싱 대상 파일의 필드 이름 집합이 Feature_Group 이 정의한 피처 이름 목록 및
    타깃 KPI 목록과 일치하지 않으면, THEN THE Feature_Store SHALL 누락된 필드 이름
    목록과 정의되지 않은 필드 이름 목록을 포함한 오류를 반환하고, Feature_Record 를
    하나도 반환하지 않는다.

``FeatureStore.parse`` reads and buffers all CSV rows *before* resolving
Feature_Group feature names and validating the header (see ``store.py``), so a
header mismatch always raises before the row-parsing list comprehension is
ever evaluated. Atomicity ("무부분결과") therefore reduces to "the call raises
instead of returning", which every assertion below is built around: there is
no code path where ``parse()`` can hand back a partial list, so a header-only
CSV (zero data rows) already exercises the full invariant.
"""

from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.feature_store import FeatureStore, FeatureStoreError
from smo.aimlfw.feature_store.store import _COUNT_SUFFIXES, _FIXED_PREFIX, _FIXED_SUFFIX
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

# Fixed by smo/aimlfw/config/feature_groups/default.json; Property 13 does not
# exercise Feature_Group selection itself, so a single fixed group keeps the
# generator focused on the field-name-mismatch invariant. `features` and
# `target_kpis` are identical for this group, matching Requirement 2.4's
# "피처 이름 목록 및 타깃 KPI 목록" wording.
_FEATURE_GROUP = "default"
_FEATURE_NAMES = (
    "cell_goodput_mbps", "avg_ue_goodput_mbps", "ue_goodput_p5_mbps", "sinr_p50_db",
    "prb_utilization_pct", "delay_p95_ms", "ho_failure_count", "pingpong_count",
    "rlf_count", "interval_energy_j",
)
_COUNT_FIELDS = tuple(f"{name}{suffix}" for name in _FEATURE_NAMES for suffix in _COUNT_SUFFIXES)
_EXPECTED_HEADER = (*_FIXED_PREFIX, *_FEATURE_NAMES, *_COUNT_FIELDS, *_FIXED_SUFFIX)


def _write_header_only_csv(path: Path, header: list[str]) -> None:
    buffer = io.StringIO()
    csv.writer(buffer).writerow(header)
    path.write_text(buffer.getvalue(), encoding="utf-8")


@st.composite
def _mismatched_headers(draw: st.DrawFn) -> tuple[list[str], list[str], list[str]]:
    """Draw (actual_header, expected_missing, expected_undefined) with >=1 mutation.

    ``drop`` removes a subset of the canonical header (missing fields);
    ``extra`` appends new, never-canonical field names (undefined fields).
    Both may be non-empty simultaneously; at least one always is, so the
    resulting header never accidentally matches the canonical one.
    """
    drop = draw(
        st.lists(st.sampled_from(_EXPECTED_HEADER), min_size=0, max_size=len(_EXPECTED_HEADER), unique=True)
    )
    extra = draw(
        st.lists(
            st.text(alphabet="abcdefghijklmnopqrstuvwxyz_", min_size=1, max_size=20).filter(
                lambda name: name not in _EXPECTED_HEADER
            ),
            min_size=0,
            max_size=5,
            unique=True,
        )
    )
    if not drop and not extra:
        drop = draw(st.lists(st.sampled_from(_EXPECTED_HEADER), min_size=1, max_size=1))

    kept = [name for name in _EXPECTED_HEADER if name not in drop]
    actual_header = [*kept, *extra]
    return actual_header, sorted(drop), sorted(extra)


# **Property 13: 필드 이름 불일치 파싱 오류**
# **Validates: Requirements 2.4**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(case=_mismatched_headers())
def test_parse_rejects_mismatched_field_names_with_no_partial_result(
    case: tuple[list[str], list[str], list[str]],
) -> None:
    actual_header, expected_missing, expected_undefined = case

    with tempfile.TemporaryDirectory() as raw_root:
        source_path = Path(raw_root) / "broken.csv"
        _write_header_only_csv(source_path, actual_header)
        store = FeatureStore(Path(raw_root) / "store")

        # `parse()` either returns a full list or raises; there is no partial-list
        # return path, so asserting the raise here *is* the "무부분결과" check.
        with pytest.raises(FeatureStoreError) as excinfo:
            store.parse(source_path, feature_group=_FEATURE_GROUP)

        error = excinfo.value
        expected_code = "missing_fields" if expected_missing else "undefined_fields"
        assert error.error_code == expected_code
        assert error.details["missing_fields"] == expected_missing
        assert error.details["undefined_fields"] == expected_undefined
