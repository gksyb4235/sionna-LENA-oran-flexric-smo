"""Property 27 test module (own module to avoid collisions with parallel Property tasks).

Task 3.16: 등록 요청의 모델 이름/Feature_Group 이름/평가 지표/산출물 URI 중 하나 이상이
누락되거나 유효하지 않은 경우, Model_Registry가 등록을 거부하고 `invalid_model_metadata`
오류와 정확한 위반 항목 목록을 반환하며 새 버전 번호를 부여하지 않는지 검증한다.

Targets ``smo.aimlfw.model_registry.registry.ModelRegistry.register`` (Task 3.2), whose
validation is delegated to the static ``_metadata_violations`` helper, as specified in
requirements.md Requirement 4.7:

    IF 등록 요청에 모델 이름, Feature_Group 이름, Target_KPI 별 평가 지표, 산출물 URI 중
    하나 이상이 누락되었거나 모델 이름 길이가 1 자 미만 또는 128 자 초과이면, THEN THE
    Model_Registry SHALL 등록을 거부하고 `invalid_model_metadata` 오류 코드와 누락 또는
    위반 항목 이름 목록을 포함한 오류 응답 본문을 반환하며 새 버전 번호를 부여하지 않는다.

design.md Property 27: 등록 메타데이터 검증
    *For any* 등록 요청에서 모델 이름/Feature_Group 이름/평가 지표/산출물 URI 중 하나
    이상이 누락되거나 모델 이름 길이가 1자 미만 또는 128자 초과인 경우, Model_Registry는
    등록을 거부하고 `invalid_model_metadata` 오류와 위반 항목 목록을 반환하며 새 버전
    번호를 부여하지 않는다.
    **Validates: Requirements 4.7**

The expected violation set below is written as an independent oracle (not imported from
``registry.py``) but is grounded in a direct reading of
``ModelRegistry._metadata_violations`` so the check order/semantics match the production
contract exactly: model_name -> feature_group -> metrics -> artifact_uri.
"""

from __future__ import annotations

import math
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st

from smo.aimlfw.common.constants import TARGET_KPIS
from smo.aimlfw.model_registry import ModelRegistry, ModelRegistryError
from smo.aimlfw.model_storage import ModelStorage
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS


class _FakeCursor:
    """Minimal in-memory stand-in for a PyMongo cursor (sort + limit only)."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents

    def sort(self, key: str, direction: int | None = None) -> "_FakeCursor":
        self.documents.sort(key=lambda item: item[key], reverse=direction == -1)
        return self

    def limit(self, count: int) -> "_FakeCursor":
        self.documents = self.documents[:count]
        return self

    def __iter__(self):
        return iter(deepcopy(self.documents))


class _FakeCollection:
    """Minimal in-memory stand-in for the PyMongo collection ``ModelRegistry`` uses."""

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []

    def create_index(self, keys: Any, **kwargs: Any) -> str:
        return kwargs.get("name", "index")

    @staticmethod
    def _matches(document: dict[str, Any], query: dict[str, Any]) -> bool:
        return all(document.get(key) == value for key, value in query.items())

    def find_one(self, query: dict[str, Any], *args: Any, **kwargs: Any):
        matching = [item for item in self.documents if self._matches(item, query)]
        sort = kwargs.get("sort")
        if sort:
            key, direction = sort[0]
            matching.sort(key=lambda item: item[key], reverse=direction == -1)
        return deepcopy(matching[0]) if matching else None

    def find(self, query: dict[str, Any]) -> _FakeCursor:
        return _FakeCursor([deepcopy(item) for item in self.documents if self._matches(item, query)])

    def update_one(self, query: dict[str, Any], update: dict[str, Any], **kwargs: Any) -> None:
        if not any(self._matches(item, query) for item in self.documents):
            document = deepcopy(query)
            document.update(deepcopy(update.get("$setOnInsert", {})))
            self.documents.append(document)

    def insert_one(self, document: dict[str, Any]) -> None:
        self.documents.append(deepcopy(document))


def _make_registry(root: Path) -> tuple[ModelRegistry, _FakeCollection]:
    collection = _FakeCollection()
    return ModelRegistry(collection, ModelStorage(root / "models")), collection


def _seed_unrelated_model(collection: _FakeCollection, model_name: str = "unrelated-model") -> None:
    """Seed an already-registered, fully valid model so "unchanged" is meaningful."""
    now = datetime.now(timezone.utc)
    collection.documents.append(
        {"_id": f"model:{model_name}", "document_type": "model", "model_name": model_name, "created_at": now}
    )
    collection.documents.append(
        {
            "_id": f"version:{model_name}:1",
            "document_type": "version",
            "model_id": f"model:{model_name}",
            "model_name": model_name,
            "version": 1,
            "created_at": now,
            "feature_group": "default",
            "metrics": {"delay_p95_ms": 1.0},
            "artifact_uri": f"models/{model_name}/1/model.pt",
        }
    )


def _expected_violations(
    model_name: Any, feature_group: Any, metrics: Any, artifact_uri: Any
) -> list[str]:
    """Independent oracle grounded in a direct reading of
    ``ModelRegistry._metadata_violations`` (Requirement 4.7)."""
    violations: list[str] = []
    if not isinstance(model_name, str) or not 1 <= len(model_name) <= 128:
        violations.append("model_name")
    if not isinstance(feature_group, str) or not 1 <= len(feature_group) <= 128:
        violations.append("feature_group")
    if not isinstance(metrics, dict) or not metrics:
        violations.append("metrics")
    elif any(
        not isinstance(kpi, str)
        or kpi not in TARGET_KPIS
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
        for kpi, value in metrics.items()
    ):
        violations.append("metrics")
    if not isinstance(artifact_uri, str) or not artifact_uri:
        violations.append("artifact_uri")
    return violations


# --- model_name strategies -------------------------------------------------

_VALID_MODEL_NAME = st.text(min_size=1, max_size=128)
_INVALID_MODEL_NAME = st.one_of(
    st.just(""),  # empty string
    st.text(min_size=129, max_size=200),  # too long
    st.none(),
    st.integers(),
    st.booleans(),
    st.lists(st.text(), max_size=3),
)
_MODEL_NAME = st.one_of(_VALID_MODEL_NAME, _INVALID_MODEL_NAME)

# --- feature_group strategies -----------------------------------------------

_VALID_FEATURE_GROUP = st.text(min_size=1, max_size=128)
_INVALID_FEATURE_GROUP = st.one_of(
    st.just(""),  # empty string
    st.text(min_size=129, max_size=200),  # too long
    st.none(),
    st.integers(),
    st.lists(st.text(), max_size=3),
)
_FEATURE_GROUP = st.one_of(_VALID_FEATURE_GROUP, _INVALID_FEATURE_GROUP)

# --- metrics strategies ------------------------------------------------------

_FINITE_NONNEGATIVE = st.floats(min_value=0, allow_nan=False, allow_infinity=False, width=32)


@st.composite
def _valid_metrics(draw: Any) -> dict[str, float]:
    kpis = draw(st.lists(st.sampled_from(TARGET_KPIS), min_size=1, max_size=len(TARGET_KPIS), unique=True))
    return {kpi: draw(_FINITE_NONNEGATIVE) for kpi in kpis}


@st.composite
def _metrics_with_invalid_kpi_key(draw: Any) -> dict[str, float]:
    bad_key = draw(st.text(min_size=1, max_size=16).filter(lambda name: name not in TARGET_KPIS))
    return {bad_key: draw(_FINITE_NONNEGATIVE)}


@st.composite
def _metrics_with_non_finite_value(draw: Any) -> dict[str, float]:
    kpi = draw(st.sampled_from(TARGET_KPIS))
    value = draw(st.sampled_from([math.nan, math.inf, -math.inf]))
    return {kpi: value}


@st.composite
def _metrics_with_negative_value(draw: Any) -> dict[str, float]:
    kpi = draw(st.sampled_from(TARGET_KPIS))
    value = draw(st.floats(max_value=-1e-9, min_value=-1_000_000, allow_nan=False, allow_infinity=False))
    return {kpi: value}


@st.composite
def _metrics_with_bool_value(draw: Any) -> dict[str, float]:
    kpi = draw(st.sampled_from(TARGET_KPIS))
    return {kpi: draw(st.booleans())}


@st.composite
def _metrics_with_non_numeric_value(draw: Any) -> dict[str, Any]:
    kpi = draw(st.sampled_from(TARGET_KPIS))
    value = draw(st.one_of(st.text(), st.none(), st.lists(st.integers(), max_size=2)))
    return {kpi: value}


_INVALID_METRICS = st.one_of(
    st.none(),
    st.just({}),  # empty dict
    st.just([1, 2, 3]),  # non-dict type
    st.just("not-a-dict"),
    _metrics_with_invalid_kpi_key(),
    _metrics_with_non_finite_value(),
    _metrics_with_negative_value(),
    _metrics_with_bool_value(),
    _metrics_with_non_numeric_value(),
)
_METRICS = st.one_of(_valid_metrics(), _INVALID_METRICS)

# --- artifact_uri strategies -------------------------------------------------

_VALID_ARTIFACT_URI = st.text(min_size=1, max_size=64).filter(str.strip)
_INVALID_ARTIFACT_URI = st.one_of(
    st.just(""),  # empty string
    st.none(),  # missing
    st.integers(),
    st.lists(st.text(), max_size=3),
)
_ARTIFACT_URI = st.one_of(_VALID_ARTIFACT_URI, _INVALID_ARTIFACT_URI)


# **Property 27: 등록 메타데이터 검증**
# **Validates: Requirements 4.7**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    model_name=_MODEL_NAME,
    feature_group=_FEATURE_GROUP,
    metrics=_METRICS,
    artifact_uri=_ARTIFACT_URI,
)
def test_invalid_registration_metadata_is_rejected_with_violation_list(
    model_name: Any,
    feature_group: Any,
    metrics: Any,
    artifact_uri: Any,
) -> None:
    expected = _expected_violations(model_name, feature_group, metrics, artifact_uri)
    # Only exercise genuinely invalid combinations here; an all-valid draw would also
    # need to satisfy the (out-of-scope for this property) artifact-availability check.
    assume(expected)

    with TemporaryDirectory() as raw_root:
        registry, collection = _make_registry(Path(raw_root))
        _seed_unrelated_model(collection)
        before_snapshot = deepcopy(collection.documents)

        with pytest.raises(ModelRegistryError) as error:
            registry.register(model_name, feature_group, metrics, artifact_uri)

        assert error.value.error_code == "invalid_model_metadata"
        assert error.value.details == {"violations": expected}
        # No new version number is assigned and the registry is left byte-for-byte
        # unchanged (no parent upsert, no version document insert).
        assert collection.documents == before_snapshot


# **Property 27: 등록 메타데이터 검증**
# **Validates: Requirements 4.7**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(model_name=st.one_of(st.just(""), st.text(min_size=129, max_size=160)))
def test_model_name_length_boundary_violation_is_reported(model_name: str) -> None:
    """Boundary-focused check: length < 1 or > 128 alone must trigger a model_name
    violation and only that field, when every other field is otherwise valid."""
    feature_group = "default"
    metrics = {"delay_p95_ms": 1.0}
    artifact_uri = "models/some-model/1/model.pt"

    with TemporaryDirectory() as raw_root:
        registry, collection = _make_registry(Path(raw_root))
        before_snapshot = deepcopy(collection.documents)

        with pytest.raises(ModelRegistryError) as error:
            registry.register(model_name, feature_group, metrics, artifact_uri)

        assert error.value.error_code == "invalid_model_metadata"
        assert error.value.details == {"violations": ["model_name"]}
        assert collection.documents == before_snapshot
