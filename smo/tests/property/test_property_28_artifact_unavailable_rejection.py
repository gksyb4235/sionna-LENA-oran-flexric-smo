"""Property 28 test module (own module to avoid collisions with parallel Property tasks).

Task 3.17: 등록 요청의 산출물 URI 가 Model_Storage 에서 조회되지 않는 경우, 모든
다른 메타데이터(모델 이름, Feature_Group, 지표)가 유효하더라도 Model_Registry 가
등록을 거부하고 부분 버전 레코드를 남기지 않는지를 검증한다.

Targets ``smo.aimlfw.model_registry.registry.ModelRegistry.register`` (Task 3.2),
specifically the ``artifact_unavailable`` check that calls
``model_storage.artifact_exists(artifact_uri)`` before allocating a new version, as
specified in requirements.md Requirement 4.8:

    IF 등록 요청의 산출물 URI 가 Model_Storage 에서 조회되지 않으면, THEN THE
    Model_Registry SHALL 등록을 거부하고 `artifact_unavailable` 오류 코드와 해당
    URI 를 포함한 오류 응답 본문을 반환하며 부분 레코드를 남기지 않는다.

design.md Property 28: 산출물 미조회 시 등록 거부
    *For any* 유효한 모델 이름/Feature_Group/지표 조합과, Model_Storage 에 실제로
    존재하지 않는 산출물 URI 에 대해, Model_Registry.register 는 `artifact_unavailable`
    오류를 발생시키고 어떤 버전 레코드도 생성하지 않는다.
    **Validates: Requirements 4.8**
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.common.constants import TARGET_KPIS
from smo.aimlfw.model_registry import ModelRegistry, ModelRegistryError
from smo.aimlfw.model_storage import ModelStorage
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

_NAME_ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. "
_MODEL_NAME = st.text(alphabet=_NAME_ALPHABET, min_size=1, max_size=128).filter(str.strip)
_FEATURE_GROUP = st.text(alphabet=_NAME_ALPHABET, min_size=1, max_size=128).filter(str.strip)
_METRICS = st.dictionaries(
    keys=st.sampled_from(TARGET_KPIS),
    values=st.floats(min_value=0.0, max_value=1_000_000.0, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=len(TARGET_KPIS),
)
# Arbitrary non-empty URIs: since the storage root is always a freshly created,
# empty temporary directory in these tests, none of these can ever resolve to an
# artifact that was actually saved, regardless of their shape.
_MISSING_ARTIFACT_URI = st.text(min_size=1, max_size=200).filter(
    lambda value: value.strip() and "\x00" not in value
)


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

    def find(self, query: dict[str, Any]):
        raise NotImplementedError("register() never calls find() before the artifact check")

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


# **Property 28: 산출물 미조회 시 등록 거부**
# **Validates: Requirements 4.8**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    model_name=_MODEL_NAME,
    feature_group=_FEATURE_GROUP,
    metrics=_METRICS,
    artifact_uri=_MISSING_ARTIFACT_URI,
)
def test_registration_rejected_when_artifact_missing_from_storage(
    model_name: str,
    feature_group: str,
    metrics: dict[str, float],
    artifact_uri: str,
) -> None:
    """Requirement 4.8: even with otherwise-valid metadata, a registration whose
    artifact_uri does not resolve to a real artifact in Model_Storage must be
    rejected with `artifact_unavailable`, and must leave no partial record behind."""
    with TemporaryDirectory() as raw_root:
        registry, collection = _make_registry(Path(raw_root))

        # Sanity check the oracle: a brand-new, empty storage root can never
        # contain this artifact_uri, so the check under test is guaranteed to
        # actually observe "missing" rather than incidentally passing.
        assert not registry.model_storage.artifact_exists(artifact_uri)

        before_snapshot = deepcopy(collection.documents)

        with pytest.raises(ModelRegistryError) as error:
            registry.register(model_name, feature_group, metrics, artifact_uri)

        assert error.value.error_code == "artifact_unavailable"
        assert error.value.details == {"artifact_uri": artifact_uri}

        # No partial model (parent) document and no version record were created.
        assert collection.documents == before_snapshot
        with pytest.raises(ModelRegistryError):
            registry.get(model_name, 1)


# **Property 28: 산출물 미조회 시 등록 거부**
# **Validates: Requirements 4.8**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(model_name=_MODEL_NAME, feature_group=_FEATURE_GROUP, metrics=_METRICS)
def test_registration_succeeds_once_artifact_is_actually_saved(
    model_name: str,
    feature_group: str,
    metrics: dict[str, float],
) -> None:
    """Contrast check: the same otherwise-valid metadata must be accepted once the
    artifact_uri does resolve to a real, saved artifact -- proving the rejection
    above is caused specifically by artifact absence, not by some other metadata
    quirk of the generated values."""
    with TemporaryDirectory() as raw_root:
        registry, collection = _make_registry(Path(raw_root))
        artifact_uri = registry.model_storage.save_artifact(model_name, 1, b"weights")

        record = registry.register(model_name, feature_group, metrics, artifact_uri)

        assert record.version == 1
        assert any(
            document.get("document_type") == "version" and document.get("version") == 1
            for document in collection.documents
        )
