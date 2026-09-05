"""Property 25 test module (independent, per tasks.md parallel-collision notes).

Task 3.14: Model_Registry 조회 시 모델 이름 또는 버전이 존재하지 않는 경우
`model_not_found` 오류와 요청된 이름/버전을 반환하고, 저장된 레코드를 변경하지
않으며 부분 메타데이터를 응답에 포함하지 않는지 검증한다.

Targets ``smo.aimlfw.model_registry.registry.ModelRegistry``'s ``list_versions``,
``latest``, and ``get``:
- Requirement 4.4: 요청된 모델 이름이 없거나 요청된 버전 번호가 해당 이름의
  저장된 버전 집합에 속하지 않으면 ``model_not_found`` 오류와 요청된 이름/버전을
  반환하고 저장된 레코드를 변경하지 않는다.
- Requirement 4.5: 버전 레코드는 존재하지만 연결된 모델 이름(parent) 레코드가
  없으면 ``model_not_found`` 오류를 반환하고 부분 메타데이터를 응답에 포함하지
  않는다.
"""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.model_registry import ModelRegistry, ModelRegistryError
from smo.aimlfw.model_storage import ModelStorage
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

# Fields that only ever appear on a fully-resolved ``ModelVersionRecord``. Property 25
# (Requirement 4.5) requires that a "not found" error never leaks any of these as
# partial metadata, even when the underlying version document happens to exist.
_VERSION_METADATA_FIELDS = ("feature_group", "metrics", "artifact_uri", "created_at")

_MODEL_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.",
    min_size=1,
    max_size=64,
).filter(str.strip)
_VERSION = st.integers(min_value=1, max_value=10_000)


class _FakeCursor:
    """Minimal in-memory stand-in for a PyMongo cursor (sort + limit only)."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents

    def sort(self, key: str, direction: int | None = None) -> _FakeCursor:
        self.documents.sort(key=lambda item: item[key], reverse=direction == -1)
        return self

    def limit(self, count: int) -> _FakeCursor:
        self.documents = self.documents[:count]
        return self

    def __iter__(self):
        return iter(deepcopy(self.documents))


class _FakeCollection:
    """Minimal in-memory stand-in for the PyMongo collection ``ModelRegistry`` uses.

    Read-only helper for this property: registration is never exercised here, so
    fixtures are seeded directly as documents rather than via ``ModelRegistry.register``.
    """

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


def _seed_registered_model(collection: _FakeCollection, model_name: str, versions: set[int]) -> None:
    """Seed a parent model document plus one version document per ``versions``."""
    now = datetime.now(timezone.utc)
    collection.documents.append(
        {"_id": f"model:{model_name}", "document_type": "model", "model_name": model_name, "created_at": now}
    )
    for version in versions:
        collection.documents.append(
            {
                "_id": f"version:{model_name}:{version}",
                "document_type": "version",
                "model_id": f"model:{model_name}",
                "model_name": model_name,
                "version": version,
                "created_at": now,
                "feature_group": "default",
                "metrics": {"delay_p95_ms": 1.0},
                "artifact_uri": f"models/{model_name}/{version}/model.pt",
            }
        )


def _seed_orphan_version(collection: _FakeCollection, model_name: str, version: int) -> None:
    """Seed a version document with no matching parent model document (Requirement 4.5)."""
    collection.documents.append(
        {
            "_id": f"version:{model_name}:{version}",
            "document_type": "version",
            "model_id": f"model:{model_name}",
            "model_name": model_name,
            "version": version,
            "created_at": datetime.now(timezone.utc),
            "feature_group": "default",
            "metrics": {"delay_p95_ms": 1.0},
            "artifact_uri": f"models/{model_name}/{version}/model.pt",
        }
    )


def _assert_not_found_without_partial_metadata(
    error: ModelRegistryError, model_name: str, version: int | None
) -> None:
    assert error.error_code == "model_not_found"
    expected_details = {"model_name": model_name}
    if version is not None:
        expected_details["version"] = version
    assert error.details == expected_details
    assert not any(field in error.details for field in _VERSION_METADATA_FIELDS)


# **Property 25: 모델/버전 not found**
# **Validates: Requirements 4.4, 4.5**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    absent_model_name=_MODEL_NAME,
    other_model_name=_MODEL_NAME,
    other_versions=st.sets(_VERSION, min_size=1, max_size=5),
    requested_version=_VERSION,
)
def test_absent_model_name_returns_not_found_and_leaves_registry_unchanged(
    absent_model_name: str,
    other_model_name: str,
    other_versions: set[int],
    requested_version: int,
) -> None:
    """Requirement 4.4: a model name the registry has never seen must not_found on
    every read API, quoting the requested name back, and must never mutate state."""
    if absent_model_name == other_model_name:
        return  # keep the "seeded, unrelated" model distinct from the absent one.

    with TemporaryDirectory() as raw_root:
        registry, collection = _make_registry(Path(raw_root))
        # Seed an unrelated, fully valid model so "unchanged" is a meaningful
        # assertion rather than a vacuous "still empty" check.
        _seed_registered_model(collection, other_model_name, other_versions)
        before_snapshot = deepcopy(collection.documents)

        with pytest.raises(ModelRegistryError) as list_error:
            registry.list_versions(absent_model_name)
        _assert_not_found_without_partial_metadata(list_error.value, absent_model_name, None)
        assert collection.documents == before_snapshot

        with pytest.raises(ModelRegistryError) as latest_error:
            registry.latest(absent_model_name)
        _assert_not_found_without_partial_metadata(latest_error.value, absent_model_name, None)
        assert collection.documents == before_snapshot

        with pytest.raises(ModelRegistryError) as get_error:
            registry.get(absent_model_name, requested_version)
        _assert_not_found_without_partial_metadata(get_error.value, absent_model_name, requested_version)
        assert collection.documents == before_snapshot


# **Property 25: 모델/버전 not found**
# **Validates: Requirements 4.4**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    model_name=_MODEL_NAME,
    existing_versions=st.sets(_VERSION, min_size=0, max_size=5),
    missing_version=_VERSION,
)
def test_existing_model_missing_version_returns_not_found_and_leaves_registry_unchanged(
    model_name: str,
    existing_versions: set[int],
    missing_version: int,
) -> None:
    """Requirement 4.4: a registered model name with a version number outside its
    stored version set must not_found with the requested name and version."""
    if missing_version in existing_versions:
        return  # the generated "missing" version must genuinely be absent.

    with TemporaryDirectory() as raw_root:
        registry, collection = _make_registry(Path(raw_root))
        _seed_registered_model(collection, model_name, existing_versions)
        before_snapshot = deepcopy(collection.documents)

        with pytest.raises(ModelRegistryError) as error:
            registry.get(model_name, missing_version)
        _assert_not_found_without_partial_metadata(error.value, model_name, missing_version)
        assert collection.documents == before_snapshot

        # Sanity check that the property is exercising a real "some versions
        # exist, this one doesn't" scenario rather than always hitting the
        # zero-versions case handled by the other tests in this module.
        if existing_versions:
            assert [record.version for record in registry.list_versions(model_name)] == sorted(
                existing_versions
            )


# **Property 25: 모델/버전 not found**
# **Validates: Requirements 4.5**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(model_name=_MODEL_NAME, version=_VERSION)
def test_orphan_version_without_parent_returns_not_found_without_partial_metadata(
    model_name: str, version: int
) -> None:
    """Requirement 4.5: a version document that exists without a linked model
    (parent) document must still not_found, never leaking the version's own
    feature_group/metrics/artifact_uri as partial metadata."""
    with TemporaryDirectory() as raw_root:
        registry, collection = _make_registry(Path(raw_root))
        _seed_orphan_version(collection, model_name, version)
        before_snapshot = deepcopy(collection.documents)
        assert before_snapshot  # the orphan version document must actually be present.

        with pytest.raises(ModelRegistryError) as error:
            registry.get(model_name, version)
        _assert_not_found_without_partial_metadata(error.value, model_name, version)
        assert collection.documents == before_snapshot
