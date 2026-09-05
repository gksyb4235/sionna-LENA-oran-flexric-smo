"""Property 24 test module (own module to avoid collisions with parallel Property tasks).

Task 3.13: 임의의 모델 이름에 대해 등록된 버전 집합에서, 최신 버전 조회 결과의 버전
번호가 해당 집합에서 가장 큰 값과 같은지를 검증한다.

Targets ``smo.aimlfw.model_registry.registry.ModelRegistry.latest`` (Task 3.2), as
specified in requirements.md Requirement 4.3:

    WHEN 모델 이름만으로 최신 버전 조회 요청이 도착하면, THE Model_Registry SHALL 해당
    이름의 버전 중 버전 번호가 가장 큰 단일 항목을 반환하며, 그 항목에 버전 번호, 생성
    시각, Feature_Group 이름, Target_KPI 별 평가 지표, 산출물 URI 를 포함한다.

design.md Property 24: 최신 버전은 최대 버전
    *For any* 모델 이름에 대해 등록된 버전 집합, 최신 버전 조회 결과의 버전 번호는
    해당 집합에서 가장 큰 값과 같다.
    **Validates: Requirements 4.3**
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.model_registry import ModelRegistry
from smo.aimlfw.model_storage import ModelStorage
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

# Model names limited to characters accepted by ``ModelStorage._validate_identity``
# (no path separators or NUL bytes) so every drawn name is a legal storage key.
_MODEL_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-",
    min_size=1,
    max_size=32,
).filter(lambda name: name.strip() and name not in {".", ".."})

# Vary the number of sequential registrations performed against the same model name.
_REGISTRATION_COUNT = st.integers(min_value=1, max_value=25)


class _FakeCursor:
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
    """Minimal in-memory stand-in for the MongoDB collection used by ModelRegistry."""

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

    def update_one(self, query: dict[str, Any], update: dict[str, Any], **kwargs: Any):
        if not any(self._matches(item, query) for item in self.documents):
            document = deepcopy(query)
            document.update(deepcopy(update.get("$setOnInsert", {})))
            self.documents.append(document)

    def insert_one(self, document: dict[str, Any]):
        self.documents.append(deepcopy(document))


# **Property 24: 최신 버전은 최대 버전**
# **Validates: Requirements 4.3**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(model_name=_MODEL_NAME, registration_count=_REGISTRATION_COUNT)
def test_latest_version_equals_max_registered_version(
    tmp_path_factory: pytest.TempPathFactory, model_name: str, registration_count: int
) -> None:
    tmp_path = tmp_path_factory.mktemp("model-registry-latest")
    storage = ModelStorage(tmp_path / "models")
    registry = ModelRegistry(_FakeCollection(), storage)

    registered_versions: list[int] = []
    for index in range(registration_count):
        artifact_uri = storage.save_artifact(model_name, index + 1, f"artifact-{index}".encode())
        record = registry.register(
            model_name,
            "default",
            {"delay_p95_ms": float(index)},
            artifact_uri,
        )
        registered_versions.append(record.version)

    latest = registry.latest(model_name)

    assert latest.version == max(registered_versions)
