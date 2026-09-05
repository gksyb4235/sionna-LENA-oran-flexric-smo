"""Property 23 test module (own module to avoid collisions with parallel Property tasks).

Task 3.12: 임의의 모델 이름에 대해 0개 이상 등록된 버전 집합에 대해 버전 목록 조회가
버전 번호 오름차순으로 정렬되고 응답 항목 수가 500개를 초과하지 않으며(0개인 경우
빈 목록을 오류 없이 반환), 각 항목이 버전 번호/생성 시각/Feature_Group 이름/평가
지표/산출물 URI를 포함하는지를 검증한다.

Targets ``smo.aimlfw.model_registry.registry.ModelRegistry.list_versions`` (Task 3.2),
as specified in requirements.md:

Requirement 4.2:
    WHEN 모델 이름으로 버전 목록 조회 요청이 도착하면, THE Model_Registry SHALL 해당
    이름의 모든 버전을 버전 번호 오름차순으로 정렬하여 한 응답에 최대 500 개까지
    반환하며, 각 항목에 버전 번호, 생성 시각, Feature_Group 이름, Target_KPI 별
    평가 지표, 산출물 URI 를 포함하고, 요청 수신 시점부터 2 초 이내에 응답한다.

Requirement 4.9:
    IF 모델 이름은 Model_Registry 에 존재하지만 저장된 버전이 0 개이면, THEN THE
    Model_Registry SHALL 항목 수가 0 인 빈 버전 목록을 반환하고 오류 코드를
    반환하지 않는다.

design.md Property 23: 버전 목록 정렬과 개수 제한
    *For any* 모델 이름에 대해 0개 이상 등록된 버전 집합, 버전 목록 조회 결과는
    버전 번호 오름차순으로 정렬되어 있고 응답 항목 수는 500개를 초과하지 않으며
    (0개인 경우 빈 목록, 오류 없음), 각 항목은 버전 번호/생성 시각/Feature_Group
    이름/평가 지표/산출물 URI를 포함한다.
    **Validates: Requirements 4.2, 4.9**
"""

from __future__ import annotations

import random
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.model_registry.registry import MAX_LIST_VERSIONS, ModelRegistry
from smo.aimlfw.model_storage import ModelStorage
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS


class _FakeCursor:
    """Minimal Mongo-cursor stand-in supporting only ``sort`` + ``limit``."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents

    def sort(self, key: str, direction: int | None = None) -> "_FakeCursor":
        self.documents = sorted(
            self.documents, key=lambda item: item[key], reverse=direction == -1
        )
        return self

    def limit(self, count: int) -> "_FakeCursor":
        self.documents = self.documents[:count]
        return self

    def __iter__(self):
        return iter(deepcopy(self.documents))


class _FakeCollection:
    """Minimal in-memory stand-in exercising only what ModelRegistry needs."""

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
        return _FakeCursor(
            [deepcopy(item) for item in self.documents if self._matches(item, query)]
        )

    def update_one(self, query: dict[str, Any], update: dict[str, Any], **kwargs: Any) -> None:
        if not any(self._matches(item, query) for item in self.documents):
            document = deepcopy(query)
            document.update(deepcopy(update.get("$setOnInsert", {})))
            self.documents.append(document)

    def insert_one(self, document: dict[str, Any]) -> None:
        self.documents.append(deepcopy(document))


# Model names limited to characters accepted by ``ModelStorage._validate_identity``
# and ``ModelRegistry`` length bounds (1..128, no path separators).
_MODEL_NAME = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-",
    min_size=1,
    max_size=32,
).filter(lambda name: name.strip() and name not in {".", ".."})

# Draw registration counts that exercise 0, small sets, and both sides of the
# 500-entry cap boundary without paying for a full 100-example sweep at large N.
_VERSION_COUNT = st.one_of(
    st.integers(min_value=0, max_value=20),
    st.sampled_from((499, 500, 501, 502, 650)),
)


def _model_parent_document(model_name: str) -> dict[str, Any]:
    return {
        "_id": f"model:{model_name}",
        "document_type": "model",
        "model_name": model_name,
        "created_at": datetime.now(timezone.utc),
    }


def _version_document(model_name: str, version: int) -> dict[str, Any]:
    return {
        "_id": f"version:{model_name}:{version}",
        "document_type": "version",
        "model_id": f"model:{model_name}",
        "model_name": model_name,
        "version": version,
        "created_at": datetime.now(timezone.utc),
        "feature_group": f"fg-{version}",
        "metrics": {"delay_p95_ms": float(version)},
        "artifact_uri": f"models/{model_name}/{version}/model.pt",
    }


# **Property 23: 버전 목록 정렬과 개수 제한**
# **Validates: Requirements 4.2, 4.9**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    model_name=_MODEL_NAME,
    version_count=_VERSION_COUNT,
    shuffle_seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_list_versions_is_sorted_ascending_and_capped_at_500(
    model_name: str, version_count: int, shuffle_seed: int
) -> None:
    with tempfile.TemporaryDirectory() as raw_root:
        storage = ModelStorage(Path(raw_root) / "models")
        collection = _FakeCollection()
        registry = ModelRegistry(collection, storage)

        # The model name exists (parent document present) even when it has zero
        # registered versions, matching Requirement 4.9's precondition.
        collection.documents.append(_model_parent_document(model_name))

        version_numbers = list(range(1, version_count + 1))
        insertion_order = list(version_numbers)
        random.Random(shuffle_seed).shuffle(insertion_order)
        for version in insertion_order:
            collection.insert_one(_version_document(model_name, version))

        result = registry.list_versions(model_name)

        # Response is capped at MAX_LIST_VERSIONS (500) regardless of how many
        # versions exist, and is empty (without error) when zero are registered.
        assert len(result) == min(version_count, MAX_LIST_VERSIONS)
        assert len(result) <= MAX_LIST_VERSIONS

        # Sorted strictly ascending by version number, independent of insertion order.
        returned_versions = [item.version for item in result]
        assert returned_versions == sorted(returned_versions)
        assert returned_versions == version_numbers[: MAX_LIST_VERSIONS]

        # Every item carries the full, correctly matched metadata contract.
        for item in result:
            assert item.feature_group == f"fg-{item.version}"
            assert item.metrics == {"delay_p95_ms": float(item.version)}
            assert item.artifact_uri == f"models/{model_name}/{item.version}/model.pt"
            assert item.created_at.tzinfo is not None
            assert item.created_at.utcoffset().total_seconds() == 0
