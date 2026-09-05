"""Property 22 test module (own module to avoid collisions with parallel Property tasks).

Task 3.11: 버전 자동 증가.

Targets the Model_Registry's version allocation, as specified in requirements.md 4.1
and implemented by ``smo.aimlfw.model_registry.registry.ModelRegistry.register`` (Task 3.2):

    WHEN 모델 이름, Feature_Group 이름, Target_KPI 별 평가 지표, Model_Storage 내 산출물
    URI 를 담은 등록 요청이 도착하면, THE Model_Registry SHALL 해당 모델 이름의 첫 등록에는
    버전 번호 1 을, 이후 등록에는 해당 이름의 기존 최대 버전 번호에 1 을 더한 값을 부여하고,
    부여된 버전 번호와 UTC 기준 생성 시각을 포함한 레코드를 저장한 뒤 저장된 모델 이름과
    버전 번호를 호출자에게 반환한다.

Design tag (design.md, "Model_Registry / Model_Storage (Requirement 4)"):

    #### Property 22: 버전 자동 증가
    *For any* 모델 이름에 대한 연속된 등록 요청 순서, 첫 등록의 버전 번호는 1이고 이후
    각 등록의 버전 번호는 해당 이름의 기존 최대 버전 번호에 1을 더한 값이다.
    **Validates: Requirements 4.1**

``ModelRegistry.register`` never accepts a caller-supplied version number, so this test
also exercises the "no manual version assignment" half of the contract simply by using
the real method signature. A minimal in-memory stand-in for the MongoDB collection is
defined locally (mirroring the fixture already used by
``smo/tests/unit/test_model_registry.py``) so this module stays independent of other
Property test modules and of task 1.2's shared ``strategies.py`` beyond the common
``PROPERTY_TEST_SETTINGS`` tuning.
"""

from __future__ import annotations

import tempfile
from copy import deepcopy
from pathlib import Path
from threading import Lock
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pymongo.errors import DuplicateKeyError

from smo.aimlfw.model_registry import ModelRegistry
from smo.aimlfw.model_storage import ModelStorage
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS

_MAX_REGISTRATIONS = 40


class _FakeCursor:
    """Minimal ``pymongo`` cursor stand-in supporting ``sort``/``limit``/iteration."""

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
    """Minimal in-memory stand-in for the MongoDB collection ``ModelRegistry`` expects."""

    def __init__(self) -> None:
        self.documents: list[dict[str, Any]] = []
        self.lock = Lock()

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

    def update_one(self, query: dict[str, Any], update: dict[str, Any], **kwargs: Any):
        with self.lock:
            if not any(self._matches(item, query) for item in self.documents):
                document = deepcopy(query)
                document.update(deepcopy(update.get("$setOnInsert", {})))
                self.documents.append(document)

    def insert_one(self, document: dict[str, Any]):
        with self.lock:
            duplicate = any(
                item.get("document_type") == "version"
                and item.get("model_name") == document.get("model_name")
                and item.get("version") == document.get("version")
                for item in self.documents
            )
            if duplicate:
                raise DuplicateKeyError("duplicate model version")
            self.documents.append(deepcopy(document))


# **Property 22: 버전 자동 증가**
# **Validates: Requirements 4.1**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(registration_count=st.integers(min_value=1, max_value=_MAX_REGISTRATIONS))
def test_sequential_registrations_auto_increment_version_starting_at_one(
    registration_count: int,
) -> None:
    model_name = "ran-gnn"

    with tempfile.TemporaryDirectory() as raw_root:
        storage = ModelStorage(Path(raw_root) / "models")
        registry = ModelRegistry(_FakeCollection(), storage)

        assigned_versions: list[int] = []
        for sequence_number in range(1, registration_count + 1):
            # The artifact version label only needs to be unique per call; it carries
            # no relationship to (and must not influence) the version Model_Registry
            # assigns, since ``register`` never accepts a caller-supplied version.
            artifact_uri = storage.save_artifact(
                model_name, sequence_number, f"artifact-{sequence_number}".encode()
            )
            record = registry.register(
                model_name,
                "default",
                {"delay_p95_ms": float(sequence_number)},
                artifact_uri,
            )
            assigned_versions.append(record.version)

        # First registration is version 1; each subsequent registration is the prior
        # maximum plus 1 -- i.e. the assigned sequence is exactly 1, 2, 3, ...
        assert assigned_versions == list(range(1, registration_count + 1))

        # The registry's own read path agrees with what was assigned during writes.
        stored_versions = [item.version for item in registry.list_versions(model_name)]
        assert stored_versions == assigned_versions
