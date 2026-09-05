from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from threading import Lock
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pymongo.errors import DuplicateKeyError

from smo.aimlfw.model_registry import ModelRegistry, ModelRegistryError, create_app
from smo.aimlfw.model_storage import ModelStorage


class FakeCursor:
    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents

    def sort(self, key: str, direction: int | None = None) -> FakeCursor:
        self.documents.sort(key=lambda item: item[key], reverse=direction == -1)
        return self

    def limit(self, count: int) -> FakeCursor:
        self.documents = self.documents[:count]
        return self

    def __iter__(self):
        return iter(deepcopy(self.documents))


class FakeCollection:
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

    def find(self, query: dict[str, Any]) -> FakeCursor:
        return FakeCursor(
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


def registry_fixture(tmp_path: Path) -> tuple[ModelRegistry, ModelStorage, FakeCollection]:
    storage = ModelStorage(tmp_path / "models")
    collection = FakeCollection()
    return ModelRegistry(collection, storage), storage, collection


def save(storage: ModelStorage, model_name: str, version: int) -> str:
    return storage.save_artifact(model_name, version, f"artifact-{version}".encode())


def test_register_assigns_versions_and_all_read_apis_are_complete(tmp_path: Path) -> None:
    registry, storage, _ = registry_fixture(tmp_path)
    first = registry.register("ran-gnn", "default", {"delay_p95_ms": 4.25}, save(storage, "ran-gnn", 1))
    second = registry.register("ran-gnn", "default", {"delay_p95_ms": 3.75}, save(storage, "ran-gnn", 2))

    assert (first.version, second.version) == (1, 2)
    assert first.created_at.utcoffset().total_seconds() == 0
    assert [item.version for item in registry.list_versions("ran-gnn")] == [1, 2]
    assert registry.latest("ran-gnn") == second
    assert registry.get("ran-gnn", 1) == first


def test_invalid_metadata_and_unavailable_artifact_leave_no_version(tmp_path: Path) -> None:
    registry, storage, collection = registry_fixture(tmp_path)

    with pytest.raises(ModelRegistryError) as invalid:
        registry.register("", None, None, None)
    assert invalid.value.error_code == "invalid_model_metadata"
    assert invalid.value.details["violations"] == [
        "model_name", "feature_group", "metrics", "artifact_uri"
    ]

    with pytest.raises(ModelRegistryError) as unavailable:
        registry.register("ran-gnn", "default", {"delay_p95_ms": 1.0}, "missing/model.pt")
    assert unavailable.value.error_code == "artifact_unavailable"
    assert not any(item.get("document_type") == "version" for item in collection.documents)


def test_existing_model_with_no_versions_returns_empty_list(tmp_path: Path) -> None:
    registry, _, collection = registry_fixture(tmp_path)
    collection.update_one(
        {"_id": "model:ran-gnn"},
        {"$setOnInsert": {"document_type": "model", "model_name": "ran-gnn"}},
        upsert=True,
    )

    assert registry.list_versions("ran-gnn") == []


def test_missing_model_version_and_orphan_never_return_partial_metadata(tmp_path: Path) -> None:
    registry, storage, collection = registry_fixture(tmp_path)
    with pytest.raises(ModelRegistryError) as missing:
        registry.get("missing", 7)
    assert missing.value.details == {"model_name": "missing", "version": 7}

    record = registry.register("ran-gnn", "default", {"delay_p95_ms": 1.0}, save(storage, "ran-gnn", 1))
    collection.documents = [item for item in collection.documents if item.get("document_type") != "model"]
    with pytest.raises(ModelRegistryError) as orphan:
        registry.get("ran-gnn", record.version)
    assert orphan.value.error_code == "model_not_found"
    assert "feature_group" not in orphan.value.details


def test_list_is_sorted_and_limited_to_500_records(tmp_path: Path) -> None:
    registry, _, collection = registry_fixture(tmp_path)
    collection.update_one(
        {"_id": "model:ran-gnn"},
        {"$setOnInsert": {"document_type": "model", "model_name": "ran-gnn"}},
        upsert=True,
    )
    for version in range(501, 0, -1):
        collection.insert_one(
            {
                "_id": f"version:ran-gnn:{version}",
                "document_type": "version",
                "model_id": "model:ran-gnn",
                "model_name": "ran-gnn",
                "version": version,
                "created_at": datetime.now(timezone.utc),
                "feature_group": "default",
                "metrics": {"delay_p95_ms": 1.0},
                "artifact_uri": f"models/ran-gnn/{version}/model.pt",
            }
        )

    versions = registry.list_versions("ran-gnn")
    assert len(versions) == 500
    assert [item.version for item in versions] == list(range(1, 501))


def test_model_registry_api_contract(tmp_path: Path) -> None:
    registry, storage, _ = registry_fixture(tmp_path)
    client = TestClient(create_app(registry))
    uri = save(storage, "ran-gnn", 1)

    created = client.post(
        "/models/ran-gnn/versions",
        json={
            "feature_group": "default",
            "metrics": {"delay_p95_ms": 2.5},
            "artifact_uri": uri,
        },
    )
    assert created.status_code == 200
    assert created.json() == {"model_name": "ran-gnn", "version": 1}

    listed = client.get("/models/ran-gnn/versions")
    assert listed.status_code == 200
    assert listed.json()["count"] == 1
    assert listed.json()["versions"][0]["version"] == 1
    assert client.get("/models/ran-gnn/latest").json()["version"] == 1
    assert client.get("/models/ran-gnn/versions/1").json()["artifact_uri"] == uri

    missing = client.get("/models/unknown/versions/9")
    assert missing.status_code == 404
    assert missing.json()["error_code"] == "model_not_found"
    assert "feature_group" not in missing.json()


def test_api_maps_missing_and_malformed_registration_to_metadata_error(tmp_path: Path) -> None:
    registry, _, _ = registry_fixture(tmp_path)
    client = TestClient(create_app(registry))

    missing = client.post("/models/ran-gnn/versions", json={})
    assert missing.status_code == 422
    assert missing.json()["error_code"] == "invalid_model_metadata"
    assert missing.json()["details"]["violations"] == [
        "feature_group", "metrics", "artifact_uri"
    ]

    malformed = client.post(
        "/models/ran-gnn/versions",
        json={"feature_group": "default", "metrics": "bad", "artifact_uri": "model.pt"},
    )
    assert malformed.status_code == 422
    assert malformed.json()["error_code"] == "invalid_model_metadata"
    assert "metrics" in malformed.json()["details"]["violations"]


def test_concurrent_registration_allocates_each_version_once(tmp_path: Path) -> None:
    registry, storage, _ = registry_fixture(tmp_path)
    artifact_uris = [save(storage, f"artifact-source-{index}", 1) for index in range(12)]

    def register(uri: str) -> int:
        return registry.register(
            "ran-gnn",
            "default",
            {"delay_p95_ms": 1.0},
            uri,
        ).version

    with ThreadPoolExecutor(max_workers=6) as executor:
        versions = list(executor.map(register, artifact_uris))

    assert sorted(versions) == list(range(1, 13))
    assert [item.version for item in registry.list_versions("ran-gnn")] == list(range(1, 13))
