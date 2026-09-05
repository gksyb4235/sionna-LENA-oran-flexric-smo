"""Task 12.7: complete Evidence Record round-trip in a real MongoDB collection."""

from __future__ import annotations

import os
import uuid
from typing import Any

import pytest
from _evidence_test_support import record
from pymongo import MongoClient

from evidence_store import EvidenceRepository, EvidenceStoreError


class _AcknowledgementFailureCollection:
    """Insert into real MongoDB, then inject failure so repository cleanup is exercised."""

    def __init__(self, collection: Any) -> None:
        self.collection = collection

    def create_index(self, *args: Any, **kwargs: Any) -> Any:
        return self.collection.create_index(*args, **kwargs)

    def insert_one(self, document: dict[str, Any]) -> Any:
        self.collection.insert_one(document)
        raise RuntimeError("injected acknowledgement failure")

    def delete_one(self, query: dict[str, Any]) -> Any:
        return self.collection.delete_one(query)

    def find_one(self, query: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
        return self.collection.find_one(query, *args, **kwargs)


@pytest.mark.skipif(not os.getenv("TEST_MONGODB_URI"), reason="TEST_MONGODB_URI is required")
def test_real_mongodb_complete_roundtrip_and_atomic_cleanup() -> None:
    client = MongoClient(os.environ["TEST_MONGODB_URI"], serverSelectionTimeoutMS=2000)
    database_name = f"evidence_contract_{uuid.uuid4().hex}"
    try:
        collection = client[database_name]["evidence_records"]
        repository = EvidenceRepository(collection)
        evidence = record(with_temporal=True)
        repository.save(evidence)
        loaded = repository.get(evidence.evidence_id)
        assert loaded == evidence
        assert loaded.temporal_plan is not None
        assert loaded.indirect_conflicts

        failing_repository = EvidenceRepository(_AcknowledgementFailureCollection(collection))
        failed = record(with_temporal=False)
        with pytest.raises(EvidenceStoreError) as excinfo:
            failing_repository.save(failed)
        assert excinfo.value.error_code == "evidence_storage_failed"
        assert collection.find_one({"_id": failed.evidence_id}) is None
    finally:
        client.drop_database(database_name)
        client.close()
