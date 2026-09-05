"""MongoDB persistence and validation for model version metadata."""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import Any, Protocol

from pymongo import ASCENDING, DESCENDING
from pymongo.errors import DuplicateKeyError

from smo.aimlfw.common.constants import MAX_SEED, TARGET_KPIS
from smo.aimlfw.common.models import ModelVersionRecord
from smo.aimlfw.model_storage import ModelStorage

from .errors import ModelRegistryError

MAX_LIST_VERSIONS = 500
MODEL_DOCUMENT = "model"
VERSION_DOCUMENT = "version"


class CursorLike(Protocol):
    def sort(self, key_or_list: Any, direction: int | None = None) -> CursorLike: ...
    def limit(self, count: int) -> CursorLike: ...
    def __iter__(self) -> Any: ...


class CollectionLike(Protocol):
    def create_index(self, keys: Any, **kwargs: Any) -> Any: ...
    def find_one(self, filter: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any] | None: ...
    def find(self, filter: dict[str, Any]) -> CursorLike: ...
    def update_one(self, filter: dict[str, Any], update: dict[str, Any], **kwargs: Any) -> Any: ...
    def insert_one(self, document: dict[str, Any]) -> Any: ...


class ModelRegistry:
    """Store model parents and immutable version records in one collection."""

    def __init__(self, collection: CollectionLike, model_storage: ModelStorage) -> None:
        self.collection = collection
        self.model_storage = model_storage
        self._indexes_ready = False

    def _ensure_indexes(self) -> None:
        if self._indexes_ready:
            return
        self.collection.create_index(
            [("model_name", ASCENDING), ("version", ASCENDING)],
            unique=True,
            name="model_version_unique",
            partialFilterExpression={"document_type": VERSION_DOCUMENT},
        )
        self.collection.create_index(
            [("model_name", ASCENDING), ("version", DESCENDING)],
            name="model_version_latest",
            partialFilterExpression={"document_type": VERSION_DOCUMENT},
        )
        self._indexes_ready = True

    @staticmethod
    def _model_id(model_name: str) -> str:
        return f"model:{model_name}"

    @staticmethod
    def _not_found(model_name: str, version: int | None = None) -> ModelRegistryError:
        details: dict[str, Any] = {"model_name": model_name}
        if version is not None:
            details["version"] = version
        return ModelRegistryError(
            "model_not_found",
            "The requested model version was not found",
            details,
        )

    @staticmethod
    def _metadata_violations(
        model_name: Any,
        feature_group: Any,
        metrics: Any,
        artifact_uri: Any,
        train_split: Any = None,
        validation_split: Any = None,
        train_samples: Any = None,
        validation_samples: Any = None,
        seed: Any = None,
    ) -> list[str]:
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
        # Requirement 3.4 completion metadata fields are optional (None is valid,
        # e.g. for registrations made outside the training pipeline) but when
        # supplied must be well-formed.
        if train_split is not None and (
            isinstance(train_split, bool) or not isinstance(train_split, (int, float))
            or not math.isfinite(train_split) or not 0.0 <= train_split <= 1.0
        ):
            violations.append("train_split")
        if validation_split is not None and (
            isinstance(validation_split, bool) or not isinstance(validation_split, (int, float))
            or not math.isfinite(validation_split) or not 0.0 <= validation_split <= 1.0
        ):
            violations.append("validation_split")
        if train_samples is not None and (
            isinstance(train_samples, bool) or not isinstance(train_samples, int) or train_samples < 0
        ):
            violations.append("train_samples")
        if validation_samples is not None and (
            isinstance(validation_samples, bool) or not isinstance(validation_samples, int) or validation_samples < 0
        ):
            violations.append("validation_samples")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= MAX_SEED
        ):
            violations.append("seed")
        return violations

    def register(
        self,
        model_name: str,
        feature_group: str | None,
        metrics: dict[str, float] | None,
        artifact_uri: str | None,
        train_split: float | None = None,
        validation_split: float | None = None,
        train_samples: int | None = None,
        validation_samples: int | None = None,
        seed: int | None = None,
    ) -> ModelVersionRecord:
        """Validate and atomically allocate the next available model version.

        ``train_split``/``validation_split``/``train_samples``/``validation_samples``/
        ``seed`` are the Requirement 3.4 completion metadata the Training_Manager
        forwards once a training job completes; they are optional so registrations
        made outside that pipeline remain unaffected.
        """
        violations = self._metadata_violations(
            model_name,
            feature_group,
            metrics,
            artifact_uri,
            train_split,
            validation_split,
            train_samples,
            validation_samples,
            seed,
        )
        if violations:
            raise ModelRegistryError(
                "invalid_model_metadata",
                "Model registration metadata is missing or invalid",
                {"violations": violations},
            )
        assert feature_group is not None and metrics is not None and artifact_uri is not None
        if not self.model_storage.artifact_exists(artifact_uri):
            raise ModelRegistryError(
                "artifact_unavailable",
                "The model artifact is unavailable in Model Storage",
                {"artifact_uri": artifact_uri},
            )

        self._ensure_indexes()
        now = datetime.now(UTC)
        self.collection.update_one(
            {"_id": self._model_id(model_name)},
            {
                "$setOnInsert": {
                    "document_type": MODEL_DOCUMENT,
                    "model_name": model_name,
                    "created_at": now,
                }
            },
            upsert=True,
        )
        while True:
            latest = self.collection.find_one(
                {"document_type": VERSION_DOCUMENT, "model_name": model_name},
                sort=[("version", DESCENDING)],
            )
            version = 1 if latest is None else int(latest["version"]) + 1
            record = ModelVersionRecord(
                model_name=model_name,
                version=version,
                created_at=now,
                feature_group=feature_group,
                metrics=metrics,
                artifact_uri=artifact_uri,
                train_split=train_split,
                validation_split=validation_split,
                train_samples=train_samples,
                validation_samples=validation_samples,
                seed=seed,
            )
            document = record.model_dump(mode="python")
            document.update(
                {
                    "_id": f"version:{model_name}:{version}",
                    "document_type": VERSION_DOCUMENT,
                    "model_id": self._model_id(model_name),
                }
            )
            try:
                self.collection.insert_one(document)
            except DuplicateKeyError:
                continue
            return record


    def _parent_exists(self, model_name: str) -> bool:
        return self.collection.find_one(
            {
                "_id": self._model_id(model_name),
                "document_type": MODEL_DOCUMENT,
                "model_name": model_name,
            }
        ) is not None

    @staticmethod
    def _to_record(document: dict[str, Any]) -> ModelVersionRecord:
        return ModelVersionRecord.model_validate(
            {
                "model_name": document["model_name"],
                "version": document["version"],
                "created_at": document["created_at"],
                "feature_group": document["feature_group"],
                "metrics": document["metrics"],
                "artifact_uri": document["artifact_uri"],
                "train_split": document.get("train_split"),
                "validation_split": document.get("validation_split"),
                "train_samples": document.get("train_samples"),
                "validation_samples": document.get("validation_samples"),
                "seed": document.get("seed"),
            }
        )

    def list_versions(self, model_name: str) -> list[ModelVersionRecord]:
        """Return at most 500 version records in ascending version order."""
        self._ensure_indexes()
        if not self._parent_exists(model_name):
            raise self._not_found(model_name)
        cursor = self.collection.find(
            {"document_type": VERSION_DOCUMENT, "model_name": model_name}
        )
        documents = cursor.sort("version", ASCENDING).limit(MAX_LIST_VERSIONS)
        return [self._to_record(document) for document in documents]

    def latest(self, model_name: str) -> ModelVersionRecord:
        """Return the greatest version for a model."""
        self._ensure_indexes()
        if not self._parent_exists(model_name):
            raise self._not_found(model_name)
        document = self.collection.find_one(
            {"document_type": VERSION_DOCUMENT, "model_name": model_name},
            sort=[("version", DESCENDING)],
        )
        if document is None:
            raise self._not_found(model_name)
        return self._to_record(document)

    def get(self, model_name: str, version: int) -> ModelVersionRecord:
        """Return one version only when both version and linked model exist."""
        self._ensure_indexes()
        document = self.collection.find_one(
            {
                "document_type": VERSION_DOCUMENT,
                "model_name": model_name,
                "version": version,
            }
        )
        if document is None or not self._parent_exists(model_name):
            raise self._not_found(model_name, version)
        return self._to_record(document)

    def serving_version(self, model_name: str) -> int | None:
        """Return the explicitly promoted serving version, if one exists."""
        document = self.collection.find_one(
            {"_id": self._model_id(model_name), "document_type": MODEL_DOCUMENT}
        )
        if document is None:
            raise self._not_found(model_name)
        value = document.get("serving_version")
        return int(value) if isinstance(value, int) else None

    def mark_serving(self, model_name: str, version: int) -> None:
        """Atomically mark an existing immutable version as the serving version."""
        self.get(model_name, version)
        self.collection.update_one(
            {"_id": self._model_id(model_name), "document_type": MODEL_DOCUMENT},
            {"$set": {"serving_version": version, "serving_updated_at": datetime.now(UTC)}},
        )

    def record_promotion_decision(
        self,
        model_name: str,
        version: int,
        *,
        promoted: bool,
        reason: str,
        error_comparison: dict[str, dict[str, float]],
    ) -> None:
        """Keep promotion/rejection evidence on the candidate version document."""
        self.get(model_name, version)
        self.collection.update_one(
            {"document_type": VERSION_DOCUMENT, "model_name": model_name, "version": version},
            {
                "$set": {
                    "promotion_status": "promoted" if promoted else "not_promoted",
                    "promotion_reason": reason,
                    "promotion_error_comparison": error_comparison,
                    "promotion_decided_at": datetime.now(UTC),
                }
            },
        )
