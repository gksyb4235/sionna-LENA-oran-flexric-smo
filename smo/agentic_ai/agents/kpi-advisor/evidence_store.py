"""Atomic MongoDB persistence and assembly for KPI Advisor evidence."""

from __future__ import annotations

import os
import uuid
from collections.abc import Callable
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from pymongo import ASCENDING, MongoClient
from pymongo.errors import DuplicateKeyError
from smo.aimlfw.common.errors import ErrorResponse
from smo.aimlfw.common.models import EvidenceRecord, MarginalEffectRecord, TemporalPlan

if TYPE_CHECKING:
    from smo.aimlfw.common.config import KpiThresholdConfig

    from agent import DegradationJudgment, ProbeExecutionResult, RecommendationResult
    from conflict_analyzer import ConflictAnalysisResult

DEFAULT_MONGODB_URI = "mongodb://127.0.0.1:27017"
DEFAULT_DATABASE = "knowledge"
DEFAULT_COLLECTION = "evidence_records"
DEFAULT_TIMEOUT_MS = 1500


class CollectionLike(Protocol):
    def create_index(self, keys: Any, **kwargs: Any) -> Any: ...

    def insert_one(self, document: dict[str, Any]) -> Any: ...

    def delete_one(self, filter: dict[str, Any]) -> Any: ...

    def find_one(self, filter: dict[str, Any], *args: Any, **kwargs: Any) -> dict[str, Any] | None: ...


class EvidenceStoreError(Exception):
    """Stable Evidence persistence failure."""

    def __init__(self, error_code: str, message: str, details: dict[str, Any] | None = None):
        self.response = ErrorResponse(error_code=error_code, message=message, details=details or {})
        super().__init__(message)

    @property
    def error_code(self) -> str:
        return self.response.error_code


class EvidenceRepository:
    """Store each EvidenceRecord as one MongoDB document."""

    def __init__(self, collection: CollectionLike, client: Any | None = None) -> None:
        self.collection = collection
        self.client = client
        self._indexes_ready = False

    def close(self) -> None:
        if self.client is not None:
            self.client.close()

    def _ensure_indexes(self) -> None:
        if not self._indexes_ready:
            self.collection.create_index([("evidence_id", ASCENDING)], unique=True, name="evidence_id_unique")
            self._indexes_ready = True

    def save(self, record: EvidenceRecord) -> EvidenceRecord:
        """Insert one complete document, cleaning it up if acknowledgement fails."""
        document = record.model_dump(mode="python")
        document["_id"] = record.evidence_id
        try:
            self._ensure_indexes()
            self.collection.insert_one(document)
        except DuplicateKeyError as exc:
            raise EvidenceStoreError(
                "evidence_id_conflict",
                "Evidence_Record identifier already exists",
                {"evidence_id": record.evidence_id},
            ) from exc
        except Exception as exc:
            try:
                self.collection.delete_one({"_id": record.evidence_id})
            except Exception:
                pass
            raise EvidenceStoreError(
                "evidence_storage_failed",
                "Evidence_Record could not be stored atomically",
                {"evidence_id": record.evidence_id, "reason": f"{type(exc).__name__}: {exc}"},
            ) from exc
        return record

    def get(self, evidence_id: str) -> EvidenceRecord | None:
        document = self.collection.find_one({"_id": evidence_id})
        if not isinstance(document, dict):
            return None
        payload = {key: value for key, value in document.items() if key != "_id"}
        # PyMongo clients created without ``tz_aware=True`` decode BSON UTC
        # datetimes as naive values. Treat those values as UTC so records
        # written by older clients remain readable under the strict domain
        # contract, while the default client below preserves timezone data.
        created_at = payload.get("created_at")
        if isinstance(created_at, datetime) and created_at.tzinfo is None:
            payload["created_at"] = created_at.replace(tzinfo=UTC)
        return EvidenceRecord.model_validate(payload)


def _default_repository_factory() -> EvidenceRepository:
    client = MongoClient(
        os.getenv("KNOWLEDGE_MONGODB_URI", DEFAULT_MONGODB_URI),
        serverSelectionTimeoutMS=int(os.getenv("KNOWLEDGE_TIMEOUT_MS", str(DEFAULT_TIMEOUT_MS))),
        tz_aware=True,
    )
    collection = client[os.getenv("KNOWLEDGE_DATABASE", DEFAULT_DATABASE)][
        os.getenv("KNOWLEDGE_EVIDENCE_COLLECTION", DEFAULT_COLLECTION)
    ]
    return EvidenceRepository(collection, client)


_repository_factory: Callable[[], EvidenceRepository] = _default_repository_factory


def set_repository_factory(factory: Callable[[], EvidenceRepository]) -> None:
    global _repository_factory
    _repository_factory = factory


def reset_repository_factory() -> None:
    global _repository_factory
    _repository_factory = _default_repository_factory


def _asdict(value: Any) -> dict[str, Any]:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return dict(value)
    raise TypeError(f"Unsupported evidence value: {type(value).__name__}")


def assemble_evidence_record(
    *,
    probe_result: ProbeExecutionResult,
    thresholds: KpiThresholdConfig,
    judgment: DegradationJudgment,
    recommendation_result: RecommendationResult,
    marginal_effects: list[MarginalEffectRecord] | None = None,
    conflict_result: ConflictAnalysisResult | None = None,
    temporal_plan: TemporalPlan | None = None,
) -> EvidenceRecord:
    """Assemble all available decision inputs into one immutable record."""
    if not probe_result.baseline_prediction.target_kpi:
        raise EvidenceStoreError(
            "no_prediction_evidence",
            "A degradation verdict requires at least one prediction",
        )
    predictions = [
        {
            "parameter_set_index": 0,
            "kind": "baseline",
            "parameter_set": probe_result.plan.baseline.model_dump(mode="python"),
            "target_kpi": dict(probe_result.baseline_prediction.target_kpi),
            "cell_kpi": probe_result.baseline_prediction.cell_kpi,
        }
    ]
    for index, variant_result in enumerate(probe_result.variant_results, start=1):
        predictions.append(
            {
                "parameter_set_index": index,
                "kind": "variant",
                "variant_id": variant_result.variant.variant_id,
                "parameter_set": variant_result.variant.parameter_set.model_dump(mode="python"),
                "target_kpi": dict(variant_result.prediction.target_kpi),
                "cell_kpi": variant_result.prediction.cell_kpi,
                "percent_change": dict(variant_result.percent_change),
            }
        )
    conflicts = None if conflict_result is None else [_asdict(item) for item in conflict_result.indirect_conflicts]
    created_at = datetime.now(UTC)
    # BSON datetime precision is milliseconds. Normalize before persistence so
    # an immediate save/get round-trip reproduces the same domain record.
    created_at = created_at.replace(microsecond=(created_at.microsecond // 1000) * 1000)
    return EvidenceRecord(
        evidence_id=f"evidence-{uuid.uuid4().hex}",
        model_name=probe_result.model_name,
        model_version=probe_result.model_version,
        probe_plan=probe_result.plan,
        predictions=predictions,
        applied_thresholds=list(thresholds.thresholds.values()),
        degradation_verdict=judgment.verdict,
        marginal_effects=marginal_effects,
        conflict_threshold=None if conflict_result is None else conflict_result.conflict_threshold,
        indirect_conflicts=conflicts,
        judgment_details={
            "violations": [_asdict(item) for item in judgment.violations],
            "unknown_reasons": [_asdict(item) for item in judgment.unknown_reasons],
        },
        recommendations=[item.model_dump(mode="python") for item in recommendation_result.recommendations],
        temporal_plan=None if temporal_plan is None else temporal_plan.model_dump(mode="python"),
        temporal_plan_ref=None,
        created_at=created_at,
    )


def persist_evidence(record: EvidenceRecord) -> EvidenceRecord:
    repository = _repository_factory()
    try:
        return repository.save(record)
    finally:
        repository.close()


__all__ = [
    "EvidenceRepository",
    "EvidenceStoreError",
    "assemble_evidence_record",
    "persist_evidence",
    "reset_repository_factory",
    "set_repository_factory",
]
