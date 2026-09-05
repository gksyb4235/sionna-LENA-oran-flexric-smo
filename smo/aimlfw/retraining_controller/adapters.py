"""Concrete local adapters for the Retraining Controller."""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from smo.aimlfw.common.constants import TARGET_KPIS
from smo.aimlfw.common.models import FeatureRecord, ParameterSet
from smo.aimlfw.feature_store.store import FeatureStore
from smo.aimlfw.inference_service.engine import InferenceEngine
from smo.aimlfw.inference_service.schemas import ParameterSetInput
from smo.aimlfw.model_registry.registry import ModelRegistry

from .service import prediction_error_percent


class FeatureStoreRecordsProvider:
    """Return a stable Feature Store snapshot for the controller poll loop."""

    def __init__(self, store: FeatureStore, feature_group: str = "default") -> None:
        self.store = store
        self.feature_group = feature_group

    async def __call__(self) -> list[FeatureRecord]:
        return await asyncio.to_thread(self.store.records, self.feature_group)


class HttpPredictionClient:
    def __init__(self, base_url: str, timeout_seconds: float = 30.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def predict(self, parameter_set: ParameterSet, time_step: int) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_seconds) as client:
            response = await client.post(
                "/predict/batch",
                json={
                    "parameter_sets": [parameter_set.model_dump(mode="json")],
                    "time_step": time_step,
                },
            )
        response.raise_for_status()
        payload = response.json()
        prediction = payload["predictions"][0]
        return {
            "model_name": payload["model_name"],
            "model_version": payload["model_version"],
            **prediction,
        }


class HttpTrainingClient:
    def __init__(self, base_url: str, timeout_seconds: float = 10.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds

    async def create_job(
        self,
        feature_group: str,
        model_name: str,
        seed: int,
        metadata: dict[str, Any],
    ) -> str:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_seconds) as client:
            response = await client.post(
                "/jobs",
                json={
                    "feature_group": feature_group,
                    "model_name": model_name,
                    "seed": seed,
                    "metadata": metadata,
                },
            )
        response.raise_for_status()
        return str(response.json()["job_id"])

    async def get_job(self, job_id: str) -> dict[str, Any]:
        async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout_seconds) as client:
            response = await client.get(f"/jobs/{job_id}")
        response.raise_for_status()
        return response.json()


class InProcessPredictionClient:
    def __init__(self, engine: InferenceEngine) -> None:
        self.engine = engine

    async def predict(self, parameter_set: ParameterSet, time_step: int) -> dict[str, Any]:
        inputs = [ParameterSetInput.model_validate(parameter_set.model_dump(mode="python"))]
        prediction = self.engine.predict_batch(inputs, time_step)[0]
        return {
            "model_name": self.engine.loaded_model_name,
            "model_version": self.engine.loaded_model_version,
            **prediction,
        }


class InProcessTrainingClient:
    def __init__(self, job_manager: Any) -> None:
        self.job_manager = job_manager

    async def create_job(
        self,
        feature_group: str,
        model_name: str,
        seed: int,
        metadata: dict[str, Any],
    ) -> str:
        return await self.job_manager.create_job(feature_group, model_name, seed, metadata)

    async def get_job(self, job_id: str) -> dict[str, Any]:
        return self.job_manager.get_job(job_id).model_dump(mode="python")


class LocalModelEvaluator:
    """Evaluate any registered version without disturbing the serving engine."""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry

    async def errors(
        self,
        model_name: str,
        model_version: int,
        records: list[FeatureRecord],
    ) -> dict[str, float]:
        engine = InferenceEngine(self.registry, model_name, model_version)
        await engine.load()
        if engine.model_load_status != "loaded":
            raise RuntimeError(f"model {model_name}:{model_version} could not be loaded for evaluation")
        errors: dict[str, list[float]] = {target_kpi: [] for target_kpi in TARGET_KPIS}
        for record in records:
            parameter_set = ParameterSet(cells={record.cell_id: record.control_parameters})
            inputs = [ParameterSetInput.model_validate(parameter_set.model_dump(mode="python"))]
            prediction = engine.predict_batch(inputs, record.time_step)[0]
            predicted_kpis = prediction["cell_kpi"].get(record.cell_id, prediction["target_kpi"])
            for target_kpi in TARGET_KPIS:
                actual = record.features.get(target_kpi)
                predicted = predicted_kpis.get(target_kpi)
                if actual is None or predicted is None or actual == 0:
                    continue
                errors[target_kpi].append(prediction_error_percent(float(predicted), float(actual)))
        return {
            target_kpi: round(sum(values) / len(values), 1)
            for target_kpi, values in errors.items()
            if values
        }


class LocalPromotionClient:
    """Coordinate registry state and in-process model swap with rollback."""

    def __init__(self, registry: ModelRegistry, engine: InferenceEngine) -> None:
        self.registry = registry
        self.engine = engine
        self._lock = asyncio.Lock()

    async def serving_version(self, model_name: str) -> int:
        explicit = self.registry.serving_version(model_name)
        current = explicit if explicit is not None else self.engine.loaded_model_version
        if current is None:
            raise RuntimeError("no serving model version is available")
        return current

    async def promote(
        self,
        model_name: str,
        model_version: int,
        error_comparison: dict[str, dict[str, float]],
    ) -> None:
        async with self._lock:
            previous_version = await self.serving_version(model_name)
            await self.engine.switch_version(model_name, model_version)
            try:
                self.registry.mark_serving(model_name, model_version)
                self.registry.record_promotion_decision(
                    model_name,
                    model_version,
                    promoted=True,
                    reason="all_kpis_not_worse_and_trigger_improved_at_least_1pp",
                    error_comparison=error_comparison,
                )
            except Exception as exc:
                # Keep the two externally visible serving pointers aligned even
                # when recording the promotion decision itself fails.
                rollback_errors: list[str] = []
                try:
                    self.registry.mark_serving(model_name, previous_version)
                except Exception as rollback_exc:  # noqa: BLE001
                    rollback_errors.append(f"registry rollback failed: {rollback_exc}")
                try:
                    await self.engine.switch_version(model_name, previous_version)
                except Exception as rollback_exc:  # noqa: BLE001
                    rollback_errors.append(f"inference rollback failed: {rollback_exc}")
                if rollback_errors:
                    raise RuntimeError("; ".join(rollback_errors)) from exc
                raise

    async def reject(
        self,
        model_name: str,
        model_version: int,
        reason: str,
        error_comparison: dict[str, dict[str, float]],
    ) -> None:
        self.registry.record_promotion_decision(
            model_name,
            model_version,
            promoted=False,
            reason=reason,
            error_comparison=error_comparison,
        )


class HttpInferencePromotionClient:
    """Switch a remote inference process and compensate on registry failure."""

    def __init__(
        self,
        registry: ModelRegistry,
        inference_base_url: str,
        timeout_seconds: float = 60.0,
    ) -> None:
        self.registry = registry
        self.inference_base_url = inference_base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._lock = asyncio.Lock()

    async def serving_version(self, model_name: str) -> int:
        version = self.registry.serving_version(model_name)
        if version is None:
            raise RuntimeError("no serving model version is available")
        return version

    async def _switch(self, model_name: str, model_version: int) -> None:
        async with httpx.AsyncClient(
            base_url=self.inference_base_url,
            timeout=self.timeout_seconds,
        ) as client:
            response = await client.put(
                "/serving-version",
                json={"model_name": model_name, "model_version": model_version},
            )
        response.raise_for_status()

    async def promote(
        self,
        model_name: str,
        model_version: int,
        error_comparison: dict[str, dict[str, float]],
    ) -> None:
        async with self._lock:
            previous_version = await self.serving_version(model_name)
            await self._switch(model_name, model_version)
            try:
                self.registry.mark_serving(model_name, model_version)
                self.registry.record_promotion_decision(
                    model_name,
                    model_version,
                    promoted=True,
                    reason="all_kpis_not_worse_and_trigger_improved_at_least_1pp",
                    error_comparison=error_comparison,
                )
            except Exception as exc:
                rollback_errors: list[str] = []
                try:
                    self.registry.mark_serving(model_name, previous_version)
                except Exception as rollback_exc:  # noqa: BLE001
                    rollback_errors.append(f"registry rollback failed: {rollback_exc}")
                try:
                    await self._switch(model_name, previous_version)
                except Exception as rollback_exc:  # noqa: BLE001
                    rollback_errors.append(f"inference rollback failed: {rollback_exc}")
                if rollback_errors:
                    raise RuntimeError("; ".join(rollback_errors)) from exc
                raise

    async def reject(
        self,
        model_name: str,
        model_version: int,
        reason: str,
        error_comparison: dict[str, dict[str, float]],
    ) -> None:
        self.registry.record_promotion_decision(
            model_name,
            model_version,
            promoted=False,
            reason=reason,
            error_comparison=error_comparison,
        )


__all__ = [
    "FeatureStoreRecordsProvider",
    "HttpInferencePromotionClient",
    "HttpPredictionClient",
    "HttpTrainingClient",
    "InProcessPredictionClient",
    "InProcessTrainingClient",
    "LocalModelEvaluator",
    "LocalPromotionClient",
]
