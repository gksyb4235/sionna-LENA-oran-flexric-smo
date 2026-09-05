"""Runnable polling worker for the local multi-process AIMLFW stack."""

from __future__ import annotations

import asyncio
import os
import signal
from pathlib import Path

from pymongo import MongoClient

from smo.aimlfw.feature_store.store import FeatureStore
from smo.aimlfw.model_registry.registry import ModelRegistry
from smo.aimlfw.model_storage.storage import ModelStorage

from .adapters import (
    FeatureStoreRecordsProvider,
    HttpInferencePromotionClient,
    HttpPredictionClient,
    HttpTrainingClient,
    LocalModelEvaluator,
)
from .service import RetrainingController

_AIMLFW_ROOT = Path(__file__).resolve().parents[1]


def build_controller() -> tuple[RetrainingController, FeatureStoreRecordsProvider]:
    """Build the worker from the same storage and service settings as the HTTP processes."""
    mongo_client = MongoClient(
        os.getenv("MODEL_REGISTRY_MONGODB_URI", "mongodb://127.0.0.1:27017"),
        serverSelectionTimeoutMS=int(os.getenv("MODEL_REGISTRY_MONGODB_TIMEOUT_MS", "5000")),
    )
    collection = mongo_client[
        os.getenv("MODEL_REGISTRY_DATABASE", "aimlfw")
    ][os.getenv("MODEL_REGISTRY_COLLECTION", "model_registry")]
    storage = ModelStorage(os.getenv("MODEL_STORAGE_ROOT", str(_AIMLFW_ROOT / "models")))
    registry = ModelRegistry(collection, storage)
    feature_group = os.getenv("RETRAINING_FEATURE_GROUP", "default")
    provider = FeatureStoreRecordsProvider(
        FeatureStore(os.getenv("FEATURE_STORE_ROOT", str(_AIMLFW_ROOT / "feature_store" / "data"))),
        feature_group,
    )
    inference_url = os.getenv("INFERENCE_SERVICE_BASE_URL", "http://127.0.0.1:8105")
    threshold_value: object = os.getenv("RETRAINING_THRESHOLD")
    if isinstance(threshold_value, str):
        try:
            threshold_value = float(threshold_value)
        except ValueError:
            pass
    controller = RetrainingController(
        HttpPredictionClient(inference_url),
        HttpTrainingClient(os.getenv("TRAINING_MANAGER_BASE_URL", "http://127.0.0.1:8103")),
        LocalModelEvaluator(registry),
        HttpInferencePromotionClient(registry, inference_url),
        feature_group=feature_group,
        model_name=os.getenv("RETRAINING_MODEL_NAME", "gnn"),
        seed=int(os.getenv("RETRAINING_SEED", "0")),
        retraining_threshold=threshold_value,
        poll_seconds=float(os.getenv("RETRAINING_POLL_SECONDS", "60")),
    )
    return controller, provider


async def run_worker() -> None:
    controller, provider = build_controller()
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(signum, stop_event.set)
        except NotImplementedError:
            pass
    await controller.run(provider, stop_event)


def main() -> None:
    asyncio.run(run_worker())


if __name__ == "__main__":
    main()
