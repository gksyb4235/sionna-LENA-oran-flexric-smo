"""Inference_Service: deterministic batch GNN prediction API (Requirement 5).

Loads one registered GNN artifact (the format ``train_gnn.py`` writes via
Training_Manager -> Model_Storage/Model_Registry) at startup and serves
``GET /status`` and ``POST /predict/batch`` on port 8105.
"""

from .engine import InferenceEngine
from .errors import InferenceServiceError
from .server import create_app

__all__ = ["InferenceEngine", "InferenceServiceError", "create_app"]
