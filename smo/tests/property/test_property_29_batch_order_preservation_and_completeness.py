"""Property 29 test module (own module to avoid collisions with parallel Property tasks).

Task 5.2: 배치 예측 순서 보존과 완전성.

Targets ``smo.aimlfw.inference_service`` end-to-end through the real FastAPI
``POST /predict/batch`` route (``server.py``) backed by a real
``InferenceEngine`` (``engine.py``) with a real, loaded
``train_gnn.CellGraphGnn`` checkpoint registered through the real
``ModelRegistry``/``ModelStorage``, as specified in requirements.md:

Requirement 5.2:
    WHEN Batch_Prediction_Request 가 1 개 이상 64 개 이하인 N 개의 Parameter_Set 을
    담고 도착하면, THE Inference_Service SHALL 요청과 동일한 순서로 N 개의 예측
    결과를 반환하고 각 결과에 요청 내 0 기반 인덱스를 포함한다.

Requirement 5.3:
    WHEN Inference_Service 가 하나의 Parameter_Set 을 예측하면, THE Inference_Service
    SHALL Glossary 에 정의된 모든 Target_KPI 각각에 대해 하나의 유한 실수 예측값과,
    요청된 대상 셀 집합의 각 cell_id 에 대해 하나의 유한 실수 셀별 예측값을 누락
    없이 반환한다.

Requirement 5.9:
    WHEN 예측 응답이 생성되면, THE Inference_Service SHALL 사용된 모델 이름과 버전을
    응답에 포함한다.

Design tag (design.md, "Inference_Service (Requirement 5)"):

    #### Property 29: 배치 예측 순서 보존과 완전성
    *For any* 1개 이상 64개 이하의 Parameter_Set으로 구성된 Batch_Prediction_Request,
    응답의 예측 결과는 요청과 동일한 순서의 N개이며 각 결과는 0 기반 인덱스, 모든
    Target_KPI에 대한 유한 실수 예측값, 요청된 각 cell_id에 대한 유한 실수 셀별
    예측값, 사용된 모델 이름과 버전을 누락 없이 포함한다.
    **Validates: Requirements 5.2, 5.3, 5.9**

Oracle grounding: this test drives the real ``POST /predict/batch`` HTTP
route (``smo.aimlfw.inference_service.server.create_app``) against a real
``InferenceEngine`` holding a real (untrained-but-real, since Property 29 is
about structural completeness/order, not prediction accuracy)
``train_gnn.CellGraphGnn`` checkpoint. The checkpoint is produced with the
exact ``torch.save({"state_dict": ...}, ...)`` shape ``engine._load_sync``
expects and is registered via the real ``ModelRegistry.register`` /
``ModelStorage.save_artifact``, using the same in-memory
``_FakeCollection``/``_FakeCursor`` MongoDB stand-in established by
``test_property_25_model_version_not_found.py`` (no local MongoDB is
available in this environment). Only the storage/registry MongoDB
*transport* is faked -- ``InferenceEngine``, ``ModelRegistry``,
``ModelStorage``, and the FastAPI app are all exercised unmodified.
"""

from __future__ import annotations

import io
import math
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.common.constants import (
    CONTROL_PARAMETER_NAMES,
    MAX_TIME_STEP,
    MIN_TIME_STEP,
    TARGET_KPIS,
)
from smo.aimlfw.common.models import ParameterSet
from smo.aimlfw.inference_service.engine import InferenceEngine
from smo.aimlfw.inference_service.schemas import MAX_BATCH_SIZE
from smo.aimlfw.inference_service.server import create_app
from smo.aimlfw.model_registry import ModelRegistry
from smo.aimlfw.model_storage import ModelStorage
from smo.aimlfw.training_manager.train_gnn import CellGraphGnn
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS, valid_parameter_sets

_MODEL_NAME = "gnn-property-29"
_IN_DIM = len(CONTROL_PARAMETER_NAMES) + 1  # +1 for the normalized Time_Step input (matches engine.py)
_OUT_DIM = len(TARGET_KPIS)
# Cap per-batch cell counts well below MAX_TARGET_CELLS: Property 29 is about
# order preservation/completeness, not cell-count scaling, so this keeps the
# up-to-64-entry batches this property exercises fast without losing coverage.
_MAX_CELLS_PER_PARAMETER_SET = 3


class _FakeCursor:
    """Minimal in-memory stand-in for a PyMongo cursor (sort + limit only)."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents

    def sort(self, key: str, direction: int | None = None) -> _FakeCursor:
        self.documents.sort(key=lambda item: item[key], reverse=direction == -1)
        return self

    def limit(self, count: int) -> _FakeCursor:
        self.documents = self.documents[:count]
        return self

    def __iter__(self):
        return iter(deepcopy(self.documents))


class _FakeCollection:
    """Minimal in-memory stand-in for the PyMongo collection ``ModelRegistry`` uses."""

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

    def update_one(self, query: dict[str, Any], update: dict[str, Any], **kwargs: Any) -> None:
        if not any(self._matches(item, query) for item in self.documents):
            document = deepcopy(query)
            document.update(deepcopy(update.get("$setOnInsert", {})))
            self.documents.append(document)

    def insert_one(self, document: dict[str, Any]) -> None:
        self.documents.append(deepcopy(document))


def _build_and_register_checkpoint(model_storage: ModelStorage, registry: ModelRegistry) -> int:
    """Build a real (untrained) ``CellGraphGnn`` checkpoint and register it.

    Property 29 concerns response structure/order/completeness, not
    prediction accuracy, so an untrained model is a legitimate "real tiny
    GNN checkpoint" -- it is the identical class, input/output dimensions,
    and ``torch.save`` shape ``train_gnn.py`` produces and ``engine.py``
    expects to load.
    """
    model = CellGraphGnn(_IN_DIM, _OUT_DIM)
    buffer = io.BytesIO()
    torch.save({"state_dict": model.state_dict()}, buffer)
    artifact_uri = model_storage.save_artifact(_MODEL_NAME, 1, buffer.getvalue())
    record = registry.register(
        _MODEL_NAME,
        "default",
        {kpi: 1.0 for kpi in TARGET_KPIS},
        artifact_uri,
    )
    return record.version


def _parameter_set_payload(parameter_set: ParameterSet) -> dict[str, Any]:
    """Convert a domain ``ParameterSet`` into a ``BatchPredictionRequest`` JSON entry."""
    return {"cells": parameter_set.model_dump(mode="json")["cells"]}


_temp_dir = TemporaryDirectory()
_root = Path(_temp_dir.name)
_model_storage = ModelStorage(_root / "models")
_registry = ModelRegistry(_FakeCollection(), _model_storage)
_REGISTERED_VERSION = _build_and_register_checkpoint(_model_storage, _registry)
_engine = InferenceEngine(_registry, _MODEL_NAME)
_app = create_app(engine=_engine)

with TestClient(_app) as _client:
    pass  # triggers the FastAPI startup hook (engine.load()) exactly once.


# **Property 29: 배치 예측 순서 보존과 완전성**
# **Validates: Requirements 5.2, 5.3, 5.9**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    parameter_sets=st.lists(
        valid_parameter_sets(min_cells=1, max_cells=_MAX_CELLS_PER_PARAMETER_SET),
        min_size=1,
        max_size=MAX_BATCH_SIZE,
    ),
    time_step=st.integers(min_value=MIN_TIME_STEP, max_value=MAX_TIME_STEP),
)
def test_batch_predictions_preserve_order_and_completeness(
    parameter_sets: list[ParameterSet], time_step: int
) -> None:
    assert _engine.model_load_status == "loaded"  # sanity: the module-level startup load succeeded.

    request_body = {
        "parameter_sets": [_parameter_set_payload(parameter_set) for parameter_set in parameter_sets],
        "time_step": time_step,
    }
    response = _client.post("/predict/batch", json=request_body)
    assert response.status_code == 200, response.text
    body = response.json()

    predictions = body["predictions"]

    # Requirement 5.2: exactly N predictions, in the same order as submitted,
    # each carrying its 0-based request index.
    assert len(predictions) == len(parameter_sets)
    for expected_index, (prediction, parameter_set) in enumerate(
        zip(predictions, parameter_sets, strict=True)
    ):
        assert prediction["index"] == expected_index

        # Requirement 5.3: every Target_KPI has one finite float prediction.
        target_kpi = prediction["target_kpi"]
        assert set(target_kpi) == set(TARGET_KPIS)
        for value in target_kpi.values():
            assert isinstance(value, float) and math.isfinite(value)

        # Requirement 5.3: exactly the requested cell_id set has one finite
        # float per-cell prediction, with no missing and no extra cell_ids.
        cell_kpi = prediction["cell_kpi"]
        assert set(cell_kpi) == set(parameter_set.cells)
        for cell_id, kpi_values in cell_kpi.items():
            assert set(kpi_values) == set(TARGET_KPIS)
            for value in kpi_values.values():
                assert isinstance(value, float) and math.isfinite(value)

    # Requirement 5.9: the response carries the model name and version
    # actually used, matching what was registered and loaded.
    assert body["model_name"] == _MODEL_NAME
    assert body["model_version"] == _REGISTERED_VERSION
    assert _engine.loaded_model_name == _MODEL_NAME
    assert _engine.loaded_model_version == _REGISTERED_VERSION
