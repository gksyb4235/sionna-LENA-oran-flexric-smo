"""Property 33 test module (own module to avoid collisions with parallel Property tasks).

Task 5.6: 예측 결정론.

Targets ``smo.aimlfw.inference_service.server``'s ``POST /predict/batch`` endpoint
(Task 5.1), which is backed by ``InferenceEngine.predict_batch``/``_predict_one``
(``engine.py``). requirements.md Requirement 5.8:

    WHEN 동일한 Batch_Prediction_Request 가 동일한 모델 이름과 버전에 두 번
    전송되면, THE Inference_Service SHALL 두 응답의 모든 Target_KPI 예측값과
    셀별 예측값이 동일하고 오류 코드가 동일한 응답을 반환한다.

Design tag (design.md "Inference_Service (Requirement 5)"):

    #### Property 33: 예측 결정론
    *For any* 동일한 Batch_Prediction_Request가 동일한 모델 이름과 버전에 두 번
    전송되는 경우, 두 응답의 모든 Target_KPI 예측값, 셀별 예측값, 오류 코드는
    동일하다.
    **Validates: Requirements 5.8**

Oracle grounding: ``engine.py`` documents its own determinism claim --
``_predict_one`` calls ``torch.manual_seed(_FORWARD_SEED)`` immediately before
every forward pass, and the model is always run in ``model.eval()`` mode
(loaded once by ``InferenceEngine.load()``), so there is no learned
randomness (no dropout, no batchnorm running-stat updates) and no
data-dependent nondeterminism across repeated calls with the same input. That
means exact (bit-identical) equality -- not a tolerance-based comparison -- is
the correct assertion for this property; any exact mismatch found by this
test is a genuine Requirement 5.8 violation, not a numerical-precision
artifact to be papered over.

This test builds one real ``InferenceEngine`` backed by a real, trained
``CellGraphGnn`` checkpoint (produced by the same ``train_gnn.run_training``
the Training_Manager uses) registered through a real ``ModelRegistry`` +
``ModelStorage`` pair, using the in-memory ``_FakeCollection``/``_FakeCursor``
MongoDB stand-in pattern from ``test_property_25_model_version_not_found.py``.
The trained artifact is built once per test module (not once per Hypothesis
example) since training is comparatively expensive and Requirement 5.8 is
about repeat *requests* against one already-loaded model, not about
retraining. Requests are sent through the full FastAPI boundary
(``fastapi.testclient.TestClient``) so the property also covers HTTP
(de)serialization, not just the in-process engine call.
"""

from __future__ import annotations

import copy
from collections.abc import Iterator
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, MAX_TIME_STEP, MIN_TIME_STEP, TARGET_KPIS
from smo.aimlfw.common.models import ParameterSet
from smo.aimlfw.inference_service.schemas import MAX_BATCH_SIZE
from smo.aimlfw.inference_service.server import create_app
from smo.aimlfw.model_registry import ModelRegistry
from smo.aimlfw.model_storage import ModelStorage
from smo.aimlfw.training_manager.train_gnn import run_training
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS, cell_ids, invalid_cell_parameters, valid_parameter_sets

_MODEL_NAME = "gnn-property-33"
_FIXED_TRAINING_CELL_IDS = ("gNB_1", "gNB_2", "gNB_3")
_FIXED_TRAINING_TIME_STEPS = 3


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
        return iter(copy.deepcopy(self.documents))


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
        return copy.deepcopy(matching[0]) if matching else None

    def find(self, query: dict[str, Any]) -> _FakeCursor:
        return _FakeCursor([copy.deepcopy(item) for item in self.documents if self._matches(item, query)])

    def update_one(self, query: dict[str, Any], update: dict[str, Any], **kwargs: Any) -> None:
        if not any(self._matches(item, query) for item in self.documents):
            document = copy.deepcopy(query)
            document.update(copy.deepcopy(update.get("$setOnInsert", {})))
            self.documents.append(document)

    def insert_one(self, document: dict[str, Any]) -> None:
        self.documents.append(copy.deepcopy(document))


def _fixed_training_records() -> list[dict[str, Any]]:
    """A small, fixed (non-random) Feature_Record-shaped dataset for ``run_training``.

    Property 33 is about repeat requests against one already-loaded model, not
    about the training process itself (that is Property 18's concern), so this
    dataset is deliberately hard-coded rather than Hypothesis-generated.
    """
    base_parameters = {
        "tx_power_dbm": 43.0,
        "ret_tilt_deg": 5.0,
        "cio_bias_db": 0.5,
        "hysteresis_db": 2.5,
        "ttt_ms": 160,
    }
    records: list[dict[str, Any]] = []
    for time_step in range(_FIXED_TRAINING_TIME_STEPS):
        for offset, cell_id in enumerate(_FIXED_TRAINING_CELL_IDS):
            records.append(
                {
                    "time_step": time_step,
                    "cell_id": cell_id,
                    "control_parameters": dict(base_parameters),
                    "features": {kpi: float(10 + offset + time_step) for kpi in TARGET_KPIS},
                }
            )
    return records


def _to_batch_payload(parameter_set: ParameterSet) -> dict[str, Any]:
    """Convert a domain ``ParameterSet`` into the raw ``ParameterSetInput``-shaped dict."""
    return {"cells": {cell_id: parameters.model_dump() for cell_id, parameters in parameter_set.cells.items()}}


@pytest.fixture(scope="module")
def loaded_client() -> Iterator[TestClient]:
    """Build one real, loaded Inference_Service app backed by a real trained artifact."""
    with TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        collection = _FakeCollection()
        model_storage = ModelStorage(root / "models")
        registry = ModelRegistry(collection, model_storage)

        model, node_order, payload = run_training(
            _fixed_training_records(), seed=0, max_epochs=3, patience=1
        )
        checkpoint = {
            "state_dict": model.state_dict(),
            "node_order": node_order,
            "target_kpis": list(TARGET_KPIS),
            "control_parameter_names": list(CONTROL_PARAMETER_NAMES),
        }
        buffer = BytesIO()
        torch.save(checkpoint, buffer)
        artifact_uri = model_storage.save_artifact(_MODEL_NAME, 1, buffer.getvalue())
        registry.register(_MODEL_NAME, "default", payload["metrics"], artifact_uri)

        from smo.aimlfw.inference_service.engine import InferenceEngine

        engine = InferenceEngine(registry, _MODEL_NAME)
        app = create_app(engine)
        with TestClient(app) as client:
            status = client.get("/status").json()
            assert status["model_load_status"] == "loaded", status
            yield client


# **Property 33: 예측 결정론**
# **Validates: Requirements 5.8**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    parameter_sets=st.lists(valid_parameter_sets(), min_size=1, max_size=MAX_BATCH_SIZE),
    time_step=st.one_of(st.none(), st.integers(min_value=MIN_TIME_STEP, max_value=MAX_TIME_STEP)),
)
def test_repeated_successful_batch_predictions_are_bit_identical(
    loaded_client: TestClient,
    parameter_sets: list[ParameterSet],
    time_step: int | None,
) -> None:
    """Requirement 5.8: two identical successful Batch_Prediction_Requests against the
    same loaded model/version must produce exactly equal Target_KPI and per-cell
    predictions (and the same model_name/model_version/applied_time_step)."""
    body = {
        "parameter_sets": [_to_batch_payload(parameter_set) for parameter_set in parameter_sets],
        "time_step": time_step,
    }

    first = loaded_client.post("/predict/batch", json=copy.deepcopy(body))
    second = loaded_client.post("/predict/batch", json=copy.deepcopy(body))

    assert first.status_code == 200, first.text
    assert second.status_code == 200, second.text
    first_body = first.json()
    second_body = second.json()

    # Exact equality of the entire response body -- not a tolerance-based
    # comparison -- per the engine's own fixed-seed determinism claim.
    assert first_body == second_body
    assert first_body["model_name"] == second_body["model_name"]
    assert first_body["model_version"] == second_body["model_version"]
    assert first_body["applied_time_step"] == second_body["applied_time_step"]
    assert len(first_body["predictions"]) == len(parameter_sets)
    for first_prediction, second_prediction in zip(
        first_body["predictions"], second_body["predictions"], strict=True
    ):
        assert first_prediction["target_kpi"] == second_prediction["target_kpi"]
        assert first_prediction["cell_kpi"] == second_prediction["cell_kpi"]


# **Property 33: 예측 결정론**
# **Validates: Requirements 5.8**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(
    cell_id=cell_ids(min_size=1, max_size=1),
    invalid_parameters=invalid_cell_parameters(),
    time_step=st.one_of(st.none(), st.integers(min_value=MIN_TIME_STEP, max_value=MAX_TIME_STEP)),
)
def test_repeated_invalid_batch_predictions_yield_identical_errors(
    loaded_client: TestClient,
    cell_id: list[str],
    invalid_parameters: dict[str, float | int],
    time_step: int | None,
) -> None:
    """Requirement 5.8: two identical Batch_Prediction_Requests that both violate
    Control_Parameter range/step (Requirement 5.5) must produce the exact same
    `parameter_out_of_range` error_code and details on both attempts."""
    body = {
        "parameter_sets": [{"cells": {cell_id[0]: invalid_parameters}}],
        "time_step": time_step,
    }

    first = loaded_client.post("/predict/batch", json=copy.deepcopy(body))
    second = loaded_client.post("/predict/batch", json=copy.deepcopy(body))

    assert first.status_code == 422, first.text
    assert second.status_code == 422, second.text
    first_body = first.json()
    second_body = second.json()

    assert first_body["error_code"] == second_body["error_code"] == "parameter_out_of_range"
    assert first_body["details"] == second_body["details"]
