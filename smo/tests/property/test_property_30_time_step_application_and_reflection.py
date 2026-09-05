"""Property 30 test module (own module to avoid collisions with parallel Property tasks).

Task 5.3: Time_Step 적용과 응답 반영.

Targets ``smo.aimlfw.inference_service.engine.InferenceEngine.predict_batch``/
``_predict_one`` and the ``POST /predict/batch`` boundary in
``smo.aimlfw.inference_service.server`` (Task 5.1), as specified in requirements.md:

Requirement 5.4:
    WHEN Batch_Prediction_Request 가 0 이상 4 이하의 정수 Time_Step 값을 포함하면,
    THE Inference_Service SHALL 해당 Time_Step 조건에서의 예측값을 반환하고
    응답에 적용된 Time_Step 값을 포함한다.

Requirement 5.13:
    WHERE Batch_Prediction_Request 에 Time_Step 값이 생략되면, THE Inference_Service
    SHALL Time_Step 0 을 적용한 예측값을 반환하고 응답에 적용된 Time_Step 값 0 을
    포함한다.

design.md Property 30: Time_Step 적용과 응답 반영
    *For any* 0 이상 4 이하의 정수 Time_Step 값(또는 생략), Inference_Service는
    해당 Time_Step(생략 시 0)에서의 예측값을 반환하고 응답에 실제 적용된 Time_Step
    값을 포함한다.
    **Validates: Requirements 5.4, 5.13**

Oracle grounding: ``train_gnn._node_input_vector`` (re-exported as
``node_input_vector``) appends ``time_step / 4.0`` as the last element of every
node's input vector, and ``InferenceEngine._predict_one`` builds that vector
identically for inference. So a genuinely applied Time_Step must both (a) be
echoed back verbatim as ``applied_time_step`` in the HTTP response, and (b)
actually participate in the forward pass -- i.e. two distinct Time_Step values
against the same Parameter_Set must be able to produce distinct predictions,
which would not hold if Time_Step were silently dropped before reaching the
model.

The GNN checkpoint used here is a single, fixed-seed *initialized* (not
trained) ``CellGraphGnn``, built once per test module run -- not per Hypothesis
example -- via the same ``ModelStorage``/``ModelRegistry`` (with an in-memory
fake Mongo collection, matching the Property 23/25 test pattern) plumbing the
real Training_Manager pipeline uses, then loaded through a real
``InferenceEngine``. Only the generated ``Parameter_Set`` and Time_Step values
vary across Hypothesis examples, so the check that Time_Step participates in
the model input is not confounded by re-randomized weights per example.
"""

from __future__ import annotations

import asyncio
import io
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, TARGET_KPIS
from smo.aimlfw.common.models import ParameterSet
from smo.aimlfw.inference_service.engine import InferenceEngine
from smo.aimlfw.inference_service.schemas import CellParametersInput, ParameterSetInput
from smo.aimlfw.inference_service.server import create_app
from smo.aimlfw.model_registry import ModelRegistry
from smo.aimlfw.model_storage import ModelStorage
from smo.aimlfw.training_manager.train_gnn import CellGraphGnn
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS, valid_parameter_sets

_MODEL_NAME = "gnn-property-30"
_MODEL_SEED = 12345
_FLOAT_TOLERANCE = 1e-9


class _FakeCursor:
    """Minimal in-memory stand-in for a PyMongo cursor (sort + limit only)."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents

    def sort(self, key: str, direction: int | None = None) -> "_FakeCursor":
        self.documents = sorted(self.documents, key=lambda item: item[key], reverse=direction == -1)
        return self

    def limit(self, count: int) -> "_FakeCursor":
        self.documents = self.documents[:count]
        return self

    def __iter__(self):
        return iter(list(self.documents))


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
        return dict(matching[0]) if matching else None

    def find(self, query: dict[str, Any]) -> _FakeCursor:
        return _FakeCursor([dict(item) for item in self.documents if self._matches(item, query)])

    def update_one(self, query: dict[str, Any], update: dict[str, Any], **kwargs: Any) -> None:
        if not any(self._matches(item, query) for item in self.documents):
            document = dict(query)
            document.update(dict(update.get("$setOnInsert", {})))
            self.documents.append(document)

    def insert_one(self, document: dict[str, Any]) -> None:
        self.documents.append(dict(document))


def _build_and_register_model(root: Path) -> tuple[ModelRegistry, str, int]:
    """Build one fixed-seed initialized ``CellGraphGnn`` and register it.

    Mirrors the artifact shape ``train_gnn.py`` writes (``state_dict`` plus the
    node-order/KPI/parameter-name metadata) so ``InferenceEngine._load_sync``
    loads it exactly as it would a real training artifact.
    """
    torch.manual_seed(_MODEL_SEED)
    in_dim = len(CONTROL_PARAMETER_NAMES) + 1  # +1 for the normalized Time_Step input
    out_dim = len(TARGET_KPIS)
    model = CellGraphGnn(in_dim, out_dim)
    model.eval()

    buffer = io.BytesIO()
    torch.save(
        {
            "state_dict": model.state_dict(),
            "node_order": [],
            "target_kpis": list(TARGET_KPIS),
            "control_parameter_names": list(CONTROL_PARAMETER_NAMES),
        },
        buffer,
    )

    storage = ModelStorage(root / "models")
    artifact_uri = storage.save_artifact(_MODEL_NAME, 1, buffer.getvalue())

    collection = _FakeCollection()
    collection.documents.append(
        {
            "_id": f"model:{_MODEL_NAME}",
            "document_type": "model",
            "model_name": _MODEL_NAME,
            "created_at": datetime.now(timezone.utc),
        }
    )
    registry = ModelRegistry(collection, storage)
    record = registry.register(
        _MODEL_NAME,
        feature_group="default",
        metrics={"delay_p95_ms": 1.0},
        artifact_uri=artifact_uri,
    )
    return registry, _MODEL_NAME, record.version


def _to_parameter_set_input(parameter_set: ParameterSet) -> ParameterSetInput:
    return ParameterSetInput(
        cells={
            cell_id: CellParametersInput(**cell_parameters.model_dump())
            for cell_id, cell_parameters in parameter_set.cells.items()
        }
    )


@pytest.fixture(scope="module")
def loaded_engine() -> InferenceEngine:
    """Build one fixed-seed GNN artifact and load it into a real InferenceEngine.

    Built once for the whole test module (not per Hypothesis example) so the
    Property 30 checks below only vary the generated Parameter_Set/Time_Step,
    per the task note about avoiding per-example weight re-randomization.
    """
    with TemporaryDirectory() as raw_root:
        registry, model_name, version = _build_and_register_model(Path(raw_root))
        engine = InferenceEngine(registry, model_name, model_version=version)
        asyncio.run(engine.load())
        assert engine.model_load_status == "loaded"
        yield engine


@pytest.fixture(scope="module")
def client(loaded_engine: InferenceEngine) -> TestClient:
    return TestClient(create_app(engine=loaded_engine))


# **Property 30: Time_Step 적용과 응답 반영**
# **Validates: Requirements 5.4**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(parameter_set=valid_parameter_sets(), time_step=st.integers(min_value=0, max_value=4))
def test_explicit_time_step_is_echoed_as_applied_time_step(
    client: TestClient, parameter_set: ParameterSet, time_step: int
) -> None:
    payload = {
        "parameter_sets": [_to_parameter_set_input(parameter_set).model_dump()],
        "time_step": time_step,
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["applied_time_step"] == time_step
    assert len(body["predictions"]) == 1


# **Property 30: Time_Step 적용과 응답 반영**
# **Validates: Requirements 5.13**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(parameter_set=valid_parameter_sets())
def test_omitted_time_step_defaults_to_zero(client: TestClient, parameter_set: ParameterSet) -> None:
    payload = {"parameter_sets": [_to_parameter_set_input(parameter_set).model_dump()]}
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["applied_time_step"] == 0
    assert len(body["predictions"]) == 1


# **Property 30: Time_Step 적용과 응답 반영**
# **Validates: Requirements 5.4, 5.13**
def test_time_step_genuinely_participates_in_predictions(loaded_engine: InferenceEngine) -> None:
    """Time_Step must not be silently dropped before reaching the model.

    Rather than asserting exact difference on every generated example (which
    would be confounded by the rare, structurally-possible case of two
    Time_Step encodings landing on the same output for a given fixed-weight
    model and Parameter_Set), this collects evidence across Hypothesis
    examples and requires at least one observed difference -- proof that
    Time_Step genuinely participates in the model input.
    """
    observed_difference = False

    @given(
        parameter_set=valid_parameter_sets(),
        time_steps=st.lists(
            st.integers(min_value=0, max_value=4), min_size=2, max_size=2, unique=True
        ),
    )
    @PROPERTY_TEST_SETTINGS
    def check(parameter_set: ParameterSet, time_steps: list[int]) -> None:
        nonlocal observed_difference
        step_a, step_b = time_steps
        parameter_set_input = _to_parameter_set_input(parameter_set)

        result_a = loaded_engine.predict_batch([parameter_set_input], step_a)[0]
        result_b = loaded_engine.predict_batch([parameter_set_input], step_b)[0]

        target_kpi_differs = any(
            abs(result_a["target_kpi"][kpi] - result_b["target_kpi"][kpi]) > _FLOAT_TOLERANCE
            for kpi in result_a["target_kpi"]
        )
        cell_kpi_differs = any(
            abs(result_a["cell_kpi"][cell_id][kpi] - result_b["cell_kpi"][cell_id][kpi]) > _FLOAT_TOLERANCE
            for cell_id in result_a["cell_kpi"]
            for kpi in result_a["cell_kpi"][cell_id]
        )
        if target_kpi_differs or cell_kpi_differs:
            observed_difference = True

    check()

    assert observed_difference, (
        "no generated (Parameter_Set, distinct Time_Step pair) example produced a "
        "different prediction -- Time_Step may be silently dropped before reaching "
        "the model"
    )
