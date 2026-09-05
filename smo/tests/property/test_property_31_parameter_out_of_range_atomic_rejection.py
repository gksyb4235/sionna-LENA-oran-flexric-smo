"""Property 31 test module (own module to avoid collisions with parallel Property tasks).

Task 5.4: 파라미터 범위 위반 시 전체 거부.

Targets ``smo.aimlfw.inference_service.engine.InferenceEngine.predict_batch`` /
``control_parameter_violations`` (Task 5.1), as specified in requirements.md
Requirement 5.5 and design.md ``Inference_Service (Requirement 5)``:

    #### Property 31: 파라미터 범위 위반 시 전체 거부
    *For any* Batch_Prediction_Request 내 하나 이상의 Parameter_Set이 허용 범위
    또는 허용 값 집합을 벗어난 Control_Parameter 값을 포함하는 경우,
    Inference_Service는 요청 전체에 대한 예측을 수행하지 않고 위반한 모든
    (인덱스, 파라미터 이름, 값) 항목을 담은 ``parameter_out_of_range`` 오류를
    반환한다.
    **Validates: Requirements 5.5**

Oracle grounding: ``engine.py``'s ``control_parameter_violations`` walks every
``(Parameter_Set index, cell_id)`` pair and, for every Control_Parameter,
either checks membership in ``TTT_ALLOWED_MS`` (``ttt_ms``) or range/step
alignment against ``CONTROL_PARAMETER_RANGES`` (the other four parameters),
collecting *every* violation rather than short-circuiting on the first one
found. ``InferenceEngine.predict_batch`` raises ``InferenceServiceError``
with ``error_code="parameter_out_of_range"`` and
``details={"violations": [...]}`` whenever that list is non-empty, before
any prediction is computed.

As ``schemas.py``'s own module docstring explains, ``ParameterSetInput`` /
``CellParametersInput`` deliberately skip range/step validation (unlike the
strict ``smo.aimlfw.common.models.CellParameters`` domain model) so that a
batch with several violations reaches the engine as ordinary input instead
of being rejected piecemeal by Pydantic. This test exploits exactly that: it
constructs request payloads as raw dicts with out-of-range values and feeds
them straight into ``ParameterSetInput.model_validate`` / the real
``/predict/batch`` endpoint.

This test drives a real ``InferenceEngine`` backed by a real
``CellGraphGnn`` artifact, a real ``ModelStorage`` filesystem root, and a
minimal in-memory ``ModelRegistry`` collection (the ``_FakeCollection`` /
``_FakeCursor`` pattern used by the Property 25/27 tests), loaded once per
test module since the model itself is irrelevant to this property -- only
whether ``predict_batch`` even reaches a forward pass matters.
"""

from __future__ import annotations

import asyncio
import io
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import pytest
import torch
from fastapi.testclient import TestClient
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, TARGET_KPIS
from smo.aimlfw.inference_service.engine import InferenceEngine
from smo.aimlfw.inference_service.errors import InferenceServiceError
from smo.aimlfw.inference_service.schemas import ParameterSetInput
from smo.aimlfw.inference_service.server import create_app
from smo.aimlfw.model_registry import ModelRegistry
from smo.aimlfw.model_storage import ModelStorage
from smo.aimlfw.training_manager.train_gnn import CellGraphGnn
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS, valid_cell_parameters, valid_parameter_sets

_IN_DIM = len(CONTROL_PARAMETER_NAMES) + 1  # +1 for the normalized Time_Step input (matches engine.py's own _IN_DIM)
_OUT_DIM = len(TARGET_KPIS)

_CELL_ID = st.text(
    alphabet="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-",
    min_size=1,
    max_size=16,
).filter(str.strip)

# One deliberately out-of-range/off-step value per Control_Parameter (mirrors the
# glossary ranges/allowed set in smo/aimlfw/common/constants.py), used to force
# known, exact violations rather than re-deriving validity with an independent
# oracle -- since every value drawn from this pool is invalid by construction,
# the expected violation set is simply "every parameter this test overrides".
_INVALID_VALUES: dict[str, SearchStrategy[float | int]] = {
    "tx_power_dbm": st.sampled_from((29.0, 30.5, 47.0)),
    "ret_tilt_deg": st.sampled_from((-1.0, 1.5, 16.0)),
    "cio_bias_db": st.sampled_from((-6.5, 0.25, 6.5)),
    "hysteresis_db": st.sampled_from((-0.5, 0.25, 10.5)),
    "ttt_ms": st.sampled_from((-1, 1, 200, 5121)),
}


class _FakeCursor:
    """Minimal in-memory stand-in for a PyMongo cursor (sort + limit only)."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents

    def sort(self, key: str, direction: int | None = None) -> "_FakeCursor":
        self.documents.sort(key=lambda item: item[key], reverse=direction == -1)
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


def _build_loaded_engine(root: Path) -> InferenceEngine:
    """Register and load one real (untrained, structurally valid) GNN artifact."""
    storage = ModelStorage(root / "models")
    model_name = "gnn-property-31"
    version = 1

    model = CellGraphGnn(_IN_DIM, _OUT_DIM)
    model.eval()
    buffer = io.BytesIO()
    torch.save({"state_dict": model.state_dict()}, buffer)
    artifact_uri = storage.save_artifact(model_name, version, buffer.getvalue())

    registry = ModelRegistry(_FakeCollection(), storage)
    registry.register(
        model_name,
        feature_group="default",
        metrics={"delay_p95_ms": 1.0},
        artifact_uri=artifact_uri,
    )

    engine = InferenceEngine(registry, model_name)
    asyncio.run(engine.load())
    if engine.model_load_status != "loaded":
        raise RuntimeError("test setup failed: InferenceEngine did not load the fixture model")
    return engine


@pytest.fixture(scope="module")
def engine() -> InferenceEngine:
    """One real, loaded InferenceEngine shared by every example in this module.

    The model weights are irrelevant to Property 31 (only whether a forward
    pass is even attempted matters), so building/loading it once per module
    keeps the Hypothesis examples fast without weakening the property.
    """
    with TemporaryDirectory() as raw_root:
        yield _build_loaded_engine(Path(raw_root))


def _valid_cell() -> dict[str, float | int]:
    return {"tx_power_dbm": 43.0, "ret_tilt_deg": 5.0, "cio_bias_db": 0.5, "hysteresis_db": 2.5, "ttt_ms": 160}


@st.composite
def _batches_with_violations(draw: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Generate a Batch_Prediction_Request payload with >=1 Control_Parameter
    violation, plus the exact (parameter_set_index, cell_id, parameter, value)
    violations it is constructed to contain.

    Every violating value comes from ``_INVALID_VALUES`` (known invalid by
    construction against the glossary ranges/allowed set), and every
    non-overridden value comes from ``valid_cell_parameters()`` (known valid),
    so the expected violation set can be computed directly from what this
    generator chose to override -- no separate range/step oracle is needed.
    """
    num_sets = draw(st.integers(min_value=1, max_value=6))
    violating_indices = set(
        draw(
            st.lists(
                st.integers(min_value=0, max_value=num_sets - 1),
                min_size=1,
                max_size=num_sets,
                unique=True,
            )
        )
    )

    parameter_sets: list[dict[str, Any]] = []
    expected_violations: list[dict[str, Any]] = []
    for index in range(num_sets):
        num_cells = draw(st.integers(min_value=1, max_value=3))
        cell_ids = draw(st.lists(_CELL_ID, min_size=num_cells, max_size=num_cells, unique=True))
        cells: dict[str, Any] = {}
        for cell_id in cell_ids:
            values = draw(valid_cell_parameters()).model_dump()
            if index in violating_indices and draw(st.booleans()):
                num_invalid = draw(st.integers(min_value=1, max_value=len(CONTROL_PARAMETER_NAMES)))
                invalid_names = draw(
                    st.lists(
                        st.sampled_from(CONTROL_PARAMETER_NAMES),
                        min_size=num_invalid,
                        max_size=num_invalid,
                        unique=True,
                    )
                )
                for name in invalid_names:
                    values[name] = draw(_INVALID_VALUES[name])
                    expected_violations.append(
                        {
                            "parameter_set_index": index,
                            "cell_id": cell_id,
                            "parameter": name,
                            "value": values[name],
                        }
                    )
            cells[cell_id] = values
        parameter_sets.append({"cells": cells})

    if not expected_violations:
        # Every violating index's coin flips landed on "stay valid" -- force
        # exactly one violation so the batch always has >=1, per this property.
        index = min(violating_indices)
        cell_id = next(iter(parameter_sets[index]["cells"]))
        value = draw(_INVALID_VALUES["ttt_ms"])
        parameter_sets[index]["cells"][cell_id]["ttt_ms"] = value
        expected_violations.append(
            {"parameter_set_index": index, "cell_id": cell_id, "parameter": "ttt_ms", "value": value}
        )
    return parameter_sets, expected_violations


def _violation_tuples(violations: list[dict[str, Any]]) -> set[tuple[int, str, str, float | int]]:
    return {
        (violation["parameter_set_index"], violation["cell_id"], violation["parameter"], violation["value"])
        for violation in violations
    }


# **Property 31: 파라미터 범위 위반 시 전체 거부**
# **Validates: Requirements 5.5**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(case=_batches_with_violations())
def test_batch_with_any_violation_is_rejected_atomically_with_every_violation_listed(
    engine: InferenceEngine, case: tuple[list[dict[str, Any]], list[dict[str, Any]]]
) -> None:
    parameter_sets, expected_violations = case
    inputs = [ParameterSetInput.model_validate(parameter_set) for parameter_set in parameter_sets]

    with pytest.raises(InferenceServiceError) as captured:
        engine.predict_batch(inputs, time_step=0)

    error = captured.value
    assert error.error_code == "parameter_out_of_range"

    actual_violations = error.details["violations"]
    # No violation is dropped, and none is duplicated: the counts and the
    # (index, cell_id, parameter, value) sets must match exactly.
    assert len(actual_violations) == len(expected_violations)
    assert _violation_tuples(actual_violations) == _violation_tuples(expected_violations)

    # The error body carries no partial/leaked prediction results.
    body = error.response.model_dump(mode="json")
    assert "predictions" not in body
    assert set(body) == {"error_code", "message", "details"}


# **Property 31: 파라미터 범위 위반 시 전체 거부**
# **Validates: Requirements 5.5**
@pytest.mark.property
def test_multi_index_multi_violation_batch_lists_every_violation_and_no_partial_predictions(
    engine: InferenceEngine,
) -> None:
    """Fixed example directly exercising the "all violations, not just first"
    atomicity claim: violations span two distinct Parameter_Set indices, and
    one of those indices carries two violations within the same cell, while a
    third, unrelated index is left fully valid."""
    parameter_sets: list[dict[str, Any]] = [
        {"cells": {"gNB_ok": _valid_cell()}},
        {"cells": {"gNB_bad": {**_valid_cell(), "tx_power_dbm": 47.0, "ttt_ms": 1}}},
        {"cells": {"gNB_other": {**_valid_cell(), "cio_bias_db": 6.5}}},
    ]
    expected = {
        (1, "gNB_bad", "tx_power_dbm", 47.0),
        (1, "gNB_bad", "ttt_ms", 1),
        (2, "gNB_other", "cio_bias_db", 6.5),
    }

    inputs = [ParameterSetInput.model_validate(parameter_set) for parameter_set in parameter_sets]
    with pytest.raises(InferenceServiceError) as captured:
        engine.predict_batch(inputs, time_step=0)

    error = captured.value
    assert error.error_code == "parameter_out_of_range"
    assert len(error.details["violations"]) == len(expected)
    assert _violation_tuples(error.details["violations"]) == expected

    # Same request through the real HTTP boundary: still one atomic error, still
    # every violation, still no "predictions" field leaking partial results.
    app = create_app(engine)
    with TestClient(app) as client:
        response = client.post("/predict/batch", json={"parameter_sets": parameter_sets, "time_step": 0})

    assert response.status_code == 422
    body = response.json()
    assert body["error_code"] == "parameter_out_of_range"
    assert "predictions" not in body
    assert _violation_tuples(body["details"]["violations"]) == expected


# **Property 31: 파라미터 범위 위반 시 전체 거부**
# **Validates: Requirements 5.5**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(parameter_set=valid_parameter_sets(min_cells=1, max_cells=3))
def test_fully_valid_batch_does_not_raise_parameter_out_of_range(
    engine: InferenceEngine, parameter_set: Any
) -> None:
    """Sanity/negative check: a batch with zero Control_Parameter violations
    must succeed normally, never with ``parameter_out_of_range``."""
    inputs = [ParameterSetInput.model_validate(parameter_set.model_dump())]

    predictions = engine.predict_batch(inputs, time_step=0)

    assert len(predictions) == 1
    assert predictions[0]["index"] == 0
    assert set(predictions[0]["target_kpi"]) == set(TARGET_KPIS)
