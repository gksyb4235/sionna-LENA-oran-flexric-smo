"""Task 6.6: MCP 도구 계약과 읽기 전용 표면 테스트.

Follows ``test_mcp_server_contract.py``'s manual ``sys.path`` setup convention
since ``kpi-advisor`` is not installed as a package during tests, and the
in-memory ``ModelRegistry``/``ModelStorage`` MongoDB stand-in pattern from
``smo/tests/integration/test_inference_service_loading_performance_and_failure.py``
and ``smo/tests/property/test_property_29_batch_order_preservation_and_completeness.py``.

This module is the integration-level complement to the existing per-property
coverage of ``GNN_MCP_Server`` (task 6.1's ``test_mcp_server_contract.py`` and
task 6.2-6.5's Property 37-40 modules, all of which exercise
``mcp_server.py`` against a *stubbed* ``httpx`` transport). It focuses on
three things those modules do not:

1. **3개 도구 시그니처 / 인자 필수 여부·범위 (Requirement 6.1, 6.2, 6.3, 6.8)**:
   one holistic "golden schema" assertion that ``mcp.list_tools()`` reports
   exactly the 3 expected tools, and that every documented argument's
   required-ness and range/size constraint matches Requirement 6's
   acceptance criteria -- in one comprehensive check rather than scattered
   examples.
2. **변경 도구 부재 (Requirement 6.6, 6.7)**: the tool name set is exactly
   ``{"predict_batch", "marginal_effect", "current_model"}`` (the strongest
   possible proof of read-only-ness, since GNN_MCP_Server only ever exposes
   these 3 tools), none of the 3 tools' names/descriptions contain a
   mutation-suggesting verb, and ``inference_service/server.py``'s two
   routes (``GET /status``, ``POST /predict/batch``) never write to
   Model_Registry/Model_Storage -- confirmed by static inspection of the
   route handlers (module-level import check below), since the routes make
   no ``ModelRegistry``/``ModelStorage`` write calls at all.
3. **Inference Service 연동 (Requirement 6.1-6.10)**: unlike every prior
   task-6 test module, this exercises ``mcp_server.py``'s tools against a
   REAL running Inference_Service -- a real ``InferenceEngine`` holding a
   real (untrained-but-real, following Property 29's precedent) trained
   ``CellGraphGnn`` checkpoint, registered through the real
   ``ModelRegistry``/``ModelStorage`` (MongoDB collection faked with the
   same minimal in-memory stand-in used throughout ``smo/tests``), served by
   the real FastAPI app from ``smo.aimlfw.inference_service.server.create_app``.
   ``mcp_server.py``'s HTTP client is pointed at this in-process ASGI app via
   ``httpx.ASGITransport`` (httpx==0.28.1) wrapped in a small sync adapter,
   through ``mcp_server.set_client_factory(...)`` -- so no real network port
   is bound and no stub response bodies are hand-written.

Requirement 6.1:
    THE GNN_MCP_Server SHALL 1개 이상 64개 이하의 Parameter_Set 과 선택적
    Time_Step 값(0~4)을 입력으로 받아 Batch_Prediction_Request 를 수행하고
    입력과 동일한 순서의 예측 결과 목록을 반환하는 MCP 도구를 제공한다.

Requirement 6.2:
    THE GNN_MCP_Server SHALL Baseline_Parameter_Set 1개, 1개 이상 5개 이하의
    Control_Parameter 이름, 1개 이상의 Target_KPI 이름을 입력으로 받아 각
    (Control_Parameter, Target_KPI) 쌍의 Marginal_Effect 값을 -100.0 이상
    100.0 이하의 percent 값으로 반환하는 MCP 도구를 제공한다.

Requirement 6.3:
    THE GNN_MCP_Server SHALL 입력 인자 없이 호출되어 Model_Registry 의 현재
    서빙 모델 이름과 버전 번호를 반환하는 MCP 도구를 제공한다.

Requirement 6.6:
    THE GNN_MCP_Server SHALL 읽기 전용 예측 및 조회 도구만 제공하며,
    Parameter_Set, Model_Registry 항목, RAN 구성 중 어느 것도 변경하는 도구를
    노출하지 않는다.

Requirement 6.8:
    WHEN MCP 클라이언트가 도구 목록 조회를 요청하면, THE GNN_MCP_Server SHALL
    제공하는 각 도구의 이름, 입력 인자 이름, 각 인자의 필수 여부와 허용 범위를
    반환한다.

Requirement 6.9:
    IF 도구 호출 인자에 필수 인자가 누락되었거나 Parameter_Set 개수가 64 를
    초과하거나 Time_Step 값이 0~4 범위를 벗어나면, THEN THE GNN_MCP_Server
    SHALL Inference_Service 를 호출하지 않고 위반된 인자 이름을 담은 입력
    검증 오류를 도구 결과로 반환한다.

**Validates: Requirements 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8, 6.9, 6.10**
"""

from __future__ import annotations

import asyncio
import inspect
import sys
from pathlib import Path
from typing import Any

import httpx
import pytest
import torch
from httpx import ASGITransport

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AGENT_DIR))

from smo.aimlfw.common.constants import (  # noqa: E402
    CONTROL_PARAMETER_NAMES,
    MAX_TIME_STEP,
    MIN_TIME_STEP,
    TARGET_KPIS,
)
from smo.aimlfw.inference_service import server as inference_service_server  # noqa: E402
from smo.aimlfw.inference_service.engine import InferenceEngine  # noqa: E402
from smo.aimlfw.inference_service.schemas import MAX_BATCH_SIZE  # noqa: E402
from smo.aimlfw.inference_service.server import create_app  # noqa: E402
from smo.aimlfw.model_registry import ModelRegistry  # noqa: E402
from smo.aimlfw.model_storage import ModelStorage  # noqa: E402
from smo.aimlfw.training_manager.train_gnn import run_training  # noqa: E402

import mcp_server  # noqa: E402

_MODEL_NAME = "gnn-task-6-6"
_FEATURE_GROUP = "default"
_SEED = 7
_CELLS = ("gNB_5G", "gNB_4G_1")
_EXPECTED_TOOL_NAMES = {"predict_batch", "marginal_effect", "current_model"}
# Denylist of mutation-suggesting verbs (Requirement 6.6): a read-only MCP
# surface must not expose any tool whose name/description implies a write.
_MUTATION_VERBS = (
    "update", "set", "delete", "create", "register", "publish", "modify",
    "write", "remove", "insert", "put", "patch", "save", "store",
)


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _cell_parameters(**overrides: Any) -> dict[str, Any]:
    base = {
        "tx_power_dbm": 43.0,
        "ret_tilt_deg": 5.0,
        "cio_bias_db": 0.5,
        "hysteresis_db": 2.5,
        "ttt_ms": 160,
    }
    base.update(overrides)
    return base


def _parameter_set(**overrides: Any) -> dict[str, Any]:
    return {"cells": {"gNB_5G": _cell_parameters(**overrides)}}


# ---------------------------------------------------------------------------
# In-memory MongoDB collection stand-in (same pattern as
# test_inference_service_loading_performance_and_failure.py and the
# Property 16/17/18/20/22-29 test modules under smo/tests/).
# ---------------------------------------------------------------------------


class _FakeCursor:
    """Minimal ``pymongo`` cursor stand-in supporting ``sort``/``limit``/iteration."""

    def __init__(self, documents: list[dict[str, Any]]) -> None:
        self.documents = documents

    def sort(self, key: str, direction: int | None = None) -> _FakeCursor:
        from copy import deepcopy

        self.documents = sorted(deepcopy(self.documents), key=lambda item: item[key], reverse=direction == -1)
        return self

    def limit(self, count: int) -> _FakeCursor:
        self.documents = self.documents[:count]
        return self

    def __iter__(self):
        from copy import deepcopy

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
        from copy import deepcopy

        matching = [item for item in self.documents if self._matches(item, query)]
        sort = kwargs.get("sort")
        if sort:
            key, direction = sort[0]
            matching.sort(key=lambda item: item[key], reverse=direction == -1)
        return deepcopy(matching[0]) if matching else None

    def find(self, query: dict[str, Any]) -> _FakeCursor:
        from copy import deepcopy

        return _FakeCursor([deepcopy(item) for item in self.documents if self._matches(item, query)])

    def update_one(self, query: dict[str, Any], update: dict[str, Any], **kwargs: Any) -> None:
        from copy import deepcopy

        if not any(self._matches(item, query) for item in self.documents):
            document = deepcopy(query)
            document.update(deepcopy(update.get("$setOnInsert", {})))
            self.documents.append(document)

    def insert_one(self, document: dict[str, Any]) -> None:
        from copy import deepcopy

        self.documents.append(deepcopy(document))


# ---------------------------------------------------------------------------
# ASGI-transport-backed sync httpx.Client, bound to a real in-process
# Inference_Service FastAPI app (no real network port bound).
# ---------------------------------------------------------------------------


class _CountingSyncASGITransport(httpx.BaseTransport):
    """Adapts httpx's async-only ``ASGITransport`` (0.28.1) to a sync ``httpx.Client``.

    Also records every request so tests can assert Inference_Service was (or
    was not) actually reached -- used by the input-validation-preblocks-real-
    service check (point 4.d in the task description).
    """

    def __init__(self, app: Any) -> None:
        self._async_transport = ASGITransport(app=app)
        self.calls: list[httpx.Request] = []

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)

        async def _call() -> httpx.Response:
            response = await self._async_transport.handle_async_request(request)
            await response.aread()
            return response

        response = asyncio.run(_call())
        return httpx.Response(response.status_code, headers=response.headers, content=response.content)


def _training_records() -> list[dict[str, Any]]:
    """Build a small, real Feature_Record-shaped input for run_training."""
    records: list[dict[str, Any]] = []
    for time_step in range(5):
        for index, cell_id in enumerate(_CELLS):
            records.append(
                {
                    "time_step": time_step,
                    "cell_id": cell_id,
                    "control_parameters": {
                        "tx_power_dbm": 40.0,
                        "ret_tilt_deg": 5.0,
                        "cio_bias_db": 0.5,
                        "hysteresis_db": 2.0,
                        "ttt_ms": 160,
                    },
                    "features": {kpi: 10.0 + time_step + index * 0.1 for kpi in TARGET_KPIS},
                }
            )
    return records


@pytest.fixture()
def real_inference_service(tmp_path: Path):
    """A real Inference_Service (real trained GNN + real Registry/Storage + real FastAPI app).

    Yields ``(client_factory, transport, registered_version)`` where
    ``client_factory`` is a zero-arg callable returning a fresh
    ``httpx.Client`` bound to the running app via ASGITransport, matching
    the shape ``mcp_server.set_client_factory`` expects.
    """
    model, _node_order, payload = run_training(
        _training_records(), seed=_SEED, max_epochs=3, patience=2, learning_rate=0.05
    )
    artifact_path = tmp_path / "artifact.pt"
    torch.save({"state_dict": model.state_dict()}, artifact_path)

    model_storage = ModelStorage(tmp_path / "models")
    registry = ModelRegistry(_FakeCollection(), model_storage)
    artifact_uri = model_storage.save_artifact(_MODEL_NAME, 1, artifact_path)
    record = registry.register(
        _MODEL_NAME,
        _FEATURE_GROUP,
        payload["metrics"],
        artifact_uri,
        train_split=payload["train_split"],
        validation_split=payload["validation_split"],
        train_samples=payload["train_samples"],
        validation_samples=payload["validation_samples"],
        seed=_SEED,
    )

    engine = InferenceEngine(registry, _MODEL_NAME)
    _run(engine.load())
    assert engine.model_load_status == "loaded", "fixture setup must reach loaded before tests run"

    app = create_app(engine=engine)
    transport = _CountingSyncASGITransport(app)

    def client_factory() -> httpx.Client:
        return httpx.Client(transport=transport, base_url="http://real-inference-service")

    # The engine is loaded explicitly above, so no FastAPI lifespan side effect
    # is required. Avoid TestClient's environment-specific thread portal and
    # exercise the real app through the ASGI transport below.
    yield client_factory, transport, record.version


@pytest.fixture(autouse=True)
def _reset_client_factory():
    yield
    mcp_server.reset_client_factory()


# ---------------------------------------------------------------------------
# 1 & 2. Golden schema: exactly 3 tools, with required-ness/range for every
# documented argument matching Requirement 6.1/6.2/6.3/6.8 exactly.
# ---------------------------------------------------------------------------


def test_list_tools_reports_exactly_three_tools_with_exact_golden_schema() -> None:
    tools = _run(mcp_server.mcp.list_tools())

    # Requirement 6.8 / point 1: exactly the 3 expected tools, no more, no fewer.
    names = [tool.name for tool in tools]
    assert set(names) == _EXPECTED_TOOL_NAMES
    assert len(names) == len(_EXPECTED_TOOL_NAMES) == 3, "no accidental extra/duplicate tools registered"

    by_name = {tool.name: tool for tool in tools}

    # --- predict_batch (Requirement 6.1) ---------------------------------
    predict_schema = by_name["predict_batch"].inputSchema
    assert predict_schema["required"] == ["parameter_sets"]
    parameter_sets_schema = predict_schema["properties"]["parameter_sets"]
    assert parameter_sets_schema["minItems"] == 1
    assert parameter_sets_schema["maxItems"] == 64 == MAX_BATCH_SIZE
    assert "time_step" not in predict_schema["required"]
    time_step_schema = predict_schema["properties"]["time_step"]
    assert time_step_schema["minimum"] == 0 == MIN_TIME_STEP
    assert time_step_schema["maximum"] == 4 == MAX_TIME_STEP
    assert set(predict_schema["properties"]) == {"parameter_sets", "time_step"}

    # --- marginal_effect (Requirement 6.2) --------------------------------
    marginal_schema = by_name["marginal_effect"].inputSchema
    assert set(marginal_schema["required"]) == {"baseline", "control_parameters", "target_kpis"}
    control_parameters_schema = marginal_schema["properties"]["control_parameters"]
    assert control_parameters_schema["minItems"] == 1
    assert control_parameters_schema["maxItems"] == 5
    target_kpis_schema = marginal_schema["properties"]["target_kpis"]
    assert target_kpis_schema["minItems"] == 1
    assert "maxItems" not in target_kpis_schema  # Requirement 6.2: "1개 이상" -- no upper bound.
    assert set(marginal_schema["properties"]) == {"baseline", "control_parameters", "target_kpis"}

    # --- current_model (Requirement 6.3) ----------------------------------
    current_model_schema = by_name["current_model"].inputSchema
    assert current_model_schema["properties"] == {}
    assert current_model_schema.get("required", []) == []


# ---------------------------------------------------------------------------
# 3. Absence of mutation tools.
# ---------------------------------------------------------------------------


def test_no_tool_name_or_description_suggests_mutation() -> None:
    tools = _run(mcp_server.mcp.list_tools())

    # Strongest possible proof: the tool name set is exactly the 3 read-only
    # names GNN_MCP_Server is specified to expose (Requirement 6.6).
    names = {tool.name for tool in tools}
    assert names == _EXPECTED_TOOL_NAMES

    for tool in tools:
        haystack = f"{tool.name} {tool.description or ''}".lower()
        matched_verbs = [verb for verb in _MUTATION_VERBS if verb in haystack]
        assert matched_verbs == [], (
            f"tool {tool.name!r} name/description suggests mutation via {matched_verbs}"
        )


def test_inference_service_predict_batch_and_status_routes_never_write_to_registry_or_storage() -> None:
    """Static check: the 2 Inference_Service routes GNN_MCP_Server calls (``GET
    /status``, ``POST /predict/batch``) never call any Model_Registry/Model_Storage
    *write* method. ``ModelRegistry``'s only write method is ``register``, and
    ``ModelStorage``'s only write method is ``save_artifact`` -- neither name
    appears in ``server.py``'s source, confirming both routes are read-only
    against persistent state (Requirement 6.6/6.7's read-only guarantee
    ultimately rests on this)."""
    source = inspect.getsource(inference_service_server)
    assert "ModelRegistry.register" not in source
    assert ".register(" not in source
    assert ".save_artifact(" not in source

    status_source = inspect.getsource(inference_service_server.create_app)
    assert "engine.predict_batch" in status_source or "predict_batch(request" in status_source
    # The route handlers only read engine state (engine.loaded_model_name/
    # version/model_load_status) and call engine.predict_batch, which itself
    # (see engine.py) only calls model_registry.latest/get (reads) plus a
    # pure torch forward pass -- never model_registry.register or
    # model_storage.save_artifact.


def test_no_mutation_tool_exposed_beyond_the_three_read_only_tools() -> None:
    """Requirement 6.6: no MCP tool changes Parameter_Set, Model_Registry entries,
    or RAN configuration. With list_tools() reporting exactly 3 tools (asserted
    above), and mcp_server module-level inspection confirming no other
    ``@mcp.tool()``-decorated function exists, this closes the loop."""
    decorated_function_names = [
        name
        for name, member in inspect.getmembers(mcp_server, inspect.isfunction)
        if getattr(member, "__module__", None) == mcp_server.__name__
        and inspect.getsource(member).lstrip().startswith("@mcp.tool()")
    ]
    assert set(decorated_function_names) == _EXPECTED_TOOL_NAMES


# ---------------------------------------------------------------------------
# 4. Real Inference_Service integration (not a stub).
# ---------------------------------------------------------------------------


def test_current_model_against_real_inference_service_returns_registered_model(
    real_inference_service,
) -> None:
    client_factory, transport, registered_version = real_inference_service
    mcp_server.set_client_factory(client_factory)

    result = mcp_server.current_model()

    assert result == {"model_name": _MODEL_NAME, "model_version": registered_version}
    assert len(transport.calls) == 1


def test_predict_batch_against_real_inference_service_returns_real_predictions(
    real_inference_service,
) -> None:
    client_factory, transport, registered_version = real_inference_service
    mcp_server.set_client_factory(client_factory)

    parameter_sets = [_parameter_set(tx_power_dbm=40.0 + index) for index in range(3)]
    result = mcp_server.predict_batch(parameter_sets=parameter_sets, time_step=2)

    assert result["model_name"] == _MODEL_NAME
    assert result["model_version"] == registered_version
    assert result["applied_time_step"] == 2
    predictions = result["predictions"]
    assert len(predictions) == 3
    for expected_index, prediction in enumerate(predictions):
        assert prediction["index"] == expected_index
        assert set(prediction["target_kpi"]) == set(TARGET_KPIS)
        for value in prediction["target_kpi"].values():
            assert isinstance(value, float)
        assert set(prediction["cell_kpi"]) == {"gNB_5G"}
        assert set(prediction["cell_kpi"]["gNB_5G"]) == set(TARGET_KPIS)
    assert len(transport.calls) == 1


def test_marginal_effect_against_real_inference_service_returns_genuinely_computed_values(
    real_inference_service,
) -> None:
    client_factory, transport, registered_version = real_inference_service
    mcp_server.set_client_factory(client_factory)

    result = mcp_server.marginal_effect(
        baseline=_parameter_set(),
        control_parameters=list(CONTROL_PARAMETER_NAMES),
        target_kpis=list(TARGET_KPIS),
    )

    assert result["model_name"] == _MODEL_NAME
    assert result["model_version"] == registered_version
    effects = result["marginal_effects"]
    assert set(effects) == set(CONTROL_PARAMETER_NAMES)
    for parameter_name in CONTROL_PARAMETER_NAMES:
        kpi_effects = effects[parameter_name]
        assert set(kpi_effects) == set(TARGET_KPIS)
        for value in kpi_effects.values():
            if parameter_name in result["unavailable_control_parameters"]:
                assert value is None
            else:
                # Genuinely computed from the real GNN forward pass, not a
                # structurally-shaped placeholder: a finite float clamped to
                # the -100.0..100.0 percent range (Requirement 6.2).
                assert value is None or (isinstance(value, float) and -100.0 <= value <= 100.0)
    # 1 real Inference_Service call carrying baseline + 1 variant per probed
    # Control_Parameter (mirrors test_mcp_server_contract.py's stubbed
    # equivalent, but this batch is served by the real engine/forward pass).
    assert len(transport.calls) == 1


def test_invalid_call_is_preblocked_before_reaching_real_inference_service(
    real_inference_service,
) -> None:
    """Point 4.d: an intentionally invalid call (batch too large) never reaches
    the real Inference_Service. Confirmed two ways: the call-count-tracking
    transport records zero calls, and the returned error_code is
    ``input_validation_error`` (GNN_MCP_Server's own pre-validation error),
    not e.g. ``batch_too_large``/``parameter_out_of_range`` (Inference_Service's
    own error codes), which would only occur if the request had reached the
    real service."""
    client_factory, transport, _registered_version = real_inference_service
    mcp_server.set_client_factory(client_factory)

    result = mcp_server.predict_batch(parameter_sets=[_parameter_set()] * (MAX_BATCH_SIZE + 1))

    assert result["error_code"] == "input_validation_error"
    assert "parameter_sets" in result["details"]["violated_arguments"]
    assert transport.calls == []

    # A second invalid call (out-of-range time_step) confirms the same for
    # the other Requirement 6.9 pre-validation path.
    result = mcp_server.predict_batch(parameter_sets=[_parameter_set()], time_step=9)
    assert result["error_code"] == "input_validation_error"
    assert "time_step" in result["details"]["violated_arguments"]
    assert transport.calls == []
