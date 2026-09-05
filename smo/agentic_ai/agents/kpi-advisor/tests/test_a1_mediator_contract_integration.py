"""Task 9.11: contract test against the repository's actual A1 Mediator ASGI app."""

from __future__ import annotations

import asyncio
import importlib.util
import re
import sys
from pathlib import Path
from typing import Any

import httpx

TEST_DIR = Path(__file__).resolve().parent
AGENT_DIR = TEST_DIR.parent
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path[:0] = [str(REPO_ROOT), str(AGENT_DIR), str(TEST_DIR)]

from _policy_test_support import temporal_plan  # noqa: E402

import policy_manager  # noqa: E402


class _SyncAsgiClient:
    def __init__(self, app: Any, calls: list[tuple[str, str]]) -> None:
        self.app = app
        self.calls = calls

    def __enter__(self) -> _SyncAsgiClient:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        self.calls.append((method, path))

        async def send() -> httpx.Response:
            transport = httpx.ASGITransport(app=self.app)
            async with httpx.AsyncClient(transport=transport, base_url="http://a1") as client:
                return await client.request(method, path, **kwargs)

        return asyncio.run(send())


def _load_actual_a1_module() -> Any:
    path = REPO_ROOT / "A1_Mediator_standalone" / "A1_Mediator" / "app" / "main.py"
    spec = importlib.util.spec_from_file_location("contract_test_a1_main", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.policy_types.clear()
    module.policy_instances.clear()
    return module


def _is_allowed_call(method: str, path: str) -> bool:
    patterns = {
        ("GET", r"/a1-p/policytypes"),
        ("GET", r"/a1-p/policytypes/\d+"),
        ("PUT", r"/a1-p/policytypes/\d+"),
        ("DELETE", r"/a1-p/policytypes/\d+"),
        ("PUT", r"/a1-p/policytypes/\d+/policies/[^/]+"),
        ("GET", r"/a1-p/policytypes/\d+/policies/[^/]+/status"),
        ("GET", r"/a1-p/policytypes/\d+/policies"),
        ("DELETE", r"/a1-p/policytypes/\d+/policies/[^/]+"),
    }
    return any(method == allowed_method and re.fullmatch(pattern, path) for allowed_method, pattern in patterns)


def test_actual_a1_accepts_flat_integer_schema_and_exactly_five_temporal_instances() -> None:
    module = _load_actual_a1_module()
    calls: list[tuple[str, str]] = []
    policy_manager.set_client_factory(lambda: _SyncAsgiClient(module.app, calls))
    policy_manager.clear_publish_attempts()
    try:
        assert policy_manager.register_policy_type() == 20100
        assert policy_manager.register_policy_type() == 20100
        properties = module.policy_types[20100].create_schema.properties
        assert len(properties) == 15
        assert all(item["type"] == "integer" for item in properties.values())

        results = policy_manager.publish_temporal_plan(temporal_plan())
        assert isinstance(results, list)
        assert len(results) == 5
        assert all(result.status == "enforced" for result in results)
        assert len(module.policy_instances[20100]) == 5
        assert all(len(item["data"]) == 15 for item in module.policy_instances[20100].values())
        assert all(
            type(value) is int for item in module.policy_instances[20100].values() for value in item["data"].values()
        )
        assert all(_is_allowed_call(method, path) for method, path in calls)
    finally:
        policy_manager.reset_client_factory()
        policy_manager.clear_publish_attempts()
