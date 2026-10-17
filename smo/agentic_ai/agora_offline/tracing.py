"""Optional LangSmith observation of public decisions and numerical evidence."""

from contextlib import ExitStack, contextmanager
from functools import lru_cache
import importlib.util
import logging
import os
from pathlib import Path
import re

from dotenv import dotenv_values
from .settings import ROOT

LOG = logging.getLogger(__name__)
SECRET_KEYS = {"key", "api_key", "authorization", "password", "access_token"}
PRIVATE_KEYS = {"_response_output", "encrypted_content", "reasoning"}
SECRET_TEXT = re.compile(r"(?:lsv2_[A-Za-z0-9_-]+|sk-[A-Za-z0-9_-]+|Bearer\s+\S+)")


def public_payload(value):
    """Never export credentials or opaque provider continuation/reasoning items."""
    if isinstance(value, dict):
        if value.get("type") == "reasoning":
            return {"type": "omitted_private_provider_item"}
        return {
            str(k): ("[redacted]" if str(k).lower() in SECRET_KEYS
                     or str(k).lower().endswith("api_key") else public_payload(v))
            for k, v in value.items() if str(k).lower() not in PRIVATE_KEYS
        }
    if isinstance(value, (list, tuple)):
        return [public_payload(item) for item in value]
    if isinstance(value, str):
        return SECRET_TEXT.sub("[redacted]", value)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return public_payload(str(value))


def configuration():
    parent = Path(os.environ.get("AGENTIC_AI_ROOT", ROOT.parent))
    env = {k: v for k, v in dotenv_values(parent / ".env").items() if v}
    env.update(os.environ)
    enabled = env.get("LANGSMITH_TRACING", env.get("LANGSMITH_TRACING_V2", "false"))
    return (
        str(enabled).lower() in {"true", "1", "yes"},
        env.get("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").rstrip("/"),
        env.get("LANGSMITH_API_KEY", ""),
        env.get("LANGSMITH_PROJECT", "AGORA"),
    )


class Span:
    def __init__(self, run=None, owner=None):
        self.run, self.owner = run, owner

    def finish(self, outputs=None, error=None, metadata=None):
        if self.run is not None:
            try:
                self.run.end(outputs=public_payload(outputs), error=public_payload(error),
                             metadata=public_payload(metadata))
            except Exception as exc:
                self.owner.record_error(exc)

    def identity(self):
        if self.run is None:
            return {"enabled": False}
        return {"enabled": True, "project": self.owner.project,
                "run_id": str(self.run.id), "trace_id": str(self.run.trace_id)}


class Observer:
    def __init__(self, enabled, endpoint, key, project):
        self.enabled, self.endpoint, self.project = enabled, endpoint, project
        self.configured = bool(key)
        self.client = None
        self.last_error = None
        self.error_count = 0
        self.sdk_available = importlib.util.find_spec("langsmith") is not None
        if enabled and key and self.sdk_available:
            try:
                from langsmith import Client
                self.client = Client(
                    api_url=endpoint, api_key=key, timeout_ms=10000,
                    hide_inputs=public_payload, hide_outputs=public_payload,
                    hide_metadata=public_payload, omit_traced_runtime_info=True,
                    tracing_error_callback=self.record_error,
                )
            except Exception as exc:
                self.record_error(exc)

    def record_error(self, exc):
        # SDK errors must not leak credentials or break the planning workflow.
        self.last_error = type(exc).__name__
        self.error_count += 1
        LOG.warning("LangSmith observation failed (%s)", self.last_error)

    def public_status(self):
        return {
            "enabled": self.enabled, "configured": self.configured,
            "sdk_available": self.sdk_available, "active": self.client is not None,
            "project": self.project, "endpoint": self.endpoint,
            "last_error": self.last_error, "error_count": self.error_count,
            "scope": "public prompts, summaries, tool inputs/results and usage; no private reasoning",
        }

    @contextmanager
    def span(self, name, *, inputs=None, metadata=None, run_type="chain", root=False):
        if self.client is None:
            yield Span()
            return
        stack = ExitStack()
        try:
            from langsmith import trace, tracing_context
            options = dict(enabled=True, client=self.client, project_name=self.project)
            if root:
                options["parent"] = False
            stack.enter_context(tracing_context(**options))
            run = stack.enter_context(trace(
                name, run_type=run_type, inputs=public_payload(inputs or {}),
                metadata=public_payload(metadata or {}), tags=["agora", "offline"],
                client=self.client, project_name=self.project,
            ))
            span = Span(run, self)
        except Exception as exc:
            self.record_error(exc)
            stack.close()
            span = Span()
        try:
            yield span
        except BaseException as exc:
            span.finish(error=f"{type(exc).__name__}: {public_payload(str(exc))}")
            raise
        finally:
            try:
                # End with our redacted error instead of SDK traceback serialization.
                stack.close()
            except Exception as exc:
                self.record_error(exc)

    def flush(self):
        if self.client is not None:
            try:
                self.client.flush(timeout=10)
            except Exception as exc:
                self.record_error(exc)


@lru_cache(maxsize=4)
def _observer(enabled, endpoint, key, project):
    return Observer(enabled, endpoint, key, project)


def observer():
    return _observer(*configuration())
