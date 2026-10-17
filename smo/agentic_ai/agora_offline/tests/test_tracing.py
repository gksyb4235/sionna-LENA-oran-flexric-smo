import asyncio
from unittest.mock import Mock

import pytest

from agora_offline import tracing


def test_public_payload_excludes_credentials_and_private_continuations():
    value = {
        "messages": [{"role": "assistant", "content": "public decision",
                      "_response_output": [{"type": "reasoning", "text": "private"}]}],
        "api_key": "secret", "Authorization": "Bearer private",
        "nested": {"OPENAI_API_KEY": "secret", "encrypted_content": "opaque"},
        "error": "token lsv2_pt_example_key and sk-example_key",
    }
    public = tracing.public_payload(value)
    assert public["messages"] == [{"role": "assistant", "content": "public decision"}]
    assert public["api_key"] == public["Authorization"] == "[redacted]"
    assert public["nested"] == {"OPENAI_API_KEY": "[redacted]"}
    assert public["error"] == "token [redacted] and [redacted]"


def test_env_file_and_explicit_process_disable(tmp_path, monkeypatch):
    (tmp_path / ".env").write_text(
        "LANGSMITH_TRACING_V2=true\nLANGSMITH_API_KEY=private\nLANGSMITH_PROJECT=AGORA\n"
    )
    monkeypatch.setenv("AGENTIC_AI_ROOT", str(tmp_path))
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_TRACING_V2", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    assert tracing.configuration()[0] is True
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    assert tracing.configuration()[0] is False


@pytest.fixture
def observed(monkeypatch):
    from langsmith import Client
    created, updated = [], []
    # Exercise the real SDK context propagation without any network transport.
    monkeypatch.setattr(Client, "create_run", lambda self, *a, **kw: created.append(kw))
    monkeypatch.setattr(Client, "update_run", lambda self, *a, **kw: updated.append(kw))
    client = Client(api_url="http://127.0.0.1:1", api_key="test", auto_batch_tracing=False)
    instance = tracing.Observer(False, "http://127.0.0.1:1", "test", "unit-test")
    instance.client = client
    return instance, created, updated


def test_async_sessions_have_separate_roots_and_nested_children(observed):
    instance, created, _ = observed

    async def session(label):
        with instance.span(label, root=True) as root:
            await asyncio.sleep(0)
            with instance.span(label + " tool", run_type="tool") as tool:
                tool.finish({"ok": True})
            root.finish({"ok": True})

    async def run():
        await asyncio.gather(session("A"), session("B"))

    asyncio.run(run())
    rows = {row["name"]: row for row in created}
    assert set(rows) == {"A", "B", "A tool", "B tool"}
    assert rows["A tool"]["parent_run_id"] == rows["A"]["id"]
    assert rows["B tool"]["parent_run_id"] == rows["B"]["id"]
    assert rows["A"]["trace_id"] != rows["B"]["trace_id"]


def test_operation_error_is_preserved_and_trace_error_redacted(observed):
    instance, _, updated = observed
    with pytest.raises(ValueError, match="lsv2_pt_private"):
        with instance.span("failure"):
            raise ValueError("lsv2_pt_private")
    assert any(row.get("error") == "ValueError: [redacted]" for row in updated)


def test_disabled_and_failing_tracing_do_not_change_result(monkeypatch):
    instance = tracing.Observer(False, "", "", "test")
    with instance.span("disabled") as span:
        assert span.identity() == {"enabled": False}
        span.finish({"value": 7})
    from langsmith import Client
    monkeypatch.setattr(Client, "__init__", Mock(side_effect=RuntimeError("secret")))
    instance = tracing.Observer(True, "http://localhost", "private", "test")
    with instance.span("failed setup"):
        result = 7
    assert result == 7
    assert instance.public_status()["last_error"] == "RuntimeError"
    assert "api_key" not in instance.public_status()
    assert instance.public_status()["configured"] is True
