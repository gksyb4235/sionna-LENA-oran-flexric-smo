"""Stream public assistant content and function calls. Credentials stay server-side."""

import json
import os
from pathlib import Path
import httpx
from dotenv import dotenv_values
from .settings import ROOT, INVARIANTS


def configuration(settings=None):
    merged = {}
    parent = Path(os.environ.get("AGENTIC_AI_ROOT", ROOT.parent))
    for path in (
        parent / "agents/decomposition/.env",
        parent / "agents/planning/.env",
        parent / ".env",
    ):
        if path.is_file():
            merged.update({k: v for k, v in dotenv_values(path).items() if v})
    merged.update(os.environ)
    model = (
        (settings or {}).get("provider", {}).get("model")
        or merged.get("AGORA_MODEL")
        or merged.get("OPENAI_MODEL", "gpt-5-mini")
    )
    if model.startswith("openai:"):
        model = model.split(":", 1)[1]
    return {
        "key": merged.get("OPENAI_API_KEY", ""),
        "model": model,
        "base_url": merged.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/"),
    }


def public_configuration(settings=None):
    conf = configuration(settings)
    return {
        "configured": bool(conf["key"]),
        "model": conf["model"],
        "credential_source": "existing Agentic AI environment; server only",
        "api": "responses",
        "streaming": True,
        "reasoning_display": "public decision summaries and tool evidence",
    }


def function(name, description, properties, required=()):
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(required),
                "additionalProperties": False,
            },
        },
    }


class Provider:
    def __init__(self, settings):
        self.settings = settings
        self.conf = configuration(settings)

    async def complete(self, role, messages, emit, tools=None, forced=None):
        if not self.conf["key"]:
            raise ValueError(
                "OPENAI_API_KEY is not configured in the existing Agentic AI environment; choose deterministic mode or configure the server environment"
            )
        await emit("agent_start", role, f"{role} agent", {"model": self.conf["model"]})
        inputs = []
        for message in messages:
            if message.get("_response_output") is not None:
                # Opaque provider continuation items stay in memory only. They
                # are not rendered, logged or included in another agent's context.
                inputs.extend(message["_response_output"])
            elif message["role"] == "tool":
                inputs.append(
                    {
                        "type": "function_call_output",
                        "call_id": message["tool_call_id"],
                        "output": message["content"],
                    }
                )
            else:
                if message.get("content"):
                    inputs.append({"role": message["role"], "content": message["content"]})
                for call in message.get("tool_calls", []):
                    inputs.append(
                        {"type": "function_call", "call_id": call["id"], **call["function"]}
                    )
        body = {
            "model": self.conf["model"],
            "input": inputs,
            "instructions": INVARIANTS + "\n" + self.settings["prompts"][role],
            "stream": True,
            "store": False,
            "max_output_tokens": 6000,
        }
        if self.conf["model"].startswith(("gpt-5", "o3", "o4")):
            body["reasoning"] = {"effort": "low"}
        if tools:
            body.update(
                tools=[{"type": "function", **tool["function"], "strict": False} for tool in tools],
                parallel_tool_calls=False,
            )
            if forced:
                body["tool_choice"] = {"type": "function", "name": forced}
        pieces, output, completed = [], [], False
        async with httpx.AsyncClient(timeout=httpx.Timeout(180, connect=20)) as client:
            async with client.stream(
                "POST",
                self.conf["base_url"] + "/responses",
                headers={"Authorization": "Bearer " + self.conf["key"]},
                json=body,
            ) as response:
                if response.status_code >= 400:
                    await response.aread()
                    try:
                        message = (
                            response.json()
                            .get("error", {})
                            .get("message", "Provider request failed")
                        )
                    except (ValueError, AttributeError):
                        message = "Provider request failed"
                    message = str(message).replace(self.conf["key"], "[redacted]")[:600]
                    raise RuntimeError(f"LLM HTTP {response.status_code}: {message}")
                async for line in response.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        break
                    item = json.loads(payload)
                    kind = item.get("type")
                    if kind in ("error", "response.failed", "response.incomplete"):
                        error = (
                            item.get("response", {}).get("error")
                            or item.get("error")
                            or item.get("response", {}).get("incomplete_details")
                        )
                        raise RuntimeError("LLM response did not complete: " + str(error)[:500])
                    if kind == "response.output_text.delta":
                        delta = item.get("delta", "")
                        pieces.append(delta)
                        await emit("summary_delta", role, delta)
                    elif kind == "response.completed":
                        output = item["response"]["output"]
                        completed = True
                        await emit(
                            "status",
                            role,
                            "LLM response completed",
                            {"usage": item["response"].get("usage"), "api": "responses"},
                        )
        if not completed:
            raise RuntimeError(
                "LLM stream ended before response.completed; no incomplete tool call accepted"
            )
        message = {
            "role": "assistant",
            "content": "".join(pieces) or None,
            "_response_output": output,
        }
        calls = [
            {
                "id": item["call_id"],
                "type": "function",
                "function": {"name": item["name"], "arguments": item["arguments"]},
            }
            for item in output
            if item.get("type") == "function_call"
        ]
        if calls:
            message["tool_calls"] = calls
        return message
