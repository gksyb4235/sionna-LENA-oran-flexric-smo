"""Central model selection for the existing deep-agents runtime."""

from __future__ import annotations

import os


def model_for(agent_name: str):
    """Build the configured ChatOpenAI model lazily at invocation time."""
    from langchain_openai import ChatOpenAI

    env_prefix = agent_name.upper().replace("-", "_")
    model = os.getenv(f"{env_prefix}_MODEL", os.getenv("OPENAI_MODEL", "gpt-5-mini"))
    return ChatOpenAI(model=model)


__all__ = ["model_for"]
