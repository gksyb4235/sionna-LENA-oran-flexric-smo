"""Shared service error response contract."""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ErrorResponse(BaseModel):
    """Atomic error body returned by AIMLFW HTTP and MCP boundaries."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    error_code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    details: dict[str, Any] = Field(default_factory=dict)
