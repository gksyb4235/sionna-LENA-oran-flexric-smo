"""Feature Store service exceptions."""

from __future__ import annotations

from typing import Any

from smo.aimlfw.common.errors import ErrorResponse


class FeatureStoreError(Exception):
    """An expected Feature Store failure with a stable API error body."""

    def __init__(self, error_code: str, message: str, details: dict[str, Any] | None = None):
        self.response = ErrorResponse(
            error_code=error_code,
            message=message,
            details=details or {},
        )
        super().__init__(message)

    @property
    def error_code(self) -> str:
        return self.response.error_code

    @property
    def details(self) -> dict[str, Any]:
        return self.response.details
