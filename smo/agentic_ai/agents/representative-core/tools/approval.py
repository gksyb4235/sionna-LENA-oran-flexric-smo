"""Approval state helpers for Representative Core Agent."""

from __future__ import annotations

from typing import Any


def pending_approval(*, risk_level: str) -> dict[str, Any]:
    return {
        "required": True,
        "state": "pending",
        "approved": False,
        "risk_level": risk_level,
        "approved_by": None,
        "reason": None,
    }


def approved_approval(*, approved_by: str | None = None, reason: str | None = None, risk_level: str | None = None) -> dict[str, Any]:
    return {
        "required": True,
        "state": "approved",
        "approved": True,
        "risk_level": risk_level,
        "approved_by": approved_by or "user",
        "reason": reason,
    }


def rejected_approval(*, approved_by: str | None = None, reason: str | None = None, risk_level: str | None = None) -> dict[str, Any]:
    return {
        "required": True,
        "state": "rejected",
        "approved": False,
        "risk_level": risk_level,
        "approved_by": approved_by or "user",
        "reason": reason,
    }
