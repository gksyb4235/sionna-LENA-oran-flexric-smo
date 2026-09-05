"""Single-sourced Control_Parameter step-adjustment helper.

Both ``mcp_server.py``'s ``marginal_effect`` tool (task 6.1, Requirement 6.2)
and ``agent.py``'s Probe_Plan construction (task 7.2, Requirement 7.2/7.3)
need the exact same "what is the adjacent allowed value for this
Control_Parameter, in this direction, or is it out of range" logic:

- TxP/RET/CIO/HYS step by the fixed step size in
  ``smo.aimlfw.common.constants.CONTROL_PARAMETER_RANGES`` and are rejected
  if the stepped value leaves ``[min, max]``.
- TTT steps to the adjacent entry of
  ``smo.aimlfw.common.constants.TTT_ALLOWED_MS`` and is rejected if there is
  no adjacent entry in that direction.

This module exists purely so that rule lives in exactly one place, per this
codebase's existing preference for single-sourcing domain constants (see
``smo.aimlfw.common.constants``). ``mcp_server.py`` originally defined this
logic as a private ``_step_value`` helper; it now imports ``step_value`` from
here instead of duplicating it.
"""

from __future__ import annotations

from typing import Any, Literal

from smo.aimlfw.common.constants import CONTROL_PARAMETER_RANGES, TTT_ALLOWED_MS

StepDirection = Literal["+1", "-1"]

STEP_DIRECTIONS: tuple[StepDirection, StepDirection] = ("+1", "-1")


def step_value(parameter_name: str, current_value: Any, direction: StepDirection) -> float | int | None:
    """Return the adjacent allowed value for one Control_Parameter, or None if out of range."""
    if parameter_name == "ttt_ms":
        try:
            current_index = TTT_ALLOWED_MS.index(int(current_value))
        except (ValueError, TypeError):
            return None
        next_index = current_index + (1 if direction == "+1" else -1)
        if not 0 <= next_index < len(TTT_ALLOWED_MS):
            return None
        return TTT_ALLOWED_MS[next_index]
    rule = CONTROL_PARAMETER_RANGES[parameter_name]
    step = float(rule["step"]) * (1 if direction == "+1" else -1)
    candidate = float(current_value) + step
    if not float(rule["min"]) <= candidate <= float(rule["max"]):
        return None
    return candidate


__all__ = ["STEP_DIRECTIONS", "StepDirection", "step_value"]
