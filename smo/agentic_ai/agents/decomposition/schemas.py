"""Output schema helpers for the Decomposition Agent."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

OUTPUT_KEYS = ("intent", "subtasks", "golden_goal_context_used")


@dataclass(frozen=True)
class DecompositionResult:
    """Normalized output for the Decomposition Agent."""

    intent: str
    subtasks: list[str]
    golden_goal_context_used: bool

    def to_dict(self) -> dict[str, Any]:
        """Return the exact public JSON contract."""

        return {
            "intent": self.intent,
            "subtasks": self.subtasks,
            "golden_goal_context_used": self.golden_goal_context_used,
        }

    @classmethod
    def from_model_payload(
        cls,
        payload: dict[str, Any],
        *,
        intent: str,
        golden_goal_context_used: bool,
    ) -> "DecompositionResult":
        """Normalize a model JSON payload into the exact public contract."""

        subtasks = payload.get("subtasks")
        if not isinstance(subtasks, list):
            raise ValueError("Model output must contain a 'subtasks' array.")

        normalized_subtasks: list[str] = []
        for item in subtasks:
            if isinstance(item, str):
                task = item.strip()
            else:
                task = str(item).strip()
            if task:
                normalized_subtasks.append(task)

        if not normalized_subtasks:
            raise ValueError("Model output must contain at least one non-empty subtask.")

        result = cls(
            intent=intent,
            subtasks=normalized_subtasks,
            golden_goal_context_used=bool(golden_goal_context_used),
        )
        validate_output_contract(result.to_dict())
        return result


def validate_output_contract(data: dict[str, Any]) -> None:
    """Validate that a dict matches the exact public output contract."""

    if tuple(data.keys()) != OUTPUT_KEYS:
        raise ValueError(f"Output keys must be exactly {OUTPUT_KEYS}; got {tuple(data.keys())}")
    if not isinstance(data["intent"], str):
        raise TypeError("'intent' must be a string.")
    if not isinstance(data["subtasks"], list) or not all(isinstance(item, str) for item in data["subtasks"]):
        raise TypeError("'subtasks' must be a list of strings.")
    if not isinstance(data["golden_goal_context_used"], bool):
        raise TypeError("'golden_goal_context_used' must be a boolean.")
