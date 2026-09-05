"""Conflict_Analyzer: pure-function library for Requirement 9's indirect conflict detection.

Task 8.1. Design tag (design.md, "9. Conflict_Analyzer (kpi_advisor 내부 모듈)"):

    `agents/kpi-advisor/conflict_analyzer.py`, 순수 함수 라이브러리(별도 REST 없음,
    `KPI_Advisor_Agent`가 직접 호출):

        def analyze(effects: list[MarginalEffectRecord], conflict_threshold: float | None)
            -> ConflictAnalysisResult: ...

This module has no server of its own -- it follows the same "kpi_advisor-internal pure
function module" convention already used by ``probing.py``, and is meant to be imported
directly by ``agent.py`` (task 8.5/Requirement 12) once that wiring task lands. Given a
Probe_Plan execution's per-(cell_id, control_parameter, target_kpi) Marginal_Effect values
(``agent.ProbeExecutionResult`` -- task 7.2/7.7), ``analyze`` finds every pair of
Control_Parameters that pull the same Target_KPI in opposite, threshold-exceeding directions
(an Indirect_Conflict), while excluding pairs a missing prediction makes unanalyzable and
reporting Target_KPIs left with too few usable Control_Parameters to compare.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations

from smo.aimlfw.common.config import RuntimeThresholdConfig
from smo.aimlfw.common.constants import DEFAULT_RETRAINING_THRESHOLD
from smo.aimlfw.common.models import MarginalEffectRecord


@dataclass(frozen=True)
class IndirectConflict:
    """One Indirect_Conflict item: a pair of Control_Parameters pulling one Target_KPI apart.

    ``control_parameter_a``/``control_parameter_b`` are always in lexicographic order (the same
    order the deterministic sort in ``analyze`` uses as its tie-break), so
    ``marginal_effect_a``/``marginal_effect_b`` correspond 1:1 with them rather than with any
    notion of "the positive one" / "the negative one" (Requirement 9.3).
    """

    cell_id: str
    control_parameter_a: str
    control_parameter_b: str
    target_kpi: str
    marginal_effect_a: float
    marginal_effect_b: float
    conflict_threshold: float


@dataclass(frozen=True)
class InsufficientParameters:
    """Requirement 9.10: fewer than 2 usable Control_Parameters for one (cell, Target_KPI)."""

    cell_id: str
    target_kpi: str
    valid_control_parameter_count: int
    reason: str


@dataclass(frozen=True)
class ConflictAnalysisResult:
    """Requirement 9's complete analysis output for one Probe_Plan execution.

    ``unavailable_control_parameters`` lists every ``(cell_id, control_parameter, target_kpi)``
    triple whose Marginal_Effect could not be computed (Requirement 9.8's ``effect_unavailable``
    marking) -- only pairs involving one of these are excluded, the rest of the analysis
    continues unaffected.
    """

    indirect_conflicts: list[IndirectConflict] = field(default_factory=list)
    insufficient_parameters: list[InsufficientParameters] = field(default_factory=list)
    unavailable_control_parameters: list[tuple[str, str, str]] = field(default_factory=list)
    conflict_threshold: float = 5.0
    threshold_warning: str | None = None


def _resolve_threshold(conflict_threshold: float | None) -> tuple[float, str | None]:
    """Resolve Conflict_Threshold, reusing ``RuntimeThresholdConfig``'s 9.5/9.6/9.9 defaulting.

    ``RuntimeThresholdConfig`` already implements exactly this rule: a value in [0.1, 100.0] is
    used as-is (9.5); an out-of-range or non-numeric value falls back to 5.0 *with* a warning
    (9.9); and an omitted key also falls back to 5.0, but -- unlike the out-of-range case --
    without a warning (9.6, "설정되지 않으면" is a distinct, non-warning condition from
    "지정되었으나 ... 범위를 벗어나면"). ``conflict_threshold=None`` (this function's "unset"
    signal) is therefore mapped to omitting the key entirely, rather than to an explicit
    ``None`` value, so that distinction reaches ``RuntimeThresholdConfig`` unchanged. Any other
    value (including a non-numeric one, should a caller pass one through by mistake) is passed
    through unchanged so the same invalid/out-of-range warning path fires.

    ``retraining_threshold`` is fixed at its own default here purely to satisfy
    ``RuntimeThresholdConfig``'s schema without ever triggering its unrelated warning; the
    result's ``retraining_threshold`` field is not read.
    """
    payload: dict[str, object] = {"retraining_threshold": DEFAULT_RETRAINING_THRESHOLD}
    if conflict_threshold is not None:
        payload["conflict_threshold"] = conflict_threshold
    resolved = RuntimeThresholdConfig.model_validate(payload)
    warning = next(
        (message for message in resolved.warnings if message.startswith("conflict_threshold")),
        None,
    )
    return resolved.conflict_threshold, warning


def _is_indirect_conflict(effect_a: float, effect_b: float, threshold: float) -> bool:
    """Requirement 9.2/9.4: boundary-inclusive, opposite-direction Indirect_Conflict test."""
    return (effect_a >= threshold and effect_b <= -threshold) or (effect_b >= threshold and effect_a <= -threshold)


def analyze(effects: list[MarginalEffectRecord], conflict_threshold: float | None) -> ConflictAnalysisResult:
    """Find every Indirect_Conflict across ``effects`` (Requirement 9.1-9.11).

    Single pass over ``effects`` to group by (cell_id, target_kpi), then one pass per group over
    that group's Control_Parameter combinations -- O(n) grouping plus O(m) pair comparisons per
    group, well within the 200-pair/2000ms bound of Requirement 9.11 for the stated 20
    Control_Parameter x 10 Target_KPI x 200-pair scale.
    """
    threshold, warning = _resolve_threshold(conflict_threshold)

    # (cell_id, target_kpi) -> {control_parameter: value_percent or None (missing prediction)}
    groups: dict[tuple[str, str], dict[str, float | None]] = {}
    for record in effects:
        groups.setdefault((record.cell_id, record.target_kpi), {})[record.control_parameter] = record.value_percent

    conflicts: list[IndirectConflict] = []
    insufficient: list[InsufficientParameters] = []
    unavailable: list[tuple[str, str, str]] = []

    for (cell_id, target_kpi), parameter_values in groups.items():
        valid_effects: dict[str, float] = {}
        for control_parameter, value in parameter_values.items():
            if value is None:
                # Requirement 9.8: missing Probe_Variant prediction -> effect_unavailable,
                # excludes only pairs involving this Control_Parameter.
                unavailable.append((cell_id, control_parameter, target_kpi))
                continue
            # Requirement 9.1: clamp to [-100.0, 100.0] and round to 1 decimal place. The
            # MarginalEffectRecord model already enforces this range at construction time, so
            # the clamp below is a defensive no-op for well-formed input; it is kept because the
            # requirement defines Conflict_Analyzer's own output value as the clamped+rounded
            # one, independent of what any upstream validation happens to already guarantee.
            valid_effects[control_parameter] = round(max(-100.0, min(100.0, value)), 1)

        if len(valid_effects) < 2:
            # Requirement 9.10: fewer than 2 usable Control_Parameters -> empty conflict list
            # for this Target_KPI plus a reason; no pairs to compare.
            insufficient.append(
                InsufficientParameters(
                    cell_id=cell_id,
                    target_kpi=target_kpi,
                    valid_control_parameter_count=len(valid_effects),
                    reason=(
                        f"only {len(valid_effects)} Control_Parameter(s) with an available "
                        f"Marginal_Effect for target_kpi={target_kpi!r} at cell_id={cell_id!r}; "
                        "at least 2 are required to compare"
                    ),
                )
            )
            continue

        # sorted() here guarantees control_parameter_a < control_parameter_b lexicographically,
        # matching the deterministic sort tie-break (Requirement 9.7) without a second sort pass.
        for control_parameter_a, control_parameter_b in combinations(sorted(valid_effects), 2):
            effect_a = valid_effects[control_parameter_a]
            effect_b = valid_effects[control_parameter_b]
            if _is_indirect_conflict(effect_a, effect_b, threshold):
                conflicts.append(
                    IndirectConflict(
                        cell_id=cell_id,
                        control_parameter_a=control_parameter_a,
                        control_parameter_b=control_parameter_b,
                        target_kpi=target_kpi,
                        marginal_effect_a=effect_a,
                        marginal_effect_b=effect_b,
                        conflict_threshold=threshold,
                    )
                )

    # Requirement 9.7: Target_KPI name ascending, then the two Control_Parameter names
    # lexicographically ascending. cell_id is appended only to keep ordering fully deterministic
    # when the same Target_KPI/Control_Parameter pair conflicts in more than one cell; it never
    # overrides the two keys the requirement specifies.
    conflicts.sort(key=lambda c: (c.target_kpi, (c.control_parameter_a, c.control_parameter_b), c.cell_id))
    insufficient.sort(key=lambda item: (item.target_kpi, item.cell_id))
    unavailable.sort()

    return ConflictAnalysisResult(
        indirect_conflicts=conflicts,
        insufficient_parameters=insufficient,
        unavailable_control_parameters=unavailable,
        conflict_threshold=threshold,
        threshold_warning=warning,
    )


__all__ = [
    "ConflictAnalysisResult",
    "IndirectConflict",
    "InsufficientParameters",
    "analyze",
]
