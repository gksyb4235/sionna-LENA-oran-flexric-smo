"""Pure aggregation-rule helper for the Data_Extractor pipeline.

design.md, Data_Extractor 처리 파이프라인 5단계: "집계 규칙 적용 후 소수점 6자리
반올림" — 이 모듈은 그 단계만을 CSV 파싱/Time_Step 매핑과 독립된 순수 함수로
분리한 것이다 (Requirement 1.2).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

AggregationRule = Literal["sum", "mean"]

FEATURE_VALUE_ROUND_NDIGITS = 6


def aggregate_values(values: Sequence[float], rule: AggregationRule) -> float:
    """Aggregate a non-empty population of valid samples per ``rule``.

    Requirement 1.2: THE Data_Extractor SHALL apply the Feature_Group's
    per-feature aggregation rule (sum or mean) to the feature column and
    round the resulting real value to 6 decimal places.

    Args:
        values: the valid (non-excluded) numeric samples for one
            (Time_Step, cell_id, feature) group. Must be non-empty; callers
            are responsible for the Requirement 1.4 "insufficient" branch
            when the valid sample count is 0.
        rule: ``"sum"`` or ``"mean"``.

    Returns:
        The aggregated value rounded to 6 decimal places.

    Raises:
        ValueError: if ``rule`` is not ``"sum"``/``"mean"`` or ``values`` is empty.
    """
    if rule not in ("sum", "mean"):
        raise ValueError(f"unsupported aggregation rule: {rule!r}")
    if not values:
        raise ValueError("aggregate_values requires a non-empty population of valid samples")
    total = sum(values)
    result = total if rule == "sum" else total / len(values)
    return round(result, FEATURE_VALUE_ROUND_NDIGITS)


__all__ = ["AggregationRule", "FEATURE_VALUE_ROUND_NDIGITS", "aggregate_values"]
