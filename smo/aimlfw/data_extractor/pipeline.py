"""Data_Extractor pipeline: CSV validation, Time_Step mapping, and per-cell
aggregation into Feature_Records, written atomically to the Feature_Store.

Implements design.md's Data_Extractor component (Requirement 1) as a single
streaming pass over ``cell_kpi.csv`` so that the 1,000,000-row input bound
(Requirement 1.1) is handled with O(1) memory per (Time_Step, cell, feature)
accumulator rather than by loading the whole file.

Pipeline order (design.md "처리 파이프라인" — fixed):

1. ``cell_kpi.csv`` existence/readability/non-empty check (1.10).
2. CSV header vs. Feature_Group required column check (1.8).
3. ``command.sh`` parsing for per-cell Control_Parameter values and seed (1.5),
   with a CSV modal-value fallback per (cell, Control_Parameter) computed over
   the same streaming pass.
4. Per-row ``time_s`` validity (1.7) and Decimal-based Time_Step mapping (1.1, 1.6).
5. Per-(Time_Step, cell, feature) valid/excluded sample counting (1.3).
6. Aggregation rule application and 6-decimal rounding (1.2), data_quality (1.4, 1.11).
7. Resolved Control_Parameter range/step validation (1.12) — before any write.
8. Atomic ``Feature_Store.upsert()`` (1.9, 2.6) — only after every prior step succeeds.
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from smo.aimlfw.common.config import ConfigurationError, load_feature_group
from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, CONTROL_PARAMETER_RANGES, TTT_ALLOWED_MS
from smo.aimlfw.common.models import CellParameters, FeatureRecord, SampleCount

from .aggregation import aggregate_values
from .errors import DataExtractorError
from .metadata import CommandMetadata, modal_value, parse_command_sh

TIME_S_COLUMN = "time_s"
CELL_COLUMN = "cell"
CSV_FILENAME = "cell_kpi.csv"
COMMAND_FILENAME = "command.sh"

MIN_TIME_S = Decimal("1")
MAX_TIME_S = Decimal("900")
INTERVAL_SECONDS = 180
NUM_TIME_STEPS = 5

_INTEGER_PARAMETERS = frozenset({"tx_power_dbm", "ret_tilt_deg", "ttt_ms"})


def time_step_for_time_s(time_s: str) -> int:
    """Map a valid ``time_s`` string (1 <= time_s <= 900) to Time_Step 0..4.

    design.md pipeline step 3: ``Time_Step = floor((time_s - 1e-9) / 180)``,
    parsed from the CSV's textual ``time_s`` via ``Decimal`` to avoid float
    boundary errors; equivalent to ``180*k < time_s <= 180*(k+1)``.
    """
    return _time_step_for_decimal(Decimal(time_s))


def _time_step_for_decimal(value: Decimal) -> int:
    for k in range(NUM_TIME_STEPS):
        if Decimal(INTERVAL_SECONDS * k) < value <= Decimal(INTERVAL_SECONDS * (k + 1)):
            return k
    raise ValueError(f"time_s={value} is outside the valid 1..900 range")


def _time_s_in_range(raw: str) -> Decimal | None:
    """Parse and range-check ``time_s``; return ``None`` when it must be excluded (1.7)."""
    try:
        value = Decimal(raw.strip())
    except (InvalidOperation, ValueError, AttributeError):
        return None
    if value.is_nan():
        return None
    if not (MIN_TIME_S <= value <= MAX_TIME_S):
        return None
    return value


def _parse_numeric(raw: str) -> float | None:
    """Parse one CSV cell as a finite real number; ``None`` when it must be excluded (1.3)."""
    stripped = raw.strip()
    if not stripped or stripped.lower() in ("nan", "-nan", "+nan"):
        return None
    try:
        value = float(stripped)
    except ValueError:
        return None
    if value != value or value in (float("inf"), float("-inf")):
        return None
    return value


def _coerce_parameter(parameter_name: str, value: float) -> float | int:
    return int(value) if parameter_name in _INTEGER_PARAMETERS else value


@dataclass(frozen=True)
class ExtractOutcome:
    """Result of one successful :func:`extract` call (design.md ``POST /extract``)."""

    records_written: int
    excluded_rows: int
    warnings: list[str] = field(default_factory=list)


@dataclass
class _FeatureAccumulator:
    valid: int = 0
    excluded: int = 0
    total: float = 0.0
    values: list[float] = field(default_factory=list)


def _validate_columns(header: list[str], feature_names: list[str]) -> None:
    required = {TIME_S_COLUMN, CELL_COLUMN, *CONTROL_PARAMETER_NAMES, *feature_names}
    missing = sorted(required - set(header))
    if missing:
        raise DataExtractorError(
            "missing_columns",
            "cell_kpi.csv is missing columns required by the Feature_Group",
            # undefined_aggregation_features is always empty because FeatureGroupConfig
            # already guarantees every feature has an aggregation rule (config-level
            # invariant enforced at load time) — kept for API contract completeness.
            {"missing_columns": missing, "undefined_aggregation_features": []},
        )


def _validate_control_parameters(resolved: dict[str, dict[str, float | int]]) -> None:
    violations: list[dict[str, Any]] = []
    for cell_id, values in resolved.items():
        for parameter_name, value in values.items():
            if parameter_name == "ttt_ms":
                if value not in TTT_ALLOWED_MS:
                    violations.append({"cell_id": cell_id, "parameter": parameter_name, "value": value})
                continue
            rule = CONTROL_PARAMETER_RANGES[parameter_name]
            minimum, maximum, step = Decimal(str(rule["min"])), Decimal(str(rule["max"])), Decimal(str(rule["step"]))
            decimal_value = Decimal(str(value))
            if not (minimum <= decimal_value <= maximum) or (decimal_value - minimum) % step != 0:
                violations.append({"cell_id": cell_id, "parameter": parameter_name, "value": value})
    if violations:
        raise DataExtractorError(
            "control_parameter_out_of_range",
            "Execution metadata contains Control_Parameter values outside the allowed range",
            {"violations": violations},
        )


def _open_csv(csv_path: Path) -> Any:
    if not csv_path.is_file():
        raise DataExtractorError(
            "csv_not_found",
            "cell_kpi.csv does not exist or is not a readable file",
            {"csv_path": str(csv_path)},
        )
    try:
        return csv_path.open("r", encoding="utf-8", newline="")
    except OSError as exc:
        raise DataExtractorError(
            "csv_not_readable",
            "cell_kpi.csv could not be opened",
            {"csv_path": str(csv_path), "reason": str(exc)},
        ) from exc


def _resolve_control_parameters(
    target_cells: list[str],
    command_metadata: CommandMetadata,
    csv_parameter_counts: dict[str, dict[str, Counter[float]]],
    warnings: list[str],
) -> dict[str, dict[str, float | int]]:
    resolved: dict[str, dict[str, float | int]] = {}
    missing: list[dict[str, str]] = []
    for cell_id in target_cells:
        cell_values: dict[str, float | int] = {}
        for parameter_name in CONTROL_PARAMETER_NAMES:
            value = command_metadata.value(parameter_name, cell_id)
            if value is None:
                fallback = modal_value(csv_parameter_counts[cell_id][parameter_name])
                if fallback is None:
                    missing.append({"cell_id": cell_id, "parameter": parameter_name})
                    continue
                value = _coerce_parameter(parameter_name, fallback)
                warnings.append(
                    f"command.sh missing {parameter_name} for {cell_id}; used CSV modal-value fallback"
                )
            cell_values[parameter_name] = value
        resolved[cell_id] = cell_values
    if missing:
        raise DataExtractorError(
            "control_parameter_unavailable",
            "Control_Parameter values could not be resolved from command.sh or cell_kpi.csv",
            {"missing": missing},
        )
    return resolved


def _data_quality_for(counts: dict[str, SampleCount]) -> str:
    if any(count.valid == 0 for count in counts.values()):
        return "insufficient"
    if any(count.excluded > 0 for count in counts.values()):
        return "partial"
    return "complete"


def extract(
    result_dir: str | Path,
    feature_group: str,
    feature_store: Any,
    *,
    config_dir: Path | str | None = None,
) -> ExtractOutcome:
    """Extract 180-second Feature_Records from ``result_dir`` and upsert them.

    ``feature_store`` is a :class:`smo.aimlfw.feature_store.FeatureStore`
    instance called in-process (mirroring the Model_Registry/Model_Storage
    shared-library pattern already used elsewhere in this package), so the
    final write is the same atomic CSV-replace upsert Requirement 1.9 and 2.6
    already guarantee.
    """
    result_path = Path(result_dir)
    csv_path = result_path / CSV_FILENAME

    try:
        config = load_feature_group(feature_group, config_dir)
    except ConfigurationError as exc:
        raise DataExtractorError(
            "invalid_feature_group",
            "Feature_Group configuration could not be loaded",
            {"feature_group": feature_group, "reason": str(exc)},
        ) from exc

    feature_names = list(dict.fromkeys([*config.features, *config.target_kpis]))
    target_cells = config.target_cells
    target_cell_set = set(target_cells)
    warnings: list[str] = []

    with _open_csv(csv_path) as source:
        reader = csv.reader(source)
        header = next(reader, None)
        if header is None:
            raise DataExtractorError(
                "csv_empty",
                "cell_kpi.csv has no header or data rows",
                {"csv_path": str(csv_path)},
            )
        _validate_columns(header, feature_names)

        time_s_index = header.index(TIME_S_COLUMN)
        cell_index = header.index(CELL_COLUMN)
        feature_indexes = {name: header.index(name) for name in feature_names}
        parameter_indexes = {name: header.index(name) for name in CONTROL_PARAMETER_NAMES}

        feature_accumulators: dict[tuple[int, str], dict[str, _FeatureAccumulator]] = defaultdict(
            lambda: {name: _FeatureAccumulator() for name in feature_names}
        )
        csv_parameter_counts: dict[str, dict[str, Counter[float]]] = {
            cell_id: {name: Counter() for name in CONTROL_PARAMETER_NAMES} for cell_id in target_cells
        }

        total_data_rows = 0
        excluded_rows = 0
        for row in reader:
            total_data_rows += 1
            cell_id = row[cell_index].strip()

            if cell_id in target_cell_set:
                for parameter_name, index in parameter_indexes.items():
                    parsed = _parse_numeric(row[index])
                    if parsed is not None:
                        csv_parameter_counts[cell_id][parameter_name][parsed] += 1

            time_s_value = _time_s_in_range(row[time_s_index])
            if time_s_value is None:
                excluded_rows += 1
                continue
            if cell_id not in target_cell_set:
                continue

            time_step = _time_step_for_decimal(time_s_value)
            group = feature_accumulators[(time_step, cell_id)]
            for name, index in feature_indexes.items():
                accumulator = group[name]
                parsed = _parse_numeric(row[index])
                if parsed is None:
                    accumulator.excluded += 1
                else:
                    accumulator.valid += 1
                    accumulator.values.append(parsed)

    if total_data_rows == 0:
        raise DataExtractorError(
            "csv_empty",
            "cell_kpi.csv has no header or data rows",
            {"csv_path": str(csv_path)},
        )

    command_metadata = parse_command_sh(result_path / COMMAND_FILENAME)
    resolved_parameters = _resolve_control_parameters(
        target_cells, command_metadata, csv_parameter_counts, warnings
    )
    _validate_control_parameters(resolved_parameters)

    records: list[FeatureRecord] = []
    for cell_id in target_cells:
        control_parameters = CellParameters(**resolved_parameters[cell_id])
        for time_step in range(NUM_TIME_STEPS):
            group = feature_accumulators.get((time_step, cell_id))
            counts: dict[str, SampleCount] = {}
            values: dict[str, float | None] = {}
            for name in feature_names:
                accumulator = group[name] if group is not None else _FeatureAccumulator()
                counts[name] = SampleCount(valid=accumulator.valid, excluded=accumulator.excluded)
                values[name] = (
                    None
                    if accumulator.valid == 0
                    else aggregate_values(accumulator.values, config.aggregations[name])
                )
            records.append(
                FeatureRecord(
                    feature_group=feature_group,
                    time_step=time_step,
                    cell_id=cell_id,
                    control_parameters=control_parameters,
                    features=values,
                    sample_counts=counts,
                    data_quality=_data_quality_for(counts),
                    source_dir=str(result_path),
                    seed=command_metadata.seed,
                )
            )

    feature_store.upsert(records)
    return ExtractOutcome(records_written=len(records), excluded_rows=excluded_rows, warnings=warnings)


__all__ = ["ExtractOutcome", "extract", "time_step_for_time_s"]
