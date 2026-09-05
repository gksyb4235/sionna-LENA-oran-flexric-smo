"""Execution-metadata extraction from ``command.sh`` (design.md, Data_Extractor).

design.md's "실행 메타데이터 원천" design decision: ``Data_Extractor`` parses the
result directory's ``command.sh`` for per-cell ``--cellXxx=cell:value,...`` CLI
maps (Control_Parameter values) and a ``seedNN`` pattern anywhere in the script
text (Requirement 1.5). When a cell/parameter pair is not present in
``command.sh`` (missing flag, missing cell entry, or an unreadable/missing
file), callers fall back to the CSV column's modal value for that
(cell, parameter) pair over the aggregation window, per design.md's explicit
fallback rule.
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

from smo.aimlfw.common.constants import ControlParameterName

#: Maps each command.sh ``--cellXxx=`` flag name to the Control_Parameter it
#: carries and to the identically named CSV column used as a fallback source.
CLI_FLAG_TO_PARAMETER: dict[str, ControlParameterName] = {
    "cellTxPowerDbm": "tx_power_dbm",
    "cellRetTiltDeg": "ret_tilt_deg",
    "cellCioDb": "cio_bias_db",
    "cellHysteresisDb": "hysteresis_db",
    "cellTttMs": "ttt_ms",
}

_INTEGER_PARAMETERS = frozenset({"tx_power_dbm", "ret_tilt_deg", "ttt_ms"})

_FLAG_PATTERN = re.compile(
    r"--(" + "|".join(re.escape(flag) for flag in CLI_FLAG_TO_PARAMETER) + r")=([^\s'\"]+)"
)
_SEED_PATTERN = re.compile(r"seed(\d+)", re.IGNORECASE)


def _coerce(parameter_name: ControlParameterName, raw: str) -> float | int:
    value = float(raw)
    return int(value) if parameter_name in _INTEGER_PARAMETERS else value


def _parse_cell_value_map(parameter_name: ControlParameterName, raw_map: str) -> dict[str, float | int]:
    """Parse one ``cell_a:1,cell_b:2`` argument value into a per-cell mapping.

    Malformed entries (missing ``:`` separator or a non-numeric value) are
    skipped rather than raising, so that a single corrupted flag degrades to
    the CSV fallback for that cell instead of failing extraction outright.
    """
    values: dict[str, float | int] = {}
    for entry in raw_map.split(","):
        entry = entry.strip()
        if not entry or ":" not in entry:
            continue
        cell_id, _, raw_value = entry.partition(":")
        cell_id = cell_id.strip()
        if not cell_id:
            continue
        try:
            values[cell_id] = _coerce(parameter_name, raw_value.strip())
        except ValueError:
            continue
    return values


class CommandMetadata:
    """Parsed ``command.sh`` contents: per-cell parameter maps and seed."""

    def __init__(
        self,
        parameters: dict[ControlParameterName, dict[str, float | int]],
        seed: int | None,
    ) -> None:
        self.parameters = parameters
        self.seed = seed

    def value(self, parameter_name: ControlParameterName, cell_id: str) -> float | int | None:
        """Return the command.sh value for (parameter_name, cell_id), if present."""
        return self.parameters.get(parameter_name, {}).get(cell_id)


def parse_command_sh(path: Path | str) -> CommandMetadata:
    """Parse ``command.sh`` for per-cell Control_Parameter maps and seed.

    Requirement 1.5. Returns an empty :class:`CommandMetadata` (all lookups miss,
    ``seed is None``) when the file is missing or unreadable, so callers always
    fall back to the CSV column mode without raising.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8")
    except OSError:
        return CommandMetadata(parameters={}, seed=None)

    parameters: dict[ControlParameterName, dict[str, float | int]] = {}
    for flag_name, raw_map in _FLAG_PATTERN.findall(text):
        parameter_name = CLI_FLAG_TO_PARAMETER[flag_name]
        parameters[parameter_name] = _parse_cell_value_map(parameter_name, raw_map)

    seed_match = _SEED_PATTERN.search(text)
    seed = int(seed_match.group(1)) if seed_match else None
    return CommandMetadata(parameters=parameters, seed=seed)


def modal_value(values: list[float | int]) -> float | int | None:
    """Return the most frequent value in ``values`` (ties broken by first occurrence).

    Used as the CSV fallback for a (cell, Control_Parameter) pair per
    design.md's "CSV의 해당 열 값(집계 구간 내 최빈값)" fallback rule.
    """
    if not values:
        return None
    counts = Counter(values)
    best_count = max(counts.values())
    for value in values:
        if counts[value] == best_count:
            return value
    return None  # pragma: no cover - unreachable, values is non-empty


__all__ = ["CLI_FLAG_TO_PARAMETER", "CommandMetadata", "modal_value", "parse_command_sh"]
