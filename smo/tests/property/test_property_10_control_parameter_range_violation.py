"""Property 10 test module (own module to avoid collisions with parallel Property tasks).

Task 2.12: 모든 범위/스텝 위반 조합에서 위반 상세와 무변경을 검증한다.

Targets the Data_Extractor's resolved Control_Parameter range/step validation, as
specified in requirements.md 1.12 and implemented by
``smo.aimlfw.data_extractor.pipeline._validate_control_parameters`` /
``extract`` (Task 2.2):

    결과 디렉터리 실행 메타데이터의 Control_Parameter 값 중 하나 이상이 용어집에 정의된
    허용 범위 또는 허용 값 집합(TTT_ALLOWED_MS)을 벗어나면, extract() 는 위반된
    Control_Parameter 이름과 해당 값을 담은 DataExtractorError 를 반환하고
    Feature_Store 를 변경하지 않는다.

Note: ``tx_power_dbm``, ``ret_tilt_deg``, and ``ttt_ms`` are glossary-integer
Control_Parameters (TxP/RET have step 1; TTT is a discrete integer set), so
command.sh/CSV metadata parsing truncates their parsed values to ``int`` before
resolution (Requirement 1.5's execution-metadata source). A fractional
step-violation such as ``tx_power_dbm=30.5`` therefore truncates to the *valid*
integer ``30`` before it ever reaches range/step validation. This test's oracle
simulates that same integer truncation so the expected violation set always
matches what ``extract()`` can actually observe post-resolution.
"""

from __future__ import annotations

import csv
import io
import tempfile
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, CONTROL_PARAMETER_RANGES, TTT_ALLOWED_MS
from smo.aimlfw.data_extractor import DataExtractorError, extract
from smo.aimlfw.feature_store import FeatureStore
from smo.tests.property.strategies import (
    PROPERTY_TEST_SETTINGS,
    invalid_cell_parameters,
    valid_cell_parameters,
)

# The "default" Feature_Group (smo/aimlfw/config/feature_groups/default.json) fixes
# target_cells to these three external cell identifiers; Property 10 does not exercise
# Feature_Group selection, so the fixed default cells keep the generator focused on
# the range/step-violation invariant.
_TARGET_CELLS = ("gNB_5G", "gNB_4G_1", "gNB_4G_2")

_FEATURE_NAMES = (
    "cell_goodput_mbps", "avg_ue_goodput_mbps", "ue_goodput_p5_mbps", "sinr_p50_db",
    "prb_utilization_pct", "delay_p95_ms", "ho_failure_count", "pingpong_count",
    "rlf_count", "interval_energy_j",
)

_CSV_HEADER = (
    "time_s", "interval_s", "cell_id", "cell", "band",
    *CONTROL_PARAMETER_NAMES,
    *_FEATURE_NAMES,
)

# command.sh CLI flag name for each Control_Parameter, per metadata.CLI_FLAG_TO_PARAMETER.
_CLI_FLAG_NAME = {
    "tx_power_dbm": "cellTxPowerDbm",
    "ret_tilt_deg": "cellRetTiltDeg",
    "cio_bias_db": "cellCioDb",
    "hysteresis_db": "cellHysteresisDb",
    "ttt_ms": "cellTttMs",
}

# Glossary Control_Parameters whose values are integer-valued (TxP/RET step 1;
# TTT a discrete integer set). command.sh/CSV metadata parsing truncates parsed
# values for these parameters to ``int`` before they reach range/step validation.
_INTEGER_PARAMETERS = frozenset({"tx_power_dbm", "ret_tilt_deg", "ttt_ms"})

# A single fixed valid parameter payload used only to seed the Feature_Store with
# pre-existing state before the violating extraction is attempted (so "무변경" is
# checked against non-empty prior state, not an empty store).
_SEED_VALUES: dict[str, float | int] = {
    "tx_power_dbm": 43.0,
    "ret_tilt_deg": 5.0,
    "cio_bias_db": 0.0,
    "hysteresis_db": 2.5,
    "ttt_ms": 160,
}


def _resolved_value(parameter_name: str, value: float | int) -> float | int:
    """Mirror the int-truncation ``extract()`` applies while resolving metadata."""
    return int(value) if parameter_name in _INTEGER_PARAMETERS else value


def _is_valid(parameter_name: str, value: float | int) -> bool:
    """Independent oracle mirroring the glossary's range/step/allowed-set rules (1.12).

    Reads only the canonical constants (Glossary source of truth), not the
    pipeline's private validation function, to avoid a tautological test.
    """
    if parameter_name == "ttt_ms":
        return value in TTT_ALLOWED_MS
    rule = CONTROL_PARAMETER_RANGES[parameter_name]
    minimum, maximum, step = Decimal(str(rule["min"])), Decimal(str(rule["max"])), Decimal(str(rule["step"]))
    decimal_value = Decimal(str(value))
    return minimum <= decimal_value <= maximum and (decimal_value - minimum) % step == 0


@st.composite
def _cell_payloads_with_violations(draw: Any) -> dict[str, dict[str, float | int]]:
    """Generate a per-target-cell parameter payload map with >=1 post-resolution violation.

    Each target cell independently draws either a fully valid parameter set or one
    with exactly one invalid value (per Task 1.2's ``invalid_cell_parameters``). If,
    after simulating the integer-truncation ``extract()`` applies during metadata
    resolution, no violation survives (e.g. a fractional step-violation like
    ``tx_power_dbm=30.5`` truncates to the valid ``30``), ``ttt_ms`` on the first
    cell is forced to an always-invalid value (``-1``, which int-truncation cannot
    heal) so the generated case is guaranteed to exercise at least one violation.
    """
    payloads: dict[str, dict[str, float | int]] = {}
    for cell_id in _TARGET_CELLS:
        if draw(st.booleans()):
            payloads[cell_id] = draw(invalid_cell_parameters())
        else:
            payloads[cell_id] = draw(valid_cell_parameters()).model_dump()

    any_violation_survives = any(
        not _is_valid(parameter_name, _resolved_value(parameter_name, value))
        for values in payloads.values()
        for parameter_name, value in values.items()
    )
    if not any_violation_survives:
        payloads[_TARGET_CELLS[0]] = dict(payloads[_TARGET_CELLS[0]])
        payloads[_TARGET_CELLS[0]]["ttt_ms"] = -1
    return payloads


def _command_sh_text(payloads: dict[str, dict[str, float | int]]) -> str:
    """Render a command.sh whose ``--cellXxx=`` maps carry ``payloads`` exactly."""
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", "# run tag: seed1_property10"]
    for parameter_name, flag_name in _CLI_FLAG_NAME.items():
        cell_value_map = ",".join(
            f"{cell_id}:{payloads[cell_id][parameter_name]}" for cell_id in _TARGET_CELLS
        )
        lines.append(f"'./ns3' '--{flag_name}={cell_value_map}'")
    return "\n".join(lines) + "\n"


def _cell_kpi_csv_text() -> str:
    """Render a minimal one-row-per-target-cell CSV (feature values are irrelevant here)."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_CSV_HEADER)
    for cell_id in _TARGET_CELLS:
        # Control_Parameter columns are dummy placeholders: command.sh already supplies
        # every (cell, parameter) pair above, so the CSV modal-value fallback never
        # triggers and these values cannot mask the command.sh-sourced violations.
        writer.writerow(
            ["1", "180", "1", cell_id, "NR", 43, 5, 0.0, 2.5, 160, *([1.0] * len(_FEATURE_NAMES))]
        )
    return buffer.getvalue()


def _expected_violations(payloads: dict[str, dict[str, float | int]]) -> set[tuple[str, str, float | int]]:
    violations: set[tuple[str, str, float | int]] = set()
    for cell_id, values in payloads.items():
        for parameter_name in CONTROL_PARAMETER_NAMES:
            resolved = _resolved_value(parameter_name, values[parameter_name])
            if not _is_valid(parameter_name, resolved):
                violations.add((cell_id, parameter_name, resolved))
    return violations


# **Property 10: Control_Parameter 범위 위반 시 무변경 오류**
# **Validates: Requirements 1.12**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(payloads=_cell_payloads_with_violations())
def test_control_parameter_violation_raises_with_details_and_leaves_store_unchanged(
    payloads: dict[str, dict[str, float | int]],
) -> None:
    expected_violations = _expected_violations(payloads)
    assert expected_violations  # generator guarantees at least one violation survives resolution

    with tempfile.TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        store = FeatureStore(root / "store")

        # Seed the Feature_Store with a legitimate prior extraction so "무변경" is
        # verified against non-empty pre-existing state, not just an empty store.
        seed_dir = root / "seed"
        seed_dir.mkdir()
        seed_payloads = {cell_id: dict(_SEED_VALUES) for cell_id in _TARGET_CELLS}
        (seed_dir / "command.sh").write_text(_command_sh_text(seed_payloads), encoding="utf-8")
        (seed_dir / "cell_kpi.csv").write_text(_cell_kpi_csv_text(), encoding="utf-8")
        extract(seed_dir, "default", store)
        snapshot_before = store.records("default")
        assert snapshot_before  # sanity: seeding actually populated the store

        # Now attempt the extraction that must fail due to Control_Parameter violations.
        violating_dir = root / "violating"
        violating_dir.mkdir()
        (violating_dir / "command.sh").write_text(_command_sh_text(payloads), encoding="utf-8")
        (violating_dir / "cell_kpi.csv").write_text(_cell_kpi_csv_text(), encoding="utf-8")

        with pytest.raises(DataExtractorError) as captured:
            extract(violating_dir, "default", store)

        error = captured.value
        assert error.error_code == "control_parameter_out_of_range"

        actual_violations = {
            (violation["cell_id"], violation["parameter"], violation["value"])
            for violation in error.details["violations"]
        }
        assert actual_violations == expected_violations

        # Feature_Store must be completely unchanged: same records, same order, same fields.
        snapshot_after = store.records("default")
        assert [record.model_dump() for record in snapshot_after] == [
            record.model_dump() for record in snapshot_before
        ]
