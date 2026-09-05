"""Property 9 test module (independent, per tasks.md parallel-collision notes).

Task 2.11: 미존재·읽기 불가·빈 CSV에서 원인 오류와 저장소 무변경을 검증한다.

Targets ``smo.aimlfw.data_extractor.pipeline``'s ``_open_csv`` and ``extract``:
for a non-existent ``cell_kpi.csv`` path, an unreadable file (permission
denied, or a directory in place of the file), and an empty CSV (truly empty,
or header-only with zero data rows), ``extract()`` must raise
``DataExtractorError`` with an error code identifying the specific cause, and
the Feature_Store must remain completely unchanged (design.md pipeline step
1, Requirement 1.10).
"""

from __future__ import annotations

import csv
import io
import os
import stat
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, MAX_SEED
from smo.aimlfw.common.models import CellParameters
from smo.aimlfw.data_extractor.errors import DataExtractorError
from smo.aimlfw.data_extractor.pipeline import extract
from smo.aimlfw.feature_store import FeatureStore
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS, valid_cell_parameters

# Fixed by the "default" Feature_Group (smo/aimlfw/config/feature_groups/default.json);
# Property 9 does not exercise Feature_Group selection, so a single fixed cell set keeps
# the generator focused on the invalid-input-path invariant.
_TARGET_CELLS = ("gNB_5G", "gNB_4G_1", "gNB_4G_2")
_FEATURE_GROUP = "default"

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

# The three causes Requirement 1.10 distinguishes, mapped to the discrete failure
# modes pipeline._open_csv/extract actually raise for each. Property testing
# guidance recommends parametrize for these discrete cases, combined with
# Hypothesis for the surrounding (non-discrete) state below.
_FAILURE_MODES = (
    "nonexistent_path",
    "directory_in_place_of_file",
    "permission_denied",
    "truly_empty_file",
    "header_only_csv",
)

_EXPECTED_ERROR_CODE = {
    "nonexistent_path": "csv_not_found",
    "directory_in_place_of_file": "csv_not_found",
    "permission_denied": "csv_not_readable",
    "truly_empty_file": "csv_empty",
    "header_only_csv": "csv_empty",
}


@st.composite
def _execution_metadata(draw: st.DrawFn) -> tuple[dict[str, CellParameters], int]:
    """Generate per-cell Control_Parameter values (Requirement 1.5) and a seed."""
    per_cell = {cell_id: draw(valid_cell_parameters()) for cell_id in _TARGET_CELLS}
    seed = draw(st.integers(min_value=0, max_value=MAX_SEED))
    return per_cell, seed


def _command_sh_text(per_cell: dict[str, CellParameters], seed: int) -> str:
    """Render a command.sh whose ``--cellXxx=`` maps carry ``per_cell`` exactly."""
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"# run tag: seed{seed}_property9"]
    for parameter_name, flag_name in _CLI_FLAG_NAME.items():
        cell_value_map = ",".join(
            f"{cell_id}:{getattr(per_cell[cell_id], parameter_name)}" for cell_id in _TARGET_CELLS
        )
        lines.append(f"'./ns3' '--{flag_name}={cell_value_map}'")
    return "\n".join(lines) + "\n"


def _valid_cell_kpi_csv_text() -> str:
    """Render a minimal, fully valid one-row-per-target-cell CSV."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_CSV_HEADER)
    for cell_id in _TARGET_CELLS:
        writer.writerow(
            ["1", "180", "1", cell_id, "NR", 43, 5, 0.0, 2.5, 160, *([1.0] * len(_FEATURE_NAMES))]
        )
    return buffer.getvalue()


def _write_valid_result_dir(result_dir: Path, per_cell: dict[str, CellParameters], seed: int) -> None:
    (result_dir / "command.sh").write_text(_command_sh_text(per_cell, seed), encoding="utf-8")
    (result_dir / "cell_kpi.csv").write_text(_valid_cell_kpi_csv_text(), encoding="utf-8")


def _prepare_broken_result_dir(result_dir: Path, failure_mode: str) -> None:
    """Set up ``result_dir`` so its ``cell_kpi.csv`` triggers ``failure_mode``.

    Every mode omits (or breaks) only ``cell_kpi.csv``; ``command.sh`` is never
    written because pipeline.extract() raises during the CSV-open/read step,
    strictly before command.sh is parsed (design.md pipeline order, steps 1
    precedes step 3).
    """
    csv_path = result_dir / "cell_kpi.csv"
    if failure_mode == "nonexistent_path":
        return  # cell_kpi.csv is simply never created.
    if failure_mode == "directory_in_place_of_file":
        csv_path.mkdir()
        return
    if failure_mode == "permission_denied":
        csv_path.write_text(_valid_cell_kpi_csv_text(), encoding="utf-8")
        os.chmod(csv_path, 0o000)
        return
    if failure_mode == "truly_empty_file":
        csv_path.write_text("", encoding="utf-8")
        return
    if failure_mode == "header_only_csv":
        buffer = io.StringIO()
        csv.writer(buffer).writerow(_CSV_HEADER)
        csv_path.write_text(buffer.getvalue(), encoding="utf-8")
        return
    raise ValueError(f"unknown failure_mode: {failure_mode}")  # pragma: no cover - defensive


def _cleanup_broken_result_dir(result_dir: Path, failure_mode: str) -> None:
    """Restore permissions so the temporary directory can be removed cleanly."""
    if failure_mode == "permission_denied":
        os.chmod(result_dir / "cell_kpi.csv", stat.S_IRUSR | stat.S_IWUSR)


# **Property 9: 잘못된 입력 경로 오류**
# **Validates: Requirements 1.10**
@pytest.mark.property
@pytest.mark.parametrize("failure_mode", _FAILURE_MODES)
@PROPERTY_TEST_SETTINGS
@given(metadata=_execution_metadata())
def test_invalid_input_path_raises_identified_error_and_leaves_store_unchanged(
    failure_mode: str,
    metadata: tuple[dict[str, CellParameters], int],
) -> None:
    per_cell, seed = metadata

    with tempfile.TemporaryDirectory() as raw_root:
        root = Path(raw_root)
        store = FeatureStore(root / "store")

        # Seed the Feature_Store with a real, successful extraction first so that
        # "무변경" (unchanged) is a meaningful assertion rather than a vacuous
        # "still empty" check: the failing call below must leave this prior state
        # byte-for-byte intact.
        valid_result_dir = root / "valid_run"
        valid_result_dir.mkdir()
        _write_valid_result_dir(valid_result_dir, per_cell, seed)
        extract(valid_result_dir, _FEATURE_GROUP, store)

        store_file = store.root / f"{_FEATURE_GROUP}.csv"
        before_snapshot = store_file.read_bytes()
        before_records = {
            (record.time_step, record.cell_id): record.model_dump(mode="json")
            for record in store.records(_FEATURE_GROUP)
        }
        assert before_records  # non-empty: the "unchanged" assertion below must not be vacuous.

        broken_result_dir = root / "broken_run"
        broken_result_dir.mkdir()
        _prepare_broken_result_dir(broken_result_dir, failure_mode)

        try:
            with pytest.raises(DataExtractorError) as excinfo:
                extract(broken_result_dir, _FEATURE_GROUP, store)

            error = excinfo.value
            # The error identifies the specific cause (Requirement 1.10): the code is
            # drawn from the exact discrete cause this failure_mode reproduces, and
            # the error carries the offending csv_path so the cause is diagnosable.
            assert error.error_code == _EXPECTED_ERROR_CODE[failure_mode]
            assert str(broken_result_dir / "cell_kpi.csv") in str(error.details.get("csv_path", ""))

            # Feature_Store is completely unchanged: identical bytes on disk and
            # identical records (including the ones seeded above).
            after_snapshot = store_file.read_bytes()
            assert after_snapshot == before_snapshot
            after_records = {
                (record.time_step, record.cell_id): record.model_dump(mode="json")
                for record in store.records(_FEATURE_GROUP)
            }
            assert after_records == before_records
        finally:
            _cleanup_broken_result_dir(broken_result_dir, failure_mode)
