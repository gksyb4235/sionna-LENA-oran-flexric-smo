"""Property 8 test module (independent, per tasks.md parallel-collision notes).

Task 2.10: 동일 결과 디렉터리 재추출 후 개수·키·필드와 중복 부재를 검증한다.

Targets ``smo.aimlfw.data_extractor.pipeline.extract`` (task 2.2) together with
``smo.aimlfw.feature_store.FeatureStore.upsert`` (task 2.1): running ``extract()``
against the exact same ``result_dir`` twice must leave the Feature_Store with the
same record count, the same ``(Time_Step, cell_id)`` key set, the same field
values on every record as after the first run, and no duplicate records for any
key (Requirement 1.9).
"""

from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, MAX_SEED
from smo.aimlfw.common.models import CellParameters, FeatureRecord
from smo.aimlfw.data_extractor.pipeline import extract
from smo.aimlfw.feature_store import FeatureStore
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS, csv_values, valid_cell_parameters

# The "default" Feature_Group (smo/aimlfw/config/feature_groups/default.json) fixes
# target_cells to these three external cell identifiers; Property 8 does not exercise
# Feature_Group selection, so the fixed default cells keep the generator focused on
# the rerun-idempotence invariant.
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

# Dummy placeholder values for the CSV's own Control_Parameter columns: command.sh
# always supplies every (cell, parameter) pair below, so the CSV modal-value
# fallback never triggers and these values cannot leak into the FeatureRecords
# compared across the two extract() runs.
_DUMMY_PARAMETER_ROW = (43, 5, 0.0, 2.5, 160)

_ROW = tuple[int, tuple[str, ...]]


@st.composite
def _execution_data(draw: st.DrawFn) -> tuple[dict[str, CellParameters], int, dict[str, list[_ROW]]]:
    """Generate per-cell Control_Parameter values, a seed, and per-cell CSV rows.

    At least one data row overall is required: a CSV with zero data rows is the
    ``csv_empty`` error case (Requirement 1.10), which is out of scope here.
    """
    per_cell = {cell_id: draw(valid_cell_parameters()) for cell_id in _TARGET_CELLS}
    seed = draw(st.integers(min_value=0, max_value=MAX_SEED))
    rows: dict[str, list[_ROW]] = {}
    for cell_id in _TARGET_CELLS:
        row_count = draw(st.integers(min_value=0, max_value=4))
        cell_rows: list[_ROW] = []
        for _ in range(row_count):
            time_s = draw(st.integers(min_value=1, max_value=900))
            features = tuple(draw(csv_values()) for _ in _FEATURE_NAMES)
            cell_rows.append((time_s, features))
        rows[cell_id] = cell_rows

    if not any(rows.values()):
        forced_cell = draw(st.sampled_from(_TARGET_CELLS))
        time_s = draw(st.integers(min_value=1, max_value=900))
        features = tuple(draw(csv_values()) for _ in _FEATURE_NAMES)
        rows[forced_cell].append((time_s, features))
    return per_cell, seed, rows


def _command_sh_text(per_cell: dict[str, CellParameters], seed: int) -> str:
    """Render a command.sh whose ``--cellXxx=`` maps carry ``per_cell`` exactly."""
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"# run tag: seed{seed}_property8"]
    for parameter_name, flag_name in _CLI_FLAG_NAME.items():
        cell_value_map = ",".join(
            f"{cell_id}:{getattr(per_cell[cell_id], parameter_name)}" for cell_id in _TARGET_CELLS
        )
        lines.append(f"'./ns3' '--{flag_name}={cell_value_map}'")
    return "\n".join(lines) + "\n"


def _cell_kpi_csv_text(rows: dict[str, list[_ROW]]) -> str:
    """Render a CSV with the given per-cell (time_s, feature values) rows."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(_CSV_HEADER)
    for cell_id in _TARGET_CELLS:
        for time_s, features in rows[cell_id]:
            writer.writerow(
                [time_s, "180", "1", cell_id, "NR", *_DUMMY_PARAMETER_ROW, *features]
            )
    return buffer.getvalue()


def _sorted_dumps(records: list[FeatureRecord]) -> list[dict[str, Any]]:
    ordered = sorted(records, key=lambda record: (record.time_step, record.cell_id))
    return [record.model_dump(mode="json") for record in ordered]


# **Property 8: 재실행 멱등성**
# **Validates: Requirements 1.9**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(data=_execution_data())
def test_rerun_extract_against_same_result_dir_is_idempotent(
    data: tuple[dict[str, CellParameters], int, dict[str, list[_ROW]]],
) -> None:
    per_cell, seed, rows = data

    # A fresh temporary directory per Hypothesis example avoids reusing pytest's
    # function-scoped tmp_path fixture across examples of the same test invocation.
    with tempfile.TemporaryDirectory() as raw_result_dir:
        result_dir = Path(raw_result_dir)
        (result_dir / "command.sh").write_text(_command_sh_text(per_cell, seed), encoding="utf-8")
        (result_dir / "cell_kpi.csv").write_text(_cell_kpi_csv_text(rows), encoding="utf-8")

        store = FeatureStore(result_dir / "store")

        first_outcome = extract(result_dir, "default", store)
        first_records = store.records("default")

        second_outcome = extract(result_dir, "default", store)
        second_records = store.records("default")

        # Every target cell contributes exactly 5 Time_Step records regardless of
        # how many CSV rows fed each one (Requirement 1.6), and that count must be
        # stable across both runs (Requirement 1.9).
        assert first_outcome.records_written == len(_TARGET_CELLS) * 5
        assert second_outcome.records_written == first_outcome.records_written
        assert len(first_records) == len(second_records) == len(_TARGET_CELLS) * 5

        # (Time_Step, cell_id) key set is identical across both runs, and no key
        # appears more than once in either snapshot (no duplicate records).
        first_keys = [(record.time_step, record.cell_id) for record in first_records]
        second_keys = [(record.time_step, record.cell_id) for record in second_records]
        assert len(first_keys) == len(set(first_keys))
        assert len(second_keys) == len(set(second_keys))
        assert set(first_keys) == set(second_keys)

        # All field values on every record are identical to the first run's result.
        assert _sorted_dumps(first_records) == _sorted_dumps(second_records)
