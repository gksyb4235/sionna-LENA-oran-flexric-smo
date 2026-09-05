"""Property 5 test module (independent, per tasks.md parallel-collision notes).

Task 2.7: 셀별 파라미터, 원본 경로와 seed가 모든 FeatureRecord에 보존되는지 검증한다.

Targets the Data Extractor's execution-metadata propagation, as specified in
design.md's "실행 메타데이터 원천" design decision: ``command.sh`` is parsed
(``smo.aimlfw.data_extractor.metadata.parse_command_sh``) for per-cell
Control_Parameter values and a ``seedNN`` pattern, and
``smo.aimlfw.data_extractor.pipeline.extract`` attaches those same values —
plus the ``result_dir`` path — to every ``FeatureRecord`` it writes to the
Feature_Store (Requirement 1.5), regardless of Time_Step.
"""

from __future__ import annotations

import csv
import io
import tempfile
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, MAX_SEED
from smo.aimlfw.common.models import CellParameters
from smo.aimlfw.data_extractor.pipeline import extract
from smo.aimlfw.feature_store import FeatureStore
from smo.tests.property.strategies import PROPERTY_TEST_SETTINGS, valid_cell_parameters

# The "default" Feature_Group (smo/aimlfw/config/feature_groups/default.json) fixes
# target_cells to these three external cell identifiers; Property 5 does not exercise
# Feature_Group selection, so the fixed default cells keep the generator focused on
# the metadata-propagation invariant.
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


@st.composite
def _execution_metadata(draw: st.DrawFn) -> tuple[dict[str, CellParameters], int]:
    """Generate per-cell Control_Parameter values (Requirement 1.5) and a seed."""
    per_cell = {cell_id: draw(valid_cell_parameters()) for cell_id in _TARGET_CELLS}
    seed = draw(st.integers(min_value=0, max_value=MAX_SEED))
    return per_cell, seed


def _command_sh_text(per_cell: dict[str, CellParameters], seed: int) -> str:
    """Render a command.sh whose ``--cellXxx=`` maps carry ``per_cell`` exactly."""
    lines = ["#!/usr/bin/env bash", "set -euo pipefail", f"# run tag: seed{seed}_property5"]
    for parameter_name, flag_name in _CLI_FLAG_NAME.items():
        cell_value_map = ",".join(
            f"{cell_id}:{getattr(per_cell[cell_id], parameter_name)}" for cell_id in _TARGET_CELLS
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
        # triggers and these values cannot leak into the asserted FeatureRecords.
        writer.writerow(
            ["1", "180", "1", cell_id, "NR", 43, 5, 0.0, 2.5, 160, *([1.0] * len(_FEATURE_NAMES))]
        )
    return buffer.getvalue()


# **Property 5: 실행 메타데이터 전파**
# **Validates: Requirements 1.5**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(metadata=_execution_metadata())
def test_execution_metadata_propagates_identically_to_every_feature_record(
    metadata: tuple[dict[str, CellParameters], int],
) -> None:
    per_cell, seed = metadata

    # A fresh temporary directory per Hypothesis example avoids reusing pytest's
    # function-scoped tmp_path fixture across examples of the same test invocation.
    with tempfile.TemporaryDirectory() as raw_result_dir:
        result_dir = Path(raw_result_dir)
        (result_dir / "command.sh").write_text(_command_sh_text(per_cell, seed), encoding="utf-8")
        (result_dir / "cell_kpi.csv").write_text(_cell_kpi_csv_text(), encoding="utf-8")

        store = FeatureStore(result_dir / "store")
        outcome = extract(result_dir, "default", store)

        records = store.records("default")

        # Every target cell contributes exactly 5 Time_Step records (Requirement 1.6).
        assert outcome.records_written == len(_TARGET_CELLS) * 5
        assert len(records) == len(_TARGET_CELLS) * 5
        assert records  # non-empty: the propagation invariant below must not vacuously hold.

        for record in records:
            # The original result directory's path is preserved verbatim on every
            # FeatureRecord, independent of cell_id or Time_Step.
            assert record.source_dir == str(result_dir)
            # The seed parsed from command.sh is preserved verbatim on every FeatureRecord.
            assert record.seed == seed
            # Each FeatureRecord's Control_Parameter values equal the per-cell metadata
            # values for its own cell_id, exactly reproducing the generated CellParameters.
            expected = per_cell[record.cell_id]
            assert record.control_parameters.model_dump() == expected.model_dump()

        # For a fixed cell_id, all 5 Time_Step records must carry the *same*
        # Control_Parameter values as each other (the metadata is directory/cell-level,
        # not Time_Step-level), which is the core "propagation" invariant under test.
        for cell_id in _TARGET_CELLS:
            cell_records = [record for record in records if record.cell_id == cell_id]
            assert len(cell_records) == 5
            distinct_parameter_dumps = {
                tuple(sorted(record.control_parameters.model_dump().items())) for record in cell_records
            }
            assert len(distinct_parameter_dumps) == 1
