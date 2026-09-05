"""Task 2.17: real ns-3 data layer integration and 1,000,000-row performance test.

Part 1 exercises :func:`smo.aimlfw.data_extractor.extract` against a real
``scenarios/results/**/cell_kpi.csv`` + ``command.sh`` pair checked into this
repository, verifying the per-cell 5 Time_Step contract (Requirement 1.1,
1.6), command.sh-vs-CSV fallback metadata resolution (Requirement 1.5), and
that re-running extraction against the same directory is atomic/idempotent
(Requirement 1.9) with no partial writes on either run (Requirement 1.10 —
covered indirectly by asserting the Feature_Store is untouched before the
first successful run and stable across the second).

Part 2 is a single-shot performance test (Requirement 1.1's "1,000,000
데이터 행... 300초 이내" bound) driven by a streamed synthetic
``cell_kpi.csv`` so the row data is never held in memory all at once.

**Validates: Requirements 1.1, 1.5, 1.6, 1.9, 1.10, 2.1-2.8**
"""

from __future__ import annotations

import csv
import random
import shutil
import time
from pathlib import Path

import pytest

from smo.aimlfw.data_extractor import extract
from smo.aimlfw.data_extractor.metadata import parse_command_sh
from smo.aimlfw.feature_store import FeatureStore

# smo/tests/integration/test_data_extractor_real_ns3_and_performance.py -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[3]
_RESULTS_ROOT = _REPO_ROOT / "scenarios" / "results"

_FEATURE_GROUP = "default"
_TARGET_CELLS = ("gNB_5G", "gNB_4G_1", "gNB_4G_2")

# A real, checked-in ns-3 result directory that has both cell_kpi.csv and a
# command.sh which fully specifies all 5 per-cell Control_Parameters (so the
# happy-path assertions below do not depend on CSV modal-value fallback).
_REAL_RESULT_DIR = (
    _RESULTS_ROOT
    / "RET_test_final"
    / "RET_tilt_1deg_all3_seed12_bw10M_ttt256ms_hys2p5dB"
)

# A real result directory whose command.sh omits --cellRetTiltDeg entirely,
# forcing the ret_tilt_deg Control_Parameter to resolve via the CSV
# modal-value fallback for every target cell (Requirement 1.5 fallback rule).
_FALLBACK_RESULT_DIR = _RESULTS_ROOT / "seed12_dual_nr_final_900s_20260827_163942"


def _require_real_fixture(path: Path) -> None:
    if not (path / "cell_kpi.csv").is_file() or not (path / "command.sh").is_file():
        pytest.skip(f"real ns-3 fixture not available at {path}")


class TestRealNs3ResultIntegration:
    """Part 1: extract() against real scenarios/results/**/cell_kpi.csv + command.sh."""

    def test_extract_produces_five_time_steps_per_target_cell(self, tmp_path: Path) -> None:
        _require_real_fixture(_REAL_RESULT_DIR)
        store = FeatureStore(tmp_path / "store")

        outcome = extract(_REAL_RESULT_DIR, _FEATURE_GROUP, store)

        assert outcome.records_written == len(_TARGET_CELLS) * 5
        for cell_id in _TARGET_CELLS:
            records = store.records(_FEATURE_GROUP, cell_id=cell_id)
            assert len(records) == 5
            assert sorted(record.time_step for record in records) == [0, 1, 2, 3, 4]

    def test_extract_uses_command_sh_metadata_when_fully_specified(self, tmp_path: Path) -> None:
        _require_real_fixture(_REAL_RESULT_DIR)
        store = FeatureStore(tmp_path / "store")
        expected = parse_command_sh(_REAL_RESULT_DIR / "command.sh")

        extract(_REAL_RESULT_DIR, _FEATURE_GROUP, store)

        for cell_id in _TARGET_CELLS:
            record = store.records(_FEATURE_GROUP, cell_id=cell_id, time_step=0)[0]
            assert record.seed == expected.seed
            for parameter_name in (
                "tx_power_dbm",
                "ret_tilt_deg",
                "cio_bias_db",
                "hysteresis_db",
                "ttt_ms",
            ):
                command_value = expected.value(parameter_name, cell_id)
                assert command_value is not None, (
                    f"fixture command.sh must specify {parameter_name} for {cell_id}"
                )
                assert getattr(record.control_parameters, parameter_name) == command_value

    def test_extract_falls_back_to_csv_modal_value_for_parameter_missing_from_command_sh(
        self, tmp_path: Path
    ) -> None:
        _require_real_fixture(_FALLBACK_RESULT_DIR)
        command_metadata = parse_command_sh(_FALLBACK_RESULT_DIR / "command.sh")
        # This fixture's command.sh does not carry a --cellRetTiltDeg flag at all,
        # so every target cell must resolve ret_tilt_deg via the CSV fallback.
        assert all(
            command_metadata.value("ret_tilt_deg", cell_id) is None for cell_id in _TARGET_CELLS
        )

        modal_ret_tilt: dict[str, float] = {}
        with (_FALLBACK_RESULT_DIR / "cell_kpi.csv").open("r", encoding="utf-8", newline="") as source:
            reader = csv.DictReader(source)
            counters: dict[str, dict[str, int]] = {cell_id: {} for cell_id in _TARGET_CELLS}
            for row in reader:
                cell_id = row["cell"].strip()
                if cell_id not in counters:
                    continue
                counters[cell_id][row["ret_tilt_deg"]] = counters[cell_id].get(row["ret_tilt_deg"], 0) + 1
            for cell_id, counts in counters.items():
                modal_ret_tilt[cell_id] = float(max(counts, key=counts.get))

        store = FeatureStore(tmp_path / "store")
        outcome = extract(_FALLBACK_RESULT_DIR, _FEATURE_GROUP, store)

        assert outcome.records_written == len(_TARGET_CELLS) * 5
        assert any("ret_tilt_deg" in warning for warning in outcome.warnings)
        for cell_id in _TARGET_CELLS:
            record = store.records(_FEATURE_GROUP, cell_id=cell_id, time_step=0)[0]
            assert record.control_parameters.ret_tilt_deg == modal_ret_tilt[cell_id]

    def test_rerunning_extract_on_same_real_directory_is_atomic_and_idempotent(
        self, tmp_path: Path
    ) -> None:
        _require_real_fixture(_REAL_RESULT_DIR)
        store_root = tmp_path / "store"
        store = FeatureStore(store_root)

        # Requirement 1.10 (no partial writes before the store exists at all).
        assert store.records(_FEATURE_GROUP) == []

        first_outcome = extract(_REAL_RESULT_DIR, _FEATURE_GROUP, store)
        first_snapshot = {
            (record.time_step, record.cell_id): record.model_dump(mode="json")
            for record in store.records(_FEATURE_GROUP)
        }
        assert len(first_snapshot) == len(_TARGET_CELLS) * 5

        second_outcome = extract(_REAL_RESULT_DIR, _FEATURE_GROUP, store)
        second_records = store.records(_FEATURE_GROUP)
        second_snapshot = {
            (record.time_step, record.cell_id): record.model_dump(mode="json")
            for record in second_records
        }

        # Requirement 1.9: identical record count, identical (Time_Step, cell_id)
        # key set, identical field values, and no duplicate records for any key.
        assert second_outcome.records_written == first_outcome.records_written
        assert len(second_records) == len(first_snapshot)
        assert set(second_snapshot) == set(first_snapshot)
        assert second_snapshot == first_snapshot


# --------------------------------------------------------------------------
# Part 2: 1,000,000-row synthetic performance test (Requirement 1.1's 300s bound)
# --------------------------------------------------------------------------

_PERFORMANCE_ROW_COUNT = 1_000_000
_PERFORMANCE_TIME_LIMIT_SECONDS = 300.0

_CSV_HEADER = (
    "time_s,interval_s,cell_id,cell,band,tx_power_dbm,ret_tilt_deg,cio_bias_db,ttt_ms,hysteresis_db,"
    "cell_goodput_mbps,avg_ue_goodput_mbps,ue_goodput_p5_mbps,sinr_p50_db,prb_utilization_pct,"
    "delay_p95_ms,ho_failure_count,pingpong_count,rlf_count,interval_energy_j\n"
)

_COMMAND_SH = (
    "#!/usr/bin/env bash\n"
    "set -euo pipefail\n"
    "'./ns3' '--runTag=seed99_performance' "
    "'--cellTxPowerDbm=gNB_5G:43,gNB_4G_1:43,gNB_4G_2:43' "
    "'--cellRetTiltDeg=gNB_5G:5,gNB_4G_1:5,gNB_4G_2:5' "
    "'--cellCioDb=gNB_5G:0,gNB_4G_1:0,gNB_4G_2:0' "
    "'--cellHysteresisDb=gNB_5G:2.5,gNB_4G_1:2.5,gNB_4G_2:2.5' "
    "'--cellTttMs=gNB_5G:160,gNB_4G_1:160,gNB_4G_2:160'\n"
)


def _write_synthetic_cell_kpi_csv(path: Path, row_count: int) -> None:
    """Stream ``row_count`` synthetic data rows to ``path`` without buffering them all."""
    rng = random.Random(0)
    # 900s simulation, one sample per cell per second across the 3 target cells
    # cycles time_s 1..900 repeatedly so every row lands in a valid Time_Step.
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(_CSV_HEADER)
        writer = csv.writer(handle, lineterminator="\n")
        for row_index in range(row_count):
            cell_id = _TARGET_CELLS[row_index % len(_TARGET_CELLS)]
            time_s = (row_index // len(_TARGET_CELLS)) % 900 + 1
            writer.writerow(
                [
                    time_s,
                    1,
                    1,
                    cell_id,
                    "NR",
                    43,
                    5,
                    0,
                    160,
                    2.5,
                    round(rng.uniform(0.0, 100.0), 3),
                    round(rng.uniform(0.0, 100.0), 3),
                    round(rng.uniform(0.0, 100.0), 3),
                    round(rng.uniform(-10.0, 40.0), 3),
                    round(rng.uniform(0.0, 100.0), 3),
                    round(rng.uniform(0.0, 50.0), 3),
                    rng.randint(0, 3),
                    rng.randint(0, 3),
                    rng.randint(0, 3),
                    round(rng.uniform(0.0, 500.0), 3),
                ]
            )


@pytest.mark.performance
class TestSyntheticPerformance:
    """Single-shot SLA check; excludable via ``pytest -m "not performance"``."""

    def test_extract_processes_one_million_rows_within_300_seconds(self, tmp_path: Path) -> None:
        result_dir = tmp_path / "performance_run"
        result_dir.mkdir()
        _write_synthetic_cell_kpi_csv(result_dir / "cell_kpi.csv", _PERFORMANCE_ROW_COUNT)
        (result_dir / "command.sh").write_text(_COMMAND_SH, encoding="utf-8")
        store = FeatureStore(tmp_path / "store")

        started = time.monotonic()
        outcome = extract(result_dir, _FEATURE_GROUP, store)
        elapsed = time.monotonic() - started

        assert outcome.records_written == len(_TARGET_CELLS) * 5
        assert elapsed <= _PERFORMANCE_TIME_LIMIT_SECONDS, (
            f"extract() took {elapsed:.1f}s for {_PERFORMANCE_ROW_COUNT} rows, "
            f"exceeding the {_PERFORMANCE_TIME_LIMIT_SECONDS:.0f}s SLA (Requirement 1.1)"
        )

        # Clean up the (potentially large) synthetic CSV promptly; tmp_path
        # fixture cleanup would eventually remove it, but this avoids letting
        # a multi-hundred-MB file linger for the rest of the test session.
        shutil.rmtree(result_dir, ignore_errors=True)
