"""Sanity/unit tests for the Data_Extractor pipeline (Task 2.2).

Full property-based coverage for Requirements 1.1-1.12 lives in the dedicated
Property task modules (2.3-2.16); this module only exercises basic
happy-path, error, and API wiring behavior.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from smo.aimlfw.data_extractor import DataExtractorError, create_app, extract, time_step_for_time_s
from smo.aimlfw.data_extractor.metadata import parse_command_sh
from smo.aimlfw.feature_store import FeatureStore

CSV_HEADER = (
    "time_s,interval_s,cell_id,cell,band,tx_power_dbm,ret_tilt_deg,cio_bias_db,ttt_ms,hysteresis_db,"
    "cell_goodput_mbps,avg_ue_goodput_mbps,ue_goodput_p5_mbps,sinr_p50_db,prb_utilization_pct,"
    "delay_p95_ms,ho_failure_count,pingpong_count,rlf_count,interval_energy_j\n"
)

COMMAND_SH = (
    "#!/usr/bin/env bash\n"
    "set -euo pipefail\n"
    "'./ns3' '--runTag=seed12_test' '--cellTxPowerDbm=gNB_5G:43,gNB_4G_1:43,gNB_4G_2:43' "
    "'--cellRetTiltDeg=gNB_5G:5,gNB_4G_1:5,gNB_4G_2:5' '--cellCioDb=gNB_5G:0,gNB_4G_1:0,gNB_4G_2:0' "
    "'--cellHysteresisDb=gNB_5G:2.5,gNB_4G_1:2.5,gNB_4G_2:2.5' '--cellTttMs=gNB_5G:160,gNB_4G_1:160,gNB_4G_2:160'\n"
)


def _row(time_s: str, cell: str, *, goodput: str = "1.0") -> str:
    return (
        f"{time_s},1,1,{cell},NR,43,5,0,160,2.5,{goodput},1.0,1.0,10.0,50.0,5.0,0,0,0,10.0\n"
    )


def _write_result_dir(tmp_path: Path, rows: list[str], *, with_command_sh: bool = True) -> Path:
    result_dir = tmp_path / "run"
    result_dir.mkdir()
    (result_dir / "cell_kpi.csv").write_text(CSV_HEADER + "".join(rows), encoding="utf-8")
    if with_command_sh:
        (result_dir / "command.sh").write_text(COMMAND_SH, encoding="utf-8")
    return result_dir


def _all_target_cell_rows(time_s: str) -> list[str]:
    return [_row(time_s, "gNB_5G"), _row(time_s, "gNB_4G_1"), _row(time_s, "gNB_4G_2")]


class TestTimeStepMapping:
    def test_boundary_values_map_to_expected_time_step(self) -> None:
        assert time_step_for_time_s("1") == 0
        assert time_step_for_time_s("180") == 0
        assert time_step_for_time_s("180.000001") == 1
        assert time_step_for_time_s("900") == 4


class TestCommandShParsing:
    def test_parses_per_cell_parameters_and_seed(self, tmp_path: Path) -> None:
        path = tmp_path / "command.sh"
        path.write_text(COMMAND_SH, encoding="utf-8")

        metadata = parse_command_sh(path)

        assert metadata.value("tx_power_dbm", "gNB_5G") == 43
        assert metadata.value("ret_tilt_deg", "gNB_4G_1") == 5
        assert metadata.value("hysteresis_db", "gNB_4G_2") == 2.5
        assert metadata.value("ttt_ms", "gNB_5G") == 160
        assert metadata.seed == 12

    def test_missing_file_yields_empty_metadata(self, tmp_path: Path) -> None:
        metadata = parse_command_sh(tmp_path / "missing_command.sh")
        assert metadata.value("tx_power_dbm", "gNB_5G") is None
        assert metadata.seed is None


class TestExtractPipeline:
    def test_extract_produces_records_for_every_target_cell_and_time_step(self, tmp_path: Path) -> None:
        rows = _all_target_cell_rows("1.5") + _all_target_cell_rows("900")
        result_dir = _write_result_dir(tmp_path, rows)
        store = FeatureStore(tmp_path / "store")

        outcome = extract(result_dir, "default", store)

        assert outcome.records_written == 3 * 5  # 3 target cells x 5 Time_Steps
        assert outcome.excluded_rows == 0

        records = store.records("default")
        assert len(records) == 15
        first_step_gnb5g = next(
            record for record in records if record.time_step == 0 and record.cell_id == "gNB_5G"
        )
        assert first_step_gnb5g.data_quality == "complete"
        assert first_step_gnb5g.features["cell_goodput_mbps"] == 1.0
        assert first_step_gnb5g.control_parameters.tx_power_dbm == 43.0
        assert first_step_gnb5g.seed == 12

    def test_extract_excludes_out_of_range_time_s_rows(self, tmp_path: Path) -> None:
        rows = _all_target_cell_rows("1.5") + [_row("901", "gNB_5G"), _row("not-a-number", "gNB_5G")]
        result_dir = _write_result_dir(tmp_path, rows)
        store = FeatureStore(tmp_path / "store")

        outcome = extract(result_dir, "default", store)

        assert outcome.excluded_rows == 2

    def test_extract_falls_back_to_csv_modal_value_without_command_sh(self, tmp_path: Path) -> None:
        rows = _all_target_cell_rows("1.5")
        result_dir = _write_result_dir(tmp_path, rows, with_command_sh=False)
        store = FeatureStore(tmp_path / "store")

        outcome = extract(result_dir, "default", store)

        assert outcome.records_written == 15
        record = store.records("default", cell_id="gNB_5G", time_step=0)[0]
        assert record.control_parameters.tx_power_dbm == 43.0
        assert record.seed is None

    def test_extract_reruns_are_idempotent(self, tmp_path: Path) -> None:
        rows = _all_target_cell_rows("1.5")
        result_dir = _write_result_dir(tmp_path, rows)
        store = FeatureStore(tmp_path / "store")

        extract(result_dir, "default", store)
        extract(result_dir, "default", store)

        assert len(store.records("default")) == 15

    def test_extract_marks_insufficient_quality_when_no_valid_samples_in_step(self, tmp_path: Path) -> None:
        rows = _all_target_cell_rows("1.5")
        result_dir = _write_result_dir(tmp_path, rows)
        store = FeatureStore(tmp_path / "store")

        extract(result_dir, "default", store)

        empty_step = store.records("default", cell_id="gNB_5G", time_step=1)[0]
        assert empty_step.data_quality == "insufficient"
        assert all(value is None for value in empty_step.features.values())

    def test_extract_raises_for_missing_csv(self, tmp_path: Path) -> None:
        result_dir = tmp_path / "missing"
        result_dir.mkdir()
        store = FeatureStore(tmp_path / "store")

        with pytest.raises(DataExtractorError) as captured:
            extract(result_dir, "default", store)
        assert captured.value.error_code == "csv_not_found"
        assert store.records("default") == []

    def test_extract_raises_for_out_of_range_control_parameter(self, tmp_path: Path) -> None:
        rows = _all_target_cell_rows("1.5")
        result_dir = _write_result_dir(tmp_path, rows, with_command_sh=False)
        bad_command = (
            "#!/usr/bin/env bash\n"
            "'./ns3' '--cellTxPowerDbm=gNB_5G:99,gNB_4G_1:43,gNB_4G_2:43'\n"
        )
        (result_dir / "command.sh").write_text(bad_command, encoding="utf-8")
        store = FeatureStore(tmp_path / "store")

        with pytest.raises(DataExtractorError) as captured:
            extract(result_dir, "default", store)
        assert captured.value.error_code == "control_parameter_out_of_range"
        assert store.records("default") == []


class TestDataExtractorApi:
    def test_extract_endpoint_returns_counts(self, tmp_path: Path) -> None:
        rows = _all_target_cell_rows("1.5")
        result_dir = _write_result_dir(tmp_path, rows)
        store = FeatureStore(tmp_path / "store")
        client = TestClient(create_app(store))

        response = client.post(
            "/extract", json={"result_dir": str(result_dir), "feature_group": "default"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["records_written"] == 15
        assert body["excluded_rows"] == 0

    def test_extract_endpoint_reports_missing_csv_as_404(self, tmp_path: Path) -> None:
        result_dir = tmp_path / "missing"
        result_dir.mkdir()
        client = TestClient(create_app(FeatureStore(tmp_path / "store")))

        response = client.post(
            "/extract", json={"result_dir": str(result_dir), "feature_group": "default"}
        )

        assert response.status_code == 404
        assert response.json()["error_code"] == "csv_not_found"
