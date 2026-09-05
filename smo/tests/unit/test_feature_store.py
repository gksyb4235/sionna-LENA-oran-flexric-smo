import csv
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from smo.aimlfw.common.config import load_feature_group
from smo.aimlfw.common.models import CellParameters, FeatureRecord, SampleCount
from smo.aimlfw.feature_store import FeatureStore, FeatureStoreError, create_app


FEATURE_NAMES = load_feature_group().features


def feature_record(
    time_step: int = 0,
    cell_id: str = "gNB_5G",
    *,
    value_offset: float = 0.0,
    source_dir: str = "/results/run",
) -> FeatureRecord:
    features = {name: index + 0.125 + value_offset for index, name in enumerate(FEATURE_NAMES)}
    counts = {name: SampleCount(valid=3, excluded=0) for name in FEATURE_NAMES}
    return FeatureRecord(
        feature_group="default",
        time_step=time_step,
        cell_id=cell_id,
        control_parameters=CellParameters(
            tx_power_dbm=43.0,
            ret_tilt_deg=5.0,
            cio_bias_db=0.5,
            hysteresis_db=2.5,
            ttt_ms=160,
        ),
        features=features,
        sample_counts=counts,
        data_quality="complete",
        source_dir=source_dir,
        seed=12,
    )


def test_serialize_sorts_and_parse_roundtrips_without_mutating_input(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "store")
    target = tmp_path / "export.csv"
    records = [feature_record(2, "z-cell"), feature_record(0, "b-cell"), feature_record(0, "a-cell")]
    original_order = [(record.time_step, record.cell_id) for record in records]

    assert store.serialize("default", records, target) == 3
    parsed = store.parse(target)

    assert [(record.time_step, record.cell_id) for record in parsed] == [(0, "a-cell"), (0, "b-cell"), (2, "z-cell")]
    assert [(record.time_step, record.cell_id) for record in records] == original_order
    assert sorted(records, key=lambda record: (record.time_step, record.cell_id)) == parsed


def test_empty_collection_has_a_valid_header_and_roundtrips(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "store")
    target = tmp_path / "empty.csv"

    assert store.serialize("default", [], target) == 0
    assert store.parse(target) == []
    with target.open(newline="", encoding="utf-8") as source:
        rows = list(csv.reader(source))
    assert len(rows) == 1
    assert set(FEATURE_NAMES).issubset(rows[0])


def test_upsert_is_idempotent_and_replaces_the_same_composite_key(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "store")
    first = feature_record(1, "gNB_5G")
    replacement = feature_record(1, "gNB_5G", value_offset=10.0, source_dir="/results/retry")
    other = feature_record(1, "gNB_4G_1")

    assert store.upsert([first, other]) == (2, 2)
    assert store.upsert([first, other]) == (2, 2)
    assert store.upsert([replacement]) == (1, 2)

    records = store.records("default", time_step=1)
    assert len(records) == 2
    replaced = next(record for record in records if record.cell_id == "gNB_5G")
    assert replaced.source_dir == "/results/retry"
    assert replaced.features[FEATURE_NAMES[0]] == pytest.approx(10.125)


def test_schema_mismatch_reports_both_field_lists_and_no_partial_records(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "store")
    target = tmp_path / "records.csv"
    store.serialize("default", [feature_record()], target)
    text = target.read_text(encoding="utf-8")
    target.write_text(text.replace(FEATURE_NAMES[0], "undefined_kpi", 1), encoding="utf-8")

    with pytest.raises(FeatureStoreError) as captured:
        store.parse(target)

    assert captured.value.error_code == "missing_fields"
    assert captured.value.details["missing_fields"] == [FEATURE_NAMES[0]]
    assert captured.value.details["undefined_fields"] == ["undefined_kpi"]


def test_truncated_record_reports_first_zero_based_index(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "store")
    target = tmp_path / "records.csv"
    store.serialize("default", [feature_record(0), feature_record(1)], target)
    lines = target.read_text(encoding="utf-8").splitlines()
    lines[2] = ",".join(lines[2].split(",")[:-1])
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(FeatureStoreError) as captured:
        store.parse(target)

    assert captured.value.error_code == "truncated_record"
    assert captured.value.details["record_index"] == 1


def test_invalid_serialize_leaves_existing_target_unchanged(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "store")
    target = tmp_path / "records.csv"
    target.write_text("original", encoding="utf-8")
    invalid = feature_record().model_copy(
        update={
            "features": {"unknown": 1.0},
            "sample_counts": {"unknown": SampleCount(valid=1, excluded=0)},
        }
    )

    with pytest.raises(FeatureStoreError, match="field names"):
        store.serialize("default", [invalid], target)

    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.glob(".records.csv.*.tmp")) == []


def test_feature_store_api_exposes_upsert_query_serialize_and_parse(tmp_path: Path) -> None:
    store = FeatureStore(tmp_path / "store")
    client = TestClient(create_app(store))
    record = feature_record(3, "gNB_5G")

    upsert = client.post("/records", json={"records": [record.model_dump(mode="json")]})
    assert upsert.status_code == 200
    assert upsert.json() == {"upserted": 1, "total": 1}

    queried = client.get("/records", params={"feature_group": "default", "time_step": 3})
    assert queried.status_code == 200
    assert queried.json()["count"] == 1
    assert queried.json()["records"][0]["cell_id"] == "gNB_5G"

    export = tmp_path / "api-export.csv"
    serialized = client.post(
        "/serialize",
        json={
            "feature_group": "default",
            "records": [record.model_dump(mode="json")],
            "target_path": str(export),
        },
    )
    assert serialized.status_code == 200
    assert serialized.json() == {"written": 1}

    parsed = client.post("/parse", json={"source_path": str(export)})
    assert parsed.status_code == 200
    assert parsed.json()["count"] == 1
    assert parsed.json()["records"][0] == record.model_dump(mode="json")


def test_feature_store_api_uses_atomic_error_contract(tmp_path: Path) -> None:
    client = TestClient(create_app(FeatureStore(tmp_path / "store")))

    missing = client.post("/parse", json={"source_path": str(tmp_path / "missing.csv")})
    assert missing.status_code == 404
    assert missing.json()["error_code"] == "source_not_found"
    assert "records" not in missing.json()

    invalid = client.post("/serialize", json={"feature_group": "default"})
    assert invalid.status_code == 422
    assert invalid.json()["error_code"] == "invalid_request"
    assert set(invalid.json()) == {"error_code", "message", "details"}
