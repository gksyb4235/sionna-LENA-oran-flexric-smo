import json
from pathlib import Path

import pytest

from smo.aimlfw.common.config import (
    ConfigurationError,
    load_feature_group,
    load_framework_config,
    load_kpi_thresholds,
    load_runtime_thresholds,
    load_target_kpi_aggregation,
)
from smo.aimlfw.common.constants import TARGET_KPIS


def test_default_framework_config_loads_complete_single_source_contracts() -> None:
    config = load_framework_config()
    assert tuple(config.feature_group.target_kpis) == TARGET_KPIS
    assert set(config.feature_group.aggregations) == set(config.feature_group.features)
    assert set(config.kpi_thresholds.thresholds) == set(TARGET_KPIS)
    assert set(config.target_kpi_aggregation.aggregations) == set(TARGET_KPIS)
    assert config.runtime_thresholds.conflict_threshold == 5.0
    assert config.runtime_thresholds.retraining_threshold == 15.0


def test_feature_group_name_rejects_path_traversal() -> None:
    with pytest.raises(ConfigurationError, match="invalid feature group name"):
        load_feature_group("../secret")


def test_runtime_threshold_loader_defaults_invalid_values_with_reasons(tmp_path: Path) -> None:
    path = tmp_path / "runtime_thresholds.json"
    path.write_text(
        json.dumps({"conflict_threshold": "bad", "retraining_threshold": 100.1}),
        encoding="utf-8",
    )
    thresholds = load_runtime_thresholds(path)
    assert thresholds.conflict_threshold == 5.0
    assert thresholds.retraining_threshold == 15.0
    assert len(thresholds.warnings) == 2


def test_kpi_threshold_loader_rejects_missing_kpis(tmp_path: Path) -> None:
    path = tmp_path / "kpi_thresholds.json"
    path.write_text(
        json.dumps(
            {
                "sinr_p50_db": {
                    "lower_bound": -20.0,
                    "improve_direction": "higher_is_better",
                }
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ConfigurationError, match="threshold KPI mismatch"):
        load_kpi_thresholds(path)


def test_kpi_threshold_loader_rejects_inverted_bounds(tmp_path: Path) -> None:
    data = {
        kpi: {"lower_bound": 0.0, "improve_direction": "higher_is_better"}
        for kpi in TARGET_KPIS
    }
    data["sinr_p50_db"] = {
        "lower_bound": 10.0,
        "upper_bound": 5.0,
        "improve_direction": "higher_is_better",
    }
    path = tmp_path / "kpi_thresholds.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="lower_bound cannot exceed upper_bound"):
        load_kpi_thresholds(path)


def test_target_aggregation_rejects_unknown_or_missing_kpis(tmp_path: Path) -> None:
    path = tmp_path / "target_kpi_aggregation.json"
    path.write_text(json.dumps({"unknown": "sum"}), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="canonical Target_KPI"):
        load_target_kpi_aggregation(path)


def test_yaml_feature_group_is_supported(tmp_path: Path) -> None:
    feature_dir = tmp_path / "feature_groups"
    feature_dir.mkdir()
    source = load_feature_group().model_dump(mode="json")
    import yaml

    (feature_dir / "custom.yaml").write_text(yaml.safe_dump(source), encoding="utf-8")
    loaded = load_feature_group("custom", feature_dir)
    assert loaded.target_cells == ["gNB_5G", "gNB_4G_1", "gNB_4G_2"]


def test_loader_wraps_missing_and_invalid_files(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="cannot read configuration"):
        load_runtime_thresholds(tmp_path / "missing.json")
    broken = tmp_path / "broken.json"
    broken.write_text("{", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="invalid configuration syntax"):
        load_runtime_thresholds(broken)
