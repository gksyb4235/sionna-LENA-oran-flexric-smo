"""Common pytest and Hypothesis configuration for SMO tests."""

from __future__ import annotations

import pytest
from hypothesis import settings

from smo.aimlfw.common.config import FrameworkConfig, load_framework_config
from smo.aimlfw.common.models import CellParameters, FeatureRecord, ParameterSet, SampleCount
from smo.tests.property.strategies import MAX_EXAMPLES

settings.register_profile("smo", max_examples=MAX_EXAMPLES, deadline=None)
settings.load_profile("smo")


@pytest.fixture(scope="session")
def framework_config() -> FrameworkConfig:
    """Return the validated canonical framework configuration."""
    return load_framework_config()


@pytest.fixture(scope="session")
def default_cell_parameters() -> CellParameters:
    """Return a stable valid parameter tuple for example and edge-case tests."""
    return CellParameters(
        tx_power_dbm=43.0,
        ret_tilt_deg=5.0,
        cio_bias_db=0.5,
        hysteresis_db=2.5,
        ttt_ms=160,
    )


@pytest.fixture(scope="session")
def default_parameter_set(
    framework_config: FrameworkConfig,
    default_cell_parameters: CellParameters,
) -> ParameterSet:
    """Return a complete canonical three-cell ParameterSet."""
    return ParameterSet(
        cells={cell_id: default_cell_parameters for cell_id in framework_config.feature_group.target_cells}
    )


@pytest.fixture
def default_feature_record(
    framework_config: FrameworkConfig,
    default_cell_parameters: CellParameters,
) -> FeatureRecord:
    """Return a complete FeatureRecord suitable for filesystem test setup."""
    names = framework_config.feature_group.features
    return FeatureRecord(
        feature_group=framework_config.feature_group.name,
        time_step=0,
        cell_id=framework_config.feature_group.target_cells[0],
        control_parameters=default_cell_parameters,
        features={name: 1.0 for name in names},
        sample_counts={name: SampleCount(valid=1, excluded=0) for name in names},
        data_quality="complete",
        source_dir="/tmp/smo-property-fixture",
        seed=0,
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register shared markers even when repository-level pytest config wins."""
    config.addinivalue_line(
        "markers",
        "property: property-based test with design Property and Validates tags",
    )
    config.addinivalue_line(
        "markers",
        "performance: single-shot SLA/performance test (may be slow; excludable via -m 'not performance')",
    )
