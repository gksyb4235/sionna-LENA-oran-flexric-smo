"""Property 41 test module (own module per the repository's Property-per-file convention).

Task 7.4: Property 41 속성 테스트 작성: Baseline_Parameter_Set 구성.

Targets ``smo/agentic_ai/agents/kpi-advisor/agent.py``'s ``build_baseline`` (task
7.2) directly, with a fake Feature_Store standing in for the real one, exactly
as ``test_agent_baseline_and_probe_plan.py`` does for its fixed-example
coverage. Follows ``test_contract.py``/``test_mcp_server_contract.py``'s manual
``sys.path`` setup convention since ``kpi-advisor`` is not installed as a
package during tests, and follows
``smo/tests/property/test_property_29_batch_order_preservation_and_completeness.py``'s
convention (also used by ``test_property_37_success_response_passthrough.py``)
of redefining small test-private helpers locally rather than importing them
from another test module.

Requirement 7.1:
    WHEN xApp 파라미터 요청이 도착하면, THE KPI_Advisor_Agent SHALL 요청에
    지정되지 않은 Control_Parameter 를 Feature_Store 의 가장 최근 Feature_Record
    값으로 채워 5개 Control_Parameter 가 모두 지정된 Baseline_Parameter_Set 1개를
    구성하고, 대상 셀 집합을 요청에 명시된 1개 이상 16개 이하의 cell_id 로 확정한다.

Design tag (design.md, "KPI_Advisor_Agent — 배치 프로빙 (Requirement 7)"):

    #### Property 41: Baseline_Parameter_Set 구성
    *For any* 부분적으로 지정된 xApp 파라미터 요청(1개 이상 16개 이하의 대상 셀),
    구성된 Baseline_Parameter_Set은 각 대상 셀마다 5개 Control_Parameter 값을 모두
    가지며, 요청에서 지정된 값은 그대로 사용되고 미지정 값은 Feature_Store의 해당
    셀 최근 Feature_Record 값과 일치한다.
    **Validates: Requirements 7.1**

Unlike ``test_agent_baseline_and_probe_plan.py``'s handful of fixed examples
(most-recent-record-wins for a single cell, full-target-cell-coverage across
two cells, and the missing-record error), this module sweeps many
Hypothesis-generated combinations of: 1..16 target cells, a random 0..5
subset of xApp overrides per cell, and 1..5 Feature_Records per cell at
distinct ``time_step`` values (0..4) with independently varying valid
Control_Parameter values -- confirming the override-wins /
highest-time_step-wins /full-cell-coverage invariants hold across the whole
generated input space, not just the fixed examples above.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest
from hypothesis import given
from hypothesis import strategies as st

AGENT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(AGENT_DIR))

from smo.aimlfw.common.constants import CONTROL_PARAMETER_NAMES, MAX_TARGET_CELLS  # noqa: E402
from smo.aimlfw.common.models import CellParameters, FeatureRecord, SampleCount  # noqa: E402
from smo.tests.property.strategies import (  # noqa: E402
    PROPERTY_TEST_SETTINGS,
    cell_ids,
    valid_cell_parameters,
)

import agent  # noqa: E402
from schemas import PartialCellParameters, XAppParameterRequest  # noqa: E402


def _feature_record(
    *, cell_id: str, time_step: int, control_parameters: dict[str, float | int]
) -> FeatureRecord:
    """Local copy of ``test_agent_baseline_and_probe_plan.py``'s helper (see module docstring)."""
    kpi_names = ("cell_goodput_mbps",)
    return FeatureRecord(
        feature_group="default",
        time_step=time_step,
        cell_id=cell_id,
        control_parameters=CellParameters.model_validate(control_parameters),
        features={name: 1.0 for name in kpi_names},
        sample_counts={name: SampleCount(valid=1, excluded=0) for name in kpi_names},
        data_quality="complete",
        source_dir="/tmp/results/example",
        seed=0,
    )


class _FakeFeatureStore:
    """Duck-typed Feature_Store stand-in: only implements ``.records(...)``.

    Local copy of ``test_agent_baseline_and_probe_plan.py``'s helper (see module docstring).
    """

    def __init__(self, records: list[FeatureRecord]) -> None:
        self._records = records

    def records(
        self, feature_group: str, *, cell_id: str | None = None, time_step: int | None = None
    ) -> list[FeatureRecord]:
        return [
            record
            for record in self._records
            if record.feature_group == feature_group
            and (cell_id is None or record.cell_id == cell_id)
            and (time_step is None or record.time_step == time_step)
        ]


@st.composite
def _baseline_construction_cases(draw: Any) -> dict[str, Any]:
    """Generate one (xApp request, Feature_Store contents, expectation) case.

    - 1..16 unique target cells.
    - Each cell gets a random 0..5 subset of the 5 Control_Parameters
      explicitly overridden (at least one cell must have >=1 override, since
      ``XAppParameterRequest.parameter_overrides`` requires a non-empty dict).
    - Each cell gets 1..5 fake Feature_Records at unique ``time_step`` values
      (0..4) with independently varying valid Control_Parameter values, so
      "가장 최근" (highest ``time_step``) is unambiguous per cell.
    """
    target_cells = draw(cell_ids(min_size=1, max_size=MAX_TARGET_CELLS))

    override_names_by_cell: dict[str, list[str]] = {}
    for cell_id in target_cells:
        count = draw(st.integers(min_value=0, max_value=len(CONTROL_PARAMETER_NAMES)))
        permuted_names = draw(st.permutations(list(CONTROL_PARAMETER_NAMES)))
        override_names_by_cell[cell_id] = permuted_names[:count]

    if not any(override_names_by_cell.values()):
        forced_cell = target_cells[0]
        count = draw(st.integers(min_value=1, max_value=len(CONTROL_PARAMETER_NAMES)))
        permuted_names = draw(st.permutations(list(CONTROL_PARAMETER_NAMES)))
        override_names_by_cell[forced_cell] = permuted_names[:count]

    override_values_by_cell: dict[str, dict[str, float | int]] = {}
    for cell_id, names in override_names_by_cell.items():
        if not names:
            continue
        sample_parameters = draw(valid_cell_parameters()).model_dump()
        override_values_by_cell[cell_id] = {name: sample_parameters[name] for name in names}

    records_by_cell: dict[str, list[FeatureRecord]] = {}
    for cell_id in target_cells:
        record_count = draw(st.integers(min_value=1, max_value=5))
        time_steps = draw(st.permutations(list(range(5))))[:record_count]
        cell_records = []
        for time_step in time_steps:
            parameters = draw(valid_cell_parameters())
            cell_records.append(
                _feature_record(cell_id=cell_id, time_step=time_step, control_parameters=parameters.model_dump())
            )
        records_by_cell[cell_id] = cell_records

    return {
        "target_cells": target_cells,
        "override_names_by_cell": override_names_by_cell,
        "override_values_by_cell": override_values_by_cell,
        "records_by_cell": records_by_cell,
    }


# **Property 41: Baseline_Parameter_Set 구성**
# **Validates: Requirements 7.1**
@pytest.mark.property
@PROPERTY_TEST_SETTINGS
@given(case=_baseline_construction_cases())
def test_build_baseline_completes_every_target_cell_with_overrides_and_most_recent_feature_records(
    case: dict[str, Any],
) -> None:
    target_cells: list[str] = case["target_cells"]
    override_names_by_cell: dict[str, list[str]] = case["override_names_by_cell"]
    override_values_by_cell: dict[str, dict[str, float | int]] = case["override_values_by_cell"]
    records_by_cell: dict[str, list[FeatureRecord]] = case["records_by_cell"]

    all_records = [record for records in records_by_cell.values() for record in records]
    store = _FakeFeatureStore(all_records)
    xapp_request = XAppParameterRequest(
        target_cells=target_cells,
        parameter_overrides={
            cell_id: PartialCellParameters(**overrides)
            for cell_id, overrides in override_values_by_cell.items()
        },
    )

    baseline = agent.build_baseline(xapp_request, store, feature_group="default")

    # (a) The target cell set is confirmed exactly -- no more, no fewer.
    assert set(baseline.cells) == set(target_cells)

    for cell_id in target_cells:
        cell = baseline.cells[cell_id]
        overridden_names = set(override_names_by_cell[cell_id])
        most_recent_record = max(records_by_cell[cell_id], key=lambda record: record.time_step)

        for name in CONTROL_PARAMETER_NAMES:
            value = getattr(cell, name)
            # (b) Every Control_Parameter is set to a non-None, valid value.
            assert value is not None

            if name in overridden_names:
                # (c) Explicit xApp overrides win verbatim.
                assert value == override_values_by_cell[cell_id][name]
            else:
                # (d) Unspecified parameters come from the highest-time_step
                # (most recent) Feature_Record for that cell.
                assert value == getattr(most_recent_record.control_parameters, name)
