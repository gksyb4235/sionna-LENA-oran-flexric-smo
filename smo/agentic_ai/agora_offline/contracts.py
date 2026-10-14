"""Validated, framework-independent contracts for the AGORA paper estimator.

The policy below is an editable research scenario, not an operator SLA. The
network has three NR cells; historical labels ``4g1`` and ``4g2`` are retained
solely for compatibility with the trained model.
"""

from __future__ import annotations

from copy import deepcopy
import math

CELLS = ("5g", "4g1", "4g2")
PARAMETER_NAMES = tuple(f"{p}_{c}" for p in ("txp", "ret", "cio") for c in CELLS) + (
    "ttt_ms",
    "hys_db",
)
METRICS = (
    "goodput_mbps",
    "prb_pct",
    "delay_ms",
    "loss_pct",
    "bler_pct",
    "rsrp_interval_mean_dbm",
    "sinr_interval_mean_db",
    "ho_attempts",
    "pingpongs",
    "rlfs",
    "power_w",
    "energy_eff_mbit_per_j",
    "ue_share_pct",
    "traffic_share_pct",
    "rsrp_p10_dbm",
    "ho_successes",
    "delay_over_100ms_pct",
    "rsrp_mean_p10_gap_db",
)
DERIVED_METRICS = (
    "goodput_retention_pct",
    "pingpong_per_attempt_pct",
    "prb_jain",
    "energy_j",
    "network_energy_eff_mbit_per_j",
    "energy_eff_retention_pct",
    "pingpong_per_success_pct",
    "ho_success_pct",
    "prb_imbalance",
)
EXPECTATION_METRICS = METRICS + DERIVED_METRICS
COUNT_METRICS = frozenset(("ho_attempts", "ho_successes", "pingpongs", "rlfs"))
DELAYS = (0, 30, 60, 90)
HORIZONS = (90, 180)
AGGREGATIONS = ("sum", "mean", "min", "max", "spread", "std")
DOMAINS = {
    **{f"txp_{c}": (30.0, 33.0, 37.0, 40.0, 43.0) for c in CELLS},
    **{f"ret_{c}": (3.0, 5.0, 7.0, 9.0, 12.0) for c in CELLS},
    "cio_5g": (5.0, 6.0, 7.0, 8.0, 9.0),
    "cio_4g1": (-2.0, -1.0, 0.0, 1.0, 2.0),
    "cio_4g2": (-2.0, -1.0, 0.0, 1.0, 2.0),
    "ttt_ms": (128.0, 256.0, 340.0, 512.0, 720.0),
    "hys_db": tuple(0.5 * i for i in range(1, 11)),
}
DEFAULT_POLICY = {
    "name": "editable_offline_research_policy",
    "kind": "research_assumption_not_operator_sla",
    "cell_ric_domains": {c: "scenario_ric_1" for c in CELLS},
    "domain_provenance": "Explicit offline scenario assumption; not discovered RIC deployment",
    "allow_continuous_txp": True,
    "allow_other_interpolation": False,
    "delay_normalizer_s": 90.0,
    "expectation_editor": "threshold-v3-app5",
}


class ContractError(ValueError):
    """A request is outside a defined mathematical or model contract."""


def finite_number(value, field):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ContractError(f"{field} must be a finite number")
    return float(value)


def policy_with_defaults(policy=None):
    result = deepcopy(DEFAULT_POLICY)
    result.update(deepcopy(policy or {}))
    # Historical guards stay in archived sessions, not active control settings.
    for name in (
        "minimum_goodput_retention",
        "rlf_allowance_per_90s",
        "pingpong_multiplier",
        "minimum_pingpong_limit_per_90s",
    ):
        result.pop(name, None)
    if set(result["cell_ric_domains"]) != set(CELLS):
        raise ContractError("cell_ric_domains must explicitly map all three model cells")
    if finite_number(result["delay_normalizer_s"], "delay_normalizer_s") <= 0:
        raise ContractError("delay_normalizer_s must be positive")
    if any(
        not isinstance(result[name], bool)
        for name in ("allow_continuous_txp", "allow_other_interpolation")
    ):
        raise ContractError("Interpolation policy options must be booleans")
    if any(not isinstance(v, str) or not v for v in result["cell_ric_domains"].values()):
        raise ContractError("Each cell needs an explicit nonempty scenario RIC domain")
    return result


def validate_parameters(parameters, policy=None):
    policy = policy_with_defaults(policy)
    if not isinstance(parameters, dict) or set(parameters) != set(PARAMETER_NAMES):
        raise ContractError("parameters must contain exactly the eleven canonical controls")
    values, warnings = {}, []
    for name in PARAMETER_NAMES:
        value = finite_number(parameters[name], name)
        domain = DOMAINS[name]
        if not domain[0] <= value <= domain[-1]:
            raise ContractError(
                f"{name}={value} is outside training bounds [{domain[0]}, {domain[-1]}]"
            )
        if not any(abs(value - point) < 1e-7 for point in domain):
            allowed = (
                policy["allow_continuous_txp"]
                if name.startswith("txp_")
                else policy["allow_other_interpolation"]
            )
            if not allowed:
                raise ContractError(
                    f"{name}={value} is not a supported trained value: {list(domain)}"
                )
            warnings.append(
                {
                    "code": "interpolated_unvalidated",
                    "parameter": name,
                    "value": value,
                    "message": "Within training bounds, but this exact value was not a training level; accuracy is unvalidated",
                }
            )
        values[name] = value
    return values, warnings


def validate_intent(intent):
    if not isinstance(intent, dict):
        raise ContractError("intent must be an object")
    result = deepcopy(intent)
    if result.get("horizon_s") not in HORIZONS:
        raise ContractError("Supported horizon_s values are 90 and 180 seconds")
    allowed_delays = result.setdefault("allowed_delays_s", list(DELAYS))
    if (
        not isinstance(allowed_delays, list)
        or not allowed_delays
        or any(d not in DELAYS for d in allowed_delays)
    ):
        raise ContractError("allowed_delays_s must be a nonempty subset of [0,30,60,90]")
    cells = result.setdefault("allowed_cells", list(CELLS))
    if not cells or len(set(cells)) != len(cells) or any(c not in CELLS for c in cells):
        raise ContractError(
            "allowed_cells must be a nonempty unique subset of the three model cells"
        )
    seen = set()
    for group in ("goals", "constraints"):
        entries = result.setdefault(group, [])
        if not isinstance(entries, list):
            raise ContractError(f"{group} must be a list")
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ContractError("Each expectation must be an object")
            ident = entry.setdefault("id", f"{group}_{index}")
            if not isinstance(ident, str) or not ident or ident in seen:
                raise ContractError("Expectation IDs must be unique nonempty strings")
            seen.add(ident)
            if entry.get("metric") not in EXPECTATION_METRICS:
                raise ContractError(f"Unsupported model KPI: {entry.get('metric')}")
            selected = entry.get("cells", list(CELLS))
            if (
                not isinstance(selected, list)
                or not selected
                or len(set(selected)) != len(selected)
                or any(c not in CELLS for c in selected)
            ):
                raise ContractError(f"Invalid cells in {ident}")
            entry["cells"] = selected
            if entry.setdefault("aggregation", "mean") not in AGGREGATIONS:
                raise ContractError(f"Unsupported aggregation in {ident}")
            if (
                entry["aggregation"] in ("spread", "std")
                or entry["metric"] in ("prb_jain", "prb_imbalance")
            ) and len(selected) < 2:
                raise ContractError("A balance metric needs at least two selected cells")
            if entry.get("operator") not in ("<=", ">="):
                raise ContractError(f"operator in {ident} must be <= or >=")
            if entry.setdefault("target_kind", "absolute") not in (
                "absolute",
                "relative_fraction",
                "absolute_delta",
            ):
                raise ContractError(f"Unsupported target_kind in {ident}")
            entry["target"] = finite_number(entry.get("target"), f"{ident}.target")
            if entry["metric"] in DERIVED_METRICS and entry["aggregation"] != "sum":
                raise ContractError(
                    "Derived metrics calculate the selected cell group directly; use aggregation=sum"
                )
            if entry["target_kind"] == "relative_fraction":
                if entry["metric"] in (
                    "rsrp_interval_mean_dbm",
                    "sinr_interval_mean_db",
                    "rsrp_p10_dbm",
                    "rsrp_mean_p10_gap_db",
                ):
                    raise ContractError(
                        "Use a dB change or absolute threshold, not a percentage of dB/dBm"
                    )
                if entry["target"] < -1:
                    raise ContractError("A proportional decrease cannot exceed 100%")
            minimum = entry.setdefault(
                "minimum_denominator",
                10.0
                if entry["metric"]
                in ("pingpong_per_attempt_pct", "pingpong_per_success_pct", "ho_success_pct")
                else 0.0,
            )
            if finite_number(minimum, f"{ident}.minimum_denominator") < 0:
                raise ContractError("minimum_denominator cannot be negative")
            if (
                result.get("policy_version") in ("threshold-v2", "threshold-v3-app5")
                and group == "constraints"
                and entry["target_kind"] != "absolute"
            ):
                raise ContractError(
                    "Threshold policy constraints require absolute thresholds; use retention metrics for ratio limits"
                )
            if (
                result.get("policy_version") in ("threshold-v2", "threshold-v3-app5")
                and group == "goals"
                and entry["target_kind"] in ("relative_fraction", "absolute_delta")
                and entry["target"] * (1 if entry["operator"] == ">=" else -1) < 0
            ):
                raise ContractError(
                    "Goal improvement sign must match its increase/decrease direction"
                )
            if entry.setdefault("reference_kind", "observed_at_issue") not in (
                "observed_at_issue",
                "keep_forecast_same_window",
            ):
                raise ContractError(f"Unsupported reference_kind in {ident}")
            if group == "goals":
                entry["weight"] = finite_number(entry.get("weight", 1.0), f"{ident}.weight")
                if entry["weight"] <= 0:
                    raise ContractError("Goal weights must be positive")
    if not result["goals"]:
        raise ContractError("At least one explicit goal is required")
    safety = result.setdefault("safety", {})
    safety["multiplier"] = finite_number(safety.get("multiplier", 0.0), "safety.multiplier")
    if safety["multiplier"] < 0:
        raise ContractError("safety.multiplier cannot be negative")
    return result


def observed_window_kpis(snapshot, horizon_s):
    """Use only past observations; counts are converted to an equal-duration rate baseline."""
    window = snapshot.get("observed_90s")
    if not window or not isinstance(window.get("kpis"), dict):
        raise ContractError(
            "snapshot.observed_90s.kpis is required for an observed-at-issue reference"
        )
    seconds = finite_number(window["end_s"], "observed.end_s") - finite_number(
        window["start_s"], "observed.start_s"
    )
    if seconds <= 0:
        raise ContractError("Observed reference window must have positive duration")
    result = deepcopy(window["kpis"])
    for cell in CELLS:
        for metric in COUNT_METRICS:
            if result.get(cell, {}).get(metric) is not None:
                result[cell][metric] *= horizon_s / seconds
    return result


def make_research_intent(
    snapshot, horizon_s=180, objective="energy_saving", target_fraction=None, policy=None
):
    """Create versioned defaults without deriving event limits from history."""
    from .intent_policy import default_spec, compile_spec

    spec = default_spec(horizon_s, objective)
    if target_fraction is not None and spec["goals"][0]["target_kind"] == "relative_fraction":
        spec["goals"][0]["target"] = abs(finite_number(target_fraction, "target_fraction")) * (
            -1 if spec["goals"][0]["operator"] == "<=" else 1
        )
    return compile_spec(spec, horizon_s)
