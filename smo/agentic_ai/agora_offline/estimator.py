"""Deterministic feasibility / goal achievement / request-deviation estimator.

Implements draft III-D and IV-E Eq. (9), not a learned conflict classifier.
LLMs can choose candidates but cannot overwrite numeric forecasts or ranking.
"""

from __future__ import annotations

from copy import deepcopy
import statistics

from .contracts import (
    CELLS,
    ContractError,
    PARAMETER_NAMES,
    finite_number,
    observed_window_kpis,
    policy_with_defaults,
    validate_intent,
    validate_parameters,
)


class MetricUnavailable(ValueError):
    """A ratio or measurement has insufficient valid evidence to be judged."""


def metric_dependencies(metric):
    return {
        "goodput_retention_pct": ("goodput_mbps",),
        "pingpong_per_attempt_pct": ("pingpongs", "ho_attempts"),
        "pingpong_per_success_pct": ("pingpongs", "ho_successes"),
        "ho_success_pct": ("ho_successes", "ho_attempts"),
        "prb_imbalance": ("prb_pct",),
        "rsrp_mean_p10_gap_db": ("rsrp_interval_mean_dbm", "rsrp_p10_dbm"),
        "prb_jain": ("prb_pct",),
        "energy_j": ("power_w",),
        "network_energy_eff_mbit_per_j": ("goodput_mbps", "power_w"),
        "energy_eff_retention_pct": ("goodput_mbps", "power_w"),
    }.get(metric, (metric,))


def analytic_power(predictions, parameters):
    result = deepcopy(predictions)
    for cell in CELLS:
        power = 130.0 + 4.7 * 10 ** ((parameters[f"txp_{cell}"] - 30.0) / 10.0)
        result[cell]["power_w"] = power
        result[cell]["energy_eff_mbit_per_j"] = result[cell]["goodput_mbps"] / power
    return result


def aggregate(predictions, expectation, keep=None, horizon=180):
    metric = expectation["metric"]

    def values(name, source=None):
        data = predictions if source is None else source
        output = []
        for cell in expectation["cells"]:
            value = data.get(cell, {}).get(name)
            try:
                value = finite_number(value, f"prediction.{cell}.{name}")
            except ContractError as exc:
                raise MetricUnavailable(str(exc)) from exc
            if (
                name
                not in (
                    "rsrp_interval_mean_dbm",
                    "sinr_interval_mean_db",
                    "rsrp_p10_dbm",
                    "rsrp_mean_p10_gap_db",
                )
                and value < 0
            ):
                raise MetricUnavailable(f"Invalid negative {name}")
            if (
                name
                in (
                    "loss_pct",
                    "bler_pct",
                    "prb_pct",
                    "ue_share_pct",
                    "traffic_share_pct",
                    "delay_over_100ms_pct",
                )
                and value > 100
            ):
                raise MetricUnavailable(f"{name} is outside 0..100 percent")
            output.append(value)
        return output

    def ratio(numerator, denominator, minimum=0.0):
        if denominator <= 1e-9 or denominator < minimum:
            raise MetricUnavailable(
                f"Insufficient denominator: {denominator:g}; minimum {minimum:g}"
            )
        return numerator / denominator

    if metric == "goodput_retention_pct":
        if keep is None:
            raise MetricUnavailable("Matched-window keep forecast is required")
        return 100 * ratio(sum(values("goodput_mbps")), sum(values("goodput_mbps", keep)))
    if metric == "pingpong_per_attempt_pct":
        pp, attempts = sum(values("pingpongs")), sum(values("ho_attempts"))
        if pp > attempts + 1e-9:
            raise MetricUnavailable("Predicted ping-pongs exceed handover attempts")
        return 100 * ratio(pp, attempts, expectation.get("minimum_denominator", 10.0))
    if metric in ("pingpong_per_success_pct", "ho_success_pct"):
        numerator = sum(
            values("pingpongs" if metric == "pingpong_per_success_pct" else "ho_successes")
        )
        denominator = sum(
            values("ho_successes" if metric == "pingpong_per_success_pct" else "ho_attempts")
        )
        if metric == "pingpong_per_success_pct" and numerator > denominator + 1e-9:
            raise MetricUnavailable("Predicted ping-pongs exceed successful handovers")
        return 100 * ratio(numerator, denominator, expectation.get("minimum_denominator", 10.0))
    if metric in ("prb_jain", "prb_imbalance"):
        utilizations = values("prb_pct")
        jain = ratio(sum(utilizations) ** 2, len(utilizations) * sum(v * v for v in utilizations))
        return 1 - jain if metric == "prb_imbalance" else jain
    if metric == "rsrp_mean_p10_gap_db":
        selected = [a - b for a, b in zip(values("rsrp_interval_mean_dbm"), values("rsrp_p10_dbm"))]
        method = expectation["aggregation"]
        return {
            "sum": sum,
            "mean": statistics.fmean,
            "min": min,
            "max": max,
            "spread": lambda x: max(x) - min(x),
            "std": statistics.pstdev,
        }[method](selected)
    if metric == "energy_j":
        return horizon * sum(values("power_w"))
    if metric in ("network_energy_eff_mbit_per_j", "energy_eff_retention_pct"):
        efficiency = ratio(sum(values("goodput_mbps")), sum(values("power_w")))
        if metric == "network_energy_eff_mbit_per_j":
            return efficiency
        if keep is None:
            raise MetricUnavailable("Matched-window keep forecast is required")
        baseline = ratio(sum(values("goodput_mbps", keep)), sum(values("power_w", keep)))
        return 100 * ratio(efficiency, baseline)
    selected = values(metric)
    method = expectation["aggregation"]
    return {
        "sum": sum,
        "mean": statistics.fmean,
        "min": min,
        "max": max,
        "spread": lambda x: max(x) - min(x),
        "std": statistics.pstdev,
    }[method](selected)


def _control_type(name):
    return name.split("_")[0]


def _control_cell(name):
    return name.split("_", 1)[1] if _control_type(name) in ("txp", "ret", "cio") else None


class Estimator:
    def __init__(self, runtime, policy=None, calibration=None):
        self.runtime = runtime
        self.policy = policy_with_defaults(policy)
        self.calibration = deepcopy(calibration)
        if self.calibration is not None and self.calibration.get("partition") not in (
            "val",
            "validation",
        ):
            raise ContractError("Safety margins may only use validation-partition calibration")

    def metadata(self):
        return {
            "selection_rule": "feasible first, U descending, D ascending (paper Eq.9)",
            "utility": "sum of positive goal weights times min(1, signed progress/requested progress)",
            "utility_lower_clipped": False,
            "deviation": "sum of per-requested-control attenuation and relocation indicators plus normalized delay shift",
            "policy": deepcopy(self.policy),
            "calibration": {
                "status": "validation_calibrated" if self.calibration else "uncalibrated",
                "calibration_id": (self.calibration or {}).get("calibration_id"),
            },
            "model": self.runtime.metadata(),
        }

    def _resolve_plan(self, plan, current, intent, issue_time_s):
        if not isinstance(plan, dict):
            raise ContractError("Each plan must be an object")
        ident = plan.get("candidate_id", plan.get("id"))
        if not isinstance(ident, str) or not ident or ident.startswith("__keep_"):
            raise ContractError(
                "A nonempty candidate_id is required (reserved __keep_ prefix forbidden)"
            )
        delay = plan.get("delay_s", 0)
        horizon = plan.get("horizon_s", intent["horizon_s"])
        if isinstance(delay, bool) or delay not in intent["allowed_delays_s"]:
            raise ContractError("Candidate delay is outside the intent's allowed trained delays")
        if horizon != intent["horizon_s"]:
            raise ContractError("All candidates must use the same intent horizon")
        if issue_time_s < 30 or issue_time_s + delay > 720:
            raise ContractError("Validated forecast start time must be between 30 and 720 seconds")
        if "parameters" in plan:
            if "changes" in plan:
                raise ContractError("Supply either parameters or changes, not both")
            parameters = deepcopy(plan["parameters"])
        else:
            changes = plan.get("changes", {})
            if not isinstance(changes, dict) or any(k not in PARAMETER_NAMES for k in changes):
                raise ContractError("changes contains an unknown or cell-specific shared control")
            parameters = dict(current, **changes)
        parameters, warnings = validate_parameters(parameters, self.policy)
        changes = {k: v for k, v in parameters.items() if abs(v - current[k]) > 1e-7}
        for key in changes:
            cell = _control_cell(key)
            if cell is not None and cell not in intent["allowed_cells"]:
                raise ContractError(f"Changing {cell} is prohibited by the intent")
        return dict(
            candidate_id=ident,
            parameters=parameters,
            changes=changes,
            delay_s=int(delay),
            horizon_s=int(horizon),
            warnings=warnings,
            axes=deepcopy(plan.get("axes", [])),
        )

    def _deviation(self, candidate, requested, current):
        """Match changed controls to requested controls without increasing magnitude."""
        sources = requested["changes"]
        unmatched = set(sources)
        mappings = []
        ordered = sorted(candidate["changes"], key=lambda key: (key not in sources, key))
        for key in ordered:
            delta = candidate["parameters"][key] - current[key]
            target_cell = _control_cell(key)
            options = []
            for source in unmatched:
                source_delta = requested["parameters"][source] - current[source]
                if _control_type(source) != _control_type(key) or delta * source_delta <= 0:
                    continue
                if abs(delta) > abs(source_delta) + 1e-7:
                    continue
                source_cell = _control_cell(source)
                if source_cell != target_cell:
                    if source_cell is None or target_cell is None:
                        continue
                    domains = self.policy["cell_ric_domains"]
                    if domains[source_cell] != domains[target_cell]:
                        continue
                options.append(source)
            if not options:
                raise ContractError(
                    f"{key}: candidate must attenuate a requested change in the same direction and RIC domain"
                )
            source = min(options, key=lambda x: (x != key, x))
            unmatched.remove(source)
            amount = abs(requested["parameters"][source] - current[source])
            mappings.append(
                dict(
                    requested_control=source,
                    candidate_control=key,
                    attenuation=max(0.0, 1 - abs(delta) / amount),
                    location_change=int(_control_cell(source) != target_cell),
                )
            )
        for source in sorted(unmatched):
            mappings.append(
                dict(
                    requested_control=source,
                    candidate_control=None,
                    attenuation=1.0,
                    location_change=0,
                )
            )
        strength = sum(m["attenuation"] for m in mappings)
        location = sum(m["location_change"] for m in mappings)
        timing = (
            abs(candidate["delay_s"] - requested["delay_s"]) / self.policy["delay_normalizer_s"]
        )
        return dict(
            total=strength + location + timing,
            strength=strength,
            location=location,
            timing=timing,
            mappings=mappings,
        )

    def _margin(self, entry, horizon, delay, multiplier):
        if multiplier == 0:
            return 0.0
        if not self.calibration:
            raise ContractError(
                "Nonzero safety multiplier requires validation calibration; use 0 for an explicitly uncalibrated point-prediction run"
            )
        if delay not in self.calibration.get("delays_s", [0]):
            raise ContractError("Validation calibration does not cover this candidate delay")
        # Exact aggregate calibration can represent nonlinear min/spread goals.
        key = "|".join(
            [
                str(horizon),
                str(delay),
                entry["metric"],
                entry["aggregation"],
                ",".join(sorted(entry["cells"])),
            ]
        )
        aggregate_rmse = self.calibration.get("aggregate_rmse", {}).get(key)
        if aggregate_rmse is not None:
            sigma = finite_number(aggregate_rmse, "aggregate validation RMSE")
        elif len(entry["cells"]) == 1:
            try:
                sigma = finite_number(
                    self.calibration["rmse"][str(horizon)][str(delay)][entry["cells"][0]][
                        entry["metric"]
                    ],
                    "validation RMSE",
                )
            except KeyError as exc:
                raise ContractError(f"Missing validation RMSE for {entry['id']}") from exc
        else:
            raise ContractError(
                "Multi-cell safety margins require validation RMSE of the actual aggregate; use per-cell guards instead"
            )
        if sigma < 0:
            raise ContractError("Validation RMSE cannot be negative")
        return multiplier * sigma

    def _expectation(self, entry, prediction, keep, observed, horizon, delay, multiplier=0.0):
        reference_data = observed if entry["reference_kind"] == "observed_at_issue" else keep
        value = reference = None
        try:
            value = aggregate(prediction, entry, keep, horizon)
            try:
                reference = aggregate(reference_data, entry, keep, horizon)
            except MetricUnavailable:
                if entry.get("constraint_group") == "goals" or entry["target_kind"] != "absolute":
                    raise
            if (
                entry["target_kind"] == "relative_fraction"
                and abs(entry["target"]) > 1e-12
                and reference <= 1e-9
            ):
                raise MetricUnavailable(
                    "A nonzero percentage improvement needs a positive baseline; use an absolute target"
                )
        except MetricUnavailable as exc:
            return dict(
                id=entry["id"],
                metric=entry["metric"],
                cells=entry["cells"],
                aggregation=entry["aggregation"],
                operator=entry["operator"],
                target_kind=entry["target_kind"],
                reference_kind=entry["reference_kind"],
                reference_value=reference,
                predicted_value=value,
                original_threshold=entry["target"] if entry["target_kind"] == "absolute" else None,
                effective_threshold=entry["target"] if entry["target_kind"] == "absolute" else None,
                safety_margin=0.0,
                satisfied=False,
                violation=None,
                normalized_violation=1.0,
                achievement=-1.0,
                weight=entry.get("weight", 1.0),
                evaluation_status="insufficient_evidence",
                reason=str(exc),
                constraint_group=entry.get("constraint_group"),
                label=entry.get("label", entry["metric"]),
            )
        if entry["target_kind"] == "relative_fraction":
            # Percentage-of-dBm is undefined operationally; require absolute dB thresholds.
            if entry["metric"] in (
                "rsrp_interval_mean_dbm",
                "sinr_interval_mean_db",
                "rsrp_p10_dbm",
                "rsrp_mean_p10_gap_db",
            ):
                raise ContractError("Use absolute dB thresholds, not fractional changes to dB/dBm")
            if reference < 0:
                raise ContractError("Relative fractions require a nonnegative reference")
            threshold = reference * (1 + entry["target"])
        elif entry["target_kind"] == "absolute_delta":
            threshold = reference + entry["target"]
        else:
            threshold = entry["target"]
        sign = 1.0 if entry["operator"] == ">=" else -1.0
        margin = self._margin(entry, horizon, delay, multiplier)
        tightened = threshold + sign * margin
        violation = max(0.0, sign * (tightened - value))
        # Used for diagnostic ordering only, not a penalty in feasible ranking.
        anchor = reference if reference is not None else threshold
        violation_scale = max(abs(threshold), abs(anchor), 1.0)
        progress = sign * (value - anchor)
        required = sign * (threshold - anchor)
        if required > 1e-12:
            achievement = min(1.0, progress / required)
            denominator_status = "positive_requested_change"
        else:
            # The goal was already true at the anchor. Preserving it earns 1;
            # losing it has signed, scale-normalized negative progress.
            achievement = (
                1.0
                if sign * (value - threshold) >= -1e-9
                else -abs(value - threshold) / violation_scale
            )
            denominator_status = "goal_already_met_at_reference"
        return dict(
            id=entry["id"],
            metric=entry["metric"],
            cells=entry["cells"],
            aggregation=entry["aggregation"],
            operator=entry["operator"],
            target_kind=entry["target_kind"],
            reference_kind=entry["reference_kind"],
            reference_value=reference,
            predicted_value=value,
            original_threshold=threshold,
            safety_margin=margin,
            effective_threshold=tightened,
            satisfied=violation <= 1e-9,
            violation=violation,
            normalized_violation=violation / violation_scale,
            signed_progress=progress,
            required_progress=required,
            achievement=achievement,
            denominator_status=denominator_status,
            weight=entry.get("weight", 1.0),
            evaluation_status="evaluated",
            constraint_group=entry.get("constraint_group"),
            label=entry.get("label", entry["metric"]),
            unit=entry.get("unit"),
            recommendation=entry.get("recommendation"),
            actual_change_pct=(
                100 * (value - reference) / reference
                if reference is not None
                and reference > 1e-9
                and entry["metric"] not in ("rsrp_interval_mean_dbm", "sinr_interval_mean_db")
                else None
            ),
        )

    def evaluate(
        self, run_id, issue_time_s, intent, requested_plan, candidates, top_k=2, explain=False
    ):
        if top_k not in (1, 2):
            raise ContractError("The estimator returns top_k 1 or 2 to the agent")
        issue_time_s = finite_number(issue_time_s, "issue_time_s")
        intent = validate_intent(intent)
        if not isinstance(candidates, list) or not candidates:
            raise ContractError("A nonempty candidate batch is required")
        snapshot = self.runtime.snapshot(run_id, issue_time_s)
        current, current_warnings = validate_parameters(snapshot["current_rcp"], self.policy)
        observed = observed_window_kpis(snapshot, intent["horizon_s"])
        requested = self._resolve_plan(requested_plan, current, intent, issue_time_s)
        if intent["safety"].get("calibration_id") and intent["safety"]["calibration_id"] != (
            self.calibration or {}
        ).get("calibration_id"):
            raise ContractError("Requested calibration_id is unavailable")
        if intent["safety"]["multiplier"] and not self.calibration:
            raise ContractError("Nonzero safety multiplier requires validation calibration")
        valid, invalid, ids = [], [], set()
        for plan in candidates:
            ident = plan.get("candidate_id", plan.get("id")) if isinstance(plan, dict) else None
            if ident in ids:
                raise ContractError("Candidate IDs must be unique within each batch")
            ids.add(ident)
            try:
                resolved = self._resolve_plan(plan, current, intent, issue_time_s)
                resolved["deviation"] = self._deviation(resolved, requested, current)
                valid.append(resolved)
            except (ContractError, TypeError) as exc:
                invalid.append(
                    dict(
                        candidate_id=ident,
                        status="invalid",
                        feasible=False,
                        all_expectations_met=False,
                        validation_error=str(exc),
                        recommended=False,
                    )
                )
        if intent["safety"]["multiplier"] and self.calibration:
            hold_only = (
                self.calibration.get("scope") == "keep_current_only"
                or self.calibration.get("switched_action_calibrated") is False
            )
            if hold_only and any(p["changes"] for p in valid):
                raise ContractError(
                    "Available validation calibration covers unchanged settings only; changed-action safety margins are not calibrated"
                )
            if self.calibration.get("interpolation_calibrated") is False and any(
                p["warnings"] for p in valid
            ):
                raise ContractError(
                    "Validation calibration does not cover interpolated control values"
                )
        delays = sorted(set(p["delay_s"] for p in valid))
        baseline_plans = [
            dict(
                candidate_id=f"__keep_{delay}",
                parameters=current,
                delay_s=delay,
                horizon_s=intent["horizon_s"],
            )
            for delay in delays
        ]
        runtime_plans = [
            {k: p[k] for k in ("candidate_id", "parameters", "delay_s", "horizon_s")} for p in valid
        ]
        raw = (
            self.runtime.predict_batch(run_id, issue_time_s, runtime_plans + baseline_plans)
            if valid
            else []
        )
        if isinstance(raw, dict):
            raw = raw.get("predictions", raw.get("results"))
        by_id = {item["candidate_id"]: item for item in raw}
        expected = {p["candidate_id"] for p in runtime_plans + baseline_plans}
        if set(by_id) != expected or len(raw) != len(expected):
            raise ContractError(
                "Runtime did not return exactly one result per requested candidate and baseline"
            )
        rows = []
        for plan in valid:
            item = by_id[plan["candidate_id"]]
            keep = by_id[f"__keep_{plan['delay_s']}"]["predictions"]
            prediction = item["predictions"]
            use_formula = intent.get("policy_version") in ("threshold-v2", "threshold-v3-app5")
            if use_formula:
                prediction = analytic_power(prediction, plan["parameters"])
                keep = analytic_power(keep, current)
            arguments = (prediction, keep, observed, intent["horizon_s"], plan["delay_s"])
            constraints = [
                self._expectation(e, *arguments, multiplier=intent["safety"]["multiplier"])
                for e in intent["constraints"]
            ]
            goals = [self._expectation(e, *arguments) for e in intent["goals"]]
            feasible = all(c["satisfied"] for c in constraints)
            goal_met = all(g["satisfied"] for g in goals)
            warnings = current_warnings + plan["warnings"] + deepcopy(item.get("warnings", []))
            if not self.calibration or intent["safety"]["multiplier"] == 0:
                warnings.append(
                    {
                        "code": "point_prediction_only",
                        "message": "No calibrated safety margin is applied; feasibility is predicted, not guaranteed",
                    }
                )
            for entry in intent["goals"] + intent["constraints"]:
                if entry["metric"] in (
                    "loss_pct",
                    "rsrp_interval_mean_dbm",
                    "sinr_interval_mean_db",
                    "delay_ms",
                ):
                    warnings.append(
                        {
                            "code": "measurement_scope",
                            "metric": entry["metric"],
                            "message": entry.get(
                                "note",
                                "Mean KPI or packet-cohort limitations apply; this is not a per-UE service guarantee",
                            ),
                        }
                    )
            rows.append(
                dict(
                    candidate_id=plan["candidate_id"],
                    status="evaluated",
                    parameters=plan["parameters"],
                    changes=plan["changes"],
                    delay_s=plan["delay_s"],
                    horizon_s=intent["horizon_s"],
                    axes=plan["axes"],
                    interval_start_s=item.get("interval_start_s", issue_time_s + plan["delay_s"]),
                    interval_end_s=item.get(
                        "interval_end_s", issue_time_s + plan["delay_s"] + intent["horizon_s"]
                    ),
                    predictions=prediction,
                    keep_predictions=keep,
                    current_parameters=current,
                    model_predictions=item["predictions"] if use_formula else None,
                    power_source="analytic_on_state_130_plus_4.7_rf_w"
                    if use_formula
                    else "runtime_output",
                    delta_vs_keep={
                        c: {k: prediction[c][k] - keep[c][k] for k in prediction[c]} for c in CELLS
                    },
                    feasible=feasible,
                    goals_met=goal_met,
                    all_expectations_met=feasible and goal_met,
                    goal_achievement=sum(g["weight"] * g["achievement"] for g in goals),
                    request_deviation=plan["deviation"]["total"],
                    deviation_components=plan["deviation"],
                    constraints=constraints,
                    goals=goals,
                    warnings=warnings,
                    violation_count=sum(not c["satisfied"] for c in constraints),
                    normalized_violation=sum(c["normalized_violation"] for c in constraints),
                    conflict_status=(
                        "insufficient_evidence"
                        if any(
                            c["evaluation_status"] == "insufficient_evidence" for c in constraints
                        )
                        else "constraint_conflict"
                    )
                    if not feasible
                    else ("none" if goal_met else "goal_unmet"),
                    recommended=False,
                )
            )
        feasible_rows = sorted(
            (r for r in rows if r["feasible"]),
            key=lambda r: (-r["goal_achievement"], r["request_deviation"], r["candidate_id"]),
        )
        recommendations = feasible_rows[:top_k]
        for row in recommendations:
            row["recommended"] = True
        diagnostics = []
        if not feasible_rows:
            diagnostics = sorted(
                rows,
                key=lambda r: (
                    r["violation_count"],
                    r["normalized_violation"],
                    -r["goal_achievement"],
                    r["request_deviation"],
                ),
            )[:top_k]
        selected = recommendations or diagnostics
        attribution_status = "not_requested"
        if explain and selected:
            attribution_status = "unavailable"
            if hasattr(self.runtime, "explain_batch"):
                selected_ids = {r["candidate_id"] for r in selected}
                explanations = self.runtime.explain_batch(
                    run_id,
                    issue_time_s,
                    [p for p in runtime_plans if p["candidate_id"] in selected_ids],
                )
                if isinstance(explanations, dict):
                    explanations = explanations.get("results", explanations.get("explanations", []))
                explanations = {r["candidate_id"]: r for r in explanations}
                for row in selected:
                    row["attribution"] = explanations.get(row["candidate_id"])
                    if row["attribution"]:
                        self._attribution_diagnostics(row, intent)
                attribution_status = "model_relative_coordinate_reset_not_causal"
        return dict(
            run_id=run_id,
            issue_time_s=issue_time_s,
            horizon_s=intent["horizon_s"],
            intent=intent,
            requested_plan=requested,
            requested_count=len(candidates),
            evaluated_count=len(rows),
            baseline_count=len(baseline_plans),
            feasible_count=len(feasible_rows),
            invalid_count=len(invalid),
            all_expectations_met=bool(
                recommendations and recommendations[0]["all_expectations_met"]
            ),
            top_candidates=recommendations,
            recommended_candidates=recommendations,
            infeasible_diagnostics=diagnostics,
            all_candidates=rows + invalid,
            baseline_predictions={
                str(d): (
                    dict(
                        by_id[f"__keep_{d}"],
                        predictions=analytic_power(by_id[f"__keep_{d}"]["predictions"], current),
                        model_predictions=by_id[f"__keep_{d}"]["predictions"],
                        power_source="analytic_on_state_130_plus_4.7_rf_w",
                    )
                    if intent.get("policy_version") in ("threshold-v2", "threshold-v3-app5")
                    else by_id[f"__keep_{d}"]
                )
                for d in delays
            },
            selection_rule="feasible first; goal_achievement descending; request_deviation ascending",
            infeasible_order="violation_count, normalized_violation, negative goal_achievement, request_deviation; diagnostic only",
            safety={
                "multiplier": intent["safety"]["multiplier"],
                "status": "validation_margin_applied"
                if intent["safety"]["multiplier"]
                else "uncalibrated_point_prediction",
                "calibration_id": (self.calibration or {}).get("calibration_id"),
            },
            attribution_status=attribution_status,
            policy=deepcopy(self.policy),
            evidence_kind="model_forecast_offline_replay_not_measured_control_effect",
        )

    @staticmethod
    def _attribution_diagnostics(row, intent):
        contrasts = row["attribution"].get("contrasts", {})
        diagnostics = []
        # Aggregate reset predictions, not the contrast itself: min/spread/std
        # are nonlinear, so aggregate(phi) is generally the wrong attribution.
        for entry in intent["goals"] + intent["constraints"]:
            try:
                predicted = aggregate(
                    row["predictions"], entry, row["keep_predictions"], row.get("horizon_s", 180)
                )
                reference = aggregate(
                    row["keep_predictions"],
                    entry,
                    row["keep_predictions"],
                    row.get("horizon_s", 180),
                )
            except MetricUnavailable:
                diagnostics.append(
                    dict(
                        expectation_id=entry["id"],
                        metric=entry["metric"],
                        effects=[],
                        status="insufficient_evidence",
                    )
                )
                continue
            sign = 1.0 if entry["operator"] == ">=" else -1.0
            effects = []
            for control in row["changes"]:
                if control not in contrasts:
                    continue
                reset = deepcopy(row["predictions"])
                for cell in entry["cells"]:
                    for metric in metric_dependencies(entry["metric"]):
                        reset[cell][metric] -= contrasts[control][cell][metric]
                if row.get("power_source", "").startswith("analytic"):
                    parameters = dict(
                        row["parameters"], **{control: row["current_parameters"][control]}
                    )
                    reset = analytic_power(reset, parameters)
                try:
                    contribution = predicted - aggregate(
                        reset, entry, row["keep_predictions"], row.get("horizon_s", 180)
                    )
                except MetricUnavailable:
                    continue
                effects.append(
                    dict(
                        control=control,
                        contribution=contribution,
                        intent_aligned_contribution=sign * contribution,
                        opposes_intent=sign * contribution < -1e-9,
                    )
                )
            effects.sort(key=lambda effect: effect["intent_aligned_contribution"])
            diagnostics.append(
                dict(
                    expectation_id=entry["id"],
                    metric=entry["metric"],
                    effects=effects,
                    net_delta_vs_keep=predicted - reference,
                    sum_of_coordinate_effects=sum(e["contribution"] for e in effects),
                    interaction_residual=sum(e["contribution"] for e in effects)
                    - (predicted - reference),
                    opposing_effects_present=any(
                        e["intent_aligned_contribution"] < -1e-9 for e in effects
                    )
                    and any(e["intent_aligned_contribution"] > 1e-9 for e in effects),
                )
            )
        row["attribution_diagnostics"] = diagnostics

    @staticmethod
    def analyze(evaluation, candidate_ids=None):
        """Return numeric diagnostics and public decision summaries, not hidden reasoning."""
        selected = set(candidate_ids) if candidate_ids is not None else None
        result = []
        for row in evaluation["all_candidates"]:
            if selected is not None and row["candidate_id"] not in selected:
                continue
            if row["status"] == "invalid":
                result.append(
                    {
                        "candidate_id": row["candidate_id"],
                        "validation_error": row["validation_error"],
                    }
                )
                continue
            result.append(
                dict(
                    candidate_id=row["candidate_id"],
                    conflict_status=row["conflict_status"],
                    violated_constraints=[c for c in row["constraints"] if not c["satisfied"]],
                    unmet_goals=[g for g in row["goals"] if not g["satisfied"]],
                    goal_achievement=row["goal_achievement"],
                    request_deviation=row["request_deviation"],
                    delta_vs_keep=row["delta_vs_keep"],
                    attribution=row.get("attribution"),
                    attribution_diagnostics=row.get("attribution_diagnostics", []),
                    classification_limit="App-pair conflict categories require explicit ownership, goal and RIC descriptors",
                )
            )
        return {"analyses": result, "evidence_kind": evaluation["evidence_kind"]}
