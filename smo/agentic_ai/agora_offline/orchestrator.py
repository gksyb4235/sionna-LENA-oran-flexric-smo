"""Five role agents, bounded tools, immutable estimator evidence and SSE events."""

import asyncio
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import time
import uuid
from .contracts import (
    CELLS,
    PARAMETER_NAMES,
    METRICS,
    EXPECTATION_METRICS,
    DOMAINS,
    make_research_intent,
    validate_intent,
)
from .estimator import Estimator, metric_dependencies
from .intent_policy import compile_spec
from .operator_review import shortlist
from .planning import alternatives, parse_deterministic
from .search import AxisSearch
from .provider import Provider, function, public_configuration
from .settings import STATE


def compact_snapshot(snapshot):
    return {k: v for k, v in snapshot.items() if k not in ("history", "kpi_history")}


def compact_expectation(entry):
    keys = (
        "id",
        "metric",
        "cells",
        "aggregation",
        "operator",
        "reference_kind",
        "reference_value",
        "predicted_value",
        "effective_threshold",
        "safety_margin",
        "satisfied",
        "violation",
        "signed_progress",
        "achievement",
        "weight",
        "evaluation_status",
        "reason",
        "constraint_group",
        "label",
        "unit",
        "actual_change_pct",
    )
    return {k: entry[k] for k in keys if k in entry}


def compact_candidate(row):
    keys = (
        "candidate_id",
        "changes",
        "parameters",
        "current_parameters",
        "delay_s",
        "horizon_s",
        "interval_start_s",
        "interval_end_s",
        "status",
        "feasible",
        "goals_met",
        "all_expectations_met",
        "goal_achievement",
        "request_deviation",
        "conflict_status",
        "validation_error",
        "warnings",
        "operator_score",
        "operator_rank",
        "review_kind",
        "suggested_ric",
    )
    reduced = {k: row[k] for k in keys if k in row}
    reduced["violated_constraints"] = [
        compact_expectation(c) for c in row.get("constraints", []) if not c["satisfied"]
    ]
    reduced["constraint_headroom"] = [compact_expectation(c) for c in row.get("constraints", [])]
    reduced["goals"] = [compact_expectation(g) for g in row.get("goals", [])]
    return reduced


def compact_tool_result(name, result):
    if name == "Analyze":
        analyses = []
        for row in result["analyses"]:
            item = {
                k: row[k]
                for k in (
                    "candidate_id",
                    "conflict_status",
                    "goal_achievement",
                    "request_deviation",
                    "classification_limit",
                    "validation_error",
                )
                if k in row
            }
            for field in ("violated_constraints", "unmet_goals"):
                item[field] = [compact_expectation(c) for c in row.get(field, [])]
            item["attribution_diagnostics"] = row.get("attribution_diagnostics", [])
            analyses.append(item)
        return {
            "analyses": analyses,
            "evidence_kind": result["evidence_kind"],
            "attribution_kind": "model coordinate reset contrast; not SHAP or identified causality",
        }
    if name == "lookup_context":
        fields = (
            "id",
            "feature",
            "cell",
            "kpi",
            "effect_unit",
            "effect_900s",
            "shap_status",
            "derived",
            "flags",
        )
        summary = {
            k: result[k]
            for k in (
                "artifact_kind",
                "reference_action",
                "applicability",
                "scope",
                "reasoning_rules",
                "decision_status",
                "retrieval",
            )
        }
        summary["results"] = [{k: r[k] for k in fields if k in r} for r in result["results"]]
        return summary
    return result


def compact_evaluation(evaluation):
    finalists = evaluation["top_candidates"] or evaluation["infeasible_diagnostics"]
    rows = [compact_candidate(row) for row in finalists[:2]]
    return {
        "candidates": rows,
        "top_candidate_ids": [c["candidate_id"] for c in evaluation["top_candidates"]],
        "feasible_count": evaluation["feasible_count"],
        "evaluated_count": evaluation["evaluated_count"],
        "invalid_candidates": [
            {"candidate_id": c["candidate_id"], "error": c["validation_error"]}
            for c in evaluation["all_candidates"]
            if c.get("status") == "invalid"
        ],
        "selection_rule": evaluation["selection_rule"],
        "compression": "Up to two feasible recommendations or infeasible diagnostics; full batch retained in UI audit",
    }


EXPECTATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "id": {"type": "string"},
        "metric": {"type": "string", "enum": list(EXPECTATION_METRICS)},
        "cells": {"type": "array", "items": {"type": "string", "enum": list(CELLS)}},
        "aggregation": {"type": "string", "enum": ["sum", "mean", "min", "max", "spread", "std"]},
        "operator": {"type": "string", "enum": ["<=", ">="]},
        "target_kind": {
            "type": "string",
            "enum": ["absolute", "relative_fraction", "absolute_delta"],
        },
        "target": {"type": "number"},
        "weight": {"type": "number"},
        "reference_kind": {
            "type": "string",
            "enum": ["observed_at_issue", "keep_forecast_same_window"],
        },
    },
    "required": [
        "id",
        "metric",
        "cells",
        "aggregation",
        "operator",
        "target_kind",
        "target",
        "reference_kind",
    ],
}
INTENT_TOOL = function(
    "submit_intent",
    "Preserve the exact requested action. Parameters belong only in changes, NEVER in KPI constraints. Only add goals/constraints for numeric KPI expectations EXPLICITLY stated by the user. When using the default research policy omit goals/constraints or return empty arrays; the server instantiates defaults. No prediction occurs here.",
    {
        "changes": {
            "type": "object",
            "properties": {k: {"type": "number"} for k in PARAMETER_NAMES},
            "additionalProperties": False,
        },
        "expected_current": {
            "type": "object",
            "properties": {k: {"type": "number"} for k in PARAMETER_NAMES},
            "additionalProperties": False,
        },
        "horizon_s": {"type": "integer"},
        "delay_s": {"type": "integer"},
        "objective": {
            "type": "string",
            "enum": [
                "energy_saving",
                "throughput",
                "coverage",
                "load_balancing",
                "handover_control",
                "custom",
            ],
        },
        "target_fraction": {
            "type": "number",
            "description": "Positive proportional improvement. Without a UI policy or explicit target, use0.05 for energy or0.20 for load/mobility and disclose research default. The locked UI policy takes precedence.",
        },
        "interpretation": {"type": "string"},
        "goals": {
            "type": "array",
            "items": EXPECTATION_SCHEMA,
            "description": "Only explicit user numeric KPI goals; omit for default research objective",
        },
        "constraints": {
            "type": "array",
            "items": EXPECTATION_SCHEMA,
            "description": "Only explicit user numeric KPI guards; omit for default research policy. Not parameter limits.",
        },
    },
    ["changes", "horizon_s", "delay_s", "objective", "target_fraction", "interpretation"],
)
SEARCH_TOOL = function(
    "estimate_batch",
    "Evaluate a vectorized batch. The server adds mandatory time/intensity/cell grid candidates so no axis or weaker supported value is silently omitted. Your proposals supplement this grid. Subsequent calls evaluate untested combinations until a goal-meeting feasible candidate is found or the declared grid/budget is exhausted. Results include actual coverage, top2 candidates and numeric analysis. Preserve action type/direction/magnitude.",
    {
        "axis": {"type": "string", "enum": ["all", "time", "intensity", "cell", "combined"]},
        "base_candidate_id": {"type": "string"},
        "values": {"type": "array", "items": {"type": "number"}, "maxItems": 10},
        "cells": {"type": "array", "items": {"type": "string", "enum": list(CELLS)}, "maxItems": 3},
        "delays_s": {
            "type": "array",
            "items": {"type": "integer", "enum": [0, 30, 60, 90]},
            "maxItems": 4,
        },
        "fractions": {
            "type": "array",
            "items": {"type": "number", "minimum": 0, "maximum": 1},
            "maxItems": 10,
        },
        "candidates": {
            "type": "array",
            "maxItems": 48,
            "items": {"type": "object"},
            "description": "Optional explicit plans: changes object, delay_s, horizon_s. Use this to mix all three axes in one batch.",
        },
        "summary": {"type": "string", "description": "Short public purpose of this batch"},
    },
    ["axis", "summary"],
)
ANALYZE_TOOL = function(
    "Analyze",
    "Return deterministic constraint and goal diagnostics with signed model sensitivities for finalists.",
    {"candidate_ids": {"type": "array", "items": {"type": "string"}, "maxItems": 8}},
)
CONTEXT_TOOL = function(
    "lookup_context",
    "Retrieve scoped SHAP-screened simulator DAG evidence; not a GNN input or a transported causal estimate.",
    {
        "feature": {"type": "string", "enum": list(PARAMETER_NAMES)},
        "kpi": {
            "type": "string",
            "enum": list(METRICS),
            "description": "Canonical KPI name, such as goodput_mbps; do not use a constraint ID such as goodput_5g",
        },
    },
    ["feature"],
)


class Session:
    def __init__(self, body, runtime, settings):
        self.id = uuid.uuid4().hex
        self.body, self.runtime, self.settings = deepcopy(body), runtime, deepcopy(settings)
        self.events, self.evaluations, self.plan_index = [], [], {}
        self.changed = asyncio.Event()
        self.status, self.result = "queued", None
        self.task = None
        self.cancelled = False
        self.requested = self.intent = self.snapshot = None
        self.provider = Provider(settings)
        self.estimator = Estimator(runtime, settings["policy"])
        self.batch_count = self.candidate_count = 0
        self.shared_evidence = []
        self.search = None
        self.directory = STATE / "sessions" / self.id

    async def emit(self, kind, agent="system", content="", data=None):
        if self.cancelled and kind not in ("done", "error"):
            raise asyncio.CancelledError()
        event = {
            "seq": len(self.events) + 1,
            "type": kind,
            "agent": agent,
            "content": content,
            "data": data,
            "ts": datetime.now(timezone.utc).isoformat(),
        }
        self.events.append(event)
        self.directory.mkdir(parents=True, exist_ok=True)
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        self.changed.set()

    def state(self):
        return {
            "session_id": self.id,
            "status": self.status,
            "request": self.body,
            "result": self.result,
            "events": self.events,
        }

    def persist(self):
        (self.directory / "session.json").write_text(
            json.dumps(self.state(), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )

    async def invoke(self, name, arguments, role, action):
        await self.emit("tool_call", role, name, {"tool": name, "arguments": arguments})
        begin = time.perf_counter()
        try:
            result = await action()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.emit("tool_result", role, name, {"tool": name, "error": str(exc)})
            raise
        await self.emit(
            "tool_result",
            role,
            name,
            {
                "tool": name,
                "result": result,
                "elapsed_ms": round((time.perf_counter() - begin) * 1000, 2),
            },
        )
        if name in ("Analyze", "lookup_context"):
            self.shared_evidence.append(
                {
                    "tool": name,
                    "source_event_seq": self.events[-1]["seq"],
                    "arguments": deepcopy(arguments),
                    "result": compact_tool_result(name, result),
                }
            )
        return result

    async def evaluate(self, plans, role="mitigation"):
        if not plans or len(plans) > 48 or self.candidate_count + len(plans) > 160:
            raise ValueError(
                "Each batch must contain1..48 candidates, with at most160 total per session"
            )
        if self.batch_count >= 5:
            raise ValueError(
                "At most five batches (original plus four alternative rounds) are allowed"
            )
        self.batch_count += 1
        self.candidate_count += len(plans)
        for i, plan in enumerate(plans):
            plan = deepcopy(plan)
            plan["candidate_id"] = (
                "requested" if role == "conflict" else f"b{self.batch_count}_c{i + 1}"
            )
            self.plan_index[plan["candidate_id"]] = plan
            plans[i] = plan

        async def run():
            return await asyncio.to_thread(
                self.estimator.evaluate,
                self.body["run_id"],
                self.body["issue_time_s"],
                self.intent,
                self.requested,
                plans,
                2,
                True,
            )

        result = await self.invoke(
            "estimate_batch",
            {
                "candidates": plans,
                "candidate_count": len(plans),
                "internal_keep_reference": "one per distinct delay; not an alternative candidate",
            },
            role,
            run,
        )
        self.evaluations.append(result)
        if self.search:
            self.search.record(result)
        await self.emit("candidates", role, f"{len(plans)} candidates evaluated", result)
        return result

    def search_report(self):
        feasible, _ = self.ranked()
        met = bool(feasible and feasible[0]["all_expectations_met"])
        return self.search.report(met, self.batch_count < 5 and self.candidate_count < 160)

    async def publish_search_report(self):
        report = self.search_report()
        await self.emit(
            "search_coverage", "mitigation", report["stop_reason"], {"search_coverage": report}
        )
        return report

    async def prepare_mitigation_context(self, first, analyst_summary):
        # Always deliver broad evidence for original and permitted relocated
        # controls, even when the analyst looked up only one KPI or skipped DAG.
        features = set(self.requested["changes"])
        for plan in self.search.first:
            if "cell" in plan["axes"]:
                features.update(plan["changes"])
        for feature in sorted(features):
            broad = any(
                e["tool"] == "lookup_context"
                and e["arguments"].get("feature") == feature
                and not e["arguments"].get("kpi")
                for e in self.shared_evidence
            )
            if not broad:
                await self.tool("lookup_context", {"feature": feature}, "analyst")
        contexts = {}
        numeric = []
        for item in self.shared_evidence:
            if item["tool"] == "lookup_context":
                feature = item["arguments"]["feature"]
                if feature not in contexts or not item["arguments"].get("kpi"):
                    contexts[feature] = item
            else:
                numeric.append(item)
        evidence = numeric + list(contexts.values())
        context = {
            "request": self.requested,
            "intent": self.intent,
            "current_rcp": self.snapshot["current_rcp"],
            "check": compact_evaluation(first),
            "trained_parameter_levels": DOMAINS,
            "continuous_txp_policy": "In-range non-grid TxP values are accepted as unvalidated interpolation; the search includes weaker integer-dBm values."
            if self.settings["policy"]["allow_continuous_txp"]
            else "Non-grid TxP interpolation is disabled; only trained levels are allowed.",
            "analyst_summary": analyst_summary,
            "shared_numeric_and_scoped_context": evidence,
            "search_coverage": self.search_report(),
            "context_provenance": {
                "dag_features": sorted(contexts),
                "dag_relations": sum(len(e["result"]["results"]) for e in contexts.values()),
                "source_event_sequences": [e["source_event_seq"] for e in evidence],
                "sha256": hashlib.sha256(json.dumps(evidence, sort_keys=True).encode()).hexdigest(),
            },
            "instructions": "Use estimate_batch to evaluate all three independent axes together. Read coverage and numeric analysis. Continue with untested combinations when goals remain unmet. Do not conclude no feasible plan or offer optional future work while search_coverage.stop_allowed is false. Feasibility claims apply only to evaluated candidates.",
        }

        async def delivered():
            return deepcopy(context)

        return await self.invoke(
            "prepare_mitigation_context",
            {"features": sorted(features), "includes_scoped_DAG": True},
            "mitigation",
            delivered,
        )

    def ranked(self):
        dedup = {}
        for result in self.evaluations:
            for row in result["all_candidates"]:
                if row.get("status") == "invalid":
                    continue
                key = (tuple(sorted(row["parameters"].items())), row["delay_s"])
                dedup.setdefault(key, row)
        rows = list(dedup.values())
        feasible = sorted(
            (r for r in rows if r["feasible"]),
            key=lambda r: (-r["goal_achievement"], r["request_deviation"], r["candidate_id"]),
        )
        diagnostics = sorted(
            (r for r in rows if not r["feasible"]),
            key=lambda r: (
                r["violation_count"],
                r["normalized_violation"],
                -r["goal_achievement"],
                r["request_deviation"],
            ),
        )
        return feasible, diagnostics

    async def lookup_context(self, arguments):
        from .context.query_dag import load_bundle, query

        graph, details = await asyncio.to_thread(load_bundle)
        feature = arguments["feature"]
        result = query(
            graph,
            details,
            feature=feature,
            kpi=arguments.get("kpi"),
            current_rcp=self.snapshot["current_rcp"],
            after=self.requested["changes"].get(feature),
            mode="switched_action_offline_forecast",
            duration_s=self.intent["horizon_s"],
            include_derived=True,
        )
        relevant = {
            m
            for e in self.intent["goals"] + self.intent["constraints"]
            for m in metric_dependencies(e["metric"])
        }
        result["results"] = sorted(
            result["results"],
            key=lambda r: (r.get("kpi") not in relevant, r.get("cell", ""), r.get("kpi", "")),
        )[:15]
        result["returned_in_this_view"] = len(result["results"])
        return result

    async def tool(self, name, args, role):
        allowed = {
            "analyst": {"Analyze", "lookup_context"},
            "mitigation": {"estimate_batch", "Analyze", "lookup_context"},
        }
        if name not in allowed.get(role, set()):
            raise ValueError("Tool is not available to this agent role")
        if name == "estimate_batch":
            base_id = args.get("base_candidate_id", "requested")
            if base_id not in self.plan_index:
                raise ValueError("Unknown base_candidate_id")
            plans = args.get("candidates")
            if plans is None:
                plans = alternatives(
                    self.requested,
                    self.snapshot["current_rcp"],
                    axis=args.get("axis", "all"),
                    base=self.plan_index[base_id],
                    **{
                        k: args[k]
                        for k in ("values", "cells", "delays_s", "fractions")
                        if k in args
                    },
                )
            plans = [p for p in plans if self.body["issue_time_s"] + p.get("delay_s", 0) <= 720]
            report = self.search_report()
            stage = report["next_stage"]
            if not stage:
                return {
                    "search_coverage": report,
                    "message": "Declared search is complete; use the evaluated result",
                }
            required = self.search.pending(stage)
            limit = min(48, 160 - self.candidate_count)
            if self.batch_count >= 5 or limit <= 0:
                return {
                    "search_coverage": report,
                    "message": "Search budget exhausted; remaining candidates are explicitly unsearched",
                }
            # Mandatory coverage takes precedence over optional LLM proposals.
            # Actual parameter vectors deduplicate repeats across all rounds.
            proposed_count = len(plans)
            plans = self.search.deduplicate(required + plans, self.search.seen)[:limit]
            await self.emit(
                "status",
                role,
                args.get("summary", "Batch exploration"),
                {"axis": args.get("axis"), "base_candidate_id": base_id},
            )
            await self.emit(
                "status",
                role,
                "Three-axis coverage guard",
                {
                    "stage": stage,
                    "required_candidates_remaining": len(required),
                    "agent_proposed_candidates": proposed_count,
                    "submitted_candidates": len(plans),
                    "batch_execution": "shared-history vectorized GNN",
                },
            )
            result = await self.evaluate(plans, role)
            shortlist = result["top_candidates"] or result["infeasible_diagnostics"]
            analysis = await self.tool(
                "Analyze", {"candidate_ids": [c["candidate_id"] for c in shortlist]}, role
            )
            return {
                **compact_evaluation(result),
                "analysis": analysis,
                "search_coverage": await self.publish_search_report(),
            }
        if name == "Analyze":

            async def analyze():
                results = [
                    self.estimator.analyze(e, args.get("candidate_ids")) for e in self.evaluations
                ]
                return {
                    "analyses": [a for r in results for a in r["analyses"]],
                    "evidence_kind": "frozen_model_numeric_diagnostics",
                }

            return compact_tool_result(name, await self.invoke(name, args, role, analyze))
        if name == "lookup_context":
            return compact_tool_result(
                name, await self.invoke(name, args, role, lambda: self.lookup_context(args))
            )
        raise ValueError("Unknown or unauthorized tool")

    async def llm_role(self, role, context, tools=None, forced=None, turns=1):
        messages = [{"role": "user", "content": json.dumps(context, ensure_ascii=False)}]
        next_forced = forced
        for turn in range(turns):
            if role == "mitigation":
                report = self.search_report()
                if report["stop_allowed"]:
                    return report["stop_reason"]
                next_forced = "estimate_batch"
            message = await self.provider.complete(role, messages, self.emit, tools, next_forced)
            next_forced = None
            messages.append(message)
            calls = message.get("tool_calls", [])
            if not calls:
                if (
                    role == "mitigation"
                    and not self.search_report()["stop_allowed"]
                    and turn < turns - 1
                ):
                    messages.append(
                        {
                            "role": "user",
                            "content": json.dumps(
                                {
                                    "instruction": "Search is incomplete. Continue estimate_batch; do not stop based only on the number of calls.",
                                    "search_coverage": self.search_report(),
                                }
                            ),
                        }
                    )
                    next_forced = "estimate_batch"
                    continue
                return message.get("content", "")
            for call in calls:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"])
                if role == "intent" and name == "submit_intent":
                    await self.emit("tool_call", role, name, {"tool": name, "arguments": args})
                    try:
                        self.prepare_intent(args)
                    except (ValueError, KeyError, TypeError) as exc:
                        await self.emit(
                            "tool_result", role, name, {"tool": name, "error": str(exc)}
                        )
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": call["id"],
                                "content": json.dumps(
                                    {
                                        "validation_error": str(exc),
                                        "instruction": "Repair the contract. Preserve the user's exact action; only explicit numeric KPI requirements belong in goals/constraints. Omit these arrays for the default research policy.",
                                    }
                                ),
                            }
                        )
                        next_forced = "submit_intent"
                        continue
                    return args
                try:
                    result = await self.tool(name, args, role)
                except (ValueError, KeyError, TypeError) as exc:
                    result = {
                        "error": str(exc),
                        "instruction": "Repair the tool request without altering the original intent",
                    }
                    await self.emit("error", role, str(exc), {"recoverable": True})
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )
                if role == "mitigation":
                    if self.search_report()["stop_allowed"]:
                        # Let this agent read the actual final tool result before
                        # summarizing; candidate proposals are not evidence.
                        message = await self.provider.complete(role, messages, self.emit)
                        return message.get("content", "")
            if self.batch_count >= 5 and role == "mitigation":
                break
        if role == "analyst" and tools:
            message = await self.provider.complete(role, messages, self.emit)
            return message.get("content", "")
        return "Tool budget completed"

    def prepare_intent(self, parsed):
        for key, value in parsed.get("expected_current", {}).items():
            if (
                key not in self.snapshot["current_rcp"]
                or abs(float(value) - self.snapshot["current_rcp"][key]) > 1e-5
            ):
                raise ValueError(
                    f"Prompt starting value {key}={value} differs from replay value {self.snapshot['current_rcp'].get(key)}; select a matching run or change the prompt"
                )
        if not parsed.get("changes"):
            raise ValueError("No explicit parameter change was found")
        self.requested = {
            "candidate_id": "requested",
            "changes": parsed["changes"],
            "delay_s": parsed.get("delay_s", 0),
            "horizon_s": parsed["horizon_s"],
            "axes": ["original"],
        }
        editor = self.body.get("intent_spec")
        if editor is not None:
            if self.body.get("intent_override") is not None:
                raise ValueError("Policy editor cannot be combined with a legacy JSON override")
            self.intent = compile_spec(editor, parsed["horizon_s"])
        else:
            self.intent = make_research_intent(
                self.snapshot,
                parsed["horizon_s"],
                parsed["objective"],
                parsed.get("target_fraction"),
                self.settings["policy"],
            )
        if not editor and parsed.get("goals"):
            self.intent["goals"] = parsed["goals"]
        if not editor and parsed.get("constraints"):
            merged = {c["id"]: c for c in self.intent["constraints"]}
            merged.update({c["id"]: c for c in parsed["constraints"]})
            self.intent["constraints"] = list(merged.values())
        override = self.body.get("intent_override") or {}
        if set(override) - {"goals", "constraints", "safety", "allowed_cells", "allowed_delays_s"}:
            raise ValueError(
                "intent_override can override goals, constraints, safety and allowed cells/delays only"
            )
        self.intent.update(override)
        if not editor and (parsed.get("goals") or parsed.get("constraints") or override):
            self.intent.pop("editor_spec", None)
        self.intent = validate_intent(self.intent)
        definition = {
            k: v for k, v in self.intent.items() if k not in ("definition_sha256", "editor_spec")
        }
        self.intent["definition_sha256"] = hashlib.sha256(
            json.dumps(definition, sort_keys=True).encode()
        ).hexdigest()
        self.search = AxisSearch(
            self.requested,
            self.snapshot["current_rcp"],
            self.intent,
            self.settings["policy"],
            self.body["issue_time_s"],
        )

    async def run(self):
        self.status = "running"
        try:
            await self.emit(
                "status",
                "system",
                "Offline replay started",
                {
                    "mode": self.body["mode"],
                    "run_id": self.body["run_id"],
                    "issue_time_s": self.body["issue_time_s"],
                },
            )
            self.snapshot = await self.invoke(
                "get_network_state",
                {"run_id": self.body["run_id"], "t": self.body["issue_time_s"]},
                "intent",
                lambda: asyncio.to_thread(
                    lambda: compact_snapshot(
                        self.runtime.snapshot(self.body["run_id"], self.body["issue_time_s"])
                    )
                ),
            )
            context = {
                "user_prompt": self.body["prompt"],
                "network_state": self.snapshot,
                "supported_parameters": DOMAINS,
                "default_horizon_s": self.body.get("horizon_s", 180),
                "expectation_contract": "{id,metric,cells,aggregation:sum|mean|min|max|spread|std,operator:<=|>=,target_kind:absolute|relative_fraction,target,reference_kind:observed_at_issue|keep_forecast_same_window,weight:1}",
                "research_policy": self.settings["policy"],
            }
            if self.body.get("intent_spec") is not None:
                context["locked_user_policy"] = compile_spec(
                    self.body["intent_spec"], self.body["horizon_s"]
                )
                context["policy_instruction"] = (
                    "The UI policy is authoritative for KPI goals and thresholds. Parse only the requested action; do not invent or change policy rows. Conflicting KPI prose does not override the editor."
                )
            if self.body["mode"] == "llm":
                parsed = await self.llm_role(
                    "intent", context, [INTENT_TOOL], "submit_intent", turns=3
                )
                if not isinstance(parsed, dict):
                    raise ValueError("Intent agent did not return a structured action")
            else:
                await self.emit("agent_start", "intent", "Deterministic parser (no LLM)")
                parsed = parse_deterministic(
                    self.body["prompt"],
                    self.snapshot["current_rcp"],
                    self.body.get("horizon_s", 180),
                )
            self.prepare_intent(parsed)
            await self.emit(
                "tool_result",
                "intent",
                "submit_intent",
                {
                    "tool": "submit_intent",
                    "result": {
                        "requested_plan": self.requested,
                        "intent": self.intent,
                        "interpretation": parsed.get("interpretation"),
                    },
                },
            )
            await self.emit(
                "agent_start",
                "conflict",
                "Checking the original request with the frozen Temporal GNN",
            )
            first = await self.evaluate([deepcopy(self.requested)], "conflict")
            if first["invalid_count"]:
                raise ValueError(
                    "Original requested plan is outside the model/action contract: "
                    + str(first["all_candidates"][0].get("validation_error"))
                )
            if self.body["mode"] == "llm":
                await self.llm_role(
                    "conflict",
                    {
                        "original_check": compact_evaluation(first),
                        "current_parameters": self.snapshot["current_rcp"],
                    },
                )
            else:
                await self.emit(
                    "summary_delta",
                    "conflict",
                    "Original request checked. " + first["all_candidates"][0]["conflict_status"],
                )
            if not first["all_expectations_met"]:
                if self.body["mode"] == "llm":
                    analyst_summary = await self.llm_role(
                        "analyst",
                        {
                            "request": self.requested,
                            "current_parameters": self.snapshot["current_rcp"],
                            "check": compact_evaluation(first),
                        },
                        [ANALYZE_TOOL, CONTEXT_TOOL],
                        "Analyze",
                        turns=3,
                    )
                    context = await self.prepare_mitigation_context(first, analyst_summary)
                    await self.llm_role(
                        "mitigation",
                        context,
                        [SEARCH_TOOL, ANALYZE_TOOL, CONTEXT_TOOL],
                        "estimate_batch",
                        turns=7,
                    )
                else:
                    await self.emit("agent_start", "analyst", "Deterministic diagnostics (no LLM)")
                    await self.tool("Analyze", {}, "analyst")
                    await self.tool(
                        "lookup_context",
                        {"feature": next(iter(self.requested["changes"]))},
                        "analyst",
                    )
                    await self.prepare_mitigation_context(
                        first, "Deterministic numeric diagnostics; no LLM reasoning"
                    )
                    await self.emit(
                        "agent_start", "mitigation", "Deterministic axis search (no LLM)"
                    )
                    await self.tool(
                        "estimate_batch",
                        {
                            "axis": "all",
                            "summary": "Explore independent time, intensity and cell alternatives",
                        },
                        "mitigation",
                    )
                # If an LLM stops returning valid tools, finish the mandatory
                # bounded search explicitly instead of silently losing an axis.
                while not self.search_report()["stop_allowed"]:
                    stage = self.search_report()["next_stage"]
                    await self.tool(
                        "estimate_batch",
                        {
                            "axis": "all" if stage == "axes" else "combined",
                            "candidates": [],
                            "summary": "Coverage guard continues untested combinations and analyzes the result",
                        },
                        "mitigation",
                    )
            feasible, diagnostics = self.ranked()
            selected = deepcopy(feasible[0]) if feasible else None
            operator_review = shortlist(
                [row for evaluation in self.evaluations for row in evaluation["all_candidates"]],
                intent=self.intent,
            )
            coverage = (
                await self.publish_search_report()
                if not first["all_expectations_met"]
                else {
                    "stop_allowed": True,
                    "stop_reason": "original_request_met_all_expectations",
                    "independent_axes_complete": False,
                    "search_required": False,
                }
            )
            self.result = {
                "session_id": self.id,
                "status": "recommended"
                if selected and selected["all_expectations_met"]
                else "partial_goal"
                if selected
                else "no_feasible_candidate",
                "mode": self.body["mode"],
                "user_prompt": self.body["prompt"],
                "requested_plan": self.requested,
                "intent": self.intent,
                "selected": selected,
                "top_candidates": feasible[:2],
                "infeasible_diagnostics": diagnostics[:2],
                "operator_review": operator_review,
                "original": first["all_candidates"][0],
                "candidate_count": self.candidate_count,
                "batch_count": self.batch_count,
                "selection_rule": first["selection_rule"],
                "evidence_kind": "offline_model_forecast_not_measured_control_effect",
                "applied": False,
                "ns3_validated": False,
                "requires_ns3_validation": True,
                "execution_plan": self.export_plan(selected),
                "search_coverage": coverage,
                "settings_sha256": hashlib.sha256(
                    json.dumps(self.settings, sort_keys=True).encode()
                ).hexdigest(),
            }
            if coverage["stop_reason"] == "budget_exhausted_with_unsearched_candidates":
                self.result["status"] = "search_incomplete"
            await self.emit("decision", "deployment", self.result["status"], self.result)
            if self.body["mode"] == "llm":
                await self.llm_role(
                    "deployment",
                    {
                        "status": self.result["status"],
                        "requested_plan": self.requested,
                        "current_parameters": self.snapshot["current_rcp"],
                        "selected_control_transitions": []
                        if selected is None
                        else [
                            {
                                "parameter": k,
                                "current": self.snapshot["current_rcp"][k],
                                "selected": v,
                                "requested_value_on_this_control": self.requested["changes"].get(k),
                            }
                            for k, v in selected["changes"].items()
                        ],
                        "selected": compact_candidate(selected) if selected else None,
                        "operator_candidates": [
                            compact_candidate(row) for row in operator_review["candidates"]
                        ],
                        "operator_ranking": operator_review["ranking"],
                        "operator_action": "The UI shows up to three alternatives. When none is feasible, explain the least-violation options with exact remaining violations. The operator can select one for a RIC-specific offline queue. Do not stop at saying no recommendation. No threshold is relaxed and no actual RIC application occurs.",
                        "original": compact_candidate(self.result["original"]),
                        "intent": self.intent,
                        "search_coverage": coverage,
                        "applied": False,
                        "ns3_validated": False,
                        "evidence_kind": self.result["evidence_kind"],
                    },
                )
            else:
                await self.emit("agent_start", "deployment", "Offline plan export (no LLM)")
                await self.emit(
                    "summary_delta",
                    "deployment",
                    f"{len(operator_review['candidates'])}개 운영자 검토안을 정리했습니다. 제약 충족 후보를 우선하고, 위반 후보는 정규화 위반 합이 작은 순으로 표시합니다. 하단에서 목표·위반 수치를 비교한 뒤 RIC별 배포 대기열에 등록할 수 있습니다. 실제 제어는 적용되지 않았습니다.",
                )
            self.status = "completed"
        except asyncio.CancelledError:
            self.status = "cancelled"
        except Exception as exc:
            self.status = "failed"
            await self.emit("error", "system", str(exc))
        finally:
            await self.emit(
                "done", "system", self.status, {"status": self.status, "session_id": self.id}
            )
            self.persist()

    def export_plan(self, selected):
        snapshot = self.snapshot
        return {
            "schema_version": "agora-offline-plan-v1",
            "session_id": self.id,
            "run_id": self.body["run_id"],
            "issue_time_s": self.body["issue_time_s"],
            "user_prompt": self.body["prompt"],
            "observed_history": {
                "start_s": snapshot["history_start_s"],
                "end_s_exclusive": snapshot["history_end_s"],
            },
            "current_parameters": snapshot["current_rcp"],
            "requested_plan": self.requested,
            "selected_plan": selected,
            "model": self.runtime.metadata(),
            "intent": self.intent,
            "policy": self.settings["policy"],
            "prompts": self.settings["prompts"],
            "provider": public_configuration(self.settings),
            "applied": False,
            "ns3_validated": False,
            "schedule": []
            if selected is None
            else [
                {
                    "time_s": selected["interval_start_s"],
                    "action": "set_parameters",
                    "parameters": selected["parameters"],
                },
                {
                    "time_s": selected["interval_end_s"],
                    "action": "restore_pre_request_parameters",
                    "parameters": snapshot["current_rcp"],
                    "note": "Duration semantics for later simulator validation; restoration effect is not forecast",
                },
            ],
            "control_domains": {
                "txp": "rApp-style",
                "ret": "rApp-style",
                "cio": "xApp-style",
                "ttt": "xApp-style shared",
                "hys": "xApp-style shared",
            },
            "execution_checks": [
                "Offline artifact only; no A1/E2/RIC or ns3 action was performed",
                "Verify live control support, TTT allowed values and RET cache for the selected parameters before simulator validation",
                "Fixed-run replay is not a branched intervention experiment; switched-action causal benefit remains unvalidated",
            ],
        }
