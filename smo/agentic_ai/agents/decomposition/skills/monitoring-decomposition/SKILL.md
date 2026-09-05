---
name: monitoring-decomposition
description: Use this skill when decomposing monitoring, maintain/keep, metrics, CPU usage, Grafana, Prometheus, NF, pod, container, or free5gc intents.
---

# Monitoring Decomposition

## Purpose

Use this skill only when the user intent asks to monitor, query, measure, or report runtime metrics for an NF, pod, container, or free5gc component.

The Planning Agent has a Probe Agent downstream. The Probe Agent owns target discovery, Grafana/Prometheus querying, metric calculation, and result formatting for one monitoring measurement.

## Decomposition Rule

Treat one requested monitoring measurement as one executable subtask.

For a single CPU monitoring request, output exactly one monitoring subtask, even when the request says each NF, all free5gc NFs, every pod, or one named NF.

Good single-subtask examples:

- Run the Probe Agent to measure the last 1 minute average CPU usage for each requested free5gc NF.
- Run the Probe Agent to measure AMF CPU usage over the requested time window.
- Run the Probe Agent to monitor CPU usage for all requested free5gc pods.

## What Not To Split

Do not turn one monitoring request into procedural subtasks such as:

- Identify the NFs to monitor.
- Query Grafana or Prometheus for each NF.
- Collect CPU metrics.
- Calculate the average.
- Evaluate health or performance criteria unless the user explicitly requested an evaluation rule.
- Report the monitoring results.
- Trigger alerts unless the user explicitly requested alerting.

Those are internal responsibilities of the Probe Agent and its tools, not independent decomposition subtasks.

## When To Split

Split monitoring into multiple subtasks only when the user explicitly asks for distinct independent work, for example:

- Different metrics that should be handled separately, such as CPU and memory.
- Monitoring plus a separate follow-up action, such as alerting, remediation, scaling, restart, storage, or dashboard creation.
- Separate scopes that the user intentionally distinguishes, such as AMF monitoring and UPF remediation.

## Maintained State

An intent that asks to keep or maintain a condition for N minutes is not a snapshot measurement.

- Keep any initial state change as its own subtask.
- Create one subsequent monitoring subtask that preserves the exact target, expected condition, and N-minute duration.
- State that Monitoring Agent uses its observation tools throughout the requested window.
- If the expected condition is not satisfied, Monitoring Agent sends fresh evidence to Planning Agent.
- Do not create an unconditional or direct remediation subtask. Planning Agent decides the approval-gated recovery after receiving confirmed mismatch evidence.
- Observation-unavailable or unknown evidence is not a mismatch.

## Output Reminder

The final Decomposition Agent output still follows `agent.md`: exactly `intent`, `subtasks`, and `golden_goal_context_used`.
