# Monitoring Agent

The Monitoring Agent compares a Planning Agent expectation with fresh network
evidence. It delegates all Grafana/Prometheus/InfluxDB work to the existing
Probe Agent. A confirmed mismatch is sent to Planning Agent `/feedback`, which
adds a recovery subtask that must not repeat the previous action unchanged.

## Input

```json
{
  "planning_report": { "run_id": "planning-...", "intent": "...", "subtask_plans": [] },
  "subtask_id": "subtask-002",
  "notify_planner": true
}
```

`subtask_id` is optional and defaults to the last plan. The expected outcome is
read from `langgraph_spec.evaluation_request`.

## CLI

```bash
cd /home/ubuntu/jaechan/agents/monitoring
.venv/bin/python agent.py --json --input request.json
```

Use `--no-notify` to inspect without creating a recovery plan.

## HTTP

```bash
.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8083
```

- `GET /health`
- `POST /invoke` with `Authorization: Bearer $MONITORING_AGENT_TOKEN`

Set the same `MONITORING_AGENT_TOKEN` in Monitoring and Planning. Configured
HTTP transports fall back to their local agent CLI only when the service is
unreachable; HTTP validation/authentication errors are returned as errors.

## Statuses

- `healthy`: threshold expectation passed.
- `observed`: values were requested without a pass/fail threshold.
- `replan_required`: Probe collected valid evidence and the expectation failed.
- `observation_unavailable`: Probe could not provide decisive evidence.
- `awaiting_execution`: the action still needs approval/execution.
- `stopped`: the action was rejected.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```
