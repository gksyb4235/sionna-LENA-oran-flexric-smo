# Probe Agent

The Probe Agent is a sub-agent for live NF monitoring. v1 supports one operation:

- monitor one NF CPU usage average over the most recent 1 minute

All files live under:

```text
/home/ubuntu/jaechan/agents/probe
```

## Grafana MCP Tools

This project includes a local Grafana MCP server in `mcp_server.py` with tools:

- `query_prometheus_tool`: Prometheus instant query through the configured Grafana datasource
- `monitor_nf_cpu_average_tool`: latest 1-minute average CPU for one NF

The implementation follows the Grafana MCP `query_prometheus` tool model and uses Grafana datasource proxy access to the configured Prometheus datasource.

## Environment

Create `.env` from `.env.example` and set:

```text
GRAFANA_URL=https://your-grafana.example
GRAFANA_API_KEY=...
GRAFANA_PROMETHEUS_DATASOURCE_UID=...
GRAFANA_CPU_METRIC=nf_cpu_usage_percent
GRAFANA_NF_LABEL=nf
GRAFANA_CPU_PROMQL_TEMPLATE=avg_over_time({metric_name}{label_selector}[{window}])
MONITORING_WINDOW_SECONDS=60
```

If your Prometheus metric or NF label is different, change `GRAFANA_CPU_METRIC`, `GRAFANA_NF_LABEL`, or the full `GRAFANA_CPU_PROMQL_TEMPLATE`.

## CLI

Human-readable output:

```bash
cd /home/ubuntu/jaechan/agents/probe
.venv/bin/python agent.py "Monitor AMF cpu"
```

Machine-readable output:

```bash
.venv/bin/python agent.py --json "Monitor AMF cpu"
```

Explicit NF override:

```bash
.venv/bin/python agent.py --nf AMF "Monitor cpu"
```

## HTTP

```bash
cd /home/ubuntu/jaechan/agents/probe
.venv/bin/uvicorn server:app --host 0.0.0.0 --port 8082
```

Endpoints:

- `GET /health`
- `POST /invoke` with `{ "intent": "Monitor AMF cpu" }`
- `POST /invoke` with `{ "intent": "Monitor cpu", "nf": "AMF" }`

## MCP Server

```bash
cd /home/ubuntu/jaechan/agents/probe
.venv/bin/python mcp_server.py
```

## Output

```json
{
  "intent": "Monitor AMF cpu",
  "nf": "AMF",
  "metric": "cpu_usage",
  "window_seconds": 60,
  "average": 37.5,
  "unit": "percent",
  "status": "completed",
  "query": "avg_over_time(nf_cpu_usage_percent{nf="AMF"}[1m])",
  "sample_count": 1,
  "source": {
    "type": "grafana_prometheus",
    "datasource_uid": "prometheus-prod",
    "mcp_tool": "monitor_nf_cpu_average",
    "base_mcp_tool": "query_prometheus"
  },
  "time_range": {
    "start_unix": 0.0,
    "end_unix": 60.0
  }
}
```

## Tests

```bash
cd /home/ubuntu/jaechan/agents/probe
.venv/bin/python -m unittest discover -s tests
```
