---
name: probe-tools
description: Plan, execute, verify, and learn read-only Grafana monitoring queries for Prometheus/free5gc and InfluxDB RAN KPM data. Use for Probe Agent metric, health, throughput, PRB, PDCP, slice, request-rate, connectivity, and target-availability requests.
---

# Probe Tools

## Workflow

1. Match the request to the query catalog before creating a query.
2. Build the smallest read-only query plan that answers the request.
3. Execute the plan and require a `completed` report with samples.
4. If execution is not completed, use the returned error and failed command to create a corrected command. Retry at most five times after the initial attempt.
5. After a completed execution, add a previously undefined query to the learned catalog.
6. When only a field or metric name differs from an existing pattern, add only that variable below the pattern. Do not duplicate the full query.

Never learn a query that returned an error or no samples.

## Runtime Tools

- `query_prometheus`: execute a read-only PromQL instant query through Grafana.
- `query_influxql`: execute one read-only InfluxQL `SELECT` through Grafana. Keep `$timeFilter` and `$__interval` in generated commands; the runtime expands them.
- `list_free5gc_monitoring_targets`: discover free5gc targets for known scopes.
- `monitor_nf_cpu_average`: legacy shortcut for CPU average.
- `monitor_nf_replica_health`: legacy shortcut for replica health.

Prefer a query plan for new or composed monitoring behavior. Use legacy shortcuts only when they directly match the request.

## Query Plan Shape

Use `operation=prometheus_query_plan`, `tool=query_prometheus`, and `query_type=instant` for PromQL.

Use `operation=influxql_query_plan`, `tool=query_influxql`, and `query_type=influxql` for RAN KPM InfluxQL.

```json
{
  "operation": "influxql_query_plan",
  "metric": "ue_throughput",
  "scope": "per_ue",
  "nfs": [],
  "window_seconds": 60,
  "expected_replicas": null,
  "queries": [
    {
      "tool": "query_influxql",
      "label": "ue_downlink_throughput",
      "target": "kpm",
      "query": "SELECT \"thp_dl_kbps\" FROM \"kpm\" WHERE $timeFilter GROUP BY \"ue_id\"",
      "purpose": "downlink throughput grouped by UE",
      "query_type": "influxql",
      "unit": "kbps"
    }
  ],
  "evaluation": {}
}
```

## Query Catalog

### RAN KPM: Per UE

Use raw KPM fields grouped by `ue_id` for per-UE inspection.

- `query_influxql`: `SELECT <fields> FROM "kpm" WHERE $timeFilter GROUP BY "ue_id"`
  - fields: `"thp_ul_kbps"`
  - fields: `"thp_dl_kbps"`
  - fields: `"prb_dl", "prb_ul"`
  - fields: `"pdcp_vol_dl_kb", "pdcp_vol_ul_kb"`

Interpret the fields as:

- `thp_ul_kbps`, `thp_dl_kbps`: uplink and downlink UE throughput.
- `prb_dl`, `prb_ul`: downlink and uplink physical resource block usage.
- `pdcp_vol_dl_kb`, `pdcp_vol_ul_kb`: downlink and uplink PDCP traffic volume.

### RAN KPM: Per Slice

Use a time-bucketed mean grouped by `sst` for slice trends.

- `query_influxql`: `SELECT <fields> FROM "kpm" WHERE $timeFilter GROUP BY time($__interval), "sst" fill(null)`
  - fields: `mean("thp_dl_kbps")`
  - fields: `mean("thp_ul_kbps")`
  - fields: `mean("prb_dl"), mean("prb_ul")`

### free5gc: SBI Traffic

- `query_prometheus`: `sum(rate(free5gc_sbi_inbound_request_duration_seconds_count[1m])) by (nf_type, path)`

Use this for inbound SBI request rate grouped by NF type and path.

### free5gc: Target Availability

- `query_prometheus`: `up{job=~".*free5gc.*"}`

Use this to verify whether free5gc Prometheus scrape targets are up.

### free5gc AMF: Business Metrics

- `query_prometheus`: `<metric>`
  - metric: `free5gc_amf_business_ue_connectivity`
  - metric: `free5gc_amf_business_ue_gmm_state_count`

Use connectivity for UE connectivity state and GMM state count for AMF registration/mobility state counts.

### free5gc: CPU And Replica Health

CPU average for a core NF:

- `query_prometheus`: `sum(rate(container_cpu_usage_seconds_total{namespace=~"free5gc.*",container="<nf>",image!=""}[1m])) * 100`
  - nf: `amf`
  - nf: `smf`
  - nf: `upf`
  - nf: `nrf`
  - nf: `ausf`
  - nf: `udm`
  - nf: `udr`
  - nf: `pcf`
  - nf: `nssf`

Replica and pod health:

- `query_prometheus`: `max(kube_deployment_spec_replicas{namespace=~"free5gc.*",deployment=~"(free5gc-free5gc-<nf>|.*<nf>.*)"})`
  - nf: `amf`

- `query_prometheus`: `max(kube_deployment_status_replicas_available{namespace=~"free5gc.*",deployment=~"(free5gc-free5gc-<nf>|.*<nf>.*)"})`
  - nf: `amf`

- `query_prometheus`: `max(kube_deployment_status_replicas_ready{namespace=~"free5gc.*",deployment=~"(free5gc-free5gc-<nf>|.*<nf>.*)"})`
  - nf: `amf`

- `query_prometheus`: `sum(kube_pod_status_ready{namespace=~"free5gc.*",pod=~".*<nf>.*",condition="true"} == 1)`

Treat requests containing healthy, ready, available, operating properly, rollout, deployment, pod, or replica as health checks. Use CPU only when CPU is explicitly requested.
  - nf: `amf`

## Evaluation

Use `evaluation.type=thresholds` when the answer requires a health judgment. Compare query labels with `==`, `!=`, `>=`, `>`, `<=`, or `<` against:

- `value`: a numeric constant.
- `value_from`: `expected_replicas`.
- `value_from_query`: another query label.

Mark a check `optional=true` only when the metric may legitimately be absent.

## Failure Policy

Record Grafana authorization, connectivity, query, and no-sample errors. Do not fabricate values, mark a target healthy without samples, persist a failed query, generate shell commands, or modify Grafana, Prometheus, InfluxDB, Kubernetes, files outside this skill, or other external systems.

## Learned Query Patterns

The runtime maintains this section only after successful execution.

<!-- learned-query-catalog:start -->

- `query_prometheus`: `count(free5gc_<nf>_business_ue_connectivity)`
  - nf: `amf`

- `query_prometheus`: `count(kube_pod_status_phase{namespace=~"free5gc.*",pod=~".*<nf>.*",phase="Running"})`
  - nf: `amf`

- `query_prometheus`: `sum(kube_pod_status_phase{namespace=~"free5gc.*",pod=~".*<nf>.*",phase="Running"})`
  - nf: `amf`

- `query_prometheus`: `count(kube_pod_status_ready{namespace=~"free5gc.*",pod=~".*<nf>.*",condition="true"})`
  - nf: `amf`

- `query_prometheus`: `count(kube_pod_status_ready{namespace=~"free5gc.*",pod=~".*<nf>.*",condition="true"} == 1)`
  - nf: `amf`
<!-- learned-query-catalog:end -->
