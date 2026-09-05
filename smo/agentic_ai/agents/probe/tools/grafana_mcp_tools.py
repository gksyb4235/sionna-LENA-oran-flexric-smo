"""Grafana MCP-style tools for monitoring metrics."""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
JAECHAN_ROOT = Path(
    os.getenv("AGENTIC_AI_ROOT", str(Path(__file__).resolve().parents[3]))
).resolve()
DEFAULT_TIMEOUT_SEC = 20
DEFAULT_CPU_METRIC = "nf_cpu_usage_percent"
DEFAULT_NF_LABEL = "nf"
DEFAULT_PROMQL_TEMPLATE = "avg_over_time({metric_name}{label_selector}[{window}])"
DEFAULT_WINDOW_SECONDS = 60
DEFAULT_INFLUXDB_DATASOURCE_UID = "RAN-KPM-InfluxDB"
DEFAULT_INFLUXDB_DATABASE = "ran_kpm"
PROM_DURATION_UNITS = ((86400, "d"), (3600, "h"), (60, "m"), (1, "s"))
SAFE_TARGET_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]+$")
INFLUXQL_WRITE_PATTERN = re.compile(r"(?i)\b(INTO|DROP|DELETE|CREATE|ALTER|GRANT|REVOKE)\b")
CORE_FREE5GC_NFS = ("amf", "smf", "upf", "nrf", "ausf", "udm", "udr", "pcf", "nssf")
IGNORED_FREE5GC_CONTAINERS = {"", "POD"}


@dataclass(frozen=True)
class GrafanaConfig:
    url: str
    api_key: str
    datasource_uid: str
    timeout_sec: int
    cpu_metric: str
    nf_label: str
    cpu_promql_template: str
    window_seconds: int
    influxdb_datasource_uid: str = DEFAULT_INFLUXDB_DATASOURCE_UID
    influxdb_database: str = DEFAULT_INFLUXDB_DATABASE


class GrafanaConfigurationError(RuntimeError):
    """Raised when Grafana configuration is missing or invalid."""


class GrafanaQueryError(RuntimeError):
    """Raised when Grafana or Prometheus returns an unusable result."""


def _ensure_project_scope() -> None:
    project_root = PROJECT_ROOT.resolve()
    if project_root != JAECHAN_ROOT and JAECHAN_ROOT not in project_root.parents:
        raise RuntimeError(f"Project root must stay under {JAECHAN_ROOT}; got {project_root}")


def load_env_file(path: Path | None = None) -> None:
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _int_env(name: str, default: int, *, minimum: int = 1, maximum: int = 3600) -> int:
    raw_value = os.getenv(name, str(default))
    try:
        value = int(raw_value)
    except ValueError:
        return default
    return min(max(value, minimum), maximum)


def load_grafana_config() -> GrafanaConfig:
    _ensure_project_scope()
    load_env_file()
    url = os.getenv("GRAFANA_URL", "").strip().rstrip("/")
    api_key = os.getenv("GRAFANA_API_KEY", "").strip()
    datasource_uid = os.getenv("GRAFANA_PROMETHEUS_DATASOURCE_UID", "").strip()
    if not url:
        raise GrafanaConfigurationError("GRAFANA_URL is required.")
    if not api_key:
        raise GrafanaConfigurationError("GRAFANA_API_KEY is required.")
    if not datasource_uid:
        raise GrafanaConfigurationError("GRAFANA_PROMETHEUS_DATASOURCE_UID is required.")
    return GrafanaConfig(
        url=url,
        api_key=api_key,
        datasource_uid=datasource_uid,
        timeout_sec=_int_env("GRAFANA_TIMEOUT_SEC", DEFAULT_TIMEOUT_SEC, minimum=1, maximum=120),
        cpu_metric=os.getenv("GRAFANA_CPU_METRIC", DEFAULT_CPU_METRIC).strip() or DEFAULT_CPU_METRIC,
        nf_label=os.getenv("GRAFANA_NF_LABEL", DEFAULT_NF_LABEL).strip() or DEFAULT_NF_LABEL,
        cpu_promql_template=os.getenv("GRAFANA_CPU_PROMQL_TEMPLATE", DEFAULT_PROMQL_TEMPLATE).strip()
        or DEFAULT_PROMQL_TEMPLATE,
        window_seconds=_int_env("MONITORING_WINDOW_SECONDS", DEFAULT_WINDOW_SECONDS, minimum=1, maximum=3600),
        influxdb_datasource_uid=os.getenv(
            "GRAFANA_INFLUXDB_DATASOURCE_UID",
            DEFAULT_INFLUXDB_DATASOURCE_UID,
        ).strip()
        or DEFAULT_INFLUXDB_DATASOURCE_UID,
        influxdb_database=os.getenv("GRAFANA_INFLUXDB_DATABASE", DEFAULT_INFLUXDB_DATABASE).strip()
        or DEFAULT_INFLUXDB_DATABASE,
    )


def seconds_to_prometheus_duration(seconds: int) -> str:
    remaining = max(1, seconds)
    parts: list[str] = []
    for unit_seconds, suffix in PROM_DURATION_UNITS:
        count, remaining = divmod(remaining, unit_seconds)
        if count:
            parts.append(f"{count}{suffix}")
    return "".join(parts) or "1s"


def _escape_label_value(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _validate_target(value: str) -> str:
    target = value.strip()
    if not target:
        raise ValueError("Monitoring target is required.")
    if not SAFE_TARGET_PATTERN.match(target):
        raise ValueError("Monitoring target contains unsupported characters.")
    return target


def build_nf_cpu_promql(nf: str, config: GrafanaConfig | None = None) -> str:
    cfg = config or load_grafana_config()
    nf_value = _validate_target(nf)
    label_selector = f'{{{cfg.nf_label}="{_escape_label_value(nf_value)}"}}'
    return cfg.cpu_promql_template.format(
        metric_name=cfg.cpu_metric,
        nf=nf_value,
        nf_lower=nf_value.lower(),
        nf_upper=nf_value.upper(),
        nf_label=cfg.nf_label,
        label_selector=label_selector,
        window=seconds_to_prometheus_duration(cfg.window_seconds),
        window_seconds=cfg.window_seconds,
    )


def build_pod_cpu_promql(pod: str, config: GrafanaConfig | None = None) -> str:
    cfg = config or load_grafana_config()
    pod_value = _validate_target(pod)
    window = seconds_to_prometheus_duration(cfg.window_seconds)
    return (
        f'sum(rate({cfg.cpu_metric}{{namespace=~"free5gc.*",pod="{_escape_label_value(pod_value)}",image!=""}}'
        f'[{window}])) * 100'
    )


def deployment_name_for_nf(nf: str) -> str:
    nf_value = _validate_target(nf).lower()
    return f"free5gc-free5gc-{nf_value}"


def _deployment_selector(nf: str) -> str:
    nf_value = _validate_target(nf).lower()
    deployment = deployment_name_for_nf(nf_value)
    return f'namespace=~"free5gc.*",deployment=~"({deployment}|.*{_escape_label_value(nf_value)}.*)"'


def build_nf_replica_promql(nf: str, metric_name: str) -> str:
    metric = _validate_target(metric_name)
    return f'max({metric}{{{_deployment_selector(nf)}}})'


def build_nf_ready_pod_promql(nf: str) -> str:
    nf_value = _validate_target(nf).lower()
    return f'sum(kube_pod_status_ready{{namespace=~"free5gc.*",pod=~".*{_escape_label_value(nf_value)}.*",condition="true"}} == 1)'


def _query_first_value(query: str, *, config: GrafanaConfig, timestamp: float) -> tuple[float | None, int]:
    payload = query_prometheus(query, config=config, timestamp=timestamp)
    values = _extract_instant_values(payload)
    if not values:
        return None, 0
    return values[0], len(values)


def _first_successful_replica_query(
    query_names: list[tuple[str, str]],
    *,
    config: GrafanaConfig,
    timestamp: float,
) -> tuple[float | None, str | None, int, list[dict[str, Any]]]:
    attempts: list[dict[str, Any]] = []
    for label, query in query_names:
        try:
            value, sample_count = _query_first_value(query, config=config, timestamp=timestamp)
        except Exception as exc:  # noqa: BLE001 - later replica queries may still work.
            attempts.append({"label": label, "query": query, "error": str(exc)})
            continue
        attempts.append({"label": label, "query": query, "value": value, "sample_count": sample_count})
        if value is not None:
            return value, query, sample_count, attempts
    return None, None, 0, attempts


def monitor_target_replica_health(
    target: dict[str, str],
    *,
    expected_replicas: int | None = None,
    timestamp: float | None = None,
) -> dict[str, Any]:
    cfg = load_grafana_config()
    query_time = timestamp or time.time()
    nf = target["query_value"].lower()

    desired, desired_query, desired_samples, desired_attempts = _first_successful_replica_query(
        [("deployment_spec_replicas", build_nf_replica_promql(nf, "kube_deployment_spec_replicas"))],
        config=cfg,
        timestamp=query_time,
    )
    available, available_query, available_samples, available_attempts = _first_successful_replica_query(
        [
            ("deployment_available_replicas", build_nf_replica_promql(nf, "kube_deployment_status_replicas_available")),
            ("deployment_ready_replicas", build_nf_replica_promql(nf, "kube_deployment_status_replicas_ready")),
        ],
        config=cfg,
        timestamp=query_time,
    )
    ready, ready_query, ready_samples, ready_attempts = _first_successful_replica_query(
        [
            ("deployment_ready_replicas", build_nf_replica_promql(nf, "kube_deployment_status_replicas_ready")),
            ("pod_ready_count", build_nf_ready_pod_promql(nf)),
        ],
        config=cfg,
        timestamp=query_time,
    )

    if desired is None and available is None and ready is None:
        raise GrafanaQueryError(f"No replica health samples returned for target '{target['name']}'.")

    desired_int = int(round(desired)) if desired is not None else None
    available_int = int(round(available)) if available is not None else None
    ready_int = int(round(ready)) if ready is not None else None
    expected = expected_replicas if expected_replicas is not None else desired_int

    checks: list[dict[str, Any]] = []
    if expected is not None and desired_int is not None:
        checks.append({"name": "desired_matches_expected", "ok": desired_int == expected, "actual": desired_int, "expected": expected})
    if expected is not None and available_int is not None:
        checks.append({"name": "available_at_least_expected", "ok": available_int >= expected, "actual": available_int, "expected": expected})
    if expected is not None and ready_int is not None:
        checks.append({"name": "ready_at_least_expected", "ok": ready_int >= expected, "actual": ready_int, "expected": expected})
    if expected is None and desired_int is not None and available_int is not None:
        checks.append({"name": "available_matches_desired", "ok": available_int >= desired_int, "actual": available_int, "expected": desired_int})
    if expected is None and desired_int is not None and ready_int is not None:
        checks.append({"name": "ready_matches_desired", "ok": ready_int >= desired_int, "actual": ready_int, "expected": desired_int})

    health_status = "unknown"
    if checks:
        health_status = "healthy" if all(check["ok"] for check in checks) else "unhealthy"

    return {
        "target": target["name"],
        "target_type": target["target_type"],
        "query_value": target["query_value"],
        "metric": "replica_status",
        "window_seconds": cfg.window_seconds,
        "expected_replicas": expected_replicas,
        "desired_replicas": desired_int,
        "available_replicas": available_int,
        "ready_replicas": ready_int,
        "health_status": health_status,
        "healthy": health_status == "healthy",
        "checks": checks,
        "unit": "replicas",
        "queries": {
            "desired": desired_query,
            "available": available_query,
            "ready": ready_query,
        },
        "query_attempts": {
            "desired": desired_attempts,
            "available": available_attempts,
            "ready": ready_attempts,
        },
        "sample_count": desired_samples + available_samples + ready_samples,
        "time_range": {"end_unix": query_time, "start_unix": query_time - cfg.window_seconds},
    }


def monitor_nf_replica_health(nf: str, *, expected_replicas: int | None = None, timestamp: float | None = None) -> dict[str, Any]:
    target = {"name": nf.upper(), "query_value": nf.lower(), "target_type": "core_nf", "label": "container"}
    return monitor_target_replica_health(target, expected_replicas=expected_replicas, timestamp=timestamp)


def _datasource_proxy_url(config: GrafanaConfig, datasource_uid: str, path: str) -> str:
    datasource_uid = quote(datasource_uid, safe="")
    return f"{config.url}/api/datasources/proxy/uid/{datasource_uid}{path}"


def _prometheus_proxy_url(config: GrafanaConfig, path: str) -> str:
    return _datasource_proxy_url(config, config.datasource_uid, path)


def _grafana_headers(config: GrafanaConfig) -> dict[str, str]:
    return {"Authorization": f"Bearer {config.api_key}", "Accept": "application/json"}


def prometheus_get(path: str, *, config: GrafanaConfig | None = None, params: dict[str, str] | None = None) -> dict[str, Any]:
    cfg = config or load_grafana_config()
    try:
        with httpx.Client(timeout=cfg.timeout_sec) as client:
            response = client.get(_prometheus_proxy_url(cfg, path), headers=_grafana_headers(cfg), params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise GrafanaQueryError(f"Grafana Prometheus request failed: HTTP {exc.response.status_code} {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise GrafanaQueryError(f"Grafana Prometheus request failed: {exc}") from exc
    except ValueError as exc:
        raise GrafanaQueryError("Grafana Prometheus request did not return JSON.") from exc
    if not isinstance(payload, dict):
        raise GrafanaQueryError("Grafana Prometheus response must be a JSON object.")
    if payload.get("status") != "success":
        raise GrafanaQueryError(f"Grafana Prometheus request did not succeed: {payload}")
    return payload


def query_prometheus(query: str, *, config: GrafanaConfig | None = None, timestamp: float | None = None) -> dict[str, Any]:
    """Run a Prometheus instant query through Grafana's datasource proxy."""

    params: dict[str, str] = {"query": query}
    if timestamp is not None:
        params["time"] = str(timestamp)
    return prometheus_get("/api/v1/query", config=config, params=params)


def expand_influxql_macros(query: str, window_seconds: int) -> str:
    interval_seconds = max(1, window_seconds // 60)
    return (
        query.replace("$timeFilter", f"time >= now() - {max(1, window_seconds)}s")
        .replace("$__interval", f"{interval_seconds}s")
    )


def _validate_influxql(query: str) -> str:
    normalized = query.strip()
    if not re.match(r"(?is)^SELECT\b", normalized) or ";" in normalized or INFLUXQL_WRITE_PATTERN.search(normalized):
        raise ValueError("InfluxQL monitoring queries must be one read-only SELECT statement.")
    return normalized


def query_influxql(
    query: str,
    *,
    config: GrafanaConfig | None = None,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
) -> dict[str, Any]:
    """Run a read-only InfluxQL SELECT through Grafana's datasource proxy."""

    cfg = config or load_grafana_config()
    expanded_query = expand_influxql_macros(_validate_influxql(query), window_seconds)
    params = {"db": cfg.influxdb_database, "q": expanded_query, "epoch": "ms"}
    try:
        with httpx.Client(timeout=cfg.timeout_sec) as client:
            response = client.get(
                _datasource_proxy_url(cfg, cfg.influxdb_datasource_uid, "/query"),
                headers=_grafana_headers(cfg),
                params=params,
            )
            response.raise_for_status()
            payload = response.json()
    except httpx.HTTPStatusError as exc:
        raise GrafanaQueryError(f"Grafana InfluxDB request failed: HTTP {exc.response.status_code} {exc.response.text}") from exc
    except httpx.HTTPError as exc:
        raise GrafanaQueryError(f"Grafana InfluxDB request failed: {exc}") from exc
    except ValueError as exc:
        raise GrafanaQueryError("Grafana InfluxDB request did not return JSON.") from exc
    if not isinstance(payload, dict):
        raise GrafanaQueryError("Grafana InfluxDB response must be a JSON object.")
    for result in payload.get("results", []):
        if isinstance(result, dict) and result.get("error"):
            raise GrafanaQueryError(f"Grafana InfluxDB request did not succeed: {result['error']}")
    return payload


def list_prometheus_label_values(label: str, *, config: GrafanaConfig | None = None) -> list[str]:
    label_name = _validate_target(label)
    payload = prometheus_get(f"/api/v1/label/{quote(label_name, safe='')}/values", config=config)
    values = payload.get("data", [])
    if not isinstance(values, list):
        return []
    return sorted(str(value) for value in values if str(value).strip())


def list_free5gc_monitoring_targets(scope: str) -> list[dict[str, str]]:
    """List free5gc monitoring targets for a supported scope."""

    cfg = load_grafana_config()
    if scope == "core_free5gc_nfs":
        return [
            {"name": nf.upper(), "query_value": nf, "target_type": "core_nf", "label": cfg.nf_label}
            for nf in CORE_FREE5GC_NFS
        ]
    if scope == "active_free5gc_containers":
        containers = [
            value for value in list_prometheus_label_values("container", config=cfg)
            if value not in IGNORED_FREE5GC_CONTAINERS and value.lower() in set(CORE_FREE5GC_NFS)
        ]
        return [
            {"name": container.upper(), "query_value": container.lower(), "target_type": "container", "label": "container"}
            for container in containers
        ]
    if scope == "all_free5gc_pods":
        pods = [value for value in list_prometheus_label_values("pod", config=cfg) if "free5gc" in value.lower()]
        return [
            {"name": pod, "query_value": pod, "target_type": "pod", "label": "pod"}
            for pod in pods
        ]
    raise ValueError(f"Unsupported target listing scope: {scope}")


def _extract_instant_values(payload: dict[str, Any]) -> list[float]:
    data = payload.get("data")
    if not isinstance(data, dict):
        return []
    result = data.get("result")
    if not isinstance(result, list):
        return []
    values: list[float] = []
    for item in result:
        if not isinstance(item, dict):
            continue
        value = item.get("value")
        if not isinstance(value, list) or len(value) < 2:
            continue
        try:
            values.append(float(value[1]))
        except (TypeError, ValueError):
            continue
    return values


def extract_influx_series(payload: dict[str, Any]) -> list[dict[str, Any]]:
    series: list[dict[str, Any]] = []
    for result in payload.get("results", []):
        if isinstance(result, dict) and isinstance(result.get("series"), list):
            series.extend(item for item in result["series"] if isinstance(item, dict))
    return series


def extract_influx_values(payload: dict[str, Any]) -> list[float]:
    values: list[float] = []
    for series in extract_influx_series(payload):
        columns = series.get("columns", [])
        for row in series.get("values", []):
            if not isinstance(row, list):
                continue
            for index, value in enumerate(row):
                if index < len(columns) and columns[index] == "time":
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    values.append(float(value))
    return values


def _result_from_values(*, target: dict[str, str], query: str, values: list[float], config: GrafanaConfig, query_time: float) -> dict[str, Any]:
    if not values:
        raise GrafanaQueryError(f"No CPU samples returned for target '{target['name']}'.")
    return {
        "target": target["name"],
        "target_type": target["target_type"],
        "query_value": target["query_value"],
        "metric": "cpu_usage",
        "window_seconds": config.window_seconds,
        "average": sum(values) / len(values),
        "unit": "percent",
        "query": query,
        "sample_count": len(values),
        "time_range": {"end_unix": query_time, "start_unix": query_time - config.window_seconds},
    }


def monitor_target_cpu_average(target: dict[str, str], *, timestamp: float | None = None) -> dict[str, Any]:
    cfg = load_grafana_config()
    query_time = timestamp or time.time()
    if target.get("target_type") == "pod":
        query = build_pod_cpu_promql(target["query_value"], cfg)
    else:
        query = build_nf_cpu_promql(target["query_value"], cfg)
    payload = query_prometheus(query, config=cfg, timestamp=query_time)
    values = _extract_instant_values(payload)
    return _result_from_values(target=target, query=query, values=values, config=cfg, query_time=query_time)


def monitor_nf_cpu_average(nf: str, *, timestamp: float | None = None) -> dict[str, Any]:
    """Return the latest 1-minute average CPU value for one NF container."""

    target = {"name": nf.upper(), "query_value": nf.lower(), "target_type": "core_nf", "label": "container"}
    return monitor_target_cpu_average(target, timestamp=timestamp)


def monitoring_source() -> dict[str, Any]:
    cfg = load_grafana_config()
    return {
        "type": "grafana",
        "grafana_url": cfg.url,
        "datasource_uids": {
            "prometheus": cfg.datasource_uid,
            "influxdb": cfg.influxdb_datasource_uid,
        },
        "mcp_tools": [
            "query_prometheus",
            "query_influxql",
            "monitor_nf_cpu_average",
            "monitor_nf_replica_health",
            "list_free5gc_monitoring_targets",
        ],
    }
