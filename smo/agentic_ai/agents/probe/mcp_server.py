"""Grafana MCP server exposing monitoring tools."""

from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from tools.grafana_mcp_tools import list_free5gc_monitoring_targets as run_list_free5gc_monitoring_targets
from tools.grafana_mcp_tools import monitor_nf_cpu_average as run_monitor_nf_cpu_average
from tools.grafana_mcp_tools import query_influxql as run_query_influxql
from tools.grafana_mcp_tools import query_prometheus as run_query_prometheus

mcp = FastMCP("grafana-monitoring")


@mcp.tool()
def query_prometheus(query: str) -> dict[str, Any]:
    """Run a Prometheus instant query through the configured Grafana datasource."""

    return run_query_prometheus(query)


@mcp.tool()
def query_influxql(query: str, window_seconds: int = 60) -> dict[str, Any]:
    """Run a read-only InfluxQL SELECT through the configured Grafana datasource."""

    return run_query_influxql(query, window_seconds=window_seconds)


@mcp.tool()
def list_free5gc_monitoring_targets(scope: str) -> list[dict[str, str]]:
    """List free5gc targets for a supported monitoring scope."""

    return run_list_free5gc_monitoring_targets(scope)


@mcp.tool()
def monitor_nf_cpu_average(nf: str) -> dict[str, Any]:
    """Return the latest 1-minute average CPU usage for one NF."""

    return run_monitor_nf_cpu_average(nf)


if __name__ == "__main__":
    mcp.run()
