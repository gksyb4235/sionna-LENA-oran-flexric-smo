from __future__ import annotations

import contextlib
import io
import json
import os
import unittest
from unittest.mock import patch

import agent
from schemas import validate_monitoring_command, validate_monitoring_report
from tools.grafana_mcp_tools import (
    GrafanaConfig,
    GrafanaQueryError,
    _extract_instant_values,
    build_nf_cpu_promql,
    expand_influxql_macros,
    extract_influx_values,
    seconds_to_prometheus_duration,
)


def sample_target_result(name: str = "AMF", average: float = 37.5) -> dict[str, object]:
    return {
        "target": name,
        "target_type": "core_nf",
        "query_value": name.lower(),
        "metric": "cpu_usage",
        "window_seconds": 60,
        "average": average,
        "unit": "percent",
        "query": f'sum(rate(container_cpu_usage_seconds_total{{container="{name.lower()}"}}[1m])) * 100',
        "sample_count": 1,
        "time_range": {"start_unix": 10.0, "end_unix": 70.0},
    }


def prometheus_value(value: float) -> dict[str, object]:
    return {"status": "success", "data": {"resultType": "vector", "result": [{"value": [1000, str(value)]}]}}


def influx_value(value: float) -> dict[str, object]:
    return {
        "results": [
            {
                "series": [
                    {
                        "name": "kpm",
                        "tags": {"ue_id": "ue-1"},
                        "columns": ["time", "thp_dl_kbps"],
                        "values": [[1000, value]],
                    }
                ]
            }
        ]
    }


def sample_influx_query_command() -> object:
    return validate_monitoring_command(
        {
            "operation": "influxql_query_plan",
            "metric": "ue_throughput",
            "scope": "per_ue",
            "nfs": [],
            "window_seconds": 60,
            "expected_replicas": None,
            "queries": [
                {
                    "tool": "query_influxql",
                    "label": "ue_downlink_throughput",
                    "target": "kpm",
                    "query": 'SELECT "thp_dl_kbps" FROM "kpm" WHERE $timeFilter GROUP BY "ue_id"',
                    "purpose": "downlink throughput grouped by UE",
                    "query_type": "influxql",
                    "unit": "kbps",
                }
            ],
            "evaluation": {},
        }
    )


def sample_replica_query_command() -> object:
    return validate_monitoring_command(
        {
            "operation": "prometheus_query_plan",
            "metric": "replica_health",
            "scope": "single_nf",
            "nfs": ["AMF"],
            "window_seconds": 60,
            "expected_replicas": 2,
            "queries": [
                {
                    "tool": "query_prometheus",
                    "check_id": "available-check",
                    "label": "display label chosen by Probe",
                    "target": "AMF",
                    "query": "available_query",
                    "purpose": "available replicas",
                    "query_type": "instant",
                    "unit": "replicas",
                },
                {
                    "tool": "query_prometheus",
                    "check_id": "ready-check",
                    "label": "another unrelated display label",
                    "target": "AMF",
                    "query": "ready_query",
                    "purpose": "ready replicas",
                    "query_type": "instant",
                    "unit": "replicas",
                },
            ],
            "evaluation": {
                "type": "thresholds",
                "checks": [
                    {
                        "check_id": "available-check",
                        "observation": "AMF available deployment replica count",
                        "expected": {"operator": ">=", "value": 2},
                    },
                    {
                        "check_id": "ready-check",
                        "observation": "AMF ready deployment replica count",
                        "expected": {"operator": ">=", "value": 2},
                    },
                ],
            },
        }
    )


class MonitoringAgentTests(unittest.TestCase):
    def test_agent_instructions_use_check_id_contract(self) -> None:
        instructions = agent.load_agent_instructions()

        self.assertNotIn("query_label", instructions)
        self.assertIn("copy each opaque `check_id` exactly", instructions)
        self.assertIn("not a supported-NF allowlist", instructions)

    def test_command_prompt_preserves_planner_check_ids(self) -> None:
        request = {
            "type": "thresholds",
            "checks": [{"check_id": "opaque-1", "observation": "AMF pod count", "expected": {"operator": "==", "value": 2}}],
        }

        prompt = agent.build_command_prompt("Verify AMF", evaluation_request=request)

        self.assertIn('"check_id": "opaque-1"', prompt)
        self.assertIn("copy its opaque `check_id` unchanged", prompt)

    def test_validate_monitoring_command(self) -> None:
        command = validate_monitoring_command(
            {
                "operation": "cpu_average",
                "metric": "cpu_usage",
                "scope": "core_free5gc_nfs",
                "nfs": [],
                "window_seconds": 60,
            }
        )

        self.assertEqual(command.operation, "cpu_average")
        self.assertEqual(command.scope, "core_free5gc_nfs")

    def test_extract_json_object_uses_monitoring_command(self) -> None:
        payload = agent.extract_json_object(
            'ignore {"x": 1}\n{"operation":"cpu_average","metric":"cpu_usage","scope":"single_nf","nfs":["AMF"],"window_seconds":60}'
        )

        command = validate_monitoring_command(payload)
        self.assertEqual(command.nfs, ["AMF"])

    def test_infer_nf_from_intent(self) -> None:
        self.assertEqual(agent.infer_nf_from_intent("Monitor AMF cpu"), "AMF")
        self.assertEqual(agent.infer_nf_from_intent("monitor nf=UPF-1 cpu"), "UPF-1")
        self.assertEqual(agent.infer_nf_from_intent("monitor 'SMF.main' cpu"), "SMF.main")

    def test_prometheus_duration(self) -> None:
        self.assertEqual(seconds_to_prometheus_duration(60), "1m")
        self.assertEqual(seconds_to_prometheus_duration(90), "1m30s")

    def test_build_nf_cpu_promql(self) -> None:
        cfg = GrafanaConfig(
            url="http://grafana.local",
            api_key="token",
            datasource_uid="prometheus-prod",
            timeout_sec=20,
            cpu_metric="nf_cpu_usage_percent",
            nf_label="nf",
            cpu_promql_template="avg_over_time({metric_name}{label_selector}[{window}])",
            window_seconds=60,
        )

        query = build_nf_cpu_promql("AMF", cfg)

        self.assertEqual(query, 'avg_over_time(nf_cpu_usage_percent{nf="AMF"}[1m])')

    def test_build_free5gc_container_cpu_promql_with_lowercase_nf(self) -> None:
        cfg = GrafanaConfig(
            url="http://grafana.local",
            api_key="token",
            datasource_uid="prometheus-prod",
            timeout_sec=20,
            cpu_metric="container_cpu_usage_seconds_total",
            nf_label="container",
            cpu_promql_template='sum(rate({metric_name}{{namespace=~"free5gc.*",{nf_label}="{nf_lower}",image!=""}}[{window}])) * 100',
            window_seconds=60,
        )

        query = build_nf_cpu_promql("AMF", cfg)

        self.assertEqual(
            query,
            'sum(rate(container_cpu_usage_seconds_total{namespace=~"free5gc.*",container="amf",image!=""}[1m])) * 100',
        )

    def test_extract_prometheus_values(self) -> None:
        payload = {
            "status": "success",
            "data": {"resultType": "vector", "result": [{"value": [1000, "25.5"]}, {"value": [1000, "50"]}]},
        }

        self.assertEqual(_extract_instant_values(payload), [25.5, 50.0])

    def test_influxql_macros_and_values(self) -> None:
        query = 'SELECT mean("thp_dl_kbps") FROM "kpm" WHERE $timeFilter GROUP BY time($__interval), "sst" fill(null)'

        self.assertEqual(
            expand_influxql_macros(query, 120),
            'SELECT mean("thp_dl_kbps") FROM "kpm" WHERE time >= now() - 120s GROUP BY time(2s), "sst" fill(null)',
        )
        self.assertEqual(extract_influx_values(influx_value(42.5)), [42.5])

    def test_influxql_rejects_write_queries(self) -> None:
        payload = sample_influx_query_command().to_dict()
        payload["queries"][0]["query"] = 'SELECT "thp_dl_kbps" INTO "archive" FROM "kpm"'

        with self.assertRaisesRegex(ValueError, "read-only SELECT"):
            validate_monitoring_command(payload)

    def test_run_monitoring_single_nf_report(self) -> None:
        command = validate_monitoring_command(
            {"operation": "cpu_average", "metric": "cpu_usage", "scope": "single_nf", "nfs": ["AMF"], "window_seconds": 60}
        )
        with patch.object(agent, "build_monitoring_command", return_value=command):
            with patch.object(agent, "monitor_target_cpu_average", return_value=sample_target_result()):
                with patch.object(agent, "monitoring_source", return_value={"type": "test"}):
                    report = agent.run_monitoring("Monitor AMF cpu").to_dict()

        validate_monitoring_report(report)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["results"][0]["target"], "AMF")

    def test_run_monitoring_core_nf_scope(self) -> None:
        command = validate_monitoring_command(
            {"operation": "cpu_average", "metric": "cpu_usage", "scope": "core_free5gc_nfs", "nfs": [], "window_seconds": 60}
        )
        targets = [
            {"name": "AMF", "query_value": "amf", "target_type": "core_nf", "label": "container"},
            {"name": "SMF", "query_value": "smf", "target_type": "core_nf", "label": "container"},
        ]
        with patch.object(agent, "build_monitoring_command", return_value=command):
            with patch.object(agent, "list_free5gc_monitoring_targets", return_value=targets):
                with patch.object(agent, "monitor_target_cpu_average", side_effect=[sample_target_result("AMF", 1.0), sample_target_result("SMF", 2.0)]):
                    with patch.object(agent, "monitoring_source", return_value={"type": "test"}):
                        report = agent.run_monitoring("Monitor each NF cpu").to_dict()

        validate_monitoring_report(report)
        self.assertEqual(report["scope"], "core_free5gc_nfs")
        self.assertEqual([item["target"] for item in report["results"]], ["AMF", "SMF"])


    def test_validate_query_plan_command(self) -> None:
        command = sample_replica_query_command()

        self.assertEqual(command.operation, "prometheus_query_plan")
        self.assertEqual(command.metric, "replica_health")
        self.assertEqual(command.expected_replicas, 2)
        self.assertEqual(command.queries[0]["tool"], "query_prometheus")

    def test_replica_health_intent_builds_query_plan(self) -> None:
        with patch.dict(os.environ, {"MONITORING_USE_DETERMINISTIC_TEST_COMMAND": "1"}):
            command = agent.build_monitoring_command("Replica with 2 AMFs and monitor if it's going well")

        self.assertEqual(command.operation, "prometheus_query_plan")
        self.assertEqual(command.metric, "replica_health")
        self.assertEqual(command.nfs, ["AMF"])
        self.assertEqual(command.expected_replicas, 2)
        self.assertGreaterEqual(len(command.queries), 3)

    def test_llm_cpu_command_overridden_for_replica_health_intent(self) -> None:
        class FakeMessage:
            content = json.dumps(
                {
                    "operation": "cpu_average",
                    "metric": "cpu_usage",
                    "scope": "single_nf",
                    "nfs": ["AMF"],
                    "window_seconds": 60,
                    "expected_replicas": None,
                    "queries": [],
                    "evaluation": {},
                }
            )

        class FakeAgent:
            def invoke(self, _: object) -> dict[str, object]:
                return {"messages": [FakeMessage()]}

        with patch.object(agent, "create_probe_agent", return_value=FakeAgent()):
            command = agent.build_monitoring_command("Replica with 2 AMFs and monitor if it's going well")

        self.assertEqual(command.operation, "prometheus_query_plan")
        self.assertEqual(command.metric, "replica_health")
        self.assertTrue(command.queries)

    def test_execute_query_plan_thresholds(self) -> None:
        command = sample_replica_query_command()
        cfg = GrafanaConfig(
            url="http://grafana.local",
            api_key="token",
            datasource_uid="prometheus-prod",
            timeout_sec=20,
            cpu_metric="container_cpu_usage_seconds_total",
            nf_label="container",
            cpu_promql_template="",
            window_seconds=60,
        )
        with patch.object(agent, "load_grafana_config", return_value=cfg):
            with patch.object(agent, "query_prometheus", side_effect=[prometheus_value(2), prometheus_value(2)]):
                with patch.object(agent, "monitoring_source", return_value={"type": "test"}):
                    report = agent.execute_monitoring_command("Check AMF replicas", command).to_dict()

        validate_monitoring_report(report)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["metric"], "replica_health")
        self.assertEqual(report["results"][0]["evaluation"]["health_status"], "healthy")
        self.assertEqual(report["evaluation_result"]["checks"][0]["check_id"], "available-check")

    def test_optional_threshold_query_failure_is_neutral_for_known_evaluation(self) -> None:
        payload = sample_replica_query_command().to_dict()
        payload["evaluation"]["checks"][1]["optional"] = True
        command = validate_monitoring_command(payload)
        cfg = GrafanaConfig(
            url="http://grafana.local",
            api_key="token",
            datasource_uid="prometheus-prod",
            timeout_sec=20,
            cpu_metric="container_cpu_usage_seconds_total",
            nf_label="container",
            cpu_promql_template="",
            window_seconds=60,
        )
        optional_outcomes = (
            ("no_samples", {"status": "success", "data": {"resultType": "vector", "result": []}}),
            ("query_exception", GrafanaQueryError("optional query failed")),
        )

        for actual, expected_status in ((2, "passed"), (1, "failed")):
            for outcome_name, optional_outcome in optional_outcomes:
                with self.subTest(expected_status=expected_status, optional_outcome=outcome_name):
                    with patch.object(agent, "load_grafana_config", return_value=cfg):
                        with patch.object(agent, "query_prometheus", side_effect=[prometheus_value(actual), optional_outcome]):
                            with patch.object(agent, "monitoring_source", return_value={"type": "test"}):
                                report = agent.execute_monitoring_command("Check AMF replicas", command).to_dict()

                    optional_check = next(
                        check for check in report["evaluation_result"]["checks"]
                        if check["check_id"] == "ready-check"
                    )
                    self.assertEqual(report["status"], "completed")
                    self.assertEqual(report["evaluation_result"]["status"], expected_status)
                    self.assertTrue(optional_check["skipped"])
                    self.assertTrue(report["errors"])

    def test_required_threshold_query_failure_remains_partial_and_unknown(self) -> None:
        command = sample_replica_query_command()
        cfg = GrafanaConfig(
            url="http://grafana.local",
            api_key="token",
            datasource_uid="prometheus-prod",
            timeout_sec=20,
            cpu_metric="container_cpu_usage_seconds_total",
            nf_label="container",
            cpu_promql_template="",
            window_seconds=60,
        )
        with patch.object(agent, "load_grafana_config", return_value=cfg):
            with patch.object(agent, "query_prometheus", side_effect=[prometheus_value(2), GrafanaQueryError("required query failed")]):
                with patch.object(agent, "monitoring_source", return_value={"type": "test"}):
                    report = agent.execute_monitoring_command("Check AMF replicas", command).to_dict()

        self.assertEqual(report["status"], "partial")
        self.assertEqual(report["evaluation_result"]["status"], "unknown")

    def test_threshold_query_requires_matching_check_id(self) -> None:
        payload = sample_replica_query_command().to_dict()
        payload["evaluation"] = {
            "type": "thresholds",
            "checks": [{"check_id": "missing", "observation": "missing observation", "expected": {"operator": ">=", "value": 1}}],
        }

        with self.assertRaisesRegex(ValueError, "no matching query check_id"):
            validate_monitoring_command(payload)

    def test_legacy_label_threshold_contract_is_rejected(self) -> None:
        payload = sample_replica_query_command().to_dict()
        payload["evaluation"] = {
            "type": "thresholds",
            "checks": [{"query_label": "legacy", "operator": "==", "value": 2}],
        }

        with self.assertRaisesRegex(ValueError, "nested expected contract"):
            validate_monitoring_command(payload)

    def test_unsupported_threshold_operator_is_rejected(self) -> None:
        payload = sample_replica_query_command().to_dict()
        payload["evaluation"] = {
            "type": "thresholds",
            "checks": [{"check_id": "available-check", "observation": "available", "expected": {"operator": "~", "value": 1}}],
        }

        with self.assertRaisesRegex(ValueError, "Unsupported threshold operator"):
            validate_monitoring_command(payload)

    def test_multi_sample_lower_bound_uses_conservative_minimum(self) -> None:
        payload = sample_replica_query_command().to_dict()
        payload["evaluation"] = {
            "type": "thresholds",
            "checks": [{"check_id": "available-check", "observation": "available", "expected": {"operator": ">=", "value": 1}}],
        }

        result = agent._evaluate_thresholds(
            validate_monitoring_command(payload),
            [{"check_id": "available-check", "label": "anything", "first_value": 2.0, "values": [2.0, 0.0], "sample_count": 2}],
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["checks"][0]["actual"], 0.0)
        self.assertEqual(result["checks"][0]["reducer"], "min")

    def test_expected_value_can_reference_another_check_id(self) -> None:
        payload = sample_replica_query_command().to_dict()
        payload["queries"].append({
            "tool": "query_prometheus",
            "check_id": "desired-check",
            "label": "display only",
            "target": "AMF",
            "query": "desired_query",
            "purpose": "desired replicas",
            "query_type": "instant",
        })
        payload["evaluation"] = {
            "type": "thresholds",
            "checks": [{
                "check_id": "available-check",
                "observation": "AMF available replicas",
                "expected": {"operator": ">=", "value_from_check_id": "desired-check"},
            }],
        }
        command = validate_monitoring_command(payload)

        result = agent._evaluate_thresholds(command, [
            {"check_id": "available-check", "values": [2.0], "first_value": 2.0, "sample_count": 1},
            {"check_id": "desired-check", "values": [2.0], "first_value": 2.0, "sample_count": 1},
        ])

        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["checks"][0]["expected"], 2.0)
        self.assertEqual(result["query_values"]["desired-check"], 2.0)

    def test_legacy_threshold_evaluation_is_not_available(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a query plan"):
            agent._evaluate_legacy_results({"type": "thresholds"}, [sample_target_result()], "completed")

    def test_influxql_query_plan_retries_five_times_after_initial_failure(self) -> None:
        command = sample_influx_query_command()
        cfg = GrafanaConfig(
            url="http://grafana.local",
            api_key="token",
            datasource_uid="prometheus-prod",
            timeout_sec=20,
            cpu_metric="container_cpu_usage_seconds_total",
            nf_label="container",
            cpu_promql_template="",
            window_seconds=60,
        )
        with patch.object(agent, "build_monitoring_command", return_value=command) as build_command:
            with patch.object(agent, "load_grafana_config", return_value=cfg):
                with patch.object(
                    agent,
                    "query_influxql",
                    side_effect=[
                        GrafanaQueryError("failed 1"),
                        GrafanaQueryError("failed 2"),
                        GrafanaQueryError("failed 3"),
                        GrafanaQueryError("failed 4"),
                        GrafanaQueryError("failed 5"),
                        influx_value(42.5),
                    ],
                ) as query:
                    with patch.object(agent, "monitoring_source", return_value={"type": "test"}):
                        with patch.object(agent, "_learn_successful_queries", return_value=False):
                            report = agent.run_monitoring("Show per-UE downlink throughput").to_dict()

        self.assertEqual(report["status"], "completed")
        self.assertEqual(query.call_count, 6)
        self.assertEqual(build_command.call_count, 6)
        self.assertEqual(report["results"][0]["query_results"][0]["series"][0]["tags"]["ue_id"], "ue-1")

    def test_learned_influx_fields_share_one_pattern(self) -> None:
        content = "## Learned Query Patterns\n\n<!-- learned-query-catalog:start -->\n<!-- learned-query-catalog:end -->\n"
        first = sample_influx_query_command().queries[0]
        second = dict(first, query='SELECT "thp_ul_kbps" FROM "kpm" WHERE $timeFilter GROUP BY "ue_id"')

        content = agent._add_catalog_entry(content, first)
        content = agent._add_catalog_entry(content, second)

        self.assertEqual(content.count('SELECT <fields> FROM "kpm" WHERE $timeFilter GROUP BY "ue_id"'), 1)
        self.assertIn('fields: `"thp_dl_kbps"`', content)
        self.assertIn('fields: `"thp_ul_kbps"`', content)

        cpu_amf = {"tool": "query_prometheus", "query": 'sum(rate(cpu{container="amf"}[1m]))'}
        cpu_smf = {"tool": "query_prometheus", "query": 'sum(rate(cpu{container="smf"}[1m]))'}
        content = agent._add_catalog_entry(agent._add_catalog_entry(content, cpu_amf), cpu_smf)
        self.assertEqual(content.count('sum(rate(cpu{container="<nf>"}[1m]))'), 1)
        self.assertIn("nf: `amf`", content)
        self.assertIn("nf: `smf`", content)

    def test_cli_wraps_monitoring_output(self) -> None:
        command = validate_monitoring_command(
            {"operation": "cpu_average", "metric": "cpu_usage", "scope": "single_nf", "nfs": ["AMF"], "window_seconds": 60}
        )
        stdout = io.StringIO()
        with patch.object(agent, "build_monitoring_command", return_value=command):
            with patch.object(agent, "monitor_target_cpu_average", return_value=sample_target_result()):
                with patch.object(agent, "monitoring_source", return_value={"type": "test"}):
                    with contextlib.redirect_stdout(stdout):
                        self.assertEqual(agent.main(["Monitor AMF cpu"]), 0)

        output = stdout.getvalue()
        self.assertIn("------probe-agent------", output)
        self.assertIn('"target": "AMF"', output)
        self.assertTrue(output.strip().endswith("----end----"))


if __name__ == "__main__":
    unittest.main()
