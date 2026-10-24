"""Exercise a real ns-3 process, FlexRIC, execution xApp and application receipts."""

import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
import uuid
import sys

ROOT = Path(__file__).resolve().parents[3]
RUNTIME = Path(os.environ.get("AGORA_RUNTIME_DIR", ROOT / "experiment/khu/results/agora_runtime"))
GATEWAY = "http://127.0.0.1:18082"


def request(path, body=None, method=None, base=GATEWAY):
    req = urllib.request.Request(
        base + path,
        data=json.dumps(body).encode() if body else None,
        headers={"Content-Type": "application/json"},
        method=method or ("POST" if body else "GET"),
    )
    with urllib.request.urlopen(req, timeout=20) as r:
        return json.load(r)


def main():
    manifest = json.loads(
        (ROOT / "smo/agentic_ai/agora_offline/runtime/assets_v16/manifest.json").read_text()
    )
    current = next(r["current_rcp"] for r in manifest["runs"] if r["run_id"] == "gnn_0130")
    before = {r["run_id"] for r in request("/api/runs")["runs"]}
    logpath = RUNTIME / "logs/e2-integration-launch.log"
    logpath.parent.mkdir(parents=True, exist_ok=True)
    with logpath.open("w") as log:
        options = (
            ["--sim-time", "250", "--test-ue-subset", "3"]
            if "--cached" in sys.argv
            else ["--smoke", "--sim-time", "240", "--pace-ms", "200"]
        )
        process = subprocess.Popen(
            [str(ROOT / ".venv/bin/python"), str(ROOT / "scripts/agora.py"), "simulate", *options],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    live = None
    results = []
    try:
        for _ in range(100):
            runs = [
                r
                for r in request("/api/runs")["runs"]
                if r["run_id"] not in before and r["process_alive"]
            ]
            if runs:
                live = runs[0]
                break
            if process.poll() is not None:
                raise RuntimeError(logpath.read_text()[-6000:])
            time.sleep(0.2)
        if not live:
            raise RuntimeError("No live ns-3 manifest")
        print("RUN " + live["run_id"], flush=True)
        cases = [
            ("energy_saving", "non_rt", {"txp_5g": 40}),
            ("coverage", "non_rt", {"ret_4g2": 5}),
            ("throughput", "near_rt", {"txp_4g2": 33, "ret_5g": 7}),
            ("load_balancing", "near_rt", {"cio_4g1": 2}),
            ("handover_control", "near_rt", {"ttt_ms": 340, "hys_db": 3}),
        ]
        for objective, route, changes in cases:
            live = next(r for r in request("/api/runs")["runs"] if r["run_id"] == live["run_id"])
            ident = uuid.uuid4().hex
            start = math.ceil(live["sim_time_s"]) + 25
            end = start + 8
            plan = {
                "schema_version": "agora-ric-queue-v1",
                "deployment_id": ident,
                "session_id": uuid.uuid4().hex,
                "candidate_id": "e2_integration_" + objective,
                "intent": {"objective": objective},
                "target_ric": route,
                "run_id": "gnn_0130",
                "current_parameters": current,
                "candidate": {"interval_start_s": start, "interval_end_s": end, "changes": changes},
            }
            body = {"live_run_id": live["run_id"], "plan": plan}
            path = (
                "/api/near-rt/deployments"
                if route == "near_rt"
                else "/a1-p/policytypes/401/policies/" + ident
            )
            base = GATEWAY if route == "near_rt" else "http://127.0.0.1:9000"
            result = request(path, body, "POST" if route == "near_rt" else "PUT", base)
            print(
                "SUBMIT "
                + route
                + " "
                + json.dumps({"id": ident, "start": start, "end": end, "status": result["status"]}),
                flush=True,
            )
            repeat = request(path, body, "POST" if route == "near_rt" else "PUT", base)
            assert repeat["deployment_id"] == ident
            for _ in range(1200 if "--cached" in sys.argv else 240):
                result = (
                    request("/api/deployments/" + ident)
                    if route == "near_rt"
                    else request(path + "/status", base=base)
                )
                if result["status"] in (
                    "rejected",
                    "transport_failed",
                    "restore_conflict",
                    "unconfirmed_simulation_stopped",
                ):
                    raise RuntimeError(json.dumps(result, indent=2))
                if result["restored"]:
                    break
                time.sleep(0.25)
            assert result["applied"] and result["restored"], result
            applied = next(r for r in result["receipts"] if r["status"] == "applied")
            restored = next(r for r in result["receipts"] if r["status"] == "restored")
            assert abs(applied["sim_time_s"] - start) < 1e-6
            assert abs(restored["sim_time_s"] - end) < 1e-6
            values = {(r["cell"], r["kind"]): r["value"] for r in applied["readback"]}
            kinds = {"txp": 1, "ret": 2, "cio": 3, "ttt_ms": 4, "hys_db": 5}
            for key, expected in changes.items():
                if key in ("ttt_ms", "hys_db"):
                    assert all(
                        abs(values[(cell, kinds[key])] - expected) < 1e-6 for cell in (1, 2, 3)
                    )
                else:
                    family, cell = key.split("_", 1)
                    assert (
                        abs(values[({"5g": 1, "4g1": 2, "4g2": 3}[cell], kinds[family])] - expected)
                        < 1e-6
                    )
            assert result["application"]["id"] == objective
            if route == "non_rt":
                assert result["path"][0] == "Non_RT_RIC"
                assert result["rapp_invocation"]["status"] == "forwarded"
                assert result["a1_policy_id"] == ident
            assert len([r for r in result["receipts"] if r["status"] == "applied"]) == 1
            results.append(result)
            print(
                "PASS "
                + objective
                + " via "
                + route
                + " applied and restored at exact simulation times",
                flush=True,
            )
        output = (
            RUNTIME
            / (
                "cached-app-routes-integration-results.json"
                if "--cached" in sys.argv
                else "app-routes-integration-results.json"
            )
        )
        output.write_text(json.dumps(results, indent=2))
        print(str(output), flush=True)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=15)


if __name__ == "__main__":
    main()
