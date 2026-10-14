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
GATEWAY = "http://127.0.0.1:18082"


def request(path, body=None, method=None):
    req = urllib.request.Request(
        GATEWAY + path,
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
    logpath = ROOT / "experiments/agora_runtime/logs/e2-integration-launch.log"
    logpath.parent.mkdir(parents=True, exist_ok=True)
    with logpath.open("w") as log:
        options = (
            ["--sim-time", "110", "--test-ue-subset", "3"]
            if "--cached" in sys.argv
            else ["--smoke", "--sim-time", "180", "--pace-ms", "250"]
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
        for route in ("near_rt", "non_rt"):
            live = next(r for r in request("/api/runs")["runs"] if r["run_id"] == live["run_id"])
            ident = uuid.uuid4().hex
            start = math.ceil(live["sim_time_s"]) + 25
            end = start + 8
            changes = {"txp_5g": 40, "ret_4g2": 5, "cio_4g1": 2, "ttt_ms": 340, "hys_db": 3}
            plan = {
                "schema_version": "agora-ric-queue-v1",
                "deployment_id": ident,
                "session_id": uuid.uuid4().hex,
                "candidate_id": "e2_integration_all_five_controls",
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
            result = request(path, body, "POST" if route == "near_rt" else "PUT")
            print(
                "SUBMIT "
                + route
                + " "
                + json.dumps({"id": ident, "start": start, "end": end, "status": result["status"]}),
                flush=True,
            )
            repeat = request(path, body, "POST" if route == "near_rt" else "PUT")
            assert repeat["deployment_id"] == ident
            for _ in range(1200 if "--cached" in sys.argv else 240):
                result = request("/api/deployments/" + ident)
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
            assert values[(1, 1)] == 40 and abs(values[(3, 2)] - 5) < 1e-6 and values[(2, 3)] == 2
            assert all(values[(i, 4)] == 340 and values[(i, 5)] == 3 for i in (1, 2, 3))
            assert len([r for r in result["receipts"] if r["status"] == "applied"]) == 1
            results.append(result)
            print("PASS " + route + " applied and restored at exact simulation times", flush=True)
        output = (
            ROOT
            / "experiments/agora_runtime"
            / (
                "cached-e2-integration-results.json"
                if "--cached" in sys.argv
                else "e2-integration-results.json"
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
