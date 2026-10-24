#!/usr/bin/env python3
"""One entry point for the repository's AGORA services and control-enabled ns-3."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request
import uuid

ROOT = Path(__file__).resolve().parents[1]
RUNTIME = Path(os.environ.get("AGORA_RUNTIME_DIR", ROOT / "experiment/khu/results/agora_runtime"))
APP = ROOT / "smo/agentic_ai/agora_offline"
PYTHON = ROOT / ".venv/bin/python"
CONF = ROOT / "smo/ric_control/config"


def environment():
    env = dict(os.environ)
    env["PATH"] = f"{ROOT}/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    env["VIRTUAL_ENV"] = str(ROOT / ".venv")
    env["PYTHONPATH"] = ":".join(
        [str(ROOT), str(ROOT / "non-rt-ric"), str(ROOT / "smo/agentic_ai"),
         str(ROOT / ".venv/lib/python3.12/site-packages")]
    )
    return env


def execute(argv, **kw):
    return subprocess.run([str(a) for a in argv], cwd=ROOT, env=environment(), check=True, **kw)


def get(url):
    with urllib.request.urlopen(url, timeout=3) as r:
        return json.load(r)


def services():
    return {
        "non-rt": [
            PYTHON,
            "-m",
            "uvicorn",
            "app.main:app",
            "--app-dir",
            ROOT / "non-rt-ric/A1_Mediator",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
        ],
        "ui": [PYTHON, "-m", "agora_offline.server", "--port", "18081"],
        "gateway": [PYTHON, "-m", "smo.ric_control.gateway", "--port", "18082"],
        "ric": [
            ROOT / "near-rt-ric/build/examples/ric/nearRT-RIC",
            "-c",
            CONF / "flexric.conf",
            "-p",
            "/usr/local/lib/flexric/",
        ],
    }


def start(name):
    health = {
        "non-rt": "http://127.0.0.1:9000/api/non-rt/health",
        "ui": "http://127.0.0.1:18081/api/health",
        "gateway": "http://127.0.0.1:18082/health",
    }
    if name in health:
        try:
            print(f"{name}: already reachable {get(health[name])}")
            return
        except (OSError, ValueError):
            pass
    state = RUNTIME / "services"
    state.mkdir(parents=True, exist_ok=True)
    pidfile = state / f"{name}.json"
    if pidfile.exists():
        saved = json.loads(pidfile.read_text())
        try:
            process = Path(f"/proc/{saved['pid']}/cmdline").read_bytes()
            if all(str(arg).encode() in process for arg in services()[name][:3]):
                print(f"{name}: already running pid={saved['pid']}")
                return
        except OSError:
            pass
    log = RUNTIME / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("ab") as output:
        p = subprocess.Popen(
            [str(v) for v in services()[name]],
            cwd=ROOT,
            env=environment(),
            stdin=subprocess.DEVNULL,
            stdout=output,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    pidfile.write_text(
        json.dumps({"pid": p.pid, "argv": [str(v) for v in services()[name]], "log": str(log)})
    )
    print(f"{name}: pid={p.pid}, log={log}")


def simulate(args):
    if args.from_session:
        if len(args.from_session) != 32 or any(
            c not in "0123456789abcdef" for c in args.from_session
        ):
            raise ValueError("Invalid session id")
        saved = json.loads(
            (APP / ".state/sessions" / args.from_session / "session.json").read_text()
        )
        plan = saved["result"]["execution_plan"]
        run = plan["run_id"]
        current = plan["current_parameters"]
    else:
        run = args.run
        manifest = json.loads((APP / "runtime/assets_v16/manifest.json").read_text())
        current = next(r["current_rcp"] for r in manifest["runs"] if r["run_id"] == run)
    identity = uuid.uuid4().hex
    env = environment()
    env.update(
        AGORA_RUN_ID=identity,
        AGORA_CONTROL_DIR=str(RUNTIME / "runs"),
        AGORA_REPLAY_RUN_ID=run,
        AGORA_RET_CACHE_ROOT=str(ROOT / "experiment/khu/caches/ret_caches"),
    )
    directory = RUNTIME / "runs" / identity
    directory.mkdir(parents=True, exist_ok=False)
    if args.smoke:
        argv = [
            ROOT / "build/scratch/ns3.48-agora-ric-smoke-default",
            f"--simTime={args.sim_time}",
            f"--paceMs={args.pace_ms}",
        ]
        argv += [f"--{key}={value}" for key, value in current.items()]
        trace = None
    else:
        sources = json.loads((CONF / "replay_sources.json").read_text())
        trace = ROOT / sources[run]["path"]
        if hashlib.sha256(trace.read_bytes()).hexdigest() != sources[run]["sha256"]:
            raise ValueError("UE trace checksum differs from the evaluated dataset")
        if args.test_ue_subset:
            import csv

            with trace.open() as source:
                rows = list(csv.reader(source))
            data = []
            for row in rows:
                if len(row) != 6:
                    continue
                try:
                    float(row[0])
                except ValueError:
                    continue
                data.append(row)
            ids = list(dict.fromkeys(r[1] for r in data))[: args.test_ue_subset]
            subset = directory / "control-test-ue-subset.csv"
            with subset.open("w") as output:
                writer = csv.writer(output)
                writer.writerow(["time_s", "ue_id", "x", "y", "z", "active"])
                writer.writerows(r for r in data if r[1] in ids)
            trace = subset
            args.ues = len(ids)
            print(
                "CONTROL TEST ONLY: using a UE subset; GNN KPI validity is not assessed.",
                flush=True,
            )
        retroot = ROOT / "experiment/khu/caches/ret_caches"
        a, b, c = (int(current[f"ret_{cell}"]) for cell in ("4g1", "4g2", "5g"))
        low = (
            retroot / f"tilt_{a}deg/sionna_rt_cache_1p8ghz_2x2_1x1_lzf.h5"
            if a == b
            else retroot / f"merged_percell/merged_1p8ghz_4g1_{a}deg_4g2_{b}deg.h5"
        )
        high = retroot / f"tilt_{c}deg/sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5"
        if not low.exists() or not high.exists():
            raise ValueError("Required RET cache missing")
        argv = [
            ROOT / "build/scratch/ns3.48-khu-real-dual-nr-sionna-final-default",
            f"--simTime={args.sim_time}s",
            f"--N_Ues={args.ues}",
            f"--sumoTrace={trace}",
            f"--sionnaCacheFile18={low}",
            f"--sionnaCacheFile35={high}",
            "--sionnaUpdatePeriod=1s",
            "--numerology=0",
            "--kpiReportInterval=1",
            "--ns3::NrHelper::E2ModeNr=true",
            "--ns3::NrHelper::E2TermIp=127.0.0.1",
            "--ns3::NrHelper::E2LocalPort=39110",
            "--ns3::NrHelper::UseIdealRrc=true",
            f"--resultsRoot={directory}/kpi",
            f"--runTag={run}",
            f"--handoverTtt={current['ttt_ms']}",
            f"--handoverHysteresis={current['hys_db']}",
        ]
        labels = {"5g": "gNB_5G", "4g1": "gNB_4G_1", "4g2": "gNB_4G_2"}
        for kind, option in [
            ("txp", "cellTxPowerDbm"),
            ("ret", "cellRetTiltDeg"),
            ("cio", "cellCioDb"),
        ]:
            values = ",".join(
                f"{name}:{current[f'{kind}_{label}']}" for label, name in labels.items()
            )
            argv.append(f"--{option}={values}")
        if not args.no_influx:
            argv += [
                f"--influxSrc={ROOT}/scratch",
                "--influxHost=127.0.0.1",
                "--influxPort=8086",
                "--influxDb=nr_kpi",
            ]
        if args.gui:
            argv += [f"--guiSrc={ROOT}/scratch", "--guiHost=localhost"]
    (directory / "launch.json").write_text(
        json.dumps(
            {
                "run_id": identity,
                "replay_run_id": run,
                "current_parameters": current,
                "argv": [str(v) for v in argv],
                "trace": str(trace) if trace else None,
                "fixture": bool(args.smoke or args.test_ue_subset),
                "prediction_source": "offline_dataset",
                "exact_stochastic_reproduction_claimed": False,
            },
            indent=2,
        )
    )
    print(
        f"ns3 run={identity}\nReplay baseline={run}\nManifest={directory}/launch.json", flush=True
    )
    if args.dry_run:
        print(" ".join(str(v) for v in argv))
        return
    with (directory / "ns3.log").open("w") as log:
        process = subprocess.Popen(
            [str(v) for v in argv], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT
        )
        print(f"pid={process.pid}; log={directory}/ns3.log", flush=True)
        try:
            code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            process.wait(timeout=15)
            raise
    if code:
        for logfile in [directory / "ns3.log", directory / "kpi" / run / "run.log"]:
            if logfile.exists():
                print("\n".join(logfile.read_text(errors="replace").splitlines()[-35:]))
    raise SystemExit(code)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ["up", "ui", "gateway", "non-rt", "ric", "monitoring", "status", "build"]:
        sub.add_parser(name)
    sim = sub.add_parser("simulate")
    sim.add_argument("--from-session")
    sim.add_argument("--run", default="gnn_0130")
    sim.add_argument("--sim-time", type=float, default=900)
    sim.add_argument("--ues", type=int, default=300)
    sim.add_argument("--no-influx", action="store_true")
    sim.add_argument("--gui", action="store_true")
    sim.add_argument("--smoke", action="store_true")
    sim.add_argument("--pace-ms", type=int, default=100)
    sim.add_argument("--dry-run", action="store_true")
    sim.add_argument(
        "--test-ue-subset",
        type=int,
        default=0,
        help="Control test only: use the first N UE traces; not a GNN performance experiment",
    )
    args = parser.parse_args()
    if args.command == "simulate" and (
        args.sim_time <= 0 or args.ues <= 0 or args.test_ue_subset < 0
    ):
        parser.error("Simulation duration and UE capacity must be positive")
    if args.command in services():
        execute(services()[args.command])
        return
    if args.command in ("monitoring", "up"):
        execute(
            ["docker", "compose", "--project-directory", ROOT / "near-rt-ric/monitoring", "up", "-d"]
        )
        if args.command == "up":
            for name in ["ric", "gateway", "non-rt", "ui"]:
                start(name)
    elif args.command == "status":
        for name, url in [
            ("non-rt", "http://127.0.0.1:9000/api/non-rt/health"),
            ("ui", "http://127.0.0.1:18081/api/health"),
            ("gateway", "http://127.0.0.1:18082/health"),
            ("grafana", "http://127.0.0.1:3000/api/health"),
        ]:
            try:
                print(name, json.dumps(get(url)))
            except OSError as exc:
                print(name, str(exc))
        try:
            print(json.dumps(get("http://127.0.0.1:18082/api/runs"), indent=2))
        except OSError:
            pass
    elif args.command == "build":
        execute([ROOT / "near-rt-ric/build-local.sh"])
        execute(
            [
                "cmake",
                "-S",
                ".",
                "-B",
                "cmake-cache",
                "-UBoost_*",
                "-Uboost_*",
                "-DCMAKE_DISABLE_FIND_PACKAGE_Boost=TRUE",
            ]
        )
        execute(
            [
                PYTHON,
                "ns3",
                "build",
                "agora-ric-smoke",
                "agora-ret-cache-check",
                "khu-real-dual-nr-sionna-final",
                "khu-real-nr-sionna-pooled",
                "khu-real-nr-sionna",
                "-j",
                "6",
            ]
        )
    elif args.command == "simulate":
        simulate(args)


if __name__ == "__main__":
    main()
