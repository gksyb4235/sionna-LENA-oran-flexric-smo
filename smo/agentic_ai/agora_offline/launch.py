"""Start the local workbench in the background without launching ns3 or an LLM."""

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.request import urlopen


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=18081)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    state = root / ".state"
    state.mkdir(exist_ok=True)
    endpoint = f"http://127.0.0.1:{args.port}"

    def health():
        try:
            with urlopen(endpoint + "/api/health", timeout=1) as response:
                return json.load(response)
        except Exception:
            return None

    existing = health()
    if existing:
        if existing.get("app_id") != "agora-offline-final-gnn":
            raise SystemExit("Port is occupied by another application")
        print("Already ready: " + endpoint)
        return
    python = root.parents[2] / ".venv/bin/python"
    if not python.exists():
        python = Path(sys.executable)
    with (state / "server.log").open("ab") as log:
        process = subprocess.Popen(
            [str(python), "-m", "agora_offline.server", "--port", str(args.port)],
            cwd=root.parent,
            stdout=log,
            stderr=log,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=dict(os.environ, PYTHONUNBUFFERED="1"),
        )
    for _ in range(100):
        current = health()
        if current and current.get("app_id") == "agora-offline-final-gnn":
            (state / "server.json").write_text(
                json.dumps({"url": endpoint, "pid": current["pid"]}), encoding="utf-8"
            )
            print("Ready: " + endpoint)
            return
        if process.poll() is not None:
            raise SystemExit("Server exited; see " + str(state / "server.log"))
        time.sleep(0.2)
    raise SystemExit("Startup timeout; see " + str(state / "server.log"))


if __name__ == "__main__":
    main()
