"""One explicitly opted-in external LLM smoke test; not part of unittest discovery."""

import argparse
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-external-api", action="store_true")
    args = parser.parse_args()
    if not args.allow_external_api:
        parser.error(
            "This sends the selected simulation context to the configured external API. Explicit opt-in is required."
        )
    base = "http://127.0.0.1:18081"
    body = dict(
        prompt="For the next3minutes, reduce Cell5G TxP from43dBm to33dBm for energy saving. Check exactly this requested action first. If conflicting, ask the GNN in batches about application delay, weaker intensity including36/40/41dBm, and target cell4g2 or4g1. If still unsatisfactory, combine axes in a second verification. Use the default editable research KPI policy and retrieved DAG context. Respond in Korean.",
        run_id="gnn_0130",
        issue_time_s=180,
        mode="llm",
        horizon_s=180,
    )
    request = Request(
        base + "/api/sessions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    ident = json.load(urlopen(request))["session_id"]
    print("SESSION", ident, flush=True)
    last = 0
    for _ in range(360):
        state = json.load(urlopen(base + "/api/sessions/" + ident))
        for event in state["events"][last:]:
            if event["type"] in ("agent_start", "tool_call", "error", "decision", "done"):
                print(
                    json.dumps(
                        {k: event[k] for k in ("seq", "type", "agent", "content")},
                        ensure_ascii=True,
                    ),
                    flush=True,
                )
        last = len(state["events"])
        if state["status"] in ("completed", "failed", "cancelled"):
            report = {
                "session_id": ident,
                "status": state["status"],
                "agent_roles": sorted(
                    {e["agent"] for e in state["events"] if e["type"] == "agent_start"}
                ),
                "tool_calls": [e["content"] for e in state["events"] if e["type"] == "tool_call"],
                "errors": [e["content"] for e in state["events"] if e["type"] == "error"],
                "usage": [
                    e["data"]["usage"]
                    for e in state["events"]
                    if isinstance(e.get("data"), dict) and "usage" in e["data"]
                ],
                "result": state["result"],
            }
            path = Path(__file__).parent / "live_llm_report.json"
            path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
            print(
                "FINAL",
                json.dumps({k: v for k, v in report.items() if k not in ("result", "usage")}),
                flush=True,
            )
            if state["status"] != "completed":
                raise SystemExit(1)
            return
        time.sleep(1)
    raise TimeoutError("Session did not finish within6minutes; inspect the running UI session")


if __name__ == "__main__":
    main()
