"""Contract fixture only: writes deterministic parameter records, never simulates radio."""
import csv
import json
from pathlib import Path
import sys

job = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=False)
state = dict(job["initial_parameters"])
with (output / "parameter_changes.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(("scheduled_time_s", "applied_time_s", "parameter", "before", "after", "status"))
    for event in job["events"]:
        # C++ std::map order deliberately differs from the input field order.
        for p in sorted(event["updates"]):
            value = event["updates"][p]
            writer.writerow((event["time_s"], event["time_s"], p, state[p], value, "applied"))
            state[p] = value
with (output / "cell_kpi.csv").open("w", newline="", encoding="utf-8") as handle:
    writer = csv.writer(handle)
    writer.writerow(("time_s", "cell_id", "tx_power_dbm", "ret_tilt_deg", "cio_bias_db", "ttt_ms", "hysteresis_db"))
    state = dict(job["initial_parameters"])
    for second in range(900):
        for event in job["events"]:
            if event["time_s"] == second:
                state.update(event["updates"])
        for ident, cell in enumerate(("5g", "4g1", "4g2"), 1):
            writer.writerow((second + .5, ident, state[f"txp_{cell}"], state[f"ret_{cell}"],
                             state[f"cio_{cell}"], state["ttt_ms"], state["hys_db"]))
(output / "ue_kpi.csv").write_text("time_s,imsi\n899.5,1\n", encoding="utf-8")
