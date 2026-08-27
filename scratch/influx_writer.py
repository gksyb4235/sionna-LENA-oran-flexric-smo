"""
influx_writer.py
─────────────────────────────────────────────────────────────────────
Minimal InfluxDB 1.x line-protocol writer, stdlib only (no pip
dependency, no extra client library) so it can be imported directly by
ns-3's embedded Python interpreter the same way scratch/zmq_bridge.py
and scratch/gui bridging already are.

Schema (see monitoring/grafana/dashboards/*.json for the panels that
read it back):

  measurement "ue_kpi", tag ue=<external UE id>
    fields: serving_cell (string), rsrp_serving_dbm (float),
            neighbor_cell (string), rsrp_neighbor_dbm (float),
            dl_throughput_mbps (float), ho_count (int),
            seconds_since_ho (float), is_pingpong (int 0/1),
            dl_sinr_db (float), dl_mcs (int)

  measurement "cell_kpi", tag cell=<external gNB id>
    fields: tx_power_dbm, ret_tilt_deg, ret_bearing_deg, ttt_ms,
            hysteresis_db, num_ues (int), avg_rsrp_dbm,
            ho_in_count (int), ho_out_count (int), pingpong_count (int),
            aggregate_throughput_mbps, prb_utilization_pct
"""

import urllib.request
import urllib.error


def _escape_tag(value: str) -> str:
    return str(value).replace(" ", "\\ ").replace(",", "\\,").replace("=", "\\=")


def _format_field(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return f"{value}i"
    if isinstance(value, float):
        return repr(value)
    # string field -- quote and escape embedded quotes
    escaped = str(value).replace('"', '\\"')
    return f'"{escaped}"'


class InfluxWriter:
    """
    host/port: where InfluxDB's HTTP API is listening (see
    monitoring/docker-compose.yml, default 127.0.0.1:8086).
    database: must already exist (created once via `create_database`,
    or ahead of time with `influx -execute 'CREATE DATABASE ...'`).
    """

    def __init__(self, host: str = "localhost", port: int = 8086,
                 database: str = "nr_kpi", timeout_s: float = 1.0):
        self.base_url = f"http://{host}:{port}"
        self.database = database
        self.timeout_s = timeout_s
        self._db_ready = False

    def create_database(self) -> bool:
        try:
            req = urllib.request.Request(
                f"{self.base_url}/query",
                data=f"q=CREATE DATABASE {self.database}".encode("utf-8"),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s):
                pass
            self._db_ready = True
            return True
        except (urllib.error.URLError, OSError) as error:
            print(f"[influx_writer] create_database failed (ignoring): {error}")
            return False

    def write_point(self, measurement: str, tags: dict, fields: dict) -> bool:
        if not self._db_ready:
            self.create_database()

        tag_str = "".join(f",{k}={_escape_tag(v)}" for k, v in tags.items())
        field_str = ",".join(f"{k}={_format_field(v)}" for k, v in fields.items())
        line = f"{measurement}{tag_str} {field_str}"

        try:
            req = urllib.request.Request(
                f"{self.base_url}/write?db={self.database}&precision=ms",
                data=line.encode("utf-8"),
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=self.timeout_s):
                pass
            return True
        except (urllib.error.URLError, OSError) as error:
            print(f"[influx_writer] write_point failed (ignoring): {error}")
            return False

    def write_ue_kpi(self, ue: str, serving_cell: str, rsrp_serving_dbm: float,
                     neighbor_cell: str, rsrp_neighbor_dbm: float,
                     dl_throughput_mbps: float, ho_count: int,
                     seconds_since_ho: float, is_pingpong: bool,
                     dl_sinr_db: float = 0.0, dl_mcs: int = 0) -> bool:
        return self.write_point(
            "ue_kpi",
            {"ue": ue},
            {
                "serving_cell": serving_cell,
                "rsrp_serving_dbm": float(rsrp_serving_dbm),
                "neighbor_cell": neighbor_cell,
                "rsrp_neighbor_dbm": float(rsrp_neighbor_dbm),
                "dl_throughput_mbps": float(dl_throughput_mbps),
                "ho_count": int(ho_count),
                "seconds_since_ho": float(seconds_since_ho),
                "is_pingpong": bool(is_pingpong),
                "dl_sinr_db": float(dl_sinr_db),
                "dl_mcs": int(dl_mcs),
            },
        )

    def write_cell_kpi(self, cell: str, tx_power_dbm: float, ret_tilt_deg: float,
                       ret_bearing_deg: float, ttt_ms: float, hysteresis_db: float,
                       num_ues: int, avg_rsrp_dbm: float, ho_in_count: int,
                       ho_out_count: int, pingpong_count: int,
                       avg_throughput_mbps: float,
                       prb_utilization_pct: float | None = None,
                       energy_state: str | None = None,
                       tx_power_watts: float | None = None,
                       energy_efficiency_mbps_per_w: float | None = None) -> bool:
        fields = {
            "tx_power_dbm": float(tx_power_dbm),
            "ret_tilt_deg": float(ret_tilt_deg),
            "ret_bearing_deg": float(ret_bearing_deg),
            "ttt_ms": float(ttt_ms),
            "hysteresis_db": float(hysteresis_db),
            "num_ues": int(num_ues),
            "ho_in_count": int(ho_in_count),
            "ho_out_count": int(ho_out_count),
            "pingpong_count": int(pingpong_count),
            "avg_throughput_mbps": float(avg_throughput_mbps),
        }
        # No UE attached this period -> nothing was measured, so omit the
        # field entirely (InfluxDB/Grafana render a gap) instead of writing
        # a literal 0.0, which previously plotted as a fake "0 dBm" reading.
        # Callers signal this by passing NaN (avg_rsrp_dbm != avg_rsrp_dbm).
        if avg_rsrp_dbm == avg_rsrp_dbm:
            fields["avg_rsrp_dbm"] = float(avg_rsrp_dbm)
        if prb_utilization_pct is not None:
            fields["prb_utilization_pct"] = float(prb_utilization_pct)
        if energy_state is not None:
            fields["energy_state"] = str(energy_state)
        if tx_power_watts is not None:
            fields["tx_power_watts"] = float(tx_power_watts)
        if energy_efficiency_mbps_per_w is not None:
            fields["energy_efficiency_mbps_per_w"] = float(energy_efficiency_mbps_per_w)
        return self.write_point("cell_kpi", {"cell": cell}, fields)
