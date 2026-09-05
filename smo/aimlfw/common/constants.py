"""Canonical AIMLFW domain constants (the single validation source)."""

from __future__ import annotations

from typing import Final, Literal

ControlParameterName = Literal[
    "tx_power_dbm", "ret_tilt_deg", "cio_bias_db", "hysteresis_db", "ttt_ms"
]
TimeStep = Literal[0, 1, 2, 3, 4]
DataQuality = Literal["complete", "partial", "insufficient"]
DegradationVerdict = Literal["acceptable", "degrading", "unknown"]
XAppObjective = Literal[
    "energy_saving", "throughput_maximization", "mobility_robustness", "unspecified"
]

CONTROL_PARAMETER_NAMES: Final[tuple[str, ...]] = (
    "tx_power_dbm", "ret_tilt_deg", "cio_bias_db", "hysteresis_db", "ttt_ms"
)
TTT_ALLOWED_MS: Final[tuple[int, ...]] = (
    0, 40, 64, 80, 100, 128, 160, 256, 320, 480, 512, 640, 1024, 1280, 2560, 5120
)
CONTROL_PARAMETER_RANGES: Final[dict[str, dict[str, object]]] = {
    "tx_power_dbm": {"min": 30, "max": 46, "step": 1},
    "ret_tilt_deg": {"min": 0, "max": 15, "step": 1},
    "cio_bias_db": {"min": -6.0, "max": 6.0, "step": 0.5},
    "hysteresis_db": {"min": 0.0, "max": 10.0, "step": 0.5},
    "ttt_ms": {"allowed": TTT_ALLOWED_MS},
}
TARGET_KPIS: Final[tuple[str, ...]] = (
    "cell_goodput_mbps", "avg_ue_goodput_mbps", "ue_goodput_p5_mbps", "sinr_p50_db",
    "prb_utilization_pct", "delay_p95_ms", "ho_failure_count", "pingpong_count",
    "rlf_count", "interval_energy_j",
)
MIN_TIME_STEP: Final = 0
MAX_TIME_STEP: Final = 4
MAX_TARGET_CELLS: Final = 16
MAX_SEED: Final = 4_294_967_295
DEFAULT_CONFLICT_THRESHOLD: Final = 5.0
DEFAULT_RETRAINING_THRESHOLD: Final = 15.0
