"""
oai_panel.py
─────────────────────────────────────────────────────────────────────
OAI KPI 오버레이 ImGui 패널.

GUI 왼쪽 패널에 추가되어 다음을 표시:
  - UE별 MCS / BLER / PRB
  - 예상 SNR
  - ZMQ 연결 상태
  - UE 위치 수동 입력
"""

import time
import numpy as np
import polyscope.imgui as psim
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .gui import SionnaRtGui

# NVIDIA green
_GREEN  = (0.463, 0.725, 0.0, 1.0)
_YELLOW = (1.0, 0.85, 0.0, 1.0)
_RED    = (0.9, 0.2, 0.1, 1.0)
_GRAY   = (0.6, 0.6, 0.6, 1.0)
_WHITE  = (1.0, 1.0, 1.0, 1.0)

# MCS → 색상 (낮을수록 빨강, 높을수록 초록)
def _mcs_color(mcs: int):
    t = max(0.0, min(1.0, mcs / 28.0))
    return (1.0 - t, t * 0.7, 0.0, 1.0)

def _bler_color(bler: float):
    if bler < 0.05:
        return _GREEN
    elif bler < 0.15:
        return _YELLOW
    else:
        return _RED

def _sinr_color(sinr_db: float):
    if sinr_db >= 15.0:
        return _GREEN
    elif sinr_db >= 0.0:
        return _YELLOW
    else:
        return _RED


def oai_kpi_panel(gui: "SionnaRtGui", bridge=None):
    """
    OAI KPI 패널 — gui.gui() 내에서 호출된다.
    bridge: ZMQBridge 인스턴스 (None이면 연결 상태 표시 생략)
    """
    # ── 섹션 헤더 ─────────────────────────────────────────────────
    psim.PushStyleColor(psim.ImGuiCol_Header,        (0.15, 0.15, 0.2, 1.0))
    psim.PushStyleColor(psim.ImGuiCol_HeaderHovered, (0.2, 0.2, 0.28, 1.0))
    expanded = psim.TreeNode("OAI 5G RAN KPI##oai_kpi")
    psim.PopStyleColor(2)

    if not expanded:
        return

    # ── ZMQ 연결 상태 ───────────────────────────────────────────────
    if bridge is not None:
        status_str = "Connected" if bridge._running else "Disconnected"
        color = _GREEN if bridge._running else _RED
        psim.TextColored(color, f"ZMQ Bridge: {status_str}")

        if bridge._running:
            age = time.time() - bridge.kpi_timestamp
            if age < 5.0:
                psim.SameLine()
                psim.TextColored(_GRAY, f"  KPI updated {age:.1f}s ago")
    psim.Separator()

    # ── KPI 테이블 ──────────────────────────────────────────────────
    kpi_list = bridge.latest_kpi if bridge else []

    if not kpi_list:
        psim.TextColored(_GRAY, "No UE data (start iperf3 traffic)")
    else:
        # 헤더 행
        psim.TextColored(_GRAY, f"{'RNTI':>6}  {'DL MCS':>6}  {'UL MCS':>6}  "
                                f"{'DL BLER':>8}  {'DL PRB':>7}")
        psim.Separator()

        for ue in kpi_list:
            rnti    = ue.get("rnti", 0)
            dl_mcs  = ue.get("dl_mcs", 0)
            ul_mcs  = ue.get("ul_mcs", 0)
            dl_bler = ue.get("dl_bler", 0.0)
            ul_bler = ue.get("ul_bler", 0.0)
            dl_prb  = ue.get("dl_prb", 0)

            # RNTI
            psim.TextColored(_WHITE, f"{rnti:>6x}  ")
            psim.SameLine()

            # DL MCS (색상)
            psim.TextColored(_mcs_color(dl_mcs), f"{dl_mcs:>6d}  ")
            psim.SameLine()

            # UL MCS
            psim.TextColored(_mcs_color(ul_mcs), f"{ul_mcs:>6d}  ")
            psim.SameLine()

            # DL BLER
            psim.TextColored(_bler_color(dl_bler), f"{dl_bler:>8.3f}  ")
            psim.SameLine()

            # DL PRB
            psim.Text(f"{dl_prb:>7d}")

            # RSRP / 서빙셀 (Sionna 채널 소스 — kyunghee_server.py가 채워줌, 없으면 생략)
            serving_cell = ue.get("serving_cell")
            serving_cell_id = ue.get("serving_cell_id", 0)
            active = ue.get("active", True)
            best_rsrp_cell = ue.get("best_rsrp_cell")
            rsrp_by_gnb  = ue.get("rsrp_dbm")
            if not active:
                psim.TextColored(_YELLOW, "SUMO inactive")
            if serving_cell or rsrp_by_gnb:
                psim.Text("        ")
                psim.SameLine()
                if serving_cell:
                    cell_suffix = f"/cell{serving_cell_id}" if serving_cell_id else ""
                    psim.TextColored(_GREEN, f"RRC serving={serving_cell}{cell_suffix}")
                    psim.SameLine()
                if best_rsrp_cell and best_rsrp_cell != serving_cell:
                    psim.TextColored(_YELLOW, f"RT best={best_rsrp_cell}")
                    psim.SameLine()
                if rsrp_by_gnb:
                    rsrp_str = "  ".join(
                        f"{gnb}={val:.1f}dBm" for gnb, val in sorted(rsrp_by_gnb.items())
                    )
                    psim.TextColored(_GRAY, f"RSRP: {rsrp_str}")

            # Real DL SINR (ns-3 MmWaveUePhy measurement of the serving cell
            # only, signal/(interference+noise) -- distinct from rsrp_dbm
            # above, which the server derives from Sionna for every
            # neighboring gNB, serving or not).
            if ue.get("has_dl_sinr"):
                dl_sinr_db = ue.get("dl_sinr_db", 0.0)
                psim.Text("        ")
                psim.SameLine()
                psim.TextColored(_GRAY, "DL SINR (real): ")
                psim.SameLine()
                psim.TextColored(_sinr_color(dl_sinr_db), f"{dl_sinr_db:.1f} dB")

    psim.Separator()

    # ── UE 위치 정보 (씬에서 현재 위치) ────────────────────────────
    if gui.scene and gui.scene.receivers:
        psim.TextColored(_GRAY, "UE Positions:")
        for name, rx in gui.scene.receivers.items():
            pos = rx.position.numpy().squeeze()
            psim.Text(f"  {name:8s}  [{pos[0]:7.1f}, {pos[1]:7.1f}, {pos[2]:5.1f}]")

    psim.TreePop()


def oai_cir_panel(gui: "SionnaRtGui"):
    """
    CIR 시각화 패널 — 현재 경로에서 계산된 CIR을 막대 그래프로 표시.
    """
    psim.PushStyleColor(psim.ImGuiCol_Header,        (0.15, 0.15, 0.2, 1.0))
    psim.PushStyleColor(psim.ImGuiCol_HeaderHovered, (0.2, 0.2, 0.28, 1.0))
    expanded = psim.TreeNode("CIR (Channel Impulse Response)##cir_panel")
    psim.PopStyleColor(2)

    if not expanded:
        return

    if gui.paths_cir is None:
        psim.TextColored(_GRAY, "No CIR data. Enable 'Compute CIR' in Paths settings.")
        psim.TreePop()
        return

    a_np, tau_np = gui.paths_cir  # numpy arrays

    try:
        a_flat   = np.array(a_np).flatten()
        tau_flat = np.array(tau_np).flatten()

        # 유효 경로
        valid = np.isfinite(a_flat) & (np.abs(a_flat) > 1e-20)
        a_v   = a_flat[valid]
        tau_v = tau_flat[valid]

        if len(a_v) == 0:
            psim.TextColored(_GRAY, "No valid paths.")
            psim.TreePop()
            return

        n_paths = len(a_v)
        psim.Text(f"Paths: {n_paths}")

        max_tau_us = tau_v.max() * 1e6
        psim.Text(f"Max delay: {max_tau_us:.3f} us")

        # RMS delay spread
        abs_sq    = np.abs(a_v) ** 2
        total_pwr = abs_sq.sum()
        if total_pwr > 0:
            mean_tau = np.sum(abs_sq * tau_v) / total_pwr
            rms_ds   = np.sqrt(np.sum(abs_sq * (tau_v - mean_tau) ** 2) / total_pwr)
            psim.Text(f"RMS DS:   {rms_ds*1e6:.3f} µs")

        # Max path gain
        max_gain_db = 20 * np.log10(np.abs(a_v).max() + 1e-10)
        psim.Text(f"Max path: {max_gain_db:.1f} dB")

        # ImGui PlotHistogram으로 간단한 지연-전력 프로파일
        gains_db = 20 * np.log10(np.abs(a_v) + 1e-10)
        # 정규화 (0~1)
        g_min, g_max = gains_db.min(), gains_db.max()
        if g_max > g_min:
            gains_norm = (gains_db - g_min) / (g_max - g_min)
        else:
            gains_norm = np.ones_like(gains_db)

        # 탭 인덱스 기반 히스토그램 (최대 64 bins)
        # This vendored GUI must not import the old, external OAI project.
        # The GUI path configuration already defines the equivalent sampling rate.
        delay_samples = (tau_v * gui.cfg.paths.bandwidth).astype(int)
        max_delay = min(delay_samples.max() + 1, 64) if len(delay_samples) > 0 else 64
        hist = np.zeros(max_delay, dtype=np.float32)
        for i, ds in enumerate(delay_samples):
            if ds < max_delay:
                hist[ds] = max(hist[ds], float(gains_norm[i]))

        psim.PlotHistogram(
            "##cir_hist",
            hist,
            graph_size=(0, 60),
            scale_min=0.0, scale_max=1.0,
        )
        psim.Text(f"0          {max_delay//2} samples          {max_delay}")

    except Exception as e:
        psim.TextColored(_RED, f"CIR display error: {e}")

    psim.TreePop()


def oai_control_panel(gui: "SionnaRtGui", bridge=None):
    """
    OAI 제어 패널 — UE 위치 수동 입력, 트래젝토리 빠른 설정.
    """
    psim.PushStyleColor(psim.ImGuiCol_Header,        (0.15, 0.15, 0.2, 1.0))
    psim.PushStyleColor(psim.ImGuiCol_HeaderHovered, (0.2, 0.2, 0.28, 1.0))
    expanded = psim.TreeNode("OAI Control##oai_ctrl")
    psim.PopStyleColor(2)

    if not expanded:
        return

    # ZMQ Bridge start/stop
    if bridge is not None:
        if bridge._running:
            psim.TextColored(_GREEN, "ZMQ Bridge running")
            psim.Text(f"  UE port: {bridge.ue_port}")
            psim.Text(f"  KPI port: {bridge.kpi_port}")
        else:
            psim.TextColored(_RED, "ZMQ Bridge stopped")
            if psim.Button("Start ZMQ Bridge"):
                bridge.start()

    psim.Separator()

    # Radio Map manual compute
    if psim.Button("Compute Radio Map##oai"):
        rm = gui.compute_radio_map()
        if rm:
            gui.set_radio_map(rm, show=True)
            print("[OAI Panel] Radio Map computed")

    psim.SameLine()

    # Paths manual update
    if psim.Button("Update Paths##oai"):
        gui.update_paths(show=True)

    psim.TreePop()
