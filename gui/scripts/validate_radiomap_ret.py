"""
validate_radiomap_ret.py
─────────────────────────────────────────────────────────────────────
GUI RadioMap의 RET(전기적 틸트) precoding_vec 부호 규약과 per-TX 독립성을
Sionna RT로 실증하는 단독 스크립트 (Polyscope/GUI 창 없이 실행 가능).

SionnaRtGui._tilt_precoding_weights / _build_precoding_vec와 '동일한 수식'을
사용해 다음을 확인한다:

  1. RadioMapSolver.precoding_vec의 부호 규약은 ns3sionna_server.py(PathSolver
     기반)와 반대(켤레)다 -- h_n = h_n^H @ p (채널 쪽을 켤레) vs
     h_eff = sum conj(w) h (가중치 쪽을 켤레). 두 TX 배열/틸트가 같은 방향으로
     기울어지는지 삼각측량(peak depression angle)으로 검증.
  2. precoding_vec가 진짜 per-transmitter인지 -- 한 gNB(gnb2)의 틸트를 바꿔도
     다른 gNB(gnb1)의 RadioMap 기여분은 전혀 변하지 않아야 한다 (Sionna의
     tx_array 자체는 scene 전체 공유이지만, precoding_vec 텐서는
     [num_tx, num_tx_ant] shape이므로 TX별로 독립적이어야 함).

주의: 측정 평면을 TX와 정확히 같은 높이(z)에 두면 Monte Carlo grazing-incidence
degenerate case(레이가 평면과 평행이라 히트 확률이 0)가 발생해 path_gain이
항상 0이 된다 -- 실제 배포(지상 gNB, 보행자 높이 측정)에서는 발생하지 않는
테스트 아티팩트이므로, 아래 두 스크립트 모두 TX와 다른 높이의 측정점을 쓴다.

사용법:
  source .venv/bin/activate
  python contrib/sionna/gui/scripts/validate_radiomap_ret.py
"""
import numpy as np
import mitsuba as mi

from sionna.rt import load_scene, Transmitter, PlanarArray, RadioMapSolver

FC = 3.5e9
WAVELENGTH = 299_792_458.0 / FC
NUM_V = 8
R = 100.0
DEPRESSION_DEG = np.arange(-30.0, 30.5, 2.0)


def tilt_precoding_weights(z_m, tilt_deg, num_v):
    """gui.py의 SionnaRtGui._tilt_precoding_weights와 반드시 동일해야 함."""
    phase = 2.0 * np.pi * (z_m / WAVELENGTH) * np.sin(np.deg2rad(tilt_deg))
    return np.exp(1j * phase) / np.sqrt(num_v)


def build_scene(num_tx_positions):
    scene = load_scene()
    scene.frequency = FC
    scene.tx_array = PlanarArray(num_rows=NUM_V, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="tr38901",
                                 polarization="V")
    scene.rx_array = PlanarArray(num_rows=1, num_cols=1, vertical_spacing=0.5,
                                 horizontal_spacing=0.5, pattern="iso", polarization="V")
    for name, pos in num_tx_positions.items():
        scene.add(Transmitter(name=name, position=pos, orientation=[0, 0, 0]))
    return scene


def to_precoding_tensor(weights_by_tx, num_v):
    w = np.stack(weights_by_tx, axis=0).astype(np.complex64)
    return (mi.TensorXf(np.real(w)), mi.TensorXf(np.imag(w)))


def main():
    ok = True

    # ---- Part 1: peak-angle sign calibration (single TX) --------------------
    scene1 = build_scene({"tx": [0.0, 0.0, 0.0]})
    z_m = np.asarray(scene1.tx_array.positions(WAVELENGTH))[2, :].reshape(-1)
    solver = RadioMapSolver()

    def gain_at_depression(tilt_deg, dep_deg):
        w = tilt_precoding_weights(z_m, tilt_deg, NUM_V)
        precoding_vec = to_precoding_tensor([w], NUM_V)
        z = float(-R * np.sin(np.deg2rad(dep_deg)))
        rm = solver(scene1, center=[float(R), 0.0, z], orientation=[0, 0, 0],
                    size=[2.0, 2.0], cell_size=[2.0, 2.0],
                    precoding_vec=precoding_vec, samples_per_tx=300_000,
                    max_depth=0, los=True, specular_reflection=False,
                    diffuse_reflection=False, refraction=False,
                    diffraction=False, edge_diffraction=False, seed=1)
        return float(np.asarray(rm.path_gain).sum())

    print("--- Part 1: peak depression angle vs commanded tilt ---")
    for tilt in (0.0, 9.0, 18.0):
        gains = [gain_at_depression(tilt, d) for d in DEPRESSION_DEG]
        peak = DEPRESSION_DEG[int(np.argmax(gains))]
        print(f"  tilt={tilt:+.0f}deg -> peak at depression {peak:+.0f}deg")
        if tilt > 0 and abs(peak - tilt) > 10.0:
            print(f"  !! FAIL: peak too far from commanded tilt "
                  f"(sampling noise tolerance is 10deg at 2deg grid resolution)")
            ok = False

    # ---- Part 2: per-TX independence (two TX, only one retilted) ------------
    scene2 = build_scene({"gnb1": [0.0, 0.0, 0.0], "gnb2": [0.0, 20.0, 0.0]})
    center = [R, 10.0, -15.0]  # ~8.5deg depression patch, matching Part 1's geometry

    def two_tx_gains(tilt_gnb1, tilt_gnb2):
        w1 = tilt_precoding_weights(z_m, tilt_gnb1, NUM_V)
        w2 = tilt_precoding_weights(z_m, tilt_gnb2, NUM_V)
        precoding_vec = to_precoding_tensor([w1, w2], NUM_V)
        rm = solver(scene2, center=center, orientation=[0, 0, 0], size=[4.0, 4.0],
                    cell_size=[2.0, 2.0], precoding_vec=precoding_vec,
                    samples_per_tx=200_000, max_depth=1, los=True,
                    specular_reflection=False, diffuse_reflection=False,
                    refraction=False, diffraction=False, edge_diffraction=False,
                    seed=1)
        pg = np.asarray(rm.path_gain)
        return float(pg[0].sum()), float(pg[1].sum())

    def db(x):
        return 10 * np.log10(max(x, 1e-30))

    p0_flat, p1_flat = two_tx_gains(0.0, 0.0)
    p0_after, p1_after = two_tx_gains(0.0, 9.0)  # only retilt gnb2

    print("\n--- Part 2: per-transmitter independence ---")
    print(f"  gnb1 (untouched): {db(p0_flat):.2f} -> {db(p0_after):.2f} dB "
          f"(delta {db(p0_after) - db(p0_flat):+.2f}, expect ~0)")
    print(f"  gnb2 (retilted +9deg): {db(p1_flat):.2f} -> {db(p1_after):.2f} dB "
          f"(delta {db(p1_after) - db(p1_flat):+.2f}, expect clearly positive)")

    if abs(db(p0_after) - db(p0_flat)) > 0.5:
        print("  !! FAIL: retilting gnb2 leaked into gnb1's precoding row")
        ok = False
    if db(p1_after) - db(p1_flat) < 3.0:
        print("  !! FAIL: retilting gnb2 towards the patch did not increase its gain")
        ok = False

    print("\nRESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
