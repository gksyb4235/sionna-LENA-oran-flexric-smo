#!/usr/bin/env python3
"""Launch the repository-local Kyunghee GUI with the ns-3 demo topology."""

import argparse
import logging
from pathlib import Path
import sys

import numpy as np
import polyscope as ps


GUI_ROOT = Path(__file__).resolve().parents[1]
SIONNA_ROOT = GUI_ROOT.parent
REPO_ROOT = SIONNA_ROOT.parent.parent.parent
SCENE_XML = REPO_ROOT / "scenes" / "kyunghee-demo" / "Kyunghee.xml"
CONFIG = (
    GUI_ROOT / "src" / "sionna_rt_gui" / "data" / "configs" / "sionna_rt_gui"
    / "kyunghee.yaml"
)

GNB_POSITIONS = {
    "gnb1": [-150.0, -267.0, 50.0],
    "gnb2": [-159.0, 105.0, 30.0],
}

TRAJECTORIES = {
    "ue1": [[-155.0, -100.0, 1.5], [-155.0, -60.0, 1.5], [-155.0, -20.0, 1.5]],
    "ue2": [[23.0, -174.0, 1.5], [-20.0, -174.0, 1.5], [-60.0, -174.0, 1.5]],
    "ue3": [[-47.0, -3.0, 1.5], [-80.0, -80.0, 1.5], [-110.0, -160.0, 1.5]],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Kyunghee Sionna RT/ns-3 3D GUI demo")
    parser.add_argument(
        "--compute-radio-map",
        action="store_true",
        help="Compute and display an initial RadioMap before opening the window",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(GUI_ROOT / "src"))
    from sionna_rt_gui import AppHolder
    from sionna_rt_gui.config import load_config

    if not SCENE_XML.is_file():
        raise FileNotFoundError(f"Repository-local Kyunghee scene not found: {SCENE_XML}")

    cfg = load_config(str(CONFIG), scene_filename=str(SCENE_XML))
    logging.basicConfig(
        level=cfg.log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    # ns-3's Sionna server computes propagation. Avoid doing a second path solve
    # on every GUI position message; RadioMap remains available on demand.
    holder = AppHolder(
        cfg,
        scene_filename=str(SCENE_XML),
        overrides={"use_live_reload": False, "paths.auto_update": False},
    )
    gui = holder.app

    for name, position in GNB_POSITIONS.items():
        gui.add_radio_device(
            position, is_transmitter=True, allow_auto_update=False, name=name
        )

    colors = {
        "ue1": (0.20, 0.75, 1.00),
        "ue2": (1.00, 0.55, 0.15),
        "ue3": (0.65, 1.00, 0.25),
    }
    for name, waypoints in TRAJECTORIES.items():
        gui.add_radio_device(
            waypoints[0], is_transmitter=False, allow_auto_update=False, name=name
        )
        curve = ps.register_curve_network(
            f"{name} planned trajectory",
            np.asarray(waypoints, dtype=np.float32),
            edges="line",
            enabled=True,
            color=colors[name],
            transparency=0.85,
        )
        curve.set_radius(0.7, relative=False)

    gui.fit_camera_to_scene()
    if args.compute_radio_map:
        print("[kyunghee-demo] Computing initial RadioMap ...")
        radio_map = gui.compute_radio_map()
        if radio_map is not None:
            gui.set_radio_map(radio_map, show=True)

    print(f"[kyunghee-demo] scene={SCENE_XML}")
    print("[kyunghee-demo] bridge=tcp://*:5600 (position), tcp://*:5601 (KPI)")
    print("[kyunghee-demo] gNB=2, UE=3; UE positions are driven by ns-3")
    holder.show()


if __name__ == "__main__":
    main()
