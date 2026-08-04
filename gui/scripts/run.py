#!/usr/bin/env python3
#
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
import argparse
import logging
import os
import sys


def add_project_root_to_path():
    lib_path = os.path.join(os.path.dirname(__file__), "..", "src")
    if lib_path not in sys.path:
        sys.path.append(lib_path)


if __name__ == "__main__":
    add_project_root_to_path()

from sionna_rt_gui import AppHolder, DEFAULT_CONFIG_PATH
from sionna_rt_gui.config import load_config


def main():
    parser = argparse.ArgumentParser(
        prog=os.path.basename(__file__), description="Interactive Sionna RT GUI"
    )
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to the GUI configuration file to use.",
    )
    parser.add_argument(
        "scene",
        type=str,
        nargs="?",
        default=None,
        help="Path to the Sionna RT scene to load (.xml file or name of a built-in scene).",
    )
    watch_group = parser.add_mutually_exclusive_group()
    watch_group.add_argument(
        "--watch", action="store_true", dest="watch", default=False
    )
    watch_group.add_argument("--no-watch", action="store_false", dest="watch")
    parser.add_argument(
        "--live-paths",
        action="store_true",
        default=False,
        help=(
            "Let the GUI recompute paths on its own scene instance on every "
            "incoming ZMQ position update (very slow -- can drop frame time "
            "to ~1-2s under a live position feed, since it duplicates "
            "whatever the RT/NR channel server already computed). Off by "
            "default, matching run_kyunghee_demo.py's rationale: the GUI is "
            "for visualization, not a second authoritative RT computation."
        ),
    )
    parser.add_argument(
        "--live-radio-map",
        action="store_true",
        default=False,
        help=(
            "Let the GUI auto-refine its radio map heatmap every frame "
            "(RadioMapConfig.auto_update). Off by default for the same "
            "reason as --live-paths: kyunghee.yaml already disables this "
            "for performance, but this script's default config (base.yaml) "
            "never got that tuning, so a radio map computed here (manually, "
            "or via a gnb_antenna/RET update while one is already shown) "
            "re-accumulates at the untuned default of 1e8 samples/frame -- "
            "observed to drop frame time to ~7s under live position/color "
            "updates."
        ),
    )
    parser.add_argument(
        "--radio-map-log-samples-per-it",
        type=float,
        default=3.0,
        help=(
            "log10(samples per radio map refinement iteration). Only "
            "matters once a radio map exists (manual trigger, or "
            "--live-radio-map). Default 3.0 (1e3 samples/it) keeps that "
            "cheap; the underlying default is 8.0 (1e8), matching "
            "kyunghee.yaml's tuned value of 6.0 (1e6) or lower."
        ),
    )
    args = parser.parse_args()

    cfg_overrides = {
        "use_live_reload": args.watch,
        "paths.auto_update": args.live_paths,
        "radio_map.auto_update": args.live_radio_map,
        "radio_map.log_samples_per_it": args.radio_map_log_samples_per_it,
    }
    cfg = load_config(args.config, scene_filename=args.scene)

    # Configure logging
    logging.basicConfig(
        level=cfg.log_level, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # --- Initialization
    app = AppHolder(cfg, scene_filename=args.scene, overrides=cfg_overrides)

    # --- Running loop
    app.show()


if __name__ == "__main__":
    main()
