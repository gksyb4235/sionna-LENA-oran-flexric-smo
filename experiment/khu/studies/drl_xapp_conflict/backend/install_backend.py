"""Create one isolated scratch target; never change the original scenario or NR library."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one injection anchor: {old!r}")
    return text.replace(old, new, 1)


def install(repo: Path) -> dict:
    repo = repo.resolve(strict=True)
    source = repo / "scratch/khu-real-dual-nr-sionna-final.cc"
    destination = repo / "scratch/drl-xapp-conflict"
    if destination.is_symlink():
        raise ValueError("Refusing symlink installation destination")
    if destination.resolve().parent != (repo / "scratch").resolve():
        raise ValueError("Destination escapes the scratch directory")
    marker = destination / "drl_backend_manifest.json"
    if destination.exists() and any(destination.iterdir()) and not marker.is_file():
        raise ValueError("Destination exists without this installer's ownership marker")
    raw = source.read_bytes()
    text = raw.decode("utf-8")
    text = replace_once(text, '#include "scheduled-parameter-control.h"',
                        '#include "../scheduled-parameter-control.h"\n#include "drl-lockstep-control.h"')
    text = replace_once(text, 'NS_LOG_COMPONENT_DEFINE("KhuRealDualNrSionnaFinal");',
                        'NS_LOG_COMPONENT_DEFINE("KhuDrlXappConflict");')
    text = replace_once(text, '    AgoraRicController::InstallFromEnvironment(agoraCells,simTime.GetSeconds());',
                        '''    NS_ABORT_MSG_IF(std::getenv("DRL_CONTROL_DIR") && !parameterEvents.empty(),
                    "DRL control cannot be mixed with a predetermined parameter schedule");
    const auto drlController = DrlLockstepControl::Install(
        agoraCells, simTime.GetSeconds(), cellKpiCsvPath,
        []() { g_cellKpiCsv.flush(); g_ueKpiCsv.flush(); });''')
    bridge = Path(__file__).with_name("drl-lockstep-control.h").read_bytes()
    scenario = text.replace("\r\n", "\n").encode("utf-8")
    manifest = {
        "schema": "drl-isolated-backend-v1",
        "source": str(source), "source_sha256": sha256(raw),
        "scenario_sha256": sha256(scenario), "bridge_sha256": sha256(bridge),
        "target": "scratch_drl-xapp-conflict_khu-drl-xapp-conflict",
        "ns3_wrapper_target": "khu-drl-xapp-conflict",
        "binary_relative": "build/scratch/drl-xapp-conflict/ns3.48-khu-drl-xapp-conflict-default",
        "transport": "local_file_lockstep_not_E2", "original_files_modified": False,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "khu-drl-xapp-conflict.cc").write_bytes(scenario)
    (destination / "drl-lockstep-control.h").write_bytes(bridge)
    marker.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/user/LENA-oran-flexric-smo"))
    args = parser.parse_args()
    print(json.dumps(install(args.repo), indent=2))
