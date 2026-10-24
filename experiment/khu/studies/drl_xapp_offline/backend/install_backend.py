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
    destination = repo / "scratch/drl-xapp-offline"
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
                        'NS_LOG_COMPONENT_DEFINE("KhuDrlXappOffline");')
    text = replace_once(text, '    AgoraRicController::InstallFromEnvironment(agoraCells,simTime.GetSeconds());',
                        '''    BooleanValue drlIdealRrc;
    nrHelper->GetAttribute("UseIdealRrc", drlIdealRrc);
    NS_ABORT_MSG_IF(!drlIdealRrc.Get(),
                    "Offline DRL data uses Ideal RRC: nonstandard TTT 340/720 ms must not be serialized");
    NS_ABORT_MSG_IF(std::getenv("DRL_CONTROL_DIR") && !parameterEvents.empty(),
                    "DRL control cannot be mixed with a predetermined parameter schedule");
    const auto drlController = DrlLockstepControl::Install(
        agoraCells, simTime.GetSeconds(), cellKpiCsvPath,
        []() { g_cellKpiCsv.flush(); g_ueKpiCsv.flush(); });''')
    bridge = Path(__file__).with_name("drl-lockstep-control.h").read_bytes()
    scenario = text.replace("\r\n", "\n").encode("utf-8")
    manifest = {
        "schema": "drl-offline-isolated-backend-v1",
        "source": str(source), "source_sha256": sha256(raw),
        "scenario_sha256": sha256(scenario), "bridge_sha256": sha256(bridge),
        "target": "scratch_drl-xapp-offline_khu-drl-xapp-offline",
        "ns3_wrapper_target": "khu-drl-xapp-offline",
        "binary_relative": "build/scratch/drl-xapp-offline/ns3.48-khu-drl-xapp-offline-default",
        "requires_ideal_rrc": True,
        "parameter_contract": {
            "cio_5g": [5, 6, 7, 8, 9], "cio_4g1": [-2, -1, 0, 1, 2],
            "cio_4g2": [-2, -1, 0, 1, 2], "ttt_ms": [128, 256, 340, 512, 720],
            "hys_db": [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5],
        },
        "transport": "local_file_lockstep_not_E2", "original_files_modified": False,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "khu-drl-xapp-offline.cc").write_bytes(scenario)
    (destination / "drl-lockstep-control.h").write_bytes(bridge)
    marker.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path("/home/user/LENA-oran-flexric-smo"))
    args = parser.parse_args()
    print(json.dumps(install(args.repo), indent=2))
