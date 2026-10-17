"""Install an isolated sweep target, preserving every existing source and edited target.

Default is validation only; --install writes three files beneath scratch/agora-conflict-sweep.
Compilation is deliberately separate: ./ns3 build khu-agora-conflict-sweep -j 2
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

SOURCE = "scratch/khu-real-dual-nr-sionna-final.cc"
SOURCE_SHA256 = "ee9815a541cb744bab0ebf55628dc2601285f2de2e078d098e86c7b689256034"
DESTINATION = "scratch/agora-conflict-sweep"
MARKER = "conflict_backend_manifest.json"
HERE = Path(__file__).resolve().parent


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def replace_once(text: str, old: str, new: str) -> str:
    if text.count(old) != 1:
        raise ValueError(f"Expected exactly one reviewed source hook: {old!r}")
    return text.replace(old, new, 1)


def prepare(raw: bytes) -> bytes:
    if sha256(raw) != SOURCE_SHA256:
        raise ValueError("Base scenario differs from the reviewed source; do not overwrite it. "
                         f"Expected SHA256 {SOURCE_SHA256}, found {sha256(raw)}")
    text = raw.decode("utf-8")
    text = replace_once(text, '#include "ns3/agora-ric-controller.h"\n', '')
    text = replace_once(text, '#include "scheduled-parameter-control.h"',
                        '#include "conflict-schedule-control.h"')
    text = replace_once(text, 'NS_LOG_COMPONENT_DEFINE("KhuRealDualNrSionnaFinal");',
                        'NS_LOG_COMPONENT_DEFINE("KhuAgoraConflictSweep");')
    text = replace_once(text, 'ScheduledParameterControl::Load(', 'ConflictScheduleControl::Load(')
    text = replace_once(text, 'ScheduledParameterControl::Install(', 'ConflictScheduleControl::Install(')
    text = replace_once(text, '    AgoraRicController::InstallFromEnvironment(agoraCells,simTime.GetSeconds());', '')
    text = replace_once(text, '''    const auto scheduledParameters = ConflictScheduleControl::Install(
        parameterEvents, agoraCells, scheduledLookups, parameterScheduleRetCacheRoot,
        (runDirectory / "parameter_changes.csv").string());''',
                        '''    BooleanValue sweepIdealRrc;
    nrHelper->GetAttribute("UseIdealRrc", sweepIdealRrc);
    NS_ABORT_MSG_IF(!sweepIdealRrc.Get(),
                    "Conflict sweep requires Ideal RRC for exact TTT 340/720 ms");
    // A3 report configurations are registered by device initialization at time zero.
    std::shared_ptr<ConflictScheduleControl> scheduledParameters;
    Simulator::ScheduleNow([&]() {
        scheduledParameters = ConflictScheduleControl::Install(
            parameterEvents, agoraCells, scheduledLookups, parameterScheduleRetCacheRoot,
            (runDirectory / "parameter_changes.csv").string());
    });''')
    return text.replace("\r\n", "\n").encode("utf-8")


def inspect_dependencies(repo: Path) -> dict:
    required = {
        "src/spectrum/model/sionna-lookup-channel-model.h": ["PrepareRetCache", "ActivateRetCache"],
        "src/spectrum/model/sionna-lookup-channel-model.cc": ["PrepareRetCache", "ActivateRetCache"],
        "contrib/nr/model/nr-ue-rrc.h": ["SetCellIndividualOffset", "GetCellIndividualOffset"],
        "contrib/nr/model/nr-gnb-rrc.h": ["GetA3HandoverConfig", "UpdateA3HandoverConfig"],
    }
    hashes = {}
    for name, hooks in required.items():
        data = (repo / name).read_bytes()
        text = data.decode("utf-8")
        if any(hook not in text for hook in hooks):
            raise ValueError(f"Required reviewed backend API absent: {name}")
        hashes[name] = sha256(data)
    return hashes


def install(repo: Path, *, write: bool = False, upgrade_owned: bool = False) -> dict:
    repo = repo.resolve(strict=True)
    source = repo / SOURCE
    destination = repo / DESTINATION
    if source.is_symlink() or destination.is_symlink():
        raise ValueError("Refusing source/destination symlink")
    if destination.resolve().parent != (repo / "scratch").resolve():
        raise ValueError("Destination escapes scratch")
    raw = source.read_bytes()
    scenario = prepare(raw)
    header = (HERE / "conflict-schedule-control.h").read_bytes()
    dependencies = inspect_dependencies(repo)
    files = {"khu-agora-conflict-sweep.cc": scenario, "conflict-schedule-control.h": header}
    manifest = {
        "schema": "agora-conflict-sweep-backend-v1", "base_source": SOURCE,
        "base_source_sha256": sha256(raw), "original_files_modified": False,
        "files_sha256": {name: sha256(data) for name, data in files.items()},
        "dependency_sha256": dependencies, "requires_ideal_rrc": True,
        "ns3_wrapper_target": "khu-agora-conflict-sweep",
        "binary_relative": "build/scratch/agora-conflict-sweep/ns3.48-khu-agora-conflict-sweep-default",
        "schedule_header": "time_s,parameter,value",
        "receipt_header": "scheduled_time_s,applied_time_s,parameter,before,after,status",
        "receipt_filename": "parameter_changes.csv",
        "application": "one_simulator_callback_per_timestamp_all_updates_then_all_11_readback",
        "verification_scope": "gNB_TxP_RET_all_pooled_UE_CIO_common_gNB_A3_RRC_reconfiguration_scheduled",
        "transport": "predetermined_simulator_events_no_RIC_no_LLM_no_DRL",
    }
    marker_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    files[MARKER] = marker_bytes
    if destination.exists():
        actual = {p.name for p in destination.iterdir()}
        if actual != set(files):
            raise ValueError("Existing target has missing or unowned files; preserve and review it")
        identical = True
        for name, expected in files.items():
            path = destination / name
            if path.is_symlink():
                raise ValueError(f"Refusing to overwrite edited target: {path}")
            identical = identical and path.read_bytes() == expected
        if identical:
            return {**manifest, "status": "already_installed_identical", "files_written": 0}
        if not upgrade_owned:
            raise ValueError("Refusing to overwrite edited target or older target without --upgrade-owned")
        old = json.loads((destination / MARKER).read_text())
        if old.get("schema") != manifest["schema"] or set(old.get("files_sha256", {})) != set(files) - {MARKER}:
            raise ValueError("Existing target ownership manifest is invalid")
        for name, expected_hash in old["files_sha256"].items():
            if sha256((destination / name).read_bytes()) != expected_hash:
                raise ValueError(f"Refusing to overwrite edited target: {destination / name}")
        if write:
            for name, data in files.items():
                temporary = destination / (name + ".upgrade-tmp")
                with temporary.open("xb") as handle:
                    handle.write(data)
                temporary.replace(destination / name)
        return {**manifest, "status": "upgraded_owned" if write else "validated_owned_upgrade_no_writes",
                "files_written": len(files) if write else 0}
    if write:
        destination.mkdir(parents=False, exist_ok=False)
        for name, data in files.items():
            with (destination / name).open("xb") as handle:
                handle.write(data)
    return {**manifest, "status": "installed" if write else "validated_no_writes",
            "files_written": len(files) if write else 0}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--install", action="store_true", help="Write isolated target after validation")
    parser.add_argument("--upgrade-owned", action="store_true", help="Permit upgrading an unchanged installer-owned target")
    args = parser.parse_args()
    print(json.dumps(install(args.repo, write=args.install, upgrade_owned=args.upgrade_owned), indent=2))
