#!/usr/bin/env python3
"""Move legacy KHU files without deleting data; retain selected legacy links.

Default is a plan. --apply is for an existing checkout before/after the layout
change. Existing destinations are never overwritten or merged automatically.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KHU = Path("experiment/khu")


def relocate(root: Path, old: Path, new: Path, apply: bool, keep_link: bool = True) -> None:
    source, target = root / old, root / new
    if source.is_symlink():
        if source.resolve() != target.resolve():
            raise ValueError(f"Unexpected link: {old}")
        if not keep_link:
            print(f"remove obsolete link {old}; preserve {new}")
            if apply:
                source.unlink()
        return
    if not source.exists():
        return
    if target.exists() or target.is_symlink():
        raise ValueError(f"Both {old} and {new} exist; preserve and reconcile manually")
    suffix = "leave compatibility link" if keep_link else "without compatibility link"
    print(f"move {old} -> {new}; {suffix}")
    if apply:
        target.parent.mkdir(parents=True, exist_ok=True)
        source = source.parent.resolve() / source.name
        source.rename(target)
        if keep_link:
            source.symlink_to(os.path.relpath(target, source.parent), target_is_directory=target.is_dir())


def migrate(root: Path, apply: bool) -> None:
    # Parent moves preserve contents, timestamps and file bytes, including dirty files.
    relocate(root, Path("scenarios/results"), KHU / "results", apply)
    relocate(root, Path("scenarios/khu-real"), KHU / "datasets", apply)
    relocate(root, Path("experiments"), KHU / "studies", apply, keep_link=False)
    studies = KHU / "studies" if (root / KHU / "studies").exists() else Path("experiments")
    # The old paths still resolve through the links above, in either layout.
    relocate(root, Path("scenarios/khu-real/tools"), KHU / "tools", apply)
    relocate(root, Path("scenarios/khu-real/ret_caches"), KHU / "caches/ret_caches", apply)
    relocate(root, Path("scenarios/khu-real/runs"), KHU / "results/legacy_pooled", apply)
    relocate(root, Path("scenarios/khu-real/analysis"), KHU / "results/legacy_analysis", apply)
    for source in sorted((root / "scenarios/khu-real").glob("sionna_rt_cache*")):
        relocate(root, source.relative_to(root), KHU / "caches" / source.name, apply)
    relocate(root, Path("scenes/khu-real"), KHU / "datasets/scene", apply)
    for name in ("agora_runtime", "agora_conflict_sweep", "two_switch_debug", "two_switch_smoke"):
        relocate(root, studies / name, KHU / "results" / name, apply)
    for study in ("drl_xapp_offline", "drl_xapp_conflict"):
        for folder in ("results", "failed_attempts", "smoke", "training", "analysis", "data_audit", "selected"):
            # analysis contains tracked reporting code; keep it with the study.
            if folder == "analysis":
                continue
            relocate(root, studies / study / folder,
                     KHU / "results" / study / folder, apply)
    for source in sorted((root / studies / "conflict_scenarios_500").glob("backend_smoke_*")):
        relocate(root, source.relative_to(root), KHU / "results/conflict/backend_smoke" / source.name, apply)
    for name, target in (("conflict_500", "../studies/conflict_scenarios_500/sweep/data"),
                         ("conflict_reserve50", "../studies/conflict_scenarios_reserve50")):
        alias = root / KHU / "datasets" / name
        if not alias.exists() and not alias.is_symlink():
            print(f"link {alias.relative_to(root)} -> {target}")
            if apply:
                alias.parent.mkdir(parents=True, exist_ok=True)
                alias.symlink_to(target, target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    migrate(ROOT, args.apply)


if __name__ == "__main__":
    main()
