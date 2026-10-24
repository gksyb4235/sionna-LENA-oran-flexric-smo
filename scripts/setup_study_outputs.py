#!/usr/bin/env python3
"""Prepare exact canonical output directories for historical DRL study code.

Creates empty directories and compatibility links only. Existing data and
checkpoints are not moved or overwritten; use reorganize_khu.py for migration.
"""

import argparse
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STUDIES = ('drl_xapp_offline', 'drl_xapp_conflict')
FOLDERS = ('results', 'failed_attempts', 'smoke', 'training', 'data_audit', 'selected')


def prepare(root, name):
    if name not in STUDIES:
        raise ValueError(f'Unknown study: {name}')
    root = Path(root).resolve()
    study = root / 'experiment/khu/studies' / name
    output = root / 'experiment/khu/results' / name
    if not study.is_dir() or study.resolve() != study:
        raise ValueError(f'Expected a canonical study directory: {study}')
    if output.resolve() != output:
        raise ValueError(f'Unexpected output alias: {output}')
    # Validate the complete plan before creating any paths.
    for name in FOLDERS:
        alias, target = study / name, output / name
        if target.resolve() != target:
            raise ValueError(f'Unexpected output symlink: {target}')
        if alias.is_symlink():
            if alias.resolve() != target:
                raise ValueError(f'Unexpected study alias: {alias}')
        elif alias.exists():
            raise ValueError(f'Existing study data must be migrated without overwriting: {alias}')
    for name in FOLDERS:
        alias, target = study / name, output / name
        target.mkdir(parents=True, exist_ok=True)
        if not alias.is_symlink():
            alias.symlink_to(os.path.relpath(target, study), target_is_directory=True)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=(*STUDIES, 'all'), default='all')
    args = parser.parse_args()
    for name in STUDIES if args.study == 'all' else (args.study,):
        print(prepare(ROOT, name))


if __name__ == '__main__':
    main()
