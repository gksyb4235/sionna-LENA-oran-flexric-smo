"""Copy only new offline-xApp code/checkpoints from Windows into its WSL folder.

No deletion, simulator launch, old-experiment edit, or binary/cache copy occurs.
Run this script with the Linux Python interpreter in WSL, then initialize the
evaluation plan there so every recorded runtime path is a POSIX path.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

SOURCE = Path('/mnt/c/Users/gksyb/Desktop/GNN 학습 데이터셋/experiments/drl_xapp_offline')
TARGET = Path('/home/user/LENA-oran-flexric-smo/experiments/drl_xapp_offline')
FOLDERS = ('python', 'evaluation', 'selected')
IGNORED_PARTS = {'__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache'}


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def assert_quiescent(target):
    """Never mix versions into an initiated frozen evaluation, even if complete."""
    campaign = target / 'campaign_status.json'
    if (target / '.archive-in-progress').exists():
        raise RuntimeError('A failed-attempt archive is unfinished; preserve and inspect it before deployment')
    if campaign.exists() or (target / '.evaluation-launch.lock').exists():
        raise RuntimeError('Main evaluation has already been initiated; refuse to alter its frozen inputs')
    for status_path in (target / 'smoke').glob('*/status.json'):
        status = json.loads(status_path.read_text(encoding='utf-8'))
        if status.get('status') not in ('complete', 'failed', 'interrupted'):
            raise RuntimeError(f'Smoke test may still be active: {status_path}')


def deployment_plan(source, target, require_selected=True):
    source, target = Path(source).resolve(), Path(target).resolve()
    if source == target or source.is_relative_to(target) or target.is_relative_to(source):
        raise ValueError('Source and destination must be separate experiment roots')
    if not source.is_dir():
        raise FileNotFoundError(source)
    assert_quiescent(target)
    required = [source / 'python/drl_offline/controller.py', source / 'evaluation/campaign.py', source / 'evaluation/report.py']
    if require_selected:
        required += [source / f'selected/{app}.pt' for app in ('lb', 'mo')]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    records = []
    for folder in FOLDERS:
        current = source / folder
        if not current.exists():
            if folder != 'selected' or require_selected:
                raise FileNotFoundError(current)
            continue
        if current.is_symlink():
            raise ValueError(f'Source folder may not be a symbolic link: {current}')
        for path in sorted(current.rglob('*')):
            relative = path.relative_to(source)
            if any(part in IGNORED_PARTS for part in relative.parts) or path.suffix in ('.pyc', '.pyo'):
                continue
            if path.is_symlink():
                raise ValueError(f'Source entry may not be a symbolic link: {path}')
            if not path.is_file():
                continue
            destination = target / relative
            if not destination.resolve().is_relative_to(target):
                raise ValueError(f'Destination escapes isolated experiment: {destination}')
            digest = sha(path)
            old_hash = sha(destination) if destination.is_file() else None
            records.append(dict(relative_path=relative.as_posix(), bytes=path.stat().st_size,
                                sha256=digest, previous_sha256=old_hash,
                                action='unchanged' if old_hash == digest else 'copy'))
    return dict(schema='offline-xapp-isolated-deployment-v1', source=str(source), target=str(target),
                allowed_folders=list(FOLDERS), require_selected=require_selected, files=records)


def deploy(source, target, dry_run=False, require_selected=True):
    plan = deployment_plan(source, target, require_selected)
    plan.update(dry_run=dry_run, recorded_utc=datetime.now(timezone.utc).isoformat())
    if dry_run:
        return plan
    source, target = Path(plan['source']), Path(plan['target'])
    # The precomputed manifest is checked again while copying so an offline
    # trainer replacing a checkpoint cannot silently produce mixed artifacts.
    target.mkdir(parents=True, exist_ok=True)
    assert_quiescent(target)
    for entry in plan['files']:
        src, dest = source / entry['relative_path'], target / entry['relative_path']
        if sha(src) != entry['sha256']:
            raise RuntimeError(f'Source changed during deployment: {src}')
        if entry['action'] == 'unchanged':
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.resolve().is_relative_to(target):
            raise ValueError(f'Destination escapes experiment after directory creation: {dest}')
        temporary = dest.with_name(dest.name + f'.deploy-{os.getpid()}.tmp')
        if temporary.exists():
            raise FileExistsError(temporary)
        try:
            shutil.copyfile(src, temporary)
            if sha(temporary) != entry['sha256']:
                raise RuntimeError(f'Copy checksum mismatch: {src}')
            os.replace(temporary, dest)
        finally:
            temporary.unlink(missing_ok=True)
    for entry in plan['files']:
        if sha(target / entry['relative_path']) != entry['sha256']:
            raise RuntimeError(f'Deployed file changed before verification: {entry["relative_path"]}')
    receipt = target / 'deployment_receipt.json'
    temp_receipt = receipt.with_name(receipt.name + f'.tmp-{os.getpid()}')
    temp_receipt.write_text(json.dumps(plan, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temp_receipt, receipt)
    return plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=SOURCE)
    parser.add_argument('--target', type=Path, default=TARGET)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--allow-missing-selected', action='store_true', help='Prepare code only before the offline trainer has written both selected checkpoints')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Run deployment using WSL/Linux Python, not Windows Python')
    if TARGET.is_symlink():
        parser.error('The experiment destination may not be a symbolic link')
    if args.target.resolve() != TARGET.resolve():
        parser.error('Deployment target is restricted to the new WSL drl_xapp_offline experiment')
    result = deploy(args.source, args.target, args.dry_run, not args.allow_missing_selected)
    print(json.dumps({'source': result['source'], 'target': result['target'], 'dry_run': result['dry_run'],
                      'copy_files': sum(entry['action'] == 'copy' for entry in result['files']),
                      'unchanged_files': sum(entry['action'] == 'unchanged' for entry in result['files']),
                      'copy_bytes': sum(entry['bytes'] for entry in result['files'] if entry['action'] == 'copy'),
                      'preserved': ['bin', 'backend', 'results', 'smoke', 'previous_online_stop.json', 'plan.json']},
                     indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
