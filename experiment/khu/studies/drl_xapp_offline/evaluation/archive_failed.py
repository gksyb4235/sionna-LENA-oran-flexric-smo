"""Preserve a stopped failed attempt before deploying changed code or relaunching.

Run this standalone helper FROM THE MOUNTED WINDOWS SOURCE before deployment.
It snapshots the currently deployed WSL code/checkpoints/binary first, then
moves only explicitly named runtime artifacts. It never deletes evidence or
signals processes. Large caches and shared libraries are not copied.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys

EXPECTED_ROOT = Path('/home/user/LENA-oran-flexric-smo/experiment/khu/studies/drl_xapp_offline')
MOVE_ITEMS = ('results', 'job_specs', 'plan.json', 'preflight.json',
              'campaign_console.log', 'background_launch.json',
              'completion_verification.json', '.evaluation-launch.lock',
              'campaign_status.json')
SNAPSHOT_ITEMS = ('python', 'evaluation', 'selected', 'bin', 'backend', 'README.md',
                  'deployment_receipt.json', 'previous_online_stop.json', 'smoke_verification.json')
SKIP_PARTS = {'__pycache__', '.pytest_cache', '.mypy_cache', '.ruff_cache'}
RELOCATED_FOLDERS = {'results', 'failed_attempts', 'smoke', 'training', 'data_audit', 'selected',
                     'job_specs', 'provenance'}


def relocated_root(root):
    root = Path(root).resolve()
    if root.parent.name == 'studies' and root.parent.parent.name == 'khu':
        return root.parent.parent / 'results' / root.name
    return None


def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def epoch(value):
    if isinstance(value, (float, int)):
        return float(value)
    return datetime.fromisoformat(value.replace('Z', '+00:00')).timestamp() if value else None


def proc_info(pid):
    """Read identity only; never read environment variables or send a signal."""
    path = Path('/proc') / str(pid)
    if not path.exists():
        return None
    try:
        raw = (path / 'stat').read_text()
        fields = raw[raw.rfind(')') + 2:].split()
        ticks = int(fields[19])
        boot = next(int(line.split()[1]) for line in Path('/proc/stat').read_text().splitlines() if line.startswith('btime '))
        args = (path / 'cmdline').read_bytes().split(b'\0')
        return dict(pid=int(pid), state=fields[0], start_ticks=ticks,
                    start_epoch_s=boot + ticks / os.sysconf('SC_CLK_TCK'),
                    command=[arg.decode('utf-8', errors='replace') for arg in args if arg])
    except FileNotFoundError:
        return None  # Process exited between the two reads.


def command_owned(info, root):
    args = info.get('command', [])
    root = root.resolve()
    for argument in args:
        if argument.startswith('/'):
            candidate = Path(argument).resolve()
            if candidate.is_relative_to(root / 'bin'):
                return True
            if candidate in (root / 'evaluation/campaign.py', root / 'evaluation/start_background.py'):
                return True
    if 'drl_offline.controller' in args and '--control-dir' in args:
        index = args.index('--control-dir')
        if index + 1 < len(args):
            control = Path(args[index + 1]).resolve()
            output = relocated_root(root)
            if control.is_relative_to(root) or (output and output.resolve() == output and control.is_relative_to(output)):
                return True
    return False


def scan_owned(root):
    result = []
    for path in Path('/proc').iterdir():
        if not path.name.isdigit() or int(path.name) == os.getpid():
            continue
        try:
            info = proc_info(int(path.name))
        except PermissionError:
            continue  # Known recorded PIDs are checked strictly below.
        if info and info['state'] != 'Z' and command_owned(info, root):
            result.append(info)
    return result


def process_audit(root, status, probe=proc_info, scanner=scan_owned):
    known = {}
    def add(value, role, ticks=None):
        if value:
            pid = int(value)
            known.setdefault(pid, {'pid': pid, 'roles': [], 'recorded_start_ticks': ticks})['roles'].append(role)
    add(status.get('runner_pid'), 'campaign runner', status.get('runner_start_ticks'))
    for name in ('background_launch.json', '.evaluation-launch.lock'):
        path = root / name
        if path.exists():
            record = read_json(path)
            add(record.get('pid'), name, record.get('start_ticks'))
    for path in (root / 'results').glob('*/execution.json'):
        record = read_json(path)
        for role in ('worker', 'ns3', 'controller'):
            add(record.get(role + '_pid'), f'{path.parent.name}:{role}', record.get(role + '_start_ticks'))
    lower, upper = epoch(status.get('started_utc')), epoch(status.get('finished_utc'))
    observations, blocked = [], []
    for pid, record in sorted(known.items()):
        info = probe(pid)
        if not info or info.get('state') == 'Z':
            observations.append({**record, 'finding': 'not_running' if not info else 'zombie_not_running'})
            continue
        finding = 'live_owned' if command_owned(info, root) else None
        if finding is None:
            expected_ticks = record.get('recorded_start_ticks')
            if expected_ticks is not None and info['start_ticks'] != expected_ticks:
                finding = 'pid_reused'
            elif (info.get('start_epoch_s') is not None and lower is not None and upper is not None
                  and (info['start_epoch_s'] < lower - 2 or info['start_epoch_s'] > upper + 2)):
                finding = 'pid_reused'
            else:
                finding = 'live_identity_ambiguous'
        observation = {**record, 'finding': finding, 'current': info}
        observations.append(observation)
        if finding != 'pid_reused':
            blocked.append(observation)
    unrecorded = scanner(root)
    if unrecorded:
        blocked += [{'finding': 'live_owned_scan', 'current': info} for info in unrecorded]
    if blocked:
        raise RuntimeError('Cannot archive while owned/ambiguous processes remain: ' + json.dumps(blocked))
    return dict(checked_utc=now(), recorded_pids=observations, live_owned_scan=[], no_processes_signalled=True)


def contained(root, relative):
    relative = Path(relative)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Archive source escapes the experiment: {relative}')
    path = root / relative
    allowed_root = root.resolve()
    for index in range(len(relative.parts)):
        part = root.joinpath(*relative.parts[:index + 1])
        if not part.is_symlink():
            continue
        output = relocated_root(root)
        expected = output / relative.parts[0] if output else None
        if index == 0 and relative.parts[0] in RELOCATED_FOLDERS and expected is not None and part.resolve() == expected and expected.resolve() == expected:
            allowed_root = expected
        else:
            raise ValueError(f'Archive source is an unexpected symlink: {part}')
    if not path.resolve().is_relative_to(allowed_root):
        raise ValueError(f'Archive source escapes the experiment or is a symlink: {path}')
    return path


def inventory(path):
    """Hash regular files, preserving links as link metadata without following."""
    if path.is_symlink():
        return [{'relative': '.', 'type': 'symlink', 'target': os.readlink(path)}]
    if path.is_file():
        return [{'relative': '.', 'type': 'file', 'bytes': path.stat().st_size, 'sha256': sha(path)}]
    records = []
    for item in sorted(path.rglob('*')):
        if item.is_symlink():
            records.append(dict(relative=item.relative_to(path).as_posix(), type='symlink', target=os.readlink(item)))
        elif item.is_file():
            records.append(dict(relative=item.relative_to(path).as_posix(), type='file', bytes=item.stat().st_size, sha256=sha(item)))
        elif item.is_dir():
            records.append(dict(relative=item.relative_to(path).as_posix(), type='directory'))
        else:
            raise ValueError(f'Unsupported special file in archived data: {item}')
    return records


def snapshot_candidates(root):
    files = []
    for name in SNAPSHOT_ITEMS:
        path = contained(root, name)
        if not path.exists():
            continue
        entries = [path] if path.is_file() else sorted(path.rglob('*'))
        for entry in entries:
            relative = entry.relative_to(root)
            if any(part in SKIP_PARTS for part in relative.parts) or entry.suffix in ('.pyc', '.pyo'):
                continue
            if entry.is_symlink():
                raise ValueError(f'Input snapshot may not follow symbolic links: {entry}')
            if entry.is_file():
                files.append(entry)
    for folder in ('training', 'data_audit'):
        path = contained(root, folder)
        for entry in sorted(path.rglob('*')) if path.exists() else ():
            if entry.is_symlink():
                raise ValueError(f'Report snapshot may not follow symbolic links: {entry}')
            if entry.is_file() and entry.suffix.lower() in ('.json', '.txt', '.csv', '.md', '.log'):
                files.append(entry)
    return sorted(set(files))


def archive(root, attempt='attempt001', dry_run=False, probe=proc_info, scanner=scan_owned):
    root = Path(root).resolve()
    if not re.fullmatch(r'attempt\d{3,}', attempt):
        raise ValueError('Attempt name must be attempt followed by at least three digits')
    archive_root = contained(root, 'failed_attempts')
    destination = contained(root, 'failed_attempts/' + attempt)
    if destination.exists():
        raise FileExistsError(destination)
    status = read_json(contained(root, 'campaign_status.json'))
    if status.get('status') not in ('failed', 'interrupted') or not status.get('finished_utc'):
        raise RuntimeError('Only an explicitly terminated failed/interrupted campaign can be archived')
    audit = process_audit(root, status, probe, scanner)
    moved = []
    for name in MOVE_ITEMS:
        path = contained(root, name)
        if path.exists():
            payload = path.resolve() if path.is_symlink() else path
            moved.append({'path': name, 'inventory': inventory(payload)})
    snapshot = [dict(path=path.relative_to(root).as_posix(), bytes=path.stat().st_size, sha256=sha(path))
                for path in snapshot_candidates(root)]
    consistency = []
    preflight_path = root / 'preflight.json'
    if preflight_path.is_file():
        frozen = read_json(preflight_path)
        current = {entry['path']: entry['sha256'] for entry in snapshot}
        for field in ('input_hashes', 'source_hashes'):
            for entry in frozen.get(field, []):
                path = Path(entry['path'])
                path = path if path.is_absolute() else root / path
                if path.is_relative_to(root) and path.relative_to(root).as_posix() in current:
                    relative = path.relative_to(root).as_posix()
                    consistency.append(dict(path=relative, preflight_sha256=entry['sha256'],
                                            snapshot_sha256=current[relative], matches=entry['sha256'] == current[relative]))
    manifest = dict(schema='offline-drl-failed-attempt-archive-v1', created_utc=now(), source_root=str(root),
                    archive_directory=str(destination), attempt=attempt, dry_run=dry_run,
                    original_campaign_status=status.get('status'), original_status_preserved=True,
                    process_audit=audit, moved_items=moved, snapshot_files=snapshot,
                    snapshot_preflight_comparisons=consistency,
                    note='Snapshots represent deployed files before this archival step. Caches/shared libraries are referenced by the preserved preflight hashes, not copied.')
    if dry_run:
        return manifest
    archive_root.mkdir(parents=True, exist_ok=True)
    lock = contained(root, '.archive-in-progress')
    with lock.open('x', encoding='utf-8') as handle:
        json.dump({'pid': os.getpid(), 'started_utc': now(), 'destination': str(destination)}, handle)
    staging = contained(root, f'failed_attempts/.{attempt}.staging-{os.getpid()}')
    staging.mkdir(exist_ok=False)
    write_json(staging / 'archive_manifest.json', {**manifest, 'status': 'snapshotting'})
    # Any failure intentionally leaves staging/lock/evidence intact for review.
    for item in snapshot:
        source = contained(root, item['path'])
        target = staging / 'snapshot' / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        if sha(source) != item['sha256']:
            raise RuntimeError(f'Deployed input changed before snapshot: {source}')
        shutil.copy2(source, target)
        if sha(target) != item['sha256']:
            raise RuntimeError(f'Snapshot checksum mismatch: {target}')
    manifest['second_process_audit'] = process_audit(root, status, probe, scanner)
    write_json(staging / 'archive_manifest.json', {**manifest, 'status': 'moving_runtime_artifacts'})
    for item in moved:
        source = contained(root, item['path'])
        # Preserve the compatibility link; archive its validated data target.
        if source.is_symlink():
            source = source.resolve()
        if inventory(source) != item['inventory']:
            raise RuntimeError(f'Runtime artifact changed before archival: {source}')
        target = staging / item['path']
        if target.exists():
            raise FileExistsError(target)
        source.rename(target)
        if inventory(target) != item['inventory']:
            raise RuntimeError(f'Archived runtime artifact checksum mismatch: {target}')
    lock.rename(staging / 'archive_operation_lock.json')
    manifest.update(status='complete', finished_utc=now(), copied_snapshot_files=len(snapshot), moved_runtime_items=len(moved))
    write_json(staging / 'archive_manifest.json', manifest)
    staging.rename(destination)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=EXPECTED_ROOT)
    parser.add_argument('--attempt', default='attempt001')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux' or args.root.is_symlink() or args.root.resolve() != EXPECTED_ROOT.resolve():
        parser.error('Archive only the isolated WSL drl_xapp_offline experiment, using Linux Python')
    result = archive(args.root, args.attempt, args.dry_run)
    print(json.dumps({'archive_directory': result['archive_directory'], 'dry_run': result['dry_run'],
                      'snapshot_files': len(result['snapshot_files']), 'runtime_items': [entry['path'] for entry in result['moved_items']],
                      'live_owned_processes': 0, 'snapshot_preflight_mismatches': [entry['path'] for entry in result['snapshot_preflight_comparisons'] if not entry['matches']],
                      'status': result.get('status', 'dry_run')}, indent=2))


if __name__ == '__main__':
    main()
