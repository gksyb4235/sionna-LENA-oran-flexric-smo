#!/usr/bin/env python3
"""Stop only a verified native validation queue; retain all output and attempts."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import time
from agora_evaluation_allocation import load_allocation


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(path.read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f'.stop_{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


class ProcFS:
    """Identify processes by UID, start ticks and exact argv, never PID alone."""

    def __init__(self, root=Path('/proc')):
        self.root = root

    def pids(self):
        return [int(p.name) for p in self.root.iterdir() if p.name.isdigit()]

    def snapshot(self, pid):
        folder = self.root / str(pid)
        try:
            # Fields after the final ')' begin at stat field 3 (state).
            fields = (folder / 'stat').read_text().rsplit(')', 1)[1].split()
            args = tuple(os.fsdecode(x) for x in (folder / 'cmdline').read_bytes().split(b'\0') if x)
            return dict(pid=int(pid), uid=folder.stat().st_uid, state=fields[0],
                        ppid=int(fields[1]), start_ticks=int(fields[19]), args=args)
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            return None

    def alive(self, previous):
        current = self.snapshot(previous['pid'])
        return bool(current and current['state'] not in ('Z', 'X') and
                    current['start_ticks'] == previous['start_ticks'] and
                    current['uid'] == previous['uid'])

    def send(self, previous, sig):
        pid = previous['pid']
        fd = None
        try:
            if hasattr(os, 'pidfd_open') and hasattr(signal, 'pidfd_send_signal'):
                fd = os.pidfd_open(pid)
            current = self.snapshot(pid)
            if not current or current['state'] in ('Z', 'X'):
                return False
            if any(current[k] != previous[k] for k in ('uid', 'start_ticks', 'args')):
                raise RuntimeError(f'Process identity changed; refusing to signal PID {pid}')
            if current['uid'] != os.getuid():
                raise RuntimeError(f'Process UID differs; refusing to signal PID {pid}')
            if fd is not None:
                signal.pidfd_send_signal(fd, sig)
            else:
                os.kill(pid, sig)
            return True
        except ProcessLookupError:
            return False
        finally:
            if fd is not None:
                os.close(fd)


def live_dispatcher(fs, launcher, partition, output):
    if launcher.get('partition') != partition or Path(launcher['output']).resolve() != output:
        raise ValueError('Launcher partition/output does not match the requested queue')
    command = tuple(launcher['command'])
    if not any(Path(arg).name == 'agora_evaluation_sweep.py' for arg in command):
        raise ValueError('Launcher does not identify agora_evaluation_sweep.py')
    if '--partition' not in command or command[command.index('--partition') + 1] != partition:
        raise ValueError('Launcher argv has a different partition')
    if '--output' not in command or Path(command[command.index('--output') + 1]).resolve() != output:
        raise ValueError('Launcher argv has a different output directory')
    process = fs.snapshot(int(launcher['pid']))
    if not process or process['state'] in ('Z', 'X'):
        return None
    if process['uid'] != os.getuid() or process['args'] != command:
        raise ValueError('Recorded dispatcher PID now belongs to a different process; refusing to stop')
    return process


def child_specs(output, jobs):
    result = {}
    for job in jobs:
        path = output / job / 'execution.json'
        if not path.exists():
            continue
        record = read(path)
        command = tuple(record.get('argv', []))
        required = f'--resultsRoot={output / job / "simulation"}'
        if required not in command:
            continue
        binary = next((i for i, arg in enumerate(command)
                       if Path(arg).name == 'ns3.48-khu-agora-conflict-sweep-default'), None)
        if binary is None:
            continue
        suffixes = [command[binary:]]
        for i, arg in enumerate(command[:binary]):
            if Path(arg).name in ('taskset', 'prlimit'):
                suffixes.append(command[i:])
        result[job] = dict(path=path, record=record, suffixes=suffixes)
    return result


def discover_children(fs, specs):
    result = {}
    for pid in fs.pids():
        process = fs.snapshot(pid)
        if not process or process['uid'] != os.getuid() or process['state'] in ('Z', 'X'):
            continue
        for job, spec in specs.items():
            if process['args'] in spec['suffixes']:
                result[pid] = dict(process=process, job=job)
                break
    return result


def wait_dead(fs, processes, timeout):
    deadline = time.monotonic() + timeout
    while True:
        alive = [p for p in processes if fs.alive(p)]
        if not alive or time.monotonic() >= deadline:
            return alive
        time.sleep(min(.1, max(0, deadline - time.monotonic())))


def stop_queue(root, partition, output, timeout=20, force=False, dry_run=False, fs=None):
    fs = fs or ProcFS()
    output = output.resolve()
    manifest = read(root / 'experiment/khu/agora_evaluation_v2/manifest.json')
    allocation = load_allocation(root, manifest)
    jobs = allocation['partitions'][partition]
    launcher_path = output / '_launchers' / f'{partition}_latest.json'
    if not launcher_path.exists():
        raise ValueError(f'No launcher record: {launcher_path}; no process was signaled')
    launcher = read(launcher_path)
    # A later reallocation must not hide jobs belonging to this launch.
    jobs = launcher.get('jobs', jobs)
    permitted = set(j for ids in manifest['partitions'].values() for j in ids)
    if not isinstance(jobs, list) or not set(jobs).issubset(permitted):
        raise ValueError('Launcher contains jobs outside the frozen replay catalog')
    dispatcher = live_dispatcher(fs, launcher, partition, output)
    specs = child_specs(output, jobs)
    children = discover_children(fs, specs)
    report = dict(partition=partition, output=str(output), requested_at=now(), dry_run=dry_run,
                  dispatcher_pid=dispatcher['pid'] if dispatcher else None,
                  jobs={str(pid): item['job'] for pid, item in children.items()}, force=force)
    if dry_run:
        return report
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    archive = output / '_stop_requests' / f'{partition}_{stamp}'
    save(archive / 'request.json', report)
    if dispatcher:
        fs.send(dispatcher, signal.SIGSTOP)
        deadline = time.monotonic() + 3
        while fs.alive(dispatcher):
            current = fs.snapshot(dispatcher['pid'])
            if current and current['state'] in ('T', 't'):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError('Dispatcher did not pause; refusing to race against new launches')
            time.sleep(.05)
    # Include a child created just before the dispatcher was paused, even if its
    # PID was not yet saved to execution.json. Its recorded exact argv is required.
    specs = child_specs(output, jobs)
    children.update(discover_children(fs, specs))
    unknown_children = []
    if dispatcher:
        for pid in fs.pids():
            process = fs.snapshot(pid)
            if (process and process['state'] not in ('Z', 'X') and
                    process['ppid'] == dispatcher['pid'] and pid not in children):
                unknown_children.append(pid)
    if unknown_children:
        report.update(stopped=False, remaining_job_pids=[],
                      remaining_dispatcher_pids=[dispatcher['pid']],
                      unverified_child_pids=unknown_children,
                      reason='Dispatcher is paused; a child has not reached a verifiable simulator argv. Retry after inspecting.',
                      report=str(archive / 'result.json'))
        save(archive / 'result.json', report)
        return report
    for item in children.values():
        fs.send(item['process'], signal.SIGTERM)
        fs.send(item['process'], signal.SIGCONT)
    alive = wait_dead(fs, [x['process'] for x in children.values()], timeout)
    if alive and force:
        for process in alive:
            fs.send(process, signal.SIGKILL)
        alive = wait_dead(fs, alive, min(timeout, 5))
    # The frozen runner kills survivors after ten seconds during its cleanup.
    # Resume it only after all verified children have already exited.
    if not alive and dispatcher and fs.alive(dispatcher):
        fs.send(dispatcher, signal.SIGTERM)
        fs.send(dispatcher, signal.SIGCONT)
        dispatcher_alive = wait_dead(fs, [dispatcher], timeout)
        if dispatcher_alive and force:
            fs.send(dispatcher, signal.SIGKILL)
            dispatcher_alive = wait_dead(fs, [dispatcher], min(timeout, 5))
    else:
        dispatcher_alive = [dispatcher] if dispatcher and fs.alive(dispatcher) else []
    updated = []
    # A paused/dead dispatcher cannot change a receipt while it is archived.
    dispatcher_state = fs.snapshot(dispatcher['pid']) if dispatcher else None
    dispatcher_safe = not dispatcher_alive or not dispatcher_state or dispatcher_state['state'] in ('T', 't')
    if dispatcher_safe:
        for item in children.values():
            if fs.alive(item['process']):
                continue
            job = item['job']; path = specs[job]['path']; record = read(path)
            if record.get('status') not in ('running', 'launching'):
                continue
            if tuple(record.get('argv', [])) != tuple(specs[job]['record'].get('argv', [])):
                continue
            save(archive / 'receipts' / job / 'execution.json', record)
            save(path, {**record, 'status': 'interrupted', 'finished_at': now(),
                        'stop_reason': 'User requested targeted queue stop',
                        'stop_observed_pid': item['process']['pid'],
                        'stop_report': str(archive / 'result.json')})
            updated.append(job)
    report.update(jobs={str(pid): item['job'] for pid, item in children.items()},
                  finished_at=now(), interrupted_receipts=updated,
                  remaining_job_pids=[p['pid'] for p in alive],
                  remaining_dispatcher_pids=[p['pid'] for p in dispatcher_alive],
                  stopped=not alive and not dispatcher_alive, report=str(archive / 'result.json'))
    save(archive / 'result.json', report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--partition', required=True, choices=['pc1', 'pc2', 'pc3', 'pc4'])
    parser.add_argument('--output', type=Path)
    parser.add_argument('--timeout', type=float, default=20)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--force', action='store_true', help='SIGKILL only verified survivors after timeout')
    args = parser.parse_args()
    if not 0 <= args.timeout <= 30:
        parser.error('--timeout must be between 0 and 30 seconds')
    root = Path(__file__).resolve().parents[3]
    output = args.output or Path(os.environ.get('AGORA_OUTPUT', str(root / 'experiment/khu/results/agora_evaluation_v2')))
    result = stop_queue(root, args.partition, output, args.timeout, args.force, args.dry_run)
    print(json.dumps(result, indent=2))
    if not args.dry_run and not result['stopped']:
        print('Verified survivors remain; dispatcher stays paused. Inspect the report before retrying with --force.')
        raise SystemExit(2)


if __name__ == '__main__':
    main()
