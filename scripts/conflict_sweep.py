#!/usr/bin/env python3
"""Repository-relative entry point for the reviewed 500-case conflict sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'experiments/conflict_scenarios_500/sweep'
CONFIG = BUNDLE / 'config.local.json'


def call(args):
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def setup(build=False, jobs=2):
    """Prepare local paths, exact merged caches and the isolated build target."""
    if sys.platform != 'linux':
        raise ValueError('Use this command inside Linux/WSL')
    python = ROOT / '.venv/bin/python'
    if not python.is_file():
        raise FileNotFoundError('Prepare the repository .venv first; see README.md')
    probe = subprocess.run([str(python), '-c',
        'import h5py,numpy,sysconfig; print(sysconfig.get_path("purelib"))'],
        cwd=ROOT, capture_output=True, text=True, check=True)
    site = Path(probe.stdout.strip()).resolve().relative_to(ROOT).as_posix()
    expected = json.loads((BUNDLE / 'config.example.json').read_text())
    expected.update(ns3_root=str(ROOT), python_site_packages=site)
    if CONFIG.exists():
        current = json.loads(CONFIG.read_text())
        if current['ns3_root'] != str(ROOT) or current['python_site_packages'] != site:
            raise ValueError(f'Existing {CONFIG} belongs to a different environment; inspect it before reuse')
    else:
        with CONFIG.open('x', encoding='utf-8') as handle:
            json.dump(expected, handle, indent=2)
            handle.write('\n')
    call([sys.executable, BUNDLE / 'backend/install_backend.py', '--repo', ROOT, '--install'])
    cache_root = ROOT / 'scenarios/khu-real/ret_caches'
    builder = ROOT / 'scenarios/khu-real/tools/build_merged_ret_cache.py'
    for first in (3, 5, 7, 9, 12):
        for second in (3, 5, 7, 9, 12):
            if first == second:
                continue
            target = cache_root / f'merged_percell/merged_1p8ghz_4g1_{first}deg_4g2_{second}deg.h5'
            if target.exists():
                # Preserve existing caches; preflight audits their exact metadata.
                continue
            call([python, builder, '--tilt-5g', '9', '--tilt-4g1', first, '--tilt-4g2', second])
    if build:
        call([ROOT / 'ns3', 'build', 'khu-agora-conflict-sweep', '-j', jobs])
    print(json.dumps({'config': str(CONFIG), 'simulation_started': False,
                      'build_requested': build}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        epilog='Runner options follow the command; for example: run --shard-index 0 --shard-count 9 --workers 16 --pilot20 --resume')
    parser.add_argument('command', choices=('setup', 'plan', 'preflight', 'run', 'status', 'collect'))
    args, rest = parser.parse_known_args()
    if args.command == 'setup':
        options = argparse.ArgumentParser(description='Prepare this checkout; never start ns3 simulation')
        options.add_argument('--build', action='store_true')
        options.add_argument('--jobs', type=int, default=2)
        parsed = options.parse_args(rest)
        if parsed.jobs < 1:
            parser.error('--jobs must be positive')
        setup(parsed.build, parsed.jobs)
    else:
        if any(x == '--manifest' or x.startswith('--manifest=') for x in rest):
            parser.error('This entry point uses the frozen 500-case manifest; invoke sweep.py directly for another design')
        explicit_config = any(x == '--config' or x.startswith('--config=') for x in rest)
        config_args = [] if explicit_config else ['--config', str(CONFIG)]
        if not explicit_config and not CONFIG.is_file():
            parser.error('Run python3 scripts/conflict_sweep.py setup --build first')
        call([sys.executable, BUNDLE / 'sweep.py', args.command,
              '--manifest', BUNDLE / 'data/execution_manifest.json', *config_args, *rest])


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
