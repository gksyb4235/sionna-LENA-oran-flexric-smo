#!/usr/bin/env python3
"""Repository-relative entry point for the frozen original and reserve conflict sweeps."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / 'experiment/khu/studies/conflict_scenarios_500/sweep'
CONFIG = BUNDLE / 'config.local.json'


def select_dataset(start=None, end=None, dataset='auto'):
    """Keep original and reserve manifests separate, including their result provenance."""
    if start is None and end is None:
        return 'original500' if dataset == 'auto' else dataset
    if start is None or end is None:
        raise ValueError('Provide both --start-index and --end-index')
    if 1 <= start <= end <= 500:
        selected = 'original500'
    elif 501 <= start <= end <= 550:
        selected = 'reserve50'
    else:
        raise ValueError('Choose an inclusive range within 1..500 or 501..550; mixed datasets must run separately')
    if dataset not in ('auto', selected):
        raise ValueError(f'Range {start}..{end} belongs to {selected}, not {dataset}')
    return selected


def dataset_paths(dataset):
    if dataset == 'original500':
        return BUNDLE / 'data/execution_manifest.json', CONFIG
    if dataset == 'reserve50':
        directory = ROOT / 'experiment/khu/studies/conflict_scenarios_reserve50'
        return directory / 'execution_manifest.json', directory / 'config.local.json'
    raise ValueError(f'Unknown dataset: {dataset}')


def default_output(dataset):
    return f'experiment/khu/results/conflict/{dataset}'


def call(args):
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


def setup(build=False, jobs=2, dataset='original500'):
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
    _, config_path = dataset_paths(dataset)
    expected = json.loads((BUNDLE / 'config.example.json').read_text())
    expected.update(ns3_root=str(ROOT), python_site_packages=site, output_root=default_output(dataset))
    if config_path.exists():
        current = json.loads(config_path.read_text())
        if current['ns3_root'] != str(ROOT) or current['python_site_packages'] != site:
            raise ValueError(f'Existing {config_path} belongs to a different environment; inspect it before reuse')
    else:
        with config_path.open('x', encoding='utf-8') as handle:
            json.dump(expected, handle, indent=2)
            handle.write('\n')
    call([sys.executable, BUNDLE / 'backend/install_backend.py', '--repo', ROOT, '--install'])
    cache_root = ROOT / 'experiment/khu/caches/ret_caches'
    builder = ROOT / 'experiment/khu/tools/build_merged_ret_cache.py'
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
    print(json.dumps({'config': str(config_path), 'dataset': dataset, 'simulation_started': False,
                      'build_requested': build}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
        epilog='Runner options follow the command; for example: run --shard-index 0 --shard-count 9 --workers 16 --pilot20 --resume')
    parser.add_argument('command', choices=('setup', 'plan', 'preflight', 'run', 'status', 'collect'))
    parser.add_argument('--dataset', choices=('auto', 'original500', 'reserve50'), default='auto')
    args, rest = parser.parse_known_args()
    if args.command == 'setup':
        options = argparse.ArgumentParser(description='Prepare this checkout; never start ns3 simulation')
        options.add_argument('--build', action='store_true')
        options.add_argument('--jobs', type=int, default=2)
        parsed = options.parse_args(rest)
        if parsed.jobs < 1:
            parser.error('--jobs must be positive')
        setup(parsed.build, parsed.jobs, select_dataset(dataset=args.dataset))
    else:
        if any(x == '--manifest' or x.startswith('--manifest=') for x in rest):
            parser.error('Use --dataset original500 or reserve50; invoke sweep.py directly for another design')
        range_parser = argparse.ArgumentParser(add_help=False)
        range_parser.add_argument('--start-index', type=int)
        range_parser.add_argument('--end-index', type=int)
        selected_range, _ = range_parser.parse_known_args(rest)
        dataset = select_dataset(selected_range.start_index, selected_range.end_index, args.dataset)
        manifest, config = dataset_paths(dataset)
        explicit_config = any(x == '--config' or x.startswith('--config=') for x in rest)
        config_args = [] if explicit_config else ['--config', str(config)]
        if not explicit_config and not config.is_file():
            parser.error(f'Run python3 scripts/conflict_sweep.py setup --dataset {dataset} --build first')
        call([sys.executable, BUNDLE / 'sweep.py', args.command,
              '--manifest', manifest, *config_args, *rest])


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as error:
        raise SystemExit(str(error))
