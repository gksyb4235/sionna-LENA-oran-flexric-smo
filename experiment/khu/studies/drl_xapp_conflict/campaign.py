"""Isolated Linux ns-3 / Double DQN pilot campaign.

Every simulator has its own lockstep controller, outputs, and process group.
Training, validation selection, and frozen paired evaluation are separate stages.
This pilot records real transition counts; replay updates are not new samples.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parent
BACKEND = Path('/home/user/LENA-oran-flexric-smo')
SCHEMA = 'ns3-drl-xapp-pilot-v1'
ARMS = {
    'none': ('none', 'local', 'eval'),
    'lb_local': ('lb', 'local', 'eval'),
    'mo_local': ('mo', 'local', 'eval'),
    'both_local': ('both', 'local', 'eval'),
    'lb_guarded': ('lb', 'guarded', 'eval'),
    'mo_guarded': ('mo', 'guarded', 'eval'),
    'both_guarded': ('both', 'guarded', 'eval'),
    'heuristic_both': ('both', 'guarded', 'heuristic'),
}


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic(path, value):
    path = Path(path)
    temp = path.with_name(path.name + f'.tmp-{os.getpid()}')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def safe_child(root, name):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', name):
        raise ValueError(f'Unsafe identifier {name!r}')
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError('Path escapes experiment root')
    return result


def make_plan():
    jobs = []
    train_traces, validation_traces, test_traces = [1414, 1415, 1416], [1417, 1418], [1419, 1420, 1421]
    for app in ('lb', 'mo'):
        for reward in ('local', 'guarded'):
            for learning_seed in (101, 202):
                previous = None
                for episode, trace in enumerate(train_traces, 1):
                    job_id = f'train_{app}_{reward}_s{learning_seed}_e{episode:02d}'
                    jobs.append(dict(id=job_id, phase='train', app=app, mode=app, reward=reward,
                                     policy='train', trace_seed=trace, learning_seed=learning_seed,
                                     episode=episode, depends=[previous] if previous else [],
                                     load_job=previous, lb_start_s=180, mo_start_s=180))
                    previous = job_id
                for trace in validation_traces:
                    jobs.append(dict(id=f'val_{app}_{reward}_s{learning_seed}_t{trace}', phase='validation',
                                     app=app, mode=app, reward=reward, policy='eval', trace_seed=trace,
                                     learning_seed=learning_seed, depends=[previous], load_job=previous,
                                     lb_start_s=180, mo_start_s=180))
    for trace in test_traces:
        for arm, (mode, reward, policy) in ARMS.items():
            jobs.append(dict(id=f'test_{arm}_t{trace}', phase='test', arm=arm, mode=mode, reward=reward,
                             policy=policy, trace_seed=trace, learning_seed=909, depends=[],
                             lb_start_s=180, mo_start_s=540))
    return dict(schema=SCHEMA, created_utc=now(), backend_root='/home/user/LENA-oran-flexric-smo',
                binary='bin/ns3-drl-xapp-conflict', duration_s=900, n_ues=300, control_start_s=180,
                control_interval_s=30, rng_seed=1, rng_run=1, training_traces=train_traces,
                validation_traces=validation_traces, test_traces=test_traces,
                baseline={'txp_dbm': 43, 'ret_deg': 9, 'cio_db': [0, 0, 0], 'ttt_ms': 128, 'hys_db': 1},
                training_status='pilot_72_real_transitions_per_model_not_convergence_claim',
                maximum_jobs=len(jobs), requested_workers=8, jobs=jobs)


def read_plan(path):
    plan = json.loads(Path(path).read_text(encoding='utf-8'))
    if plan.get('schema') != SCHEMA:
        raise ValueError('Unsupported plan schema')
    groups = [set(plan[k]) for k in ('training_traces', 'validation_traces', 'test_traces')]
    if any(groups[a] & groups[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Training/validation/test trace leakage')
    seen = set()
    for job in plan['jobs']:
        safe_child(ROOT / 'results', job['id'])
        if job['id'] in seen:
            raise ValueError('Duplicate job id')
        seen.add(job['id'])
        split = {'train': 'training_traces', 'validation': 'validation_traces', 'test': 'test_traces'}
        if job['phase'] not in split or job['trace_seed'] not in plan[split[job['phase']]]:
            raise ValueError('Job trace does not belong to its declared phase')
        if (job['phase'] == 'train') != (job['policy'] == 'train'):
            raise ValueError('Only training-phase jobs may update policies')
    return plan


def environment():
    env = {k: v for k, v in os.environ.items() if not k.startswith(('AGORA_', 'RIC_', 'DRL_'))}
    env.pop('PYTHONHOME', None)
    env['PYTHONPATH'] = str(ROOT / 'python') + os.pathsep + str(BACKEND / '.venv/lib/python3.12/site-packages')
    env.update(OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
               PYTHONDONTWRITEBYTECODE='1')
    return env


def sim_argv(plan, spec, directory):
    cache = BACKEND / 'scenarios/khu-real/ret_caches/tilt_9deg'
    labels = ('gNB_5G', 'gNB_4G_1', 'gNB_4G_2')
    base = plan['baseline']
    def field(key, value):
        values = value if isinstance(value, list) else [value] * 3
        return '--' + key + '=' + ','.join(f'{label}:{v:g}' for label, v in zip(labels, values))
    return [str(ROOT / plan['binary']), '--centralFrequency18=1.8e9', '--bandwidth18=10e6',
            '--centralFrequency35=3.5e9', '--bandwidth35=10e6', '--numerology=0',
            f'--sionnaCacheFile18={cache}/sionna_rt_cache_1p8ghz_2x2_1x1_lzf.h5',
            f'--sionnaCacheFile35={cache}/sionna_rt_cache_3p5ghz_2x2_1x1_lzf.h5',
            '--sionnaUpdatePeriod=1s', '--udpBaseIntervalMs=200', '--trafficMultiplierEnabled=true',
            f'--simTime={spec.get("duration_s", plan["duration_s"])}', '--kpiReportInterval=1.0',
            f'--N_Ues={spec.get("n_ues", plan["n_ues"])}',
            f'--sumoTrace={BACKEND}/scenarios/khu-real/ue_position/ue_positions_seed{spec["trace_seed"]}.csv',
            f'--gnbPositions={BACKEND}/scenarios/khu-real/gnbs-ret.csv',
            f'--resultsRoot={directory}/ns3', '--runTag=simulation',
            f'--cellKpiCsvPath={directory}/cell_kpi.csv', f'--ueKpiCsvPath={directory}/ue_kpi.csv',
            f'--RngSeed={plan["rng_seed"]}', f'--RngRun={plan["rng_run"]}',
            '--ns3::NrHelper::E2ModeNr=false',
            field('cellTxPowerDbm', base['txp_dbm']), field('cellRetTiltDeg', base['ret_deg']),
            field('cellCioDb', base['cio_db']), '--cellRetBearingDeg=gNB_5G:20,gNB_4G_1:0,gNB_4G_2:10',
            f'--handoverTtt={base["ttt_ms"]}', f'--handoverHysteresis={base["hys_db"]}',
            field('cellTttMs', base['ttt_ms']), field('cellHysteresisDb', base['hys_db'])]


def controller_argv(spec, control):
    args = [sys.executable, '-m', 'drl_xapps.controller', '--control-dir', str(control),
            '--mode', spec['mode'], '--reward', spec['reward'], '--policy', spec['policy'],
            '--seed', str(spec['learning_seed']), '--checkpoint-dir', str(control / 'checkpoints'),
            '--lb-start-s', str(spec['lb_start_s']), '--mo-start-s', str(spec['mo_start_s']),
            '--epsilon-decay', '96']
    if spec.get('load_job'):
        args += ['--load-dir', str(ROOT / 'results' / spec['load_job'] / 'control/checkpoints')]
    if spec.get('selected_load_dir'):
        args += ['--load-dir', spec['selected_load_dir']]
    return args


def last_time(directory):
    try:
        with (directory / 'cell_kpi.csv').open('rb') as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 16384))
            lines = f.read().decode('utf-8').splitlines()
        for line in reversed(lines):
            try:
                return float(line.split(',')[0])
            except ValueError:
                continue
    except OSError:
        pass
    return None


def verify(directory, duration):
    issues, slots, cell_rows = [], set(), []
    try:
        with (directory / 'cell_kpi.csv').open(newline='') as f:
            for row in csv.DictReader(f):
                cell_rows.append(row)
                t, cell = float(row['time_s']), int(row['cell_id'])
                if cell not in (1, 2, 3) or t < 0 or t >= duration:
                    issues.append('invalid_cell_or_time')
                key = (t, cell)
                if key in slots:
                    issues.append('duplicate_cell_time')
                slots.add(key)
        expected = {(i + 0.5, c) for i in range(int(duration)) for c in (1, 2, 3)}
        if slots != expected:
            issues.append(f'incomplete_cell_grid:{len(slots)}/{duration*3}')
    except (OSError, KeyError, ValueError) as error:
        issues.append('cell_csv:' + str(error))
    try:
        observation = json.loads((directory / 'control/observation.json').read_text())
        if not observation['terminal']:
            issues.append('missing_terminal_observation')
        summary = json.loads((directory / 'control/summary.json').read_text())
        if summary.get('status') != 'complete' or abs(summary.get('last_time_s', -1) - duration) > 1e-6:
            issues.append('controller_not_complete_at_stop')
        receipts = [json.loads(line) for line in (directory / 'control/receipts.jsonl').read_text().splitlines()]
        if not receipts or receipts[-1]['status'] != 'terminal':
            issues.append('missing_terminal_receipt')
        if any(r['status'] not in ('applied', 'held', 'terminal') for r in receipts):
            issues.append('non_applied_control_receipt')
        actions = [json.loads(line) for line in (directory / 'control/actions.jsonl').read_text().splitlines()]
        nonterminal = [a for a in actions if not a['terminal']]
        applied = [r for r in receipts if r['status'] in ('applied', 'held')]
        if len(nonterminal) != len(applied):
            issues.append('action_receipt_count_mismatch')
        for action, receipt in zip(nonterminal, applied):
            if action['seq'] != receipt['seq'] or abs(action['time_s'] - receipt['time_s']) > 1e-6:
                issues.append('action_receipt_sequence_or_time_mismatch')
            if action['updates'] != receipt['updates']:
                issues.append('action_receipt_updates_mismatch')
            for key, value in action['updates'].items():
                if abs(receipt['after'][key] - value) > 1e-6:
                    issues.append('action_receipt_readback_mismatch')
        if applied:
            labels = {1: '5g', 2: '4g1', 3: '4g2'}
            state = dict(applied[0]['before'])
            index = 0
            for row in sorted(cell_rows, key=lambda r: float(r['time_s'])):
                while index < len(applied) and applied[index]['time_s'] <= float(row['time_s']):
                    state = dict(applied[index]['after'])
                    index += 1
                label = labels[int(row['cell_id'])]
                for column, parameter in (('cio_bias_db', 'cio_' + label), ('ttt_ms', 'ttt_ms'), ('hysteresis_db', 'hys_db')):
                    if abs(float(row[column]) - state[parameter]) > 1e-6:
                        issues.append(f'cell_parameter_state_mismatch:{row["time_s"]}:{parameter}')
    except (OSError, KeyError, ValueError) as error:
        issues.append('controller:' + str(error))
        summary = None
    return dict(passed=not issues, issues=issues, cell_rows=len(slots), controller_summary=summary)


def run_job(plan, spec, timeout=172800):
    if spec['phase'] != 'smoke':
        check_frozen_inputs()
    directory = safe_child(ROOT / 'results', spec['id'])
    directory.mkdir(parents=True, exist_ok=False)
    control = directory / 'control'
    control.mkdir()
    duration = spec.get('duration_s', plan['duration_s'])
    record = dict(spec=spec, status='launching', started_utc=now(), worker_pid=os.getpid(),
                  directory=str(directory), ns3_returncode=None, controller_returncode=None)
    atomic(directory / 'execution.json', record)
    args, ctrl_args = sim_argv(plan, spec, directory), controller_argv(spec, control)
    atomic(directory / 'command.json', {'ns3': args, 'controller': ctrl_args})
    (directory / 'command.sh').write_text('#!/usr/bin/env bash\n' + shlex.join(args) + '\n', encoding='utf-8')
    env = environment()
    env.update(DRL_CONTROL_DIR=str(control), DRL_START_S=str(spec.get('control_start_s', plan['control_start_s'])),
               DRL_INTERVAL_S=str(spec.get('control_interval_s', plan['control_interval_s'])), DRL_TIMEOUT_S='300')
    processes = []
    started = time.monotonic()
    previous_handlers = {}
    def terminate(signum=None, frame=None):
        for p in processes:
            if p.poll() is None:
                p.terminate()
        if signum:
            raise KeyboardInterrupt(f'worker signal {signum}')
    for signum in (signal.SIGTERM, signal.SIGINT):
        previous_handlers[signum] = signal.signal(signum, terminate)
    try:
        with (directory / 'controller.log').open('xb', buffering=0) as clog, (directory / 'ns3.log').open('xb', buffering=0) as slog:
            controller = subprocess.Popen(ctrl_args, cwd=BACKEND, env=env, stdout=clog, stderr=subprocess.STDOUT)
            processes.append(controller)
            simulator = subprocess.Popen(args, cwd=BACKEND, env=env, stdout=slog, stderr=subprocess.STDOUT)
            processes.append(simulator)
            record.update(status='running', controller_pid=controller.pid, ns3_pid=simulator.pid)
            while simulator.poll() is None:
                if controller.poll() is not None:
                    # The controller may exit after its terminal acknowledgement, just before ns3.
                    if not (control / 'summary.json').exists() or controller.returncode != 0:
                        raise RuntimeError(f'Controller exited early: {controller.returncode}')
                if time.monotonic() - started > timeout:
                    raise TimeoutError('ns3 job exceeded wall-clock timeout')
                record.update(wall_seconds=time.monotonic() - started, last_cell_time_s=last_time(directory), updated_utc=now())
                atomic(directory / 'execution.json', record)
                time.sleep(2)
            record['ns3_returncode'] = simulator.returncode
            if simulator.returncode != 0:
                raise RuntimeError(f'ns3 exited {simulator.returncode}')
            controller.wait(timeout=60)
            record['controller_returncode'] = controller.returncode
            if controller.returncode:
                raise RuntimeError(f'Controller exited {controller.returncode}')
            record['verification'] = verify(directory, duration)
            record['status'] = 'complete' if record['verification']['passed'] else 'failed'
    except BaseException as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
    finally:
        terminate()
        for p in processes:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=10)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        record.update(finished_utc=now(), wall_seconds=time.monotonic() - started, last_cell_time_s=last_time(directory))
        atomic(directory / 'execution.json', record)
    return record


def memory_available_gib():
    for line in Path('/proc/meminfo').read_text().splitlines():
        if line.startswith('MemAvailable:'):
            return int(line.split()[1]) / 1024 ** 2
    return 0


def executions():
    result = {}
    for path in (ROOT / 'results').glob('*/execution.json'):
        try:
            item = json.loads(path.read_text())
            result[item['spec']['id']] = item
        except (OSError, ValueError, KeyError):
            continue
    return result


def load_analysis():
    spec = importlib.util.spec_from_file_location('drl_campaign_report', ROOT / 'analysis/report.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_policies(plan):
    import shutil
    report = load_analysis()
    selected = dict(created_utc=now(), source_phase='validation_only', selections={})
    for app in ('lb', 'mo'):
        for reward in ('local', 'guarded'):
            candidates = []
            for seed in (101, 202):
                dirs = [ROOT / 'results' / f'val_{app}_{reward}_s{seed}_t{trace}' for trace in plan['validation_traces']]
                scores = [report.score_run(d, app) for d in dirs]
                if not all(x['eligible'] for x in scores):
                    raise ValueError(f'Insufficient validation support for {app}/{reward}/{seed}')
                # The same objective and guard weights rank both reward variants.
                score = sum(x['score'] for x in scores) / len(scores)
                candidates.append(dict(learning_seed=seed, score=score, details=scores))
            chosen = max(candidates, key=lambda x: (x['score'], -x['learning_seed']))
            last = len(plan['training_traces'])
            source = ROOT / 'results' / f'train_{app}_{reward}_s{chosen["learning_seed"]}_e{last:02d}' / 'control/checkpoints' / f'{app}.pt'
            destination = ROOT / 'selected' / reward / f'{app}.pt'
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists():
                raise FileExistsError('Selected checkpoint already exists')
            shutil.copyfile(source, destination)
            selected['selections'][f'{app}_{reward}'] = dict(chosen=chosen, candidates=candidates,
                                                           source=str(source), checkpoint=str(destination), sha256=sha(destination))
    atomic(ROOT / 'selection.json', selected)
    return selected


def preflight(plan):
    import numpy
    import torch
    if sys.platform != 'linux':
        raise RuntimeError('Simulation campaign requires Linux/WSL')
    paths = [ROOT / plan['binary'], ROOT / 'python/drl_xapps/controller.py', ROOT / 'analysis/report.py']
    for trace in plan['training_traces'] + plan['validation_traces'] + plan['test_traces']:
        paths.append(BACKEND / f'scenarios/khu-real/ue_position/ue_positions_seed{trace}.csv')
    paths += [BACKEND / f'scenarios/khu-real/ret_caches/tilt_9deg/sionna_rt_cache_{band}_2x2_1x1_lzf.h5'
              for band in ('1p8ghz', '3p5ghz')]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    return dict(checked_utc=now(), available_ram_gib=memory_available_gib(), jobs=len(plan['jobs']),
                versions={'python': sys.version, 'torch': torch.__version__, 'numpy': numpy.__version__},
                input_hashes=[dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in paths],
                source_hashes=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in ROOT.rglob('*.py') if 'results' not in p.parts],
                linked_library_hashes=[dict(path=str(p), sha256=sha(p)) for p in (BACKEND / 'build/lib').glob('*.so')])


def check_frozen_inputs():
    """Reject a mixed-code campaign if a subsequent build/edit changes inputs."""
    reference = json.loads((ROOT / 'preflight.json').read_text())
    for field in ('input_hashes', 'source_hashes', 'linked_library_hashes'):
        for entry in reference[field]:
            path = Path(entry['path'])
            path = path if path.is_absolute() else ROOT / path
            if not path.is_file() or sha(path) != entry['sha256']:
                raise RuntimeError(f'Frozen campaign input changed: {path}')


def campaign(plan_path, workers=8):
    plan = read_plan(plan_path)
    if (ROOT / 'campaign_status.json').exists():
        raise FileExistsError('Campaign already started. Do not overwrite or duplicate it.')
    atomic(ROOT / 'preflight.json', preflight(plan))
    pending = {j['id']: j for j in plan['jobs']}
    campaign_ids = set(pending)
    running, records = {}, {}
    state = dict(status='running', started_utc=now(), runner_pid=os.getpid(), workers=workers,
                 total_jobs=len(pending), pilot_only=True, stage='train')
    selected = False
    stopping = False
    def stop(signum, frame):
        nonlocal stopping
        stopping = True
    for number in (signal.SIGINT, signal.SIGTERM):
        signal.signal(number, stop)
    next_launch = 0
    try:
        while pending or running:
            records = {k: v for k, v in executions().items() if k in campaign_ids}
            for job_id, (p, f) in list(running.items()):
                if p.poll() is not None:
                    f.close()
                    # Read AFTER wait/poll observes exit; the worker may have
                    # atomically completed between the earlier snapshot and poll.
                    path = ROOT / 'results' / job_id / 'execution.json'
                    if path.is_file():
                        records[job_id] = json.loads(path.read_text())
                    if records.get(job_id, {}).get('status') not in ('complete', 'failed'):
                        record = records.get(job_id, {'spec': next(j for j in plan['jobs'] if j['id'] == job_id)})
                        record.update(status='failed', directory=str(ROOT / 'results' / job_id),
                                      error=f'worker exited {p.returncode} without terminal execution record')
                        records[job_id] = record
                        path = ROOT / 'results' / job_id / 'execution.json'
                        path.parent.mkdir(parents=True, exist_ok=True)
                        atomic(path, record)
                    del running[job_id]
            failed = [j for j, r in records.items() if r['status'] == 'failed']
            if failed:
                state.update(status='failed', failed_jobs=failed)
                raise RuntimeError('Campaign stopped after failed jobs; preserve evidence before retry')
            if stopping:
                state['status'] = 'interrupted'
                raise KeyboardInterrupt('Campaign stop requested')
            train_done = all(records.get(j['id'], {}).get('status') == 'complete' for j in plan['jobs'] if j['phase'] == 'train')
            validation_done = all(records.get(j['id'], {}).get('status') == 'complete' for j in plan['jobs'] if j['phase'] == 'validation')
            stage = 'test' if validation_done else 'validation' if train_done else 'train'
            if validation_done and not selected:
                select_policies(plan)
                selected = True
            ready = [j for j in pending.values() if j['phase'] == stage and all(records.get(d, {}).get('status') == 'complete' for d in j['depends'])]
            if ready and len(running) < workers and time.monotonic() >= next_launch and memory_available_gib() >= 5.5:
                job = dict(ready[0])
                if stage == 'test' and job['policy'] == 'eval' and job['mode'] != 'none':
                    job['selected_load_dir'] = str(ROOT / 'selected' / job['reward'])
                specs = ROOT / 'job_specs'
                specs.mkdir(exist_ok=True)
                specpath = specs / (job['id'] + '.json')
                atomic(specpath, job)
                log = (specs / (job['id'] + '.worker.log')).open('xb', buffering=0)
                p = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), 'run-job', '--plan', str(plan_path), '--spec', str(specpath)],
                                     cwd=BACKEND, env=environment(), stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                running[job['id']] = (p, log)
                del pending[job['id']]
                next_launch = time.monotonic() + 8
            state.update(stage=stage, updated_utc=now(), pending_ids=list(pending), running_ids=list(running),
                         completed_ids=[j for j, r in records.items() if r['status'] == 'complete'],
                         available_ram_gib=memory_available_gib(), jobs=records)
            atomic(ROOT / 'campaign_status.json', state)
            if pending or running:
                time.sleep(3)
        subprocess.run([sys.executable, str(ROOT / 'analysis/report.py'), '--root', str(ROOT)], check=True, env=environment())
        state['status'] = 'complete'
    except BaseException as error:
        state.update(status=state['status'] if state['status'] != 'running' else 'failed', error=f'{type(error).__name__}: {error}')
        for p, _ in running.values():
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
        for p, f in running.values():
            try:
                p.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.wait(timeout=10)
            f.close()
    finally:
        state.update(finished_utc=now(), jobs={k: v for k, v in executions().items() if k in campaign_ids})
        atomic(ROOT / 'campaign_status.json', state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('init', 'preflight', 'run-job', 'campaign', 'status'))
    parser.add_argument('--plan', type=Path, default=ROOT / 'plan.json')
    parser.add_argument('--spec', type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    if args.command == 'init':
        if args.plan.exists():
            raise FileExistsError(args.plan)
        atomic(args.plan, make_plan())
        print(args.plan)
    elif args.command == 'status':
        records = executions()
        print(json.dumps([dict(id=k, status=v['status'], time_s=last_time(Path(v['directory']))) for k, v in records.items()], indent=2))
    else:
        plan = read_plan(args.plan)
        if args.command == 'preflight':
            print(json.dumps(preflight(plan), indent=2))
        elif args.command == 'run-job':
            result = run_job(plan, json.loads(args.spec.read_text(encoding='utf-8')))
            print(json.dumps(result, indent=2))
            return 0 if result['status'] == 'complete' else 1
        else:
            result = campaign(args.plan.resolve(), args.workers)
            print(result['status'])
            return 0 if result['status'] == 'complete' else 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
