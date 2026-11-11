"""Test queue stopping against fake processes only; never signal real PIDs."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('stopper', Path(__file__).with_name('agora_evaluation_stop.py'))
stopper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(stopper)


class FakeProc:
    def __init__(self):
        self.processes = {}; self.signals = []; self.resistant = set(); self.pending_term = set()

    def add(self, pid, args, state='S', ppid=1, start=100):
        self.processes[pid] = dict(pid=pid, uid=1000, state=state, ppid=ppid,
                                   start_ticks=start, args=tuple(args))

    def pids(self):
        return list(self.processes)

    def snapshot(self, pid):
        return copy.deepcopy(self.processes.get(pid))

    def alive(self, previous):
        current = self.processes.get(previous['pid'])
        return bool(current and current['state'] not in ('Z', 'X') and current['start_ticks'] == previous['start_ticks'])

    def send(self, previous, sig):
        pid = previous['pid']
        if not self.alive(previous):
            return False
        current = self.processes[pid]
        if current['args'] != previous['args']:
            raise RuntimeError('Identity changed')
        self.signals.append((pid, sig))
        if sig == 19:
            current['state'] = 'T'
        elif sig == 15:
            if current['state'] == 'T':
                self.pending_term.add(pid)
            elif pid not in self.resistant:
                del self.processes[pid]
        elif sig == 18:
            if pid in self.pending_term and pid not in self.resistant:
                del self.processes[pid]
            else:
                current['state'] = 'S'
        elif sig == 9:
            del self.processes[pid]
        return True


class StopTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.out = (self.root / 'results').resolve()
        self.manifest = self.root / 'experiment/khu/agora_evaluation_v2/manifest.json'
        stopper.save(self.manifest, {'partitions': {'pc2': ['rq3_0382'], 'pc3': ['rq3_0120']}})
        self.command = ['python3', '-u', str(self.root / 'tools/agora_evaluation_sweep.py'), '--partition', 'pc2', '--workers', '22', '--output', str(self.out)]
        stopper.save(self.out / '_launchers/pc2_latest.json', dict(pid=42, partition='pc2', output=str(self.out), command=self.command))
        self.argv = ['taskset', '-c', '0', 'prlimit', '--as=8589934592', '/repo/build/ns3.48-khu-agora-conflict-sweep-default',
                     f'--resultsRoot={self.out / "rq3_0382/simulation"}', '--simTime=900']
        self.receipt = self.out / 'rq3_0382/execution.json'
        stopper.save(self.receipt, dict(status='running', pid=43, argv=self.argv))
        self.fs = FakeProc(); self.fs.add(42, self.command); self.fs.add(43, self.argv[5:], ppid=42)
        self.addCleanup(patch.stopall)
        patch.object(stopper.os, 'getuid', return_value=1000, create=True).start()
        for name, number in [('SIGSTOP', 19), ('SIGCONT', 18), ('SIGTERM', 15), ('SIGKILL', 9)]:
            patch.object(stopper.signal, name, number, create=True).start()

    def run_stop(self, **kwargs):
        return stopper.stop_queue(self.root, 'pc2', self.out, timeout=0, fs=self.fs, **kwargs)

    def test_dry_run_never_signals_or_changes_receipts(self):
        before = self.receipt.read_bytes(); result = self.run_stop(dry_run=True)
        self.assertEqual(result['jobs'], {'43': 'rq3_0382'}); self.assertFalse(self.fs.signals)
        self.assertEqual(before, self.receipt.read_bytes()); self.assertFalse((self.out / '_stop_requests').exists())

    def test_paused_child_and_dispatcher_stop_gracefully(self):
        self.fs.processes[42]['state'] = 'T'; self.fs.processes[43]['state'] = 'T'
        result = self.run_stop()
        self.assertTrue(result['stopped']); self.assertFalse(any(sig == 9 for _, sig in self.fs.signals))
        self.assertLess(self.fs.signals.index((43, 18)), self.fs.signals.index((42, 15)))
        self.assertEqual(stopper.read(self.receipt)['status'], 'interrupted')
        archived = list((self.out / '_stop_requests').glob('*/receipts/rq3_0382/execution.json'))
        self.assertEqual(len(archived), 1); self.assertEqual(stopper.read(archived[0])['status'], 'running')

    def test_other_queue_is_untouched(self):
        other = list(self.argv[5:]); other[1] = f'--resultsRoot={self.out / "rq3_0120/simulation"}'
        self.fs.add(44, other, ppid=99)
        self.assertTrue(self.run_stop()['stopped']); self.assertIn(44, self.fs.processes)
        self.assertFalse(any(pid == 44 for pid, _ in self.fs.signals))

    def test_reused_dispatcher_pid_refused_before_any_signal(self):
        self.fs.processes[42]['args'] = ('unrelated_process',)
        with self.assertRaises(ValueError): self.run_stop()
        self.assertFalse(self.fs.signals)

    def test_unrecorded_pid_launch_race_matches_exact_argv(self):
        stopper.save(self.receipt, dict(status='launching', argv=self.argv))
        self.assertTrue(self.run_stop()['stopped']); self.assertEqual(stopper.read(self.receipt)['stop_observed_pid'], 43)

    def test_unverified_direct_child_keeps_dispatcher_paused(self):
        self.fs.add(44, ['python3', 'pre_exec'], ppid=42)
        result = self.run_stop()
        self.assertFalse(result['stopped']); self.assertEqual(result['unverified_child_pids'], [44])
        self.assertEqual(self.fs.signals, [(42, 19)])

    def test_resistant_child_no_implicit_force(self):
        self.fs.resistant.add(43); result = self.run_stop()
        self.assertFalse(result['stopped']); self.assertEqual(result['remaining_job_pids'], [43])
        self.assertEqual(self.fs.processes[42]['state'], 'T'); self.assertNotIn((42, 15), self.fs.signals)
        self.assertFalse(any(sig == 9 for _, sig in self.fs.signals)); self.assertEqual(stopper.read(self.receipt)['status'], 'running')

    def test_explicit_force_only_kills_verified_survivor(self):
        self.fs.resistant.add(43); result = self.run_stop(force=True)
        self.assertTrue(result['stopped']); self.assertEqual([p for p, sig in self.fs.signals if sig == 9], [43])

    def test_terminal_receipt_is_not_rewritten(self):
        stopper.save(self.receipt, dict(status='complete', pid=43, argv=self.argv))
        before = self.receipt.read_bytes(); self.run_stop()
        self.assertEqual(self.receipt.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
