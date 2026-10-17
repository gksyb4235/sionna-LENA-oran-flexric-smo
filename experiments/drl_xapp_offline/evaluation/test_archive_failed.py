"""Archive actual temporary files; never touch or signal real simulator jobs."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('offline_archive', Path(__file__).with_name('archive_failed.py'))
archive = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(archive)


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name) / 'experiment'
        self.root.mkdir()
        self.status = {'status': 'failed', 'started_utc': '2026-10-17T00:00:00+00:00',
                       'finished_utc': '2026-10-17T00:10:00+00:00', 'runner_pid': 101,
                       'running_ids': ['stale_job'], 'jobs': {}}
        self.write('campaign_status.json', json.dumps(self.status))
        self.write('background_launch.json', json.dumps({'pid': 101}))
        self.write('.evaluation-launch.lock', json.dumps({'pid': 101}))
        self.write('results/job/cell_kpi.csv', 'raw,unchanged\n1,2\n')
        self.write('results/job/execution.json', json.dumps({'worker_pid': 102, 'ns3_pid': 103, 'controller_pid': 104, 'status': 'failed'}))
        self.write('job_specs/job.json', '{}')
        self.write('plan.json', '{"original":"plan"}')
        self.write('campaign_console.log', 'SIGILL evidence')
        for path in ('python/drl_offline/controller.py', 'evaluation/campaign.py', 'selected/lb.pt', 'bin/ns3-drl-offline',
                     'backend/scenario.cc', 'README.md', 'training/training_report.json', 'previous_online_stop.json'):
            self.write(path, 'old deployed ' + path)
        self.write('training/huge_replay.npz', 'must not snapshot this large dataset')
        self.write('smoke/failed/status.json', 'preserve old smoke')
        self.write('preflight.json', json.dumps({'source_hashes': [{'path': 'evaluation/campaign.py', 'sha256': archive.sha(self.root / 'evaluation/campaign.py')}]}))

    def tearDown(self):
        self.temp.cleanup()

    def write(self, relative, content):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')

    def run_archive(self, **options):
        return archive.archive(self.root, probe=options.pop('probe', lambda pid: None), scanner=options.pop('scanner', lambda root: []), **options)

    def test_real_archive_preserves_raw_and_original_status_with_old_code_snapshot(self):
        original_status = (self.root / 'campaign_status.json').read_bytes()
        result = self.run_archive()
        target = self.root / 'failed_attempts/attempt001'
        self.assertEqual(result['status'], 'complete')
        self.assertEqual((target / 'campaign_status.json').read_bytes(), original_status)
        self.assertEqual((target / 'results/job/cell_kpi.csv').read_text(), 'raw,unchanged\n1,2\n')
        self.assertFalse((self.root / 'results').exists())
        self.assertFalse((self.root / 'campaign_status.json').exists())
        self.assertFalse((self.root / '.evaluation-launch.lock').exists())
        self.assertFalse((self.root / '.archive-in-progress').exists())
        self.assertTrue((target / 'archive_operation_lock.json').is_file())
        for relative in ('python/drl_offline/controller.py', 'evaluation/campaign.py', 'selected/lb.pt', 'bin/ns3-drl-offline', 'README.md', 'training/training_report.json'):
            self.assertEqual((target / 'snapshot' / relative).read_bytes(), (self.root / relative).read_bytes())
        self.assertFalse((target / 'snapshot/training/huge_replay.npz').exists())
        self.assertTrue((self.root / 'smoke/failed/status.json').exists())
        self.assertTrue(result['snapshot_preflight_comparisons'][0]['matches'])
        with self.assertRaises(FileExistsError):
            self.run_archive()

    def test_dry_run_does_not_move_or_create_archive(self):
        result = self.run_archive(dry_run=True)
        self.assertTrue(result['dry_run'])
        self.assertTrue((self.root / 'results/job/cell_kpi.csv').exists())
        self.assertFalse((self.root / 'failed_attempts').exists())

    def test_live_owned_and_ambiguous_pids_rejected(self):
        started = archive.epoch(self.status['started_utc'])
        for info in ({'pid': 101, 'state': 'R', 'start_ticks': 99, 'start_epoch_s': started,
                      'command': ['python', str(self.root / 'evaluation/campaign.py'), 'launch']},
                     {'pid': 101, 'state': 'R', 'start_ticks': 99, 'start_epoch_s': started + 10,
                      'command': ['unrelated-program']}):
            with self.subTest(info=info), self.assertRaises(RuntimeError):
                self.run_archive(dry_run=True, probe=lambda pid: info if pid == 101 else None)
        self.assertTrue((self.root / 'campaign_status.json').exists())

    def test_pid_reuse_with_later_start_is_recorded_without_signalling(self):
        info = {'pid': 101, 'state': 'S', 'start_ticks': 999, 'start_epoch_s': archive.epoch(self.status['finished_utc']) + 100,
                'command': ['unrelated-program']}
        result = self.run_archive(dry_run=True, probe=lambda pid: info if pid == 101 else None)
        self.assertEqual(result['process_audit']['recorded_pids'][0]['finding'], 'pid_reused')

    def test_unrecorded_owned_worker_rejected(self):
        with self.assertRaises(RuntimeError):
            self.run_archive(dry_run=True, scanner=lambda root: [{'pid': 999, 'command': [str(root / 'bin/ns3-drl-offline')]}])

    def test_nonterminal_campaign_and_path_escape_rejected(self):
        self.write('campaign_status.json', json.dumps({**self.status, 'status': 'running'}))
        with self.assertRaises(RuntimeError):
            self.run_archive(dry_run=True)
        with self.assertRaises(ValueError):
            self.run_archive(attempt='../escape', dry_run=True)
