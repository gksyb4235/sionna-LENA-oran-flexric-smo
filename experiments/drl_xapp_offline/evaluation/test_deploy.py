"""Exercise real temporary-directory copies, preservation, and rejection paths."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location('offline_deploy', Path(__file__).with_name('deploy.py'))
deploy = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(deploy)


class DeployTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.source = Path(self.temp.name) / 'windows'
        self.target = Path(self.temp.name) / 'wsl'
        for relative in ('python/drl_offline/controller.py', 'evaluation/campaign.py', 'evaluation/report.py', 'selected/lb.pt', 'selected/mo.pt'):
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('new-' + relative).encode())
        self.preserved = ('bin/ns3-drl-offline', 'backend/scenario.cc', 'results/old/cell_kpi.csv', 'previous_online_stop.json', 'plan.json', 'smoke/complete/marker.txt')
        for relative in self.preserved:
            path = self.target / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(('preserve-' + relative).encode())

    def tearDown(self):
        self.temp.cleanup()

    def test_deploy_copies_only_allowlist_and_preserves_existing_evidence(self):
        excluded = self.source / 'results/should_not_copy.txt'
        excluded.parent.mkdir()
        excluded.write_text('never copy')
        result = deploy.deploy(self.source, self.target)
        self.assertEqual(len(result['files']), 5)
        for item in result['files']:
            self.assertEqual((self.source / item['relative_path']).read_bytes(), (self.target / item['relative_path']).read_bytes())
        for relative in self.preserved:
            self.assertEqual((self.target / relative).read_bytes(), ('preserve-' + relative).encode())
        self.assertFalse((self.target / 'results/should_not_copy.txt').exists())
        self.assertTrue((self.target / 'deployment_receipt.json').is_file())
        repeated = deploy.deploy(self.source, self.target)
        self.assertTrue(all(item['action'] == 'unchanged' for item in repeated['files']))

    def test_dry_run_does_not_create_target(self):
        target = Path(self.temp.name) / 'dry-destination'
        result = deploy.deploy(self.source, target, dry_run=True)
        self.assertEqual(len(result['files']), 5)
        self.assertFalse(target.exists())

    def test_pyc_and_caches_are_not_deployed(self):
        path = self.source / 'python/drl_offline/__pycache__/bad.pyc'
        path.parent.mkdir()
        path.write_bytes(b'cache')
        result = deploy.deploy(self.source, self.target)
        self.assertEqual(len(result['files']), 5)

    def test_started_main_or_running_smoke_refuses_source_change(self):
        status = self.target / 'campaign_status.json'
        status.write_text(json.dumps({'status': 'complete'}))
        with self.assertRaises(RuntimeError):
            deploy.deploy(self.source, self.target)
        status.unlink()
        status = self.target / 'smoke/complete/status.json'
        status.write_text(json.dumps({'status': 'running'}))
        with self.assertRaises(RuntimeError):
            deploy.deploy(self.source, self.target)
        status.write_text(json.dumps({'status': 'complete'}))
        self.assertEqual(len(deploy.deploy(self.source, self.target)['files']), 5)

    def test_missing_selected_requires_explicit_code_only_flag(self):
        (self.source / 'selected/mo.pt').unlink()
        with self.assertRaises(FileNotFoundError):
            deploy.deploy(self.source, self.target)
        self.assertEqual(len(deploy.deploy(self.source, self.target, require_selected=False)['files']), 4)

    def test_overlapping_roots_rejected(self):
        with self.assertRaises(ValueError):
            deploy.deploy(self.source, self.source)
        with self.assertRaises(ValueError):
            deploy.deploy(self.source, self.source / 'copy')


if __name__ == '__main__':
    unittest.main()
