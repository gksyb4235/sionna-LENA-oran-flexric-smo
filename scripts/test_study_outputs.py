"""Exact relocated study paths are allowed; arbitrary links stay forbidden."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import setup_study_outputs as setup


def load(name, relative):
    spec = importlib.util.spec_from_file_location(name, setup.ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE = 'experiment/khu/studies/drl_xapp_offline/evaluation/'
campaign = load('relocated_offline_campaign', BASE + 'campaign.py')
archive = load('relocated_offline_archive', BASE + 'archive_failed.py')


class StudyLayoutTests(unittest.TestCase):
    def fixture(self, root):
        study = root / 'experiment/khu/studies/drl_xapp_offline'
        study.mkdir(parents=True)
        setup.prepare(root, 'drl_xapp_offline')
        return study

    def test_fresh_clone_output_setup_and_checkpoint_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study = self.fixture(root)
            (study / 'selected/lb.pt').write_bytes(b'unchanged checkpoint')
            setup.prepare(root, 'drl_xapp_offline')
            self.assertEqual((study / 'selected/lb.pt').read_bytes(), b'unchanged checkpoint')
            plan = study / 'plan.json'
            plan.write_text(json.dumps(campaign.make_plan()))
            with patch.object(campaign, 'ROOT', study):
                self.assertEqual(len(campaign.read_plan(plan)['jobs']), 16)
            self.assertEqual(archive.contained(study, 'selected/lb.pt'), study / 'selected/lb.pt')

    def test_external_or_cross_study_symlink_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study = self.fixture(root)
            alias = study / 'selected'
            alias.unlink()
            alias.symlink_to(root)
            with self.assertRaises(ValueError):
                setup.prepare(root, 'drl_xapp_offline')
            with self.assertRaises(ValueError):
                archive.contained(study, 'selected')
            plan = study / 'plan.json'
            plan.write_text(json.dumps(campaign.make_plan()))
            with patch.object(campaign, 'ROOT', study), self.assertRaises(ValueError):
                campaign.read_plan(plan)
            with self.assertRaises(ValueError):
                archive.contained(study, '../outside')

    def test_archive_moves_payload_and_keeps_compatibility_link(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            study = self.fixture(root)
            status = {'status': 'failed', 'finished_utc': '2026-10-25T00:00:00+00:00'}
            (study / 'campaign_status.json').write_text(json.dumps(status))
            (study / 'results/raw.csv').write_bytes(b'preserved bytes')
            (study / 'selected/lb.pt').write_bytes(b'preserved checkpoint')
            result = archive.archive(study, probe=lambda pid: None, scanner=lambda root: [])
            self.assertEqual(result['status'], 'complete')
            self.assertTrue((study / 'results').is_symlink())
            self.assertFalse((study / 'results').exists())
            destination = study / 'failed_attempts/attempt001'
            self.assertEqual((destination / 'results/raw.csv').read_bytes(), b'preserved bytes')
            self.assertEqual((destination / 'snapshot/selected/lb.pt').read_bytes(), b'preserved checkpoint')
            setup.prepare(root, 'drl_xapp_offline')
            self.assertTrue((study / 'results').is_dir())


if __name__ == '__main__':
    unittest.main()
