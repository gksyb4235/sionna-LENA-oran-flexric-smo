"""Repository entry-point contracts; no radio process or external API calls."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import conflict_sweep as cli
import conflict_sweep_range as range_cli


class EntryPointTests(unittest.TestCase):
    def test_default_run_forwards_frozen_manifest_and_machine_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=Path(tmp)/'config.json'
            config.write_text('{}')
            with patch.object(cli,'CONFIG',config), patch.object(cli,'call') as call, \
                 patch('sys.argv',['conflict_sweep','run','--shard-index','3','--shard-count','9','--pilot20','--resume']):
                cli.main()
            args=[str(x) for x in call.call_args.args[0]]
            self.assertEqual(args[2],'run')
            self.assertIn(str(cli.BUNDLE/'data/execution_manifest.json'),args)
            self.assertEqual(args[-6:],['--shard-index','3','--shard-count','9','--pilot20','--resume'])

    def test_setup_preserves_config_and_existing_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); bundle=root/'bundle'; bundle.mkdir()
            python=root/'.venv/bin/python'; python.parent.mkdir(parents=True); python.touch()
            site=root/'.venv/lib/python3.12/site-packages'
            config=bundle/'config.local.json'
            original={'ns3_root':str(root),'python_site_packages':str(site.relative_to(root)),'custom':'preserve'}
            config.write_text(json.dumps(original))
            (bundle/'config.example.json').write_text('{}')
            caches=root/'experiment/khu/caches/ret_caches/merged_percell'; caches.mkdir(parents=True)
            for a in (3,5,7,9,12):
                for b in (3,5,7,9,12):
                    if a!=b:
                        (caches/f'merged_1p8ghz_4g1_{a}deg_4g2_{b}deg.h5').write_bytes(b'preserve')
            with patch.object(cli,'ROOT',root),patch.object(cli,'BUNDLE',bundle),patch.object(cli,'CONFIG',config), \
                 patch.object(cli.sys,'platform','linux'),patch.object(cli,'call') as call, \
                 patch.object(cli.subprocess,'run',return_value=Mock(stdout=str(site)+'\n')), \
                 contextlib.redirect_stdout(io.StringIO()):
                cli.setup()
            self.assertEqual(json.loads(config.read_text()),original)
            self.assertEqual(call.call_count,1)  # Installer only; no build or simulation.
            self.assertTrue(all(p.read_bytes()==b'preserve' for p in caches.iterdir()))

    def test_rejects_manifest_override(self):
        with patch('sys.argv',['conflict_sweep','run','--manifest','other.json']), \
             patch.object(cli,'call') as call, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit): cli.main()
            call.assert_not_called()

    def test_direct_entry_infers_reserve_manifest_from_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)/'config.json'
            config.write_text('{}')
            manifest, _ = cli.dataset_paths('reserve50')
            with patch.object(cli,'dataset_paths',return_value=(manifest,config)) as paths, \
                 patch.object(cli,'call') as call, \
                 patch('sys.argv',['conflict_sweep','run','--start-index','501','--end-index','512']):
                cli.main()
            paths.assert_called_once_with('reserve50')
            self.assertIn(str(manifest), [str(x) for x in call.call_args.args[0]])

    def test_manifest_selection_rejects_mixed_or_wrong_dataset(self):
        self.assertEqual(cli.select_dataset(1,500),'original500')
        self.assertEqual(cli.select_dataset(501,550),'reserve50')
        for start,end,dataset in ((500,501,'auto'),(501,512,'original500'),
                                  (0,1,'auto'),(550,551,'auto'),(502,501,'auto')):
            with self.subTest(start=start,end=end,dataset=dataset), self.assertRaises(ValueError):
                cli.select_dataset(start,end,dataset)


class RangeTests(unittest.TestCase):
    def test_legacy_config_paths_resolve_without_alias_or_contract_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            studies = root / "experiment/khu/studies"
            results = root / "experiment/khu/results/agora_conflict_sweep"
            studies.mkdir(parents=True)
            results.mkdir(parents=True)
            (studies / "agora_conflict_sweep").symlink_to("../results/agora_conflict_sweep")
            config = {"output_root": "experiments/agora_conflict_sweep/range_results_v1"}
            before = json.dumps(config)
            self.assertEqual(range_cli.sweep.contained(root, config["output_root"], "output_root"),
                             results / "range_results_v1")
            self.assertEqual(json.dumps(config), before)
            self.assertFalse((root / "experiments").exists())
            with self.assertRaises(ValueError):
                range_cli.sweep.contained(root, "experiments/../escape", "output_root")

    def test_cpu_v1_layout_keeps_original_container_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(range_cli.sweep.contained(root, "experiments/sweep", "bundle"),
                             root / "experiments/sweep")

    def plan(self, start, end, dataset, config_data=None):
        manifest, _ = cli.dataset_paths(dataset)
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp)/'config.local.json'
            if config_data is not None:
                config.with_name('config.range.local.json').write_text(json.dumps(config_data))
            output = io.StringIO()
            with patch.object(cli,'dataset_paths',return_value=(manifest,config)), \
                 patch.object(cli,'setup') as setup, \
                 patch.object(range_cli.sweep,'cpu_slots',return_value=[[0],[1]]) as slots, \
                 patch.object(range_cli.subprocess,'run') as run, \
                 patch('sys.argv',['conflict_range',str(start),str(end),'1','2','--dry-run']), \
                 contextlib.redirect_stdout(output):
                range_cli.main()
            setup.assert_not_called()
            run.assert_not_called()
            slots.assert_called_once_with(1,2)
            self.assertFalse(config.exists())
            return json.loads(output.getvalue())

    def test_original_plan_is_read_only_and_uses_canonical_output(self):
        plan = self.plan(1,12,'original500')
        self.assertEqual(plan['dataset'],'original500')
        self.assertEqual(plan['first_id'],'conflict_0001')
        self.assertEqual(plan['selected_jobs'],12)
        self.assertEqual(plan['output_root'],str(cli.ROOT/cli.default_output('original500')))

    def test_reserve_plan_selects_inclusive_range_with_separate_output(self):
        plan = self.plan(501,512,'reserve50')
        self.assertEqual(plan['dataset'],'reserve50')
        self.assertEqual(plan['job_ids'],[f'conflict_{i:04d}' for i in range(501,513)])
        self.assertEqual(plan['output_root'],str(cli.ROOT/cli.default_output('reserve50')))
        self.assertFalse(plan['simulation_launched'])

    def test_existing_range_output_is_preserved(self):
        existing = {'ns3_root':str(cli.ROOT),'output_root':'experiment/khu/results/existing'}
        plan = self.plan(501,512,'reserve50',existing)
        self.assertEqual(plan['output_root'],str(cli.ROOT/existing['output_root']))

    def test_cross_dataset_range_rejected_without_setup(self):
        with patch('sys.argv',['conflict_range','499','502','1','2','--dry-run']), \
             patch.object(cli,'setup') as setup, contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                range_cli.main()
            self.assertEqual(error.exception.code,2)
            setup.assert_not_called()


if __name__=='__main__': unittest.main()
