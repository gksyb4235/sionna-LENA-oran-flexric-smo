"""Repository entry-point contracts; no radio process or external API calls."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, Mock

import conflict_sweep as cli


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
            caches=root/'scenarios/khu-real/ret_caches/merged_percell'; caches.mkdir(parents=True)
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


if __name__=='__main__': unittest.main()
