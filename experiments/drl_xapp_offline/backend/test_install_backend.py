"""Installer safety and exact offline action-contract checks."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("offline_backend_installer", HERE / "install_backend.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.repo = Path(self.temporary.name)
        (self.repo / "scratch").mkdir()
        self.source = self.repo / "scratch/khu-real-dual-nr-sionna-final.cc"
        self.raw = ('#include "scheduled-parameter-control.h"\n'
                    'NS_LOG_COMPONENT_DEFINE("KhuRealDualNrSionnaFinal");\n'
                    '    AgoraRicController::InstallFromEnvironment(agoraCells,simTime.GetSeconds());\n').encode()
        self.source.write_bytes(self.raw)

    def tearDown(self):
        self.temporary.cleanup()

    def test_install_is_idempotent_and_keeps_original(self):
        first = installer.install(self.repo)
        second = installer.install(self.repo)
        self.assertEqual(first, second)
        self.assertEqual(self.source.read_bytes(), self.raw)
        target = self.repo / "scratch/drl-xapp-offline/khu-drl-xapp-offline.cc"
        text = target.read_text()
        self.assertIn('nrHelper->GetAttribute("UseIdealRrc", drlIdealRrc)', text)
        self.assertIn('NS_ABORT_MSG_IF(!drlIdealRrc.Get()', text)
        self.assertNotIn('AgoraRicController::InstallFromEnvironment', text)
        self.assertEqual(first['parameter_contract']['ttt_ms'], [128, 256, 340, 512, 720])
        self.assertEqual(first['parameter_contract']['hys_db'], [i / 2 for i in range(1, 11)])
        self.assertFalse((self.repo / "scratch/drl-xapp-conflict").exists())

    def test_unowned_destination_refused(self):
        target = self.repo / "scratch/drl-xapp-offline"
        target.mkdir()
        (target / "important.txt").write_text("preserve")
        with self.assertRaisesRegex(ValueError, "ownership marker"):
            installer.install(self.repo)
        self.assertEqual((target / "important.txt").read_text(), "preserve")

    def test_missing_or_duplicate_anchor_refused(self):
        self.source.write_bytes(self.raw + self.raw)
        with self.assertRaisesRegex(ValueError, "exactly one"):
            installer.install(self.repo)
        self.assertFalse((self.repo / "scratch/drl-xapp-offline").exists())


if __name__ == "__main__":
    unittest.main()
