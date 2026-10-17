"""Ensure the portable installer fails closed and never overwrites user edits."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("conflict_backend_installer", HERE / "install_backend.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.repo = Path(self.tmp.name)
        self.source = self.repo / installer.SOURCE
        self.source.parent.mkdir(parents=True)
        self.raw = (b'#include "ns3/agora-ric-controller.h"\n'
                    b'#include "scheduled-parameter-control.h"\n'
                    b'NS_LOG_COMPONENT_DEFINE("KhuRealDualNrSionnaFinal");\n'
                    b'ScheduledParameterControl::Load(\n'
                    b'    const auto scheduledParameters = ScheduledParameterControl::Install(\n'
                    b'        parameterEvents, agoraCells, scheduledLookups, parameterScheduleRetCacheRoot,\n'
                    b'        (runDirectory / "parameter_changes.csv").string());\n'
                    b'    AgoraRicController::InstallFromEnvironment(agoraCells,simTime.GetSeconds());\n')
        self.source.write_bytes(self.raw)
        self.original_hash = installer.SOURCE_SHA256
        installer.SOURCE_SHA256 = installer.sha256(self.raw)
        deps = {
            "src/spectrum/model/sionna-lookup-channel-model.h": "PrepareRetCache ActivateRetCache",
            "src/spectrum/model/sionna-lookup-channel-model.cc": "PrepareRetCache ActivateRetCache",
            "contrib/nr/model/nr-ue-rrc.h": "SetCellIndividualOffset GetCellIndividualOffset",
            "contrib/nr/model/nr-gnb-rrc.h": "GetA3HandoverConfig UpdateA3HandoverConfig",
        }
        for name, text in deps.items():
            path = self.repo / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text)

    def tearDown(self):
        installer.SOURCE_SHA256 = self.original_hash
        self.tmp.cleanup()

    def test_dryrun_and_idempotent_install_preserve_original(self):
        result = installer.install(self.repo)
        self.assertEqual(result["files_written"], 0)
        self.assertFalse((self.repo / installer.DESTINATION).exists())
        installed = installer.install(self.repo, write=True)
        self.assertEqual(installed["files_written"], 3)
        again = installer.install(self.repo, write=True)
        self.assertEqual(again["files_written"], 0)
        self.assertEqual(self.source.read_bytes(), self.raw)
        text = (self.repo / installer.DESTINATION / "khu-agora-conflict-sweep.cc").read_text()
        self.assertIn("ConflictScheduleControl::Install", text)
        self.assertIn('nrHelper->GetAttribute("UseIdealRrc", sweepIdealRrc)', text)
        self.assertIn('Simulator::ScheduleNow([&]()', text)
        self.assertNotIn("AgoraRicController", text)

    def test_source_change_is_refused(self):
        self.source.write_bytes(self.raw + b"// user change\n")
        with self.assertRaisesRegex(ValueError, "differs from the reviewed"):
            installer.install(self.repo, write=True)
        self.assertFalse((self.repo / installer.DESTINATION).exists())

    def test_target_edit_is_preserved(self):
        installer.install(self.repo, write=True)
        target = self.repo / installer.DESTINATION / "khu-agora-conflict-sweep.cc"
        edited = target.read_bytes() + b"// keep this\n"
        target.write_bytes(edited)
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            installer.install(self.repo, write=True)
        self.assertEqual(target.read_bytes(), edited)
        with self.assertRaisesRegex(ValueError, "Refusing to overwrite"):
            installer.install(self.repo, write=True, upgrade_owned=True)

    def test_unowned_destination_and_missing_api_are_refused(self):
        target = self.repo / installer.DESTINATION
        target.mkdir()
        with self.assertRaisesRegex(ValueError, "unowned files"):
            installer.install(self.repo, write=True)
        dependency = self.repo / "contrib/nr/model/nr-ue-rrc.h"
        dependency.write_text("missing API")
        with self.assertRaisesRegex(ValueError, "API absent"):
            installer.install(self.repo)

    def test_source_hooks_are_unique(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            installer.replace_once("hook hook", "hook", "replacement")


if __name__ == "__main__":
    unittest.main()
