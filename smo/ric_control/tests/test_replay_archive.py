"""Replay transfers must validate all input before changing local files."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile
from scripts import agora_replay


class ReplayArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.destination = self.root / "destination"
        for variant in agora_replay.VARIANTS:
            content = (variant + "-observations").encode()
            manifest = {
                "files": [
                    {
                        "file": "runs/gnn_0001.npz",
                        "bytes": len(content),
                        "sha256": hashlib.sha256(content).hexdigest(),
                    }
                ]
            }
            for runtime in (self.source, self.destination):
                directory = runtime / variant
                directory.mkdir(parents=True)
                (directory / "manifest.json").write_text(json.dumps(manifest))
            run = self.source / variant / "runs/gnn_0001.npz"
            run.parent.mkdir()
            run.write_bytes(content)

    def test_roundtrip_and_repeat_import(self):
        archive = self.root / "replay.zip"
        self.assertEqual(agora_replay.export_archive(archive, self.source)["files"], 2)
        self.assertEqual(agora_replay.import_archive(archive, self.destination)["restored"], 2)
        self.assertEqual(agora_replay.import_archive(archive, self.destination)["restored"], 0)
        self.assertEqual(agora_replay.verify(self.source), agora_replay.verify(self.destination))

    def test_one_corrupt_member_prevents_all_writes(self):
        archive = self.root / "corrupt.zip"
        rows = agora_replay.inventory(self.source)
        with zipfile.ZipFile(archive, "w") as bundle:
            for index, (name, row) in enumerate(rows.items()):
                bundle.writestr(name, row["path"].read_bytes() if index == 0 else b"invalid")
        with self.assertRaises(ValueError):
            agora_replay.import_archive(archive, self.destination)
        self.assertFalse(any(self.destination.rglob("*.npz")))

    def test_unexpected_archive_path_is_rejected(self):
        archive = self.root / "extra.zip"
        agora_replay.export_archive(archive, self.source)
        with zipfile.ZipFile(archive, "a") as bundle:
            bundle.writestr("../outside.npz", b"not allowed")
        with self.assertRaises(ValueError):
            agora_replay.import_archive(archive, self.destination)
        self.assertFalse(any(self.destination.rglob("*.npz")))

    def test_missing_data_explains_how_to_restore(self):
        with self.assertRaisesRegex(ValueError, "agora_replay.py import"):
            agora_replay.verify(self.destination)
