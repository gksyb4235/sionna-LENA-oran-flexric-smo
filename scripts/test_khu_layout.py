"""Check lossless relocation and canonical experiment dispatch without ns-3."""
import importlib.util
from pathlib import Path
import tempfile
import unittest


def module(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


class LayoutTests(unittest.TestCase):
    def test_experiments_is_moved_without_recreating_root_alias(self):
        migration = module("reorganize_khu")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "experiments/example"
            source.mkdir(parents=True)
            (source / "manifest.json").write_bytes(b'{"unchanged": true}\r\n')
            migration.migrate(root, True)
            migration.migrate(root, True)
            old = root / "experiments"
            self.assertFalse(old.exists() or old.is_symlink())
            self.assertEqual((root / "experiment/khu/studies/example/manifest.json").read_bytes(),
                             b'{"unchanged": true}\r\n')

    def test_only_obsolete_link_is_removed_and_target_is_preserved(self):
        migration = module("reorganize_khu")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "experiment/khu/studies"
            target.mkdir(parents=True)
            (target / "data").write_bytes(b"keep")
            alias = root / "experiments"
            alias.symlink_to("experiment/khu/studies", target_is_directory=True)
            migration.relocate(root, Path("experiments"), migration.KHU / "studies",
                               False, keep_link=False)
            self.assertTrue(alias.is_symlink())
            migration.migrate(root, True)
            migration.migrate(root, True)
            self.assertFalse(alias.exists() or alias.is_symlink())
            self.assertEqual((target / "data").read_bytes(), b"keep")

    def test_move_nested_under_legacy_alias_preserves_bytes_and_is_idempotent(self):
        migration = module("reorganize_khu")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "scenarios/khu-real"
            (data / "ret_caches").mkdir(parents=True)
            (data / "ret_caches/cache.h5").write_bytes(b"unchanged cache bytes")
            (data / "tools").mkdir()
            (data / "tools/run.sh").write_text("echo unchanged\n")
            migration.migrate(root, False)
            self.assertFalse((root / "experiment").exists())
            migration.migrate(root, True)
            migration.migrate(root, True)
            self.assertEqual((data / "ret_caches/cache.h5").read_bytes(), b"unchanged cache bytes")
            self.assertEqual((data / "tools/run.sh").resolve(), root / "experiment/khu/tools/run.sh")
            self.assertTrue((root / "scenarios/khu-real").is_symlink())

    def test_move_refuses_collision(self):
        migration = module("reorganize_khu")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "old").write_text("old")
            (root / "new").write_text("new")
            with self.assertRaises(ValueError):
                migration.relocate(root, Path("old"), Path("new"), True)
            self.assertEqual((root / "old").read_text(), "old")

    def test_distinct_fixed_and_conflict_launchers(self):
        runner = module("run_experiments")
        self.assertIn("run_gnn_dataset_sweep.sh", runner.command("fixed", ["501", "512", "1", "12"])[1])
        self.assertIn("run_conflict_dataset_sweep.sh", runner.command("conflict", ["501", "512", "1", "12"])[1])


if __name__ == "__main__":
    unittest.main()
