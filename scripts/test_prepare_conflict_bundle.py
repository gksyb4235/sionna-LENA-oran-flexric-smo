"""CPU-v1 packaging contracts; synthetic files only, no Docker or native build."""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "prepare_conflict_bundle", Path(__file__).resolve().parents[1] / "deploy/conflict-sweep/prepare.py")
prepare = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(prepare)


class BundleTests(unittest.TestCase):
    def put(self, root, relative, data="example"):
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data)
        return path

    def fixture(self, root):
        config = dict(binary="build/example", source_files=["scratch/example.cc"],
                      gnb_positions="experiment/khu/datasets/gnbs-ret.csv",
                      ret_cache_root="experiment/khu/caches/ret_caches")
        manifest = {"jobs": [{"trace_path": "scenarios/khu-real/ue_position/seed12.csv"}]}
        bundle = root / prepare.BUNDLE
        self.put(root, prepare.BUNDLE / "config.example.json", json.dumps(config))
        self.put(root, prepare.BUNDLE / "data/execution_manifest.json", json.dumps(manifest))
        for name in ("sweep.py", "backend/cache_tools.py", "resource_limits.py"):
            self.put(bundle, name)
        for name in ("config.local.json", "data/private.json", "results/private.csv", ".env"):
            self.put(bundle, name, "must-not-export")
        for name in (config["binary"], *config["source_files"], config["gnb_positions"],
                     "scratch/khu-real-dual-nr-sionna-final.cc", "deploy/conflict-sweep/entrypoint.py",
                     "deploy/conflict-sweep/Dockerfile", "experiment/khu/datasets/ue_position/seed12.csv"):
            self.put(root, name)
        (root / "scenarios").mkdir()
        (root / "scenarios/khu-real").symlink_to("../experiment/khu/datasets")
        for tilt in (3, 5, 7, 9, 12):
            for band in ("1p8ghz", "3p5ghz"):
                self.put(root, f"{config['ret_cache_root']}/tilt_{tilt}deg/sionna_rt_cache_{band}_2x2_1x1_lzf.h5")
            for other in (3, 5, 7, 9, 12):
                if tilt != other:
                    self.put(root, f"{config['ret_cache_root']}/merged_percell/merged_1p8ghz_4g1_{tilt}deg_4g2_{other}deg.h5")
        for name in ("numpy", "numpy.libs", "numpy-1.dist-info", "h5py", "h5py.libs", "h5py-1.dist-info"):
            self.put(root, f".venv/lib/python3.12/site-packages/{name}/example.py")
        return [path.relative_to(root) for path in bundle.rglob("*") if path.is_file()]

    def test_host_paths_keep_v1_wire_layout(self):
        self.assertEqual(prepare.container_path(prepare.BUNDLE / "sweep.py"),
                         prepare.CONTAINER_BUNDLE / "sweep.py")
        self.assertEqual(str(prepare.container_path("experiment/khu/caches/ret_caches/a.h5")),
                         "scenarios/khu-real/ret_caches/a.h5")
        with self.assertRaises(ValueError):
            prepare.container_path("../private")

    def test_exports_work_before_and_after_git_index_move(self):
        for legacy_index in (True, False):
            with self.subTest(legacy_index=legacy_index), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                tracked = self.fixture(root)
                if legacy_index:
                    tracked = [prepare.CONTAINER_BUNDLE / path.relative_to(prepare.BUNDLE) for path in tracked]
                def command(args, **kwargs):
                    if args[0] == "ldd":
                        return "linux-vdso.so.1"
                    if args[:2] == ["git", "ls-files"]:
                        return "\0".join(map(str, tracked)).encode() + b"\0"
                    return "test-revision\n"
                manifest = root / prepare.BUNDLE / "data/execution_manifest.json"
                original = manifest.read_bytes()
                with patch.object(prepare, "ROOT", root), patch.object(prepare.platform, "machine", return_value="x86_64"), \
                     patch.object(prepare.subprocess, "check_output", side_effect=command), \
                     patch("sys.argv", ["prepare.py", "--output", str(root / "export")]), \
                     contextlib.redirect_stdout(io.StringIO()):
                    prepare.main()
                runtime = root / "export/context/runtime"
                output_bundle = runtime / prepare.CONTAINER_BUNDLE
                self.assertEqual((output_bundle / "data/execution_manifest.json").read_bytes(), original)
                self.assertEqual(manifest.read_bytes(), original)
                self.assertTrue((output_bundle / "backend/cache_tools.py").is_file())
                self.assertFalse((output_bundle / "config.local.json").exists())
                self.assertFalse((output_bundle / "data/private.json").exists())
                self.assertFalse((output_bundle / "results/private.csv").exists())
                self.assertFalse((output_bundle / ".env").exists())
                config = json.loads((output_bundle / "config.container.json").read_text())
                self.assertEqual(config["ret_cache_root"], "scenarios/khu-real/ret_caches")
                self.assertEqual(config["gnb_positions"], "scenarios/khu-real/gnbs-ret.csv")
                with tarfile.open(root / "export/inputs.tar") as archive:
                    names = archive.getnames()
                    self.assertEqual(len(names), 32)
                    self.assertTrue(all(name.startswith("scenarios/khu-real/") for name in names))
                    self.assertTrue(all(item.isfile() for item in archive.getmembers()))


if __name__ == "__main__":
    unittest.main()
