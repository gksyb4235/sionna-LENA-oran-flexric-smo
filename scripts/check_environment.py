#!/usr/bin/env python3
"""Check the host and Python dependencies used by the KHU scenario."""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import shutil
import sys


REQUIRED_PACKAGES = {
    "cppyy": "3.5.0",
    "drjit": "1.3.1",
    "matplotlib": "3.11.1",
    "mitsuba": "3.8.0",
    "numpy": "2.5.1",
    "omegaconf": "2.3.1",
    "polyscope": "2.6.1",
    "pybind11": "2.11.1",
    "PyYAML": "6.0.3",
    "pyzmq": "27.1.0",
    "scipy": "1.17.1",
    "sionna-rt": "2.0.1",
}


def status(ok: bool, message: str) -> None:
    print(f"[{'OK' if ok else 'FAIL'}] {message}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-gpu",
        action="store_true",
        help="fail if the Dr.Jit CUDA backend is unavailable",
    )
    parser.add_argument(
        "--require-e2",
        action="store_true",
        help="fail if the installed e2sim headers/library are unavailable",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    failures = 0

    print(f"Repository: {repo_root}")
    print(f"OS: {platform.platform()}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")

    in_venv = sys.prefix != sys.base_prefix
    status(in_venv, "project virtual environment is active")
    failures += not in_venv

    expected_python = sys.version_info[:2] == (3, 12)
    status(expected_python, "Python 3.12 is in use")
    failures += not expected_python

    for package, expected in REQUIRED_PACKAGES.items():
        try:
            actual = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            status(False, f"{package} is missing (expected {expected})")
            failures += 1
            continue
        matches = actual == expected
        status(matches, f"{package}=={actual} (expected {expected})")
        failures += not matches

    scene = repo_root / "scenes/khu-real/KHU_Cropped_Sionna_RT.xml"
    status(scene.is_file(), f"KHU scene exists: {scene.relative_to(repo_root)}")
    failures += not scene.is_file()

    e2_paths = (
        Path("/usr/local/include/e2sim/e2sim.hpp"),
        Path("/usr/local/lib/libe2sim.a"),
    )
    e2_ok = all(path.exists() for path in e2_paths)
    if e2_ok or args.require_e2:
        status(e2_ok, "e2sim development headers and static library are installed")
    else:
        print("[WARN] e2sim is not installed; the E2-enabled ns-3 build will be unavailable")
    failures += args.require_e2 and not e2_ok

    flexric_root_value = os.environ.get("FLEXRIC_ROOT")
    if flexric_root_value:
        ric = Path(flexric_root_value) / "build/examples/ric/nearRT-RIC"
        status(ric.is_file(), f"nearRT-RIC binary exists: {ric}")
        failures += not ric.is_file()
    else:
        print("[INFO] FLEXRIC_ROOT is unset; skipping the nearRT-RIC binary check")

    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        print(f"[OK] nvidia-smi is available: {nvidia_smi}")
    else:
        print("[WARN] nvidia-smi is unavailable inside WSL")

    cuda_ok = False
    cuda_error: Exception | None = None
    dr = None
    mitsuba = None
    try:
        dr = importlib.import_module("drjit")
        mitsuba = importlib.import_module("mitsuba")
        mitsuba.set_variant("cuda_ad_mono_polarized")
        probe = mitsuba.Float(1.0)
        dr.eval(probe)
        dr.sync_thread()
        cuda_ok = bool(dr.has_backend(dr.JitBackend.CUDA))
    except Exception as error:  # pragma: no cover - depends on host GPU
        cuda_error = error
    if cuda_ok:
        print("[OK] Dr.Jit CUDA backend is available")
    elif args.require_gpu:
        status(False, "Dr.Jit CUDA backend is unavailable")
        if cuda_error:
            print(f"       {cuda_error}")
        failures += 1
    else:
        print("[WARN] Dr.Jit CUDA backend is unavailable; Sionna RT will use the CPU")

    if not cuda_ok and not args.require_gpu:
        llvm_ok = False
        llvm_error: Exception | None = None
        try:
            if dr is None:
                dr = importlib.import_module("drjit")
            if mitsuba is None:
                mitsuba = importlib.import_module("mitsuba")
            mitsuba.set_variant("llvm_ad_mono_polarized")
            probe = mitsuba.Float(1.0)
            dr.eval(probe)
            dr.sync_thread()
            llvm_ok = bool(dr.has_backend(dr.JitBackend.LLVM))
        except Exception as error:  # pragma: no cover - depends on host LLVM
            llvm_error = error
        status(llvm_ok, "Dr.Jit LLVM CPU backend is available")
        if llvm_error:
            print(f"       {llvm_error}")
        failures += not llvm_ok

    try:
        if mitsuba is None:
            mitsuba = importlib.import_module("mitsuba")
        cuda_variants = [name for name in mitsuba.variants() if name.startswith("cuda_")]
        status(bool(cuda_variants), f"Mitsuba CUDA variants: {', '.join(cuda_variants)}")
        failures += not bool(cuda_variants)
    except Exception as error:  # pragma: no cover - diagnostic path
        status(False, f"Mitsuba import/variant check failed: {error}")
        failures += 1

    print(f"Result: {'PASS' if failures == 0 else f'FAIL ({failures} issue(s))'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
