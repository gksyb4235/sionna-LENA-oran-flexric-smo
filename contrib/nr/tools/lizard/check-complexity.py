#!/usr/bin/env python3

# Copyright (c) 2026 Centre Tecnologic de Telecomunicacions de Catalunya (CTTC)
#
# SPDX-License-Identifier: GPL-2.0-only

"""Cyclomatic-complexity and duplicate-block gate for the nr module.

This script wraps `lizard <https://github.com/terryyin/lizard>`_ and is meant
to be run both locally and from CI (see the ``complexity`` job in
``.gitlab-ci-code-linting.yml``).

Rather than enforcing a fixed cap or relying on a hand-maintained list of
"magic" baseline numbers, the gate computes the baseline *on the fly* from a
fresh copy of the nr ``master`` branch. It then measures the same metrics on
the current working tree and fails when a metric got *worse* (a higher maximum
cyclomatic complexity, or more duplicate code blocks) than on ``master``.

This way the budget always tracks the upstream tip automatically: no file to
re-baseline, no value to keep in sync. A change is allowed as long as it does
not make the code more complex than ``master`` already is; reducing complexity
relative to ``master`` is, of course, always fine.

Two metrics are gated:

* ``max_ccn``      -- the highest cyclomatic complexity number (CCN) of any
                      single function across the analysed sources.
* ``duplicates``   -- the number of duplicate code blocks reported by lizard's
                      ``-Eduplicate`` extension.

By default the master copy is fetched with a shallow ``git clone`` into a
temporary directory that is removed when the script exits. CI clones it once
and passes it explicitly with ``--baseline-dir`` to avoid a second clone.

Usage (run from the root of the nr module)::

    ./tools/lizard/check-complexity.py                    # clone master, gate
    ./tools/lizard/check-complexity.py --baseline-dir DIR # gate against DIR
    ./tools/lizard/check-complexity.py --show             # current metrics only

Options of note::

    --baseline-ref REF   git ref to compare against (default: master)
    --baseline-url URL   git URL to clone (default: cttc-lena/nr)
    --baseline-dir DIR   use an existing checkout instead of cloning
"""

import argparse
import csv
import io
import subprocess
import sys
import tempfile
from pathlib import Path

# Directories with nr source code that we analyse. These are relative to the
# root of the nr module (the directory that contains model/, helper/, ...).
SOURCE_DIRS = ["model", "helper", "utils"]

# Third-party single-header libraries vendored under utils/. They are not nr
# code and carry enormous (and irrelevant) complexity, so they are excluded
# from every measurement. Patterns are matched by lizard's -x/--exclude option
# (shell-style globs).
EXCLUDE_GLOBS = [
    "*nr-json.hpp",  # nlohmann/json, vendored
    "*termcolor.hpp",  # termcolor, vendored
]

# Language restriction: only analyse C/C++ sources.
LANGUAGE = "cpp"

# Defaults describing the upstream branch used as the baseline. These mirror
# the CTTC upstream the rest of the CI uses (see .gitlab-ci-code-linting.yml).
DEFAULT_BASELINE_URL = "https://gitlab.com/cttc-lena/nr.git"
DEFAULT_BASELINE_REF = "master"

# Keys for the metrics we gate.
KEY_MAX_CCN = "max_ccn"
KEY_DUPLICATES = "duplicates"
METRIC_KEYS = (KEY_MAX_CCN, KEY_DUPLICATES)


def lizard_base_cmd() -> list:
    """Build the common lizard command-line prefix (language + exclusions).

    @return List of command-line tokens shared by all lizard invocations.
    """
    cmd = ["lizard", "-l", LANGUAGE]
    for glob in EXCLUDE_GLOBS:
        cmd += ["-x", glob]
    return cmd


def run_lizard(extra_args: list, root: Path) -> str:
    """Run lizard over the source dirs found under ``root`` and return stdout.

    Note: lizard returns a non-zero exit code when it emits warnings, so the
    return code is intentionally *not* checked here; callers parse stdout.

    @param extra_args Extra lizard arguments (e.g. --csv or -Eduplicate).
    @param root Directory whose SOURCE_DIRS subfolders are analysed.
    @return Captured standard output as text.
    """
    dirs = [str(root / d) for d in SOURCE_DIRS if (root / d).is_dir()]
    if not dirs:
        sys.exit(f"error: none of {SOURCE_DIRS} found under '{root}'.")
    cmd = lizard_base_cmd() + extra_args + dirs
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        sys.exit("error: 'lizard' was not found on PATH. Install it with " "'pip install lizard'.")
    return result.stdout


def measure_max_ccn(root: Path) -> int:
    """Compute the maximum cyclomatic complexity over the sources under root.

    Uses lizard's CSV output, whose second column is the per-function CCN.

    @param root Directory whose SOURCE_DIRS subfolders are analysed.
    @return The highest CCN found, or 0 when no function was analysed.
    """
    output = run_lizard(["--csv"], root)
    max_ccn = 0
    reader = csv.reader(io.StringIO(output))
    for row in reader:
        # CSV columns: nloc, CCN, token, param, length, location, ...
        if len(row) < 2:
            continue
        try:
            ccn = int(row[1])
        except ValueError:
            continue
        max_ccn = max(max_ccn, ccn)
    return max_ccn


def measure_duplicates(root: Path) -> int:
    """Count duplicate code blocks reported by lizard's duplicate extension.

    @param root Directory whose SOURCE_DIRS subfolders are analysed.
    @return Number of "Duplicate block:" entries lizard reports.
    """
    output = run_lizard(["-Eduplicate"], root)
    return sum(1 for line in output.splitlines() if line.strip() == "Duplicate block:")


def measure(root: Path) -> dict:
    """Measure both gated metrics for the sources under ``root``.

    @param root Directory whose SOURCE_DIRS subfolders are analysed.
    @return Dict mapping metric name to its current integer value.
    """
    return {
        KEY_MAX_CCN: measure_max_ccn(root),
        KEY_DUPLICATES: measure_duplicates(root),
    }


def module_root() -> Path:
    """Return the root of the nr module (two levels up from this script).

    @return Absolute path of the nr module root (contains model/, helper/).
    """
    return Path(__file__).resolve().parent.parent.parent


def clone_baseline(url: str, ref: str, dest: Path) -> None:
    """Shallow-clone a single ref of the baseline repository.

    @param url Git URL to clone from.
    @param ref Branch or tag to clone.
    @param dest Destination directory (must not yet exist or be empty).
    """
    cmd = ["git", "clone", "--depth=1", "--branch", ref, url, str(dest)]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        sys.exit("error: 'git' was not found on PATH.")
    if result.returncode != 0:
        sys.exit(f"error: failed to clone baseline '{ref}' from '{url}':\n{result.stdout}")


def main() -> int:
    """Entry point.

    @return Process exit code (0 on success, 1 when a metric regressed).
    """
    parser = argparse.ArgumentParser(
        description="Gate nr cyclomatic complexity and duplicate blocks "
        "against a fresh copy of the nr master branch using lizard.",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="print the current working-tree metrics and exit (no baseline, no gate)",
    )
    parser.add_argument(
        "--baseline-ref",
        default=DEFAULT_BASELINE_REF,
        help=f"git ref to use as the baseline (default: {DEFAULT_BASELINE_REF})",
    )
    parser.add_argument(
        "--baseline-url",
        default=DEFAULT_BASELINE_URL,
        help=f"git URL to clone the baseline from (default: {DEFAULT_BASELINE_URL})",
    )
    parser.add_argument(
        "--baseline-dir",
        default=None,
        help="path to an existing baseline checkout; if given, no clone is done",
    )
    args = parser.parse_args()

    root = module_root()
    current = measure(root)

    if args.show:
        for key in METRIC_KEYS:
            print(f"{key}={current[key]}")
        return 0

    # Obtain a baseline checkout: either the one provided, or a fresh shallow
    # clone of master into a temporary directory that is cleaned up on exit.
    if args.baseline_dir:
        baseline_root = Path(args.baseline_dir).resolve()
        if not baseline_root.is_dir():
            sys.exit(f"error: --baseline-dir '{baseline_root}' is not a directory.")
        baseline = measure(baseline_root)
        baseline_desc = str(baseline_root)
    else:
        with tempfile.TemporaryDirectory(prefix="nr-lizard-baseline-") as tmp:
            dest = Path(tmp) / "nr-baseline"
            print(f"Cloning baseline '{args.baseline_ref}' from {args.baseline_url} ...")
            clone_baseline(args.baseline_url, args.baseline_ref, dest)
            baseline = measure(dest)
        baseline_desc = f"{args.baseline_url}@{args.baseline_ref}"

    print(f"Baseline: {baseline_desc}")
    failed = False
    for key in METRIC_KEYS:
        cur = current[key]
        base = baseline[key]
        status = "OK" if cur <= base else "REGRESSION"
        print(f"{key}: current={cur} baseline={base} -> {status}")
        if cur > base:
            failed = True

    if failed:
        print()
        print(
            "error: code complexity increased relative to the baseline branch.\n"
            "Reduce the cyclomatic complexity / duplication of the offending\n"
            "functions so they are no worse than the baseline."
        )
        return 1

    print("\nComplexity gate passed: no regression against the baseline branch.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
