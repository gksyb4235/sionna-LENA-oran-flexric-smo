"""kubectl execution adapter used only after explicit approval."""

from __future__ import annotations

import subprocess
from typing import Any


def execute_kubectl_commands(argvs: list[list[str]], *, timeout_seconds: int = 120) -> dict[str, Any]:
    results: list[dict[str, Any]] = []
    for argv in argvs:
        completed = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
        results.append(
            {
                "argv": argv,
                "returncode": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
    return {
        "executed": True,
        "command_count": len(argvs),
        "results": results,
        "success": all(item["returncode"] == 0 for item in results),
    }
