#!/usr/bin/env python3
"""List one allowed logical CPU per physical core for legacy parameter sweeps."""

import os
from pathlib import Path


def physical_cpus():
    selected, seen = [], set()
    for cpu in sorted(os.sched_getaffinity(0)):
        topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        try:
            key = ((topology / "physical_package_id").read_text().strip(),
                   (topology / "core_id").read_text().strip())
        except OSError:
            key = ("logical", cpu)
        if key not in seen:
            seen.add(key)
            selected.append(cpu)
    return selected


if __name__ == "__main__":
    print("\n".join(map(str, physical_cpus())))
