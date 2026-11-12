"""Read the server's cgroup v2 memory and CPU budget before LIBERO evaluation."""

import argparse
import json
import os
from pathlib import Path
import time


GIB = 1024 ** 3


def resource_status(cgroup=Path("/sys/fs/cgroup"), meminfo=Path("/proc/meminfo")):
    """Read the namespace-root cgroup mounted in this server container."""
    current = int((cgroup / "memory.current").read_text())
    maximum = (cgroup / "memory.max").read_text().strip()
    memory_limit = None if maximum == "max" else int(maximum)
    host_available = next(int(line.split()[1]) * 1024 for line in meminfo.read_text().splitlines()
                          if line.startswith("MemAvailable:"))
    available = host_available if memory_limit is None else min(host_available, memory_limit - current)
    quota, period = (cgroup / "cpu.max").read_text().split()
    cpu_ids = sorted(os.sched_getaffinity(0))
    cpu_cores = len(cpu_ids) if quota == "max" else min(len(cpu_ids), int(quota) / int(period))
    cpu_stat = dict(line.split() for line in (cgroup / "cpu.stat").read_text().splitlines())
    events = {key: int(value) for key, value in
              (line.split() for line in (cgroup / "memory.events").read_text().splitlines())}
    return {"memory_current_bytes": current, "memory_limit_bytes": memory_limit,
            "memory_available_bytes": max(0, available), "memory_events": events,
            "cpu_quota_cores": cpu_cores, "cpu_ids": cpu_ids,
            "cpu_usage_usec": int(cpu_stat["usage_usec"]),
            "cpu_throttled_usec": int(cpu_stat.get("throttled_usec", 0))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-seconds", type=float, default=2)
    args = parser.parse_args()
    if not 0 < args.sample_seconds <= 30:
        parser.error("sample-seconds must be between 0 and 30")
    before = resource_status()
    started = time.monotonic()
    time.sleep(args.sample_seconds)
    after = resource_status()
    elapsed = time.monotonic() - started
    after["sampled_cpu_cores_used"] = (after["cpu_usage_usec"] - before["cpu_usage_usec"]) / (elapsed * 1e6)
    after["memory_available_gib"] = after["memory_available_bytes"] / GIB
    after["memory_current_gib"] = after["memory_current_bytes"] / GIB
    after["visible_cpu_count"] = len(after.pop("cpu_ids"))
    print(json.dumps(after, indent=2))


if __name__ == "__main__":
    main()
