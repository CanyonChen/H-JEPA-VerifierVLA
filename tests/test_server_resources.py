"""Verify cgroup limits take precedence over visible host resources."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.check_server_resources import GIB, apply_cpu_affinity, resource_status


class ServerResourcesTest(unittest.TestCase):
    def test_container_limits_with_176_visible_cpus_and_20_core_quota(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            values = {"memory.current": "162945130496", "memory.max": "214748364800",
                      "cpu.max": "2000000 100000", "cpu.stat": "usage_usec 123\nthrottled_usec 9\n",
                      "memory.events": "max 5\noom 0\noom_kill 0\n",
                      "meminfo": "MemTotal: 1056422572 kB\nMemAvailable: 859704416 kB\n"}
            for name, value in values.items():
                (root / name).write_text(value, encoding="utf-8")
            with mock.patch("os.sched_getaffinity", return_value=set(range(176)), create=True):
                status = resource_status(root, root / "meminfo")
            self.assertEqual(status["cpu_quota_cores"], 20)
            self.assertEqual(status["memory_available_bytes"], 214748364800 - 162945130496)
            self.assertEqual(status["memory_limit_bytes"], 200 * GIB)
            self.assertEqual(status["memory_events"]["oom_kill"], 0)

    def test_worker_affinity_uses_only_assigned_cpu_ids(self):
        with mock.patch.dict(os.environ, {"LIBERO_CPU_IDS": "4,5"}), \
                mock.patch("os.sched_setaffinity", create=True) as set_affinity:
            apply_cpu_affinity()
            set_affinity.assert_called_once_with(0, {4, 5})


if __name__ == "__main__":
    unittest.main()
