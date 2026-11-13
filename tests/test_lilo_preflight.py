"""Verify device assignment and report collection before the policy smoke run."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

from scripts import check_lilo_env


class LiloPreflightTest(unittest.TestCase):
    def test_nine_scenes_cover_both_egl_devices_with_existing_python(self):
        calls = []

        def child(command, **kwargs):
            calls.append(command)
            task = int(command[command.index("--task-id") + 1])
            device = int(command[command.index("--egl-device") + 1])
            output = Path(command[command.index("--output-root") + 1]) / "task{:02d}".format(task)
            output.mkdir()
            (output / "report.json").write_text(json.dumps({"status": "PASS", "task_id": task,
                                                           "egl_device": device}), encoding="utf-8")
            return SimpleNamespace(returncode=0)

        with tempfile.TemporaryDirectory() as temporary, \
                mock.patch.object(check_lilo_env.subprocess, "run", side_effect=child), \
                contextlib.redirect_stdout(io.StringIO()):
            result = check_lilo_env.main(["--egl-devices", "0", "1", "--output-root", temporary])
            directory, = Path(temporary).iterdir()
            report = json.loads((directory / "report.json").read_text())
        self.assertEqual(result, 0)
        self.assertEqual(report["egl_devices"], [0, 1])
        self.assertEqual(report["simulation_environment"], ".venv-libero")
        self.assertEqual([r["task_id"] for r in report["tasks"]], list(range(9)))
        self.assertEqual([r["egl_device"] for r in report["tasks"]], [0, 1, 0, 1, 0, 1, 0, 1, 0])
        self.assertTrue(all(command[0] == check_lilo_env.sys.executable for command in calls))


if __name__ == "__main__":
    unittest.main()
