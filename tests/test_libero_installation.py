"""Run with the LIBERO environment to check the installed editable package."""

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class LiberoInstallationTest(unittest.TestCase):
    def test_installed_package_without_source_path_injection(self):
        benchmark_root = PROJECT_ROOT / "openpi/third_party/libero/libero/libero"
        with tempfile.TemporaryDirectory() as temporary_dir:
            config_dir = Path(temporary_dir)
            paths = {
                "benchmark_root": str(benchmark_root),
                "bddl_files": str(benchmark_root / "bddl_files"),
                "init_states": str(benchmark_root / "init_files"),
                "datasets": temporary_dir,
                "assets": str(benchmark_root / "assets"),
            }
            (config_dir / "config.yaml").write_text(json.dumps(paths), encoding="utf-8")
            code = """
from pathlib import Path
import sys
import libero
import openpi_client
import numpy as np
from libero.libero import benchmark

assert Path(libero.__file__).resolve() == Path(sys.argv[1]).resolve()
suite = benchmark.get_benchmark_dict()["libero_10"]()
for task_id in range(suite.n_tasks):
    states = np.asarray(suite.get_task_init_states(task_id))
    assert states.shape[0] == 50 and np.isfinite(states).all()
    assert Path(suite.get_task_bddl_file_path(task_id)).is_file()
print("PASS: installed LIBERO imports and 500 initial states without source path injection")
"""
            result = subprocess.run(
                [sys.executable, "-I", "-c", code, str(benchmark_root.parent / "__init__.py")],
                cwd=temporary_dir,
                env=dict(os.environ, LIBERO_CONFIG_PATH=temporary_dir),
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
