"""Exercise the public workflow without starting a model or the simulator."""
import contextlib
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from scripts import train_verifier as entry


class VerifierLauncherTests(unittest.TestCase):
    def test_dry_run_only_inspects_data_and_creates_no_job(self):
        with tempfile.TemporaryDirectory() as directory:
            job = Path(directory) / "job with spaces"
            with patch.object(entry.subprocess, "run") as run, contextlib.redirect_stdout(io.StringIO()):
                entry.main(["--dry-run", "--cuda=2", "--job-dir", str(job)])
            run.assert_called_once()
            cmd = run.call_args.args[0]
            self.assertIn("scripts.verivla_prepare", cmd)
            self.assertIn("--plan", cmd)
            self.assertEqual(run.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "2")
            self.assertFalse(job.exists())

    def test_incomplete_data_stops_pipeline_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            weights = root / "existing.pt"
            weights.touch()
            with patch.object(entry.subprocess, "run", side_effect=subprocess.CalledProcessError(2, "data")) as run:
                with self.assertRaises(subprocess.CalledProcessError), contextlib.redirect_stdout(io.StringIO()):
                    entry.main(["--job-dir", str(root / "job"), "--checkpoint", str(weights), "--gpu=0"])
            run.assert_called_once()
            self.assertIn("scripts.verivla_prepare", run.call_args.args[0])

    def test_stage_contract_and_resume_policies(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1,3"}):
            job = Path(directory)
            args = entry.arguments(["--job-dir", str(job), "--epochs=2"])
            self.assertEqual(args.gpu, 1)
            cmds = entry.commands(args, job / "vjepa.pt")
            self.assertEqual([c[c.index("-m") + 1] for c in cmds], [
                "scripts.verivla_prepare", "scripts.b5_model", "scripts.b5_features",
                "scripts.b5_train", "scripts.verivla_evaluate",
            ])
            for c in cmds[2:4]:
                self.assertIn("joint40", c)
            self.assertEqual(cmds[3][cmds[3].index("--epochs") + 1], "2")
            self.assertEqual(cmds[4][0], str(entry.ROOT / ".venv-libero/bin/python"))
            entry.check_resume("auto", job)
            with self.assertRaises(ValueError):
                entry.check_resume("required", job)
            (job / "last.pt").touch()
            entry.check_resume("required", job)
            with self.assertRaises(ValueError):
                entry.check_resume("never", job)


if __name__ == "__main__":
    unittest.main()
