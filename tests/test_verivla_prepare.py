"""Offline checks for data reuse and interrupted collection. No simulators or models are started."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts import verivla_prepare as prep


class VerivlaPrepareTests(unittest.TestCase):
    def test_virtualenv_interpreter_is_not_resolved_to_its_base(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(prep, "prepare", return_value={}) as run:
            root = Path(tmp)
            alias = root / "sim-python"
            resolve = Path.resolve

            def resolve_alias(path, *args, **kwargs):
                return Path(sys.executable) if path == alias else resolve(path, *args, **kwargs)

            argv = ["prepare", "--long-root", tmp, "--data-root", tmp, "--plan", "--sim-python", str(alias)]
            with patch.object(sys, "argv", argv), patch.object(Path, "resolve", resolve_alias), patch.dict(os.environ), patch("builtins.print"):
                prep.main()
            self.assertEqual(run.call_args.args[4], alias)

    def test_plan_and_missing_long_never_collect(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(prep, "collect_and_index") as collect:
            root = Path(tmp)
            args = (root / "data", root / "long", 2, None, root / "sim")
            status = prep.prepare(*args, plan=True)
            self.assertEqual(list(status), list(prep.SUITES))
            self.assertTrue(all(r["status"] == "missing split" for r in status.values()))
            self.assertFalse(any(root.iterdir()))
            with self.assertRaisesRegex(ValueError, "existing Long"):
                prep.prepare(*args)
            collect.assert_not_called()

    def test_already_complete_data_reused_and_cross_suite_provenance_checked(self):
        ready = ({"status": "ready"}, "a" * 64, {"model": {"model_id": "ref"}})
        args = (Path("datasets"), Path("long"), 2, None, Path("unused"))
        with patch.object(prep, "collect_and_index") as collect:
            with patch.object(prep, "inspect_suite", return_value=ready):
                self.assertEqual(len(prep.prepare(*args)), 4)
            with patch.object(prep, "inspect_suite", side_effect=[ready, (ready[0], "b"*64, ready[2])]):
                with self.assertRaisesRegex(ValueError, "pi05 checkpoint"):
                    prep.prepare(*args)
            with patch.object(prep, "inspect_suite", side_effect=[ready, (ready[0], ready[1], {"model": "other"})]):
                with self.assertRaisesRegex(ValueError, "Robometer"):
                    prep.prepare(*args)
            collect.assert_not_called()

    def test_paths_and_h100_relocation(self):
        data, old = Path("/new/data"), Path("/new/long")
        paths = prep.suite_paths(data, old)
        self.assertEqual(paths["libero_10"], (old/"split_seed7", old/"robometer_raw_trainval_seed7_gpu0"))
        self.assertEqual(paths["libero_goal"], (data/"libero_goal/split", data/"libero_goal/labels"))
        new = Path("installed")
        self.assertEqual(prep.artifact_path("/old/repo/outputs/a", "/old/repo", new), new/"outputs/a")
        self.assertEqual(prep.artifact_path("/old/repo-other/a", "/old/repo", new), Path("/old/repo-other/a"))

    def test_collection_resumes_on_requested_physical_gpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = "libero_object"
            base = root / suite
            collection = base / "collection"
            ckpt = root / "ckpt"
            old = collection / "pi05_libero_object_gpu3_seed7_20260101T010101_000000Z"
            old.mkdir(parents=True)
            cfg = dict(suite=suite, task_ids=list(range(10)), trials_per_task=50, seed=7, checkpoint=str(ckpt))
            (old/"run_config.json").write_text(json.dumps(cfg))
            (old/"summary.json").write_text(json.dumps({"status": "FAIL", "valid_episodes": 3}))
            new = collection / "pi05_libero_object_gpu2_seed7_20260102T010101_000000Z"

            def simulate(cmd, **kw):
                if "scripts.run_libero_parallel" not in cmd:
                    self.assertEqual(cmd[cmd.index("--run-dir")+1], str(new))
                    return
                self.assertEqual(cmd[cmd.index("--resume-from")+1], str(old))
                self.assertEqual(cmd[cmd.index("--gpu")+1], "2")
                self.assertEqual(cmd[cmd.index("--egl-device")+1], "2")
                self.assertEqual(kw["env"]["CUDA_VISIBLE_DEVICES"], "2")
                self.assertNotIn("B5_SCHEDULER_GPU_UUID", kw["env"])
                new.mkdir()
                (new/"run_config.json").write_text(json.dumps(cfg))
                (new/"summary.json").write_text(json.dumps({"status":"PASS","valid_episodes":500}))
                (new/"episodes.jsonl").write_text("{}\n"*500)

            with patch.object(prep, "pi05_checkpoint", return_value=ckpt), patch.dict(os.environ, {"B5_SCHEDULER_GPU_UUID":"stale"}):
                with patch.object(prep.subprocess, "run", side_effect=simulate) as launch:
                    prep.collect_and_index(suite, base, root/"sim", 2)
                    self.assertEqual(launch.call_count, 2)
                    launch.reset_mock()
                    prep.collect_and_index(suite, base, root/"sim", 2)
                    self.assertEqual(launch.call_count, 1)


if __name__ == "__main__":
    unittest.main()
