"""Catch split leakage, executed-action/target mismatch, and partial publication."""

from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.audit_verifier_data import read_rows
from scripts.prepare_verifier_long_data import assign_splits, json_text, main, prepare_dataset


def fixture(base):
    run = base / "run"
    run.mkdir()
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{
        "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
        "observation/state": np.zeros(8), "prompt": "pick the bowl"})
    rows = []
    for initial in range(50):
        source = base / "earlier" if initial < 29 else run
        directory = source / "episodes" / str(initial)
        directory.mkdir(parents=True)
        row = dict(task_id=0, init_state_id=initial, outcome="timeout" if initial % 10 == 0 else "success",
                   task_description="pick the bowl", policy_steps=7, policy_requests=2,
                   artifacts=os.path.relpath(directory, run))
        rows.append(row)
        (directory / "result.json").write_text(json_text(row), encoding="utf-8")
        decisions = []
        for request, step, count in ((1, 0, 5), (2, 5, 2)):
            name = "observation_{}.npz".format(request)
            (directory / name).write_bytes(buffer.getvalue())
            decisions.append(dict(policy_step=step, executed_steps=count, response_received=True,
                                  actions=(np.arange(70).reshape(10, 7) + request * 100).tolist(), observation=name))
        (directory / "decisions.jsonl").write_text("".join(json_text(d) for d in decisions), encoding="utf-8")
        (directory / "video.mp4").write_bytes(b"video fixture; the preceding audit owns decoding")
    (run / "run_config.json").write_text(json_text(dict(suite="libero_10", task_ids=[0], trials_per_task=50)),
                                         encoding="utf-8")
    (run / "summary.json").write_text(json_text(dict(status="PASS", planned_episodes=50, recorded_episodes=50,
                                                     valid_episodes=50, successes=45, timeouts=5,
                                                     policy_steps=350, policy_requests=100)), encoding="utf-8")
    (run / "episodes.jsonl").write_text("".join(json_text(row) for row in rows), encoding="utf-8")
    return run, rows


class VerifierLongDataTest(unittest.TestCase):
    def test_group_assignment_is_stable_and_outcome_independent(self):
        rows = [dict(task_id=task, init_state_id=initial, outcome="success")
                for task in range(10) for initial in range(50)]
        assigned = assign_splits(rows, 7)
        reordered = [dict(row, outcome="timeout") for row in reversed(rows)]
        self.assertEqual(assigned, assign_splits(reordered, 7))
        self.assertNotEqual(assigned, assign_splits(rows, 8))
        for task in range(10):
            for split, expected in (("train", 40), ("validation", 5), ("test", 5)):
                self.assertEqual(sum(assigned[(task, initial)] == split for initial in range(50)), expected)

    def test_duplicate_or_missing_initial_state_is_rejected(self):
        rows = [dict(task_id=0, init_state_id=initial) for initial in range(50)]
        with self.assertRaisesRegex(ValueError, "Duplicate task/initial-state"):
            assign_splits(rows + [rows[0]], 7)
        with self.assertRaisesRegex(ValueError, "initial states 0..49"):
            assign_splits(rows[:-1], 7)

    def test_cli_keeps_resumed_sources_short_windows_and_identical_reruns(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run, _ = fixture(base)
            output = base / "prepared"
            before = {p: p.read_bytes() for p in base.rglob("*") if p.is_file()}
            args = ["--run-dir", str(run), "--output-dir", str(output), "--split-seed", "7"]
            stdout = io.StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(main(args), 0)
            report = json.loads(stdout.getvalue())
            self.assertEqual(report["status"], "COMPLETE")
            manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["retained_artifact_episodes"], 29)
            self.assertEqual(manifest["totals"]["full_five_step_prefixes"], 50)
            self.assertEqual(manifest["totals"]["short_prefixes"], 50)
            seen = set()
            for split, count in (("train", 40), ("validation", 5), ("test", 5)):
                episodes = read_rows(output / (split + "_episodes.jsonl"))
                windows = read_rows(output / (split + "_windows.jsonl"))
                self.assertEqual(len(episodes), count)
                ids = {ep["episode_id"] for ep in episodes}
                self.assertTrue(seen.isdisjoint(ids))
                seen.update(ids)
                self.assertEqual(len(windows), count * 2)
                for window in windows:
                    self.assertIn(window["episode_id"], ids)
                    self.assertEqual(window["split"], split)
                    self.assertTrue(Path(window["observation"]).is_file())
                    self.assertTrue(Path(window["video"]).is_file())
                    count = window["executed_steps"]
                    step = window["policy_step"]
                    self.assertEqual(window["context_frame_index"], step)
                    self.assertEqual(window["target_frame_indices"], list(range(step + 1, step + count + 1)))
                    self.assertEqual(window["valid_subchunk_mask"], [True] * count + [False] * (5 - count))
                    request = window["decision_index"] + 1
                    self.assertEqual(window["executed_actions"],
                                     (np.arange(70).reshape(10, 7) + request * 100)[:count].tolist())
            self.assertEqual(len(seen), 50)
            self.assertEqual(before, {p: p.read_bytes() for p in before})
            prepared = {p: p.read_bytes() for p in output.iterdir()}
            with redirect_stdout(io.StringIO()):
                self.assertEqual(main(args), 0)
            self.assertEqual(prepared, {p: p.read_bytes() for p in output.iterdir()})
            with self.assertRaisesRegex(ValueError, "different or incomplete split"):
                prepare_dataset(run, output, split_seed=8)
            self.assertEqual(prepared, {p: p.read_bytes() for p in output.iterdir()})

    def test_invalid_execution_has_no_published_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run, rows = fixture(base)
            path = (run / rows[0]["artifacts"]) / "decisions.jsonl"
            decisions = read_rows(path)
            decisions[0]["executed_steps"] = 10
            path.write_text("".join(json_text(d) for d in decisions), encoding="utf-8")
            output = base / "prepared"
            with self.assertRaisesRegex(ValueError, "prefix of 1..5"):
                prepare_dataset(run, output)
            self.assertFalse(output.exists())

    def test_failed_write_is_clean_and_retry_publishes_complete_dataset(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run, _ = fixture(base)
            output = base / "prepared"
            original_write = Path.write_bytes
            def interrupted_write(path, data):
                if path.name == "validation_episodes.jsonl":
                    raise OSError("simulated disk write failure")
                return original_write(path, data)
            with patch.object(Path, "write_bytes", interrupted_write):
                with self.assertRaisesRegex(OSError, "simulated disk"):
                    prepare_dataset(run, output)
            self.assertFalse(output.exists())
            self.assertEqual(list(base.glob(".verifier-long-*")), [])
            manifest = prepare_dataset(run, output)
            self.assertEqual(manifest["status"], "COMPLETE")
            for split in manifest["splits"].values():
                for name in split["files"].values():
                    self.assertTrue((output / name).is_file())


if __name__ == "__main__":
    unittest.main()
