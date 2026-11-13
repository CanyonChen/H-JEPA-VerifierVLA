"""Check executed-prefix reuse, terminal frames, resume paths and read-only audit."""

import json
import os
from pathlib import Path
import tempfile
import unittest

import imageio.v2 as imageio
import numpy as np

from scripts.audit_verifier_data import audit_run, video_info


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture(root, retained=False):
    root.mkdir(parents=True)
    source = root.parent / "earlier" if retained else root
    directory = source / "tasks/task00/client/run/episodes/episode0"
    directory.mkdir(parents=True)
    row = dict(task_id=0, init_state_id=0, outcome="success", policy_steps=7, policy_requests=2,
               artifacts=os.path.relpath(directory, root))
    write_json(root / "run_config.json", dict(suite="libero_10", task_ids=[0], trials_per_task=1))
    write_json(root / "summary.json", dict(status="PASS", planned_episodes=1, recorded_episodes=1,
                                           valid_episodes=1, policy_steps=7, policy_requests=2))
    (root / "episodes.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    write_json(directory / "result.json", row)
    decisions = []
    for request, step, count in ((1, 0, 5), (2, 5, 2)):
        name = "observation_{:06d}.npz".format(request)
        np.savez_compressed(directory / name, **{
            "observation/image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((224, 224, 3), dtype=np.uint8),
            "observation/state": np.zeros(8), "prompt": "pick the bowl"})
        decisions.append(dict(client_request_id=request, policy_step=step, executed_steps=count,
                              response_received=True, actions=np.zeros((10, 7)).tolist(), observation=name))
    (directory / "decisions.jsonl").write_text("".join(json.dumps(d) + "\n" for d in decisions), encoding="utf-8")
    (directory / "video.mp4").write_bytes(b"video fixture")
    return directory


def recorded_video(path):
    return dict(frames=8, size=(448, 224), fps=10)


class VerifierDataAuditTest(unittest.TestCase):
    def test_resumed_artifacts_full_and_short_prefixes_are_read_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "resumed"
            fixture(root, retained=True)
            before = {p: p.read_bytes() for p in base.rglob("*") if p.is_file()}
            result = audit_run(root, read_video=recorded_video)
            self.assertEqual(before, {p: p.read_bytes() for p in base.rglob("*") if p.is_file()})
        self.assertEqual(result["audit_status"], "COMPLETE")
        self.assertEqual(result["retained_artifact_episodes"], 1)
        self.assertEqual(result["verified_totals"]["full_five_step_prefixes"], 1)
        self.assertEqual(result["verified_totals"]["short_prefixes"], 1)
        self.assertEqual(result["verified_totals"]["policy_steps"], 7)
        self.assertEqual(result["verified_totals"]["observation_files"], 2)
        self.assertEqual(result["split_group_key"], ["suite", "task_id", "init_state_id"])

    def test_missing_terminal_frame_is_not_label_ready(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            fixture(root)
            result = audit_run(root, read_video=lambda path: dict(frames=7, size=(448, 224), fps=10))
        self.assertEqual(result["audit_status"], "CHECK_REQUIRED")
        self.assertIn("terminal observation", result["errors"][0]["error"])

    def test_missing_npz_reports_the_episode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            directory = fixture(root)
            (directory / "observation_000002.npz").unlink()
            result = audit_run(root, read_video=recorded_video)
        self.assertEqual(result["episodes_with_errors"], 1)
        self.assertEqual(result["errors"][0]["episode_index"], 0)
        self.assertIn("observation_000002.npz", result["errors"][0]["error"])

    def test_unexecuted_suffix_cannot_be_counted_as_a_prefix_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            directory = fixture(root)
            path = directory / "decisions.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["executed_steps"] = 10
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            result = audit_run(root, read_video=recorded_video)
        self.assertEqual(result["audit_status"], "CHECK_REQUIRED")
        self.assertIn("prefix of 1..5", result["errors"][0]["error"])

    def test_duplicate_resumed_episode_is_not_an_independent_sample(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            fixture(root)
            path = root / "episodes.jsonl"
            path.write_text(path.read_text(encoding="utf-8") * 2, encoding="utf-8")
            write_json(root / "summary.json", dict(status="PASS", planned_episodes=2, recorded_episodes=2,
                                                   valid_episodes=2, policy_steps=14, policy_requests=4))
            write_json(root / "run_config.json", dict(suite="libero_10", task_ids=[0], trials_per_task=2))
            result = audit_run(root, read_video=recorded_video)
        self.assertEqual(result["audit_status"], "CHECK_REQUIRED")
        self.assertIn("Duplicate episode ID", result["errors"][0]["error"])

    def test_actual_ffmpeg_reader_counts_saved_terminal_frame(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "video.mp4"
            with imageio.get_writer(path, fps=10) as writer:
                for _ in range(8):
                    writer.append_data(np.zeros((224, 448, 3), dtype=np.uint8))
            result = video_info(path)
        self.assertEqual(result["frames"], 8)
        self.assertEqual(tuple(result["size"]), (448, 224))
        self.assertAlmostEqual(result["fps"], 10)


if __name__ == "__main__":
    unittest.main()
