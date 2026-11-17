"""Regressions for train-only selection, frame alignment, short tails and restart."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from scripts.label_verifier_long_pilot import label_episode, load_pilot, main, prefix_indices, read_external_video, run_pilot
from scripts.prepare_verifier_long_data import json_text, sha256_file
from scripts.robometer_backend import validate_loading


def make_fixture(base):
    split = base / "split"
    split.mkdir()
    episodes, windows = [], []
    for task in (0, 8):
        for initial, outcome in ((0, "success"), (1, "timeout")):
            episode_id = "libero_10_task{:02d}_init{:02d}".format(task, initial)
            video = base / (episode_id + ".mp4")
            video.write_bytes(b"mock decoder fixture")
            episodes.append(dict(episode_id=episode_id, task_id=task, init_state_id=initial, outcome=outcome,
                                 suite="libero_10", split="train", policy_steps=7, policy_requests=2,
                                 video=str(video), prompt="pick the bowl"))
            for index, start, count in ((0, 0, 5), (1, 5, 2)):
                windows.append(dict(episode_id=episode_id, split="train", policy_step=start, decision_index=index,
                                    context_frame_index=start, executed_steps=count,
                                    target_frame_indices=list(range(start + 1, start + count + 1)),
                                    valid_subchunk_mask=[i < count for i in range(5)]))
    for name, rows in (("train_episodes.jsonl", episodes), ("train_windows.jsonl", windows)):
        (split / name).write_text("".join(json_text(row) for row in rows), encoding="utf-8")
    manifest = dict(schema="verifier_long_split_v1", status="COMPLETE", splits={"train": {
        "files": {"episodes": "train_episodes.jsonl", "windows": "train_windows.jsonl"}}},
        index_sha256={name: sha256_file(split / name) for name in ("train_episodes.jsonl", "train_windows.jsonl")})
    (split / "manifest.json").write_text(json_text(manifest), encoding="utf-8")
    frames = np.zeros((8, 224, 224, 3), dtype=np.uint8)
    frames[:] = np.array([0, 20, 40, 35, 60, 55, 70, 65], dtype=np.uint8)[:, None, None, None]
    return split, episodes, windows, frames


class FakeLabeler:
    provenance = {"model_id": "synthetic-test-model"}

    def __init__(self):
        self.calls = 0

    def predict(self, sequences, task):
        self.calls += 1
        return sequences[:, -1, 0, 0, 0] / 100.0, np.full(len(sequences), 0.5)

    def metrics(self):
        return {}


class RobometerPilotTest(unittest.TestCase):
    def test_prefix_context_has_no_future_frames(self):
        self.assertEqual(prefix_indices(0).tolist(), [0, 0, 0, 0])
        self.assertEqual(prefix_indices(1).tolist(), [0, 0, 0, 1])
        self.assertEqual(prefix_indices(9).tolist(), [0, 3, 6, 9])

    def test_selection_is_train_only_and_detects_changed_split(self):
        with tempfile.TemporaryDirectory() as temporary:
            split, episodes, _, _ = make_fixture(Path(temporary))
            selected, _ = load_pilot(split)
            self.assertEqual(selected, episodes)
            path = split / "train_episodes.jsonl"
            rows = list(episodes)
            rows[0] = dict(rows[0], split="test")
            path.write_text("".join(json_text(row) for row in rows), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Split file changed"):
                load_pilot(split)
            manifest_path = split / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["index_sha256"][path.name] = sha256_file(path)
            manifest_path.write_text(json_text(manifest), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "training episodes"):
                load_pilot(split)

    def test_adjacent_deltas_preserve_negative_values_and_short_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, episodes, windows, frames = make_fixture(Path(temporary))
            with patch("scripts.label_verifier_long_pilot.read_external_video", return_value=frames):
                trace, labels, summary = label_episode(episodes[0], windows[:2], FakeLabeler(), 3)
            np.testing.assert_allclose(labels[0]["delta_progress"], [0.2, 0.2, -0.05, 0.25, -0.05])
            np.testing.assert_allclose(labels[1]["delta_progress"], [0.15, -0.05])
            self.assertEqual(labels[1]["valid_subchunk_mask"], [True, True, False, False, False])
            self.assertEqual(trace["context_frame_indices"][7], [0, 2, 4, 7])
            self.assertAlmostEqual(summary["progress_last"], 0.65)
            self.assertEqual(summary["outcome"], "success")

    def test_bad_predictions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            _, episodes, windows, frames = make_fixture(Path(temporary))
            labeler = FakeLabeler()
            labeler.predict = lambda sequences, task: (np.full(len(sequences), np.nan), np.zeros(len(sequences)))
            with patch("scripts.label_verifier_long_pilot.read_external_video", return_value=frames):
                with self.assertRaisesRegex(ValueError, "finite progress"):
                    label_episode(episodes[0], windows[:2], labeler, 3)

    def test_cli_and_resume_reuse_completed_episodes(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            split, _, _, frames = make_fixture(base)
            output = base / "labels"
            labeler = FakeLabeler()
            with patch("scripts.label_verifier_long_pilot.read_external_video", return_value=frames), \
                    patch("scripts.label_verifier_long_pilot.plot_progress", return_value=b"plot fixture"), \
                    patch("scripts.robometer_backend.RobometerLabeler", return_value=labeler):
                stdout = io.StringIO()
                with redirect_stdout(stdout):
                    self.assertEqual(main(["--split-dir", str(split), "--output-dir", str(output)]), 0)
                self.assertEqual(json.loads(stdout.getvalue())["labeled_frames"], 32)
                calls = labeler.calls
                report = run_pilot(split, output, labeler)
                self.assertEqual(report["retained_episodes"], 4)
                self.assertIsNone(report["new_frames_per_second"])
                self.assertEqual(labeler.calls, calls)
                damaged = next(output.glob("libero_*/window_labels.jsonl"))
                damaged.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "Saved label artifact changed"):
                    run_pilot(split, output, labeler)

    def test_failure_after_first_episode_can_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            split, _, _, frames = make_fixture(base)
            output = base / "labels"
            labeler = FakeLabeler()
            actual_predict = labeler.predict
            def fail_second(sequences, task):
                if labeler.calls == 1:
                    raise RuntimeError("inference interrupted")
                return actual_predict(sequences, task)
            labeler.predict = fail_second
            with patch("scripts.label_verifier_long_pilot.read_external_video", return_value=frames), \
                    patch("scripts.label_verifier_long_pilot.plot_progress", return_value=b"plot fixture"):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    run_pilot(split, output, labeler)
                self.assertEqual(len(list(output.glob("libero_*"))), 1)
                report = run_pilot(split, output, FakeLabeler())
                self.assertEqual(report["retained_episodes"], 1)
                self.assertEqual(report["labeled_windows"], 8)

    def test_real_video_decoder_preserves_terminal_frame_and_view(self):
        import imageio_ffmpeg
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "views.mp4"
            writer = imageio_ffmpeg.write_frames(str(path), (448, 224), fps=10)
            writer.send(None)
            for value in (20, 70, 120):
                frame = np.full((224, 448, 3), 210, dtype=np.uint8)
                frame[:, :224] = value
                writer.send(frame)
            writer.close()
            frames = read_external_video(path, 2)
            self.assertEqual(frames.shape, (3, 224, 224, 3))
            np.testing.assert_allclose(frames.mean(axis=(1, 2, 3)), [20, 70, 120], atol=4)
            with self.assertRaisesRegex(ValueError, "terminal frame"):
                read_external_video(path, 3)

    def test_checkpoint_loading_requires_all_used_weights(self):
        clean = dict(missing_keys=[], unexpected_keys=["similarity_head.0.weight"], mismatched_keys=[], error_msgs=[])
        validate_loading(clean)
        with self.assertRaisesRegex(ValueError, "load completely"):
            validate_loading(dict(clean, missing_keys=["progress_head.0.weight"]))
        with self.assertRaisesRegex(ValueError, "Unexpected"):
            validate_loading(dict(clean, unexpected_keys=["different_model.weight"]))


if __name__ == "__main__":
    unittest.main()
