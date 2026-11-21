"""Train/validation selection and restart behavior for dense Robometer labeling."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.label_verifier_long_splits import MODEL_FIELDS, run_labels
from scripts.prepare_verifier_long_data import json_text, sha256_file
from tests.test_robometer_pilot import FakeLabeler, make_fixture


def make_splits(base):
    indices, episodes, windows, frames = make_fixture(base)
    manifest = dict(schema="verifier_long_split_v1", suite="libero_10", split_seed=7,
                    status="COMPLETE", splits={}, index_sha256={})
    for split, episode in zip(("train", "validation"), episodes):
        eid = episode["episode_id"]
        rows = [dict(w, split=split) for w in windows if w["episode_id"] == eid]
        names = {"episodes": split + "_episodes.jsonl", "windows": split + "_windows.jsonl"}
        for name, records in ((names["episodes"], [dict(episode, split=split)]), (names["windows"], rows)):
            path = indices / name
            path.write_text("".join(json_text(r) for r in records), encoding="utf-8")
            manifest["index_sha256"][name] = sha256_file(path)
        manifest["splits"][split] = dict(files=names, episodes=1, policy_steps=7, policy_requests=2)
    (indices / "manifest.json").write_text(json_text(manifest), encoding="utf-8")
    return indices, frames


class SplitLabelsTest(unittest.TestCase):
    def test_resume_after_failed_validation_and_check_retained_outputs(self):
        model = FakeLabeler()
        model.provenance = {k: "fixture" for k in MODEL_FIELDS}
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            indices, frames = make_splits(base)
            output = base / "labels"
            decoder = "scripts.label_verifier_long_pilot.read_external_video"
            with redirect_stdout(io.StringIO()), patch(decoder, side_effect=[frames, OSError("interrupted")]):
                with self.assertRaisesRegex(OSError, "interrupted"):
                    run_labels(indices, output, model)
            self.assertFalse((output / "complete").exists())
            self.assertEqual(model.calls, 1)
            with redirect_stdout(io.StringIO()), patch(decoder, return_value=frames):
                report = run_labels(indices, output, model)
                self.assertEqual(model.calls, 2)
                self.assertEqual(report["splits"], dict.fromkeys(("train", "validation"),
                                                                 dict(episodes=1, frames=8, windows=2)))
                self.assertEqual(report, json.loads((output / "complete" / "summary.json").read_text()))
                self.assertFalse((output / "test").exists())
                self.assertEqual(run_labels(indices, output, model), report)
                self.assertEqual(model.calls, 2)
                next((output / "train").glob("*/window_labels.jsonl")).write_text("changed")
                with self.assertRaisesRegex(ValueError, "Changed labels"):
                    run_labels(indices, output, model)


if __name__ == "__main__":
    unittest.main()
