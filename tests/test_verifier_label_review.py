"""CPU regressions for reviewing raw labels without changing their source artifacts."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.inspect_verifier_long_labels import main, run_review
from scripts.label_verifier_long_pilot import run_pilot
from scripts.prepare_verifier_long_data import json_text, sha256_file
from tests.test_robometer_pilot import FakeLabeler, make_fixture


class LabelReviewTest(unittest.TestCase):
    def make_run(self, base):
        split, episodes, _, frames = make_fixture(base)
        run = base / "pilot"
        with patch("scripts.label_verifier_long_pilot.read_external_video", return_value=frames), \
                patch("scripts.label_verifier_long_pilot.plot_progress", return_value=b"synthetic plot"):
            run_pilot(split, run, FakeLabeler())
        return run, episodes, frames

    def test_review_checks_windows_and_preserves_raw_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run, _, frames = self.make_run(base)
            before = {path: sha256_file(path) for path in run.rglob("*") if path.is_file()}
            with patch("scripts.inspect_verifier_long_labels.read_external_video", return_value=frames):
                report = run_review(run, base / "review")
            self.assertEqual((report["episodes"], report["frames"], report["windows"]), (4, 32, 8))
            self.assertEqual(report["model_requests"], 0)
            self.assertFalse(report["gpu_used"])
            self.assertEqual(report["label_quality_status"], "REVIEW_REQUIRED")
            row = report["per_episode"][0]
            self.assertAlmostEqual(row["one_step_deltas"]["negative_fraction"], 3 / 7)
            self.assertEqual(row["full_five_step_window_deltas"]["count"], 1)
            self.assertEqual(row["full_five_step_window_deltas"]["negative_fraction"], 0)
            self.assertAlmostEqual(row["minimum_delta_event"]["delta"], -0.05)
            self.assertEqual(report["paired_terminal_progress"][0]["terminal_progress_gap"], 0)
            for row in report["per_episode"]:
                self.assertTrue((base / "review" / row["figure"]).read_bytes().startswith(b"\x89PNG"))
            self.assertEqual(before, {path: sha256_file(path) for path in before})

    def test_changed_window_delta_is_rejected_even_with_updated_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run, episodes, _ = self.make_run(base)
            directory = run / episodes[0]["episode_id"]
            path = directory / "window_labels.jsonl"
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
            rows[0]["delta_progress"][0] += 0.1
            path.write_text("".join(json_text(row) for row in rows), encoding="utf-8")
            summary_path = directory / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["artifact_sha256"][path.name] = sha256_file(path)
            summary_path.write_text(json_text(summary), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "deltas differ"):
                run_review(run, base / "review")
            self.assertFalse((base / "review").exists())

    def test_source_video_change_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run, episodes, _ = self.make_run(base)
            Path(episodes[0]["video"]).write_bytes(b"different source")
            with self.assertRaisesRegex(ValueError, "Source video changed"):
                run_review(run, base / "review")

    def test_text_only_cli_preserves_checks_without_plotting(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            run, _, frames = self.make_run(base)
            output = base / "text_review"
            stdout = io.StringIO()
            with patch("scripts.inspect_verifier_long_labels.read_external_video", return_value=frames), \
                    patch("scripts.inspect_verifier_long_labels.plot_review", side_effect=AssertionError("No plot")), \
                    redirect_stdout(stdout):
                code = main(["--run-dir", str(run), "--output-dir", str(output), "--text-only"])
            self.assertEqual(code, 0)
            report = json.loads((output / "review.json").read_text(encoding="utf-8"))
            self.assertEqual((report["episodes"], report["frames"], report["windows"]), (4, 32, 8))
            self.assertEqual(list(output.glob("*.png")), [])
            self.assertTrue(report["text_only"])
            self.assertNotIn("figure", report["per_episode"][0])
            self.assertIn("Semantic label quality remains unverified", stdout.getvalue())
            self.assertIn("minimum_delta_event:", stdout.getvalue())
            self.assertIn("full_five_step_window_deltas:", stdout.getvalue())
            self.assertIn("paired_terminal_progress:", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
