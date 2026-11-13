"""Saved-trace inspection must preserve experiment results and frozen skill names."""

import json
from pathlib import Path
import tempfile
import unittest

from scripts.inspect_lilo_run import inspect_run


class LiloInspectionTest(unittest.TestCase):
    def fixture(self, root, outcome="out_of_order"):
        directory = root / "tasks/task00/client/saved_run/episodes/trial000"
        directory.mkdir(parents=True)
        config = dict(task=dict(skills=["saved first", "saved second", "saved future"],
                                predicates=[[["first"]], [["second"]], [["future"]]],
                                prompt="saved full instruction"), repo_commit="saved_commit",
                      protocol="saved_protocol", egl_device_id="1")
        (directory.parent.parent / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
        row = dict(task_id=0, trial_id=0, outcome=outcome, policy_steps=2,
                   skills_completed=1, total_skills=3, artifacts=str(directory.relative_to(root)))
        (root / "episodes.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
        (root / "run_config.json").write_text(json.dumps(dict(gpu=1, egl_device=1, max_workers=2)), encoding="utf-8")
        (root / "summary.json").write_text(json.dumps(dict(recorded_episodes=1, success_rate=0)), encoding="utf-8")
        events = [dict(kind="out_of_order", expected_skill_index=1, skill_index=2)] if outcome == "out_of_order" else []
        trace = [dict(policy_step=1, skill_predicates=[False, False, False], events=[]),
                 dict(policy_step=2, skill_predicates=[False, False, True], events=events)]
        (directory / "skills.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trace), encoding="utf-8")
        return directory

    def test_uses_frozen_task_definitions_and_does_not_modify_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root)
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            report = inspect_run(root)
            after = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(report["inspection_status"], "COMPLETE")
        row = report["episodes"][0]
        self.assertEqual(row["client"]["repo_commit"], "saved_commit")
        self.assertEqual(row["prompt"], "saved full instruction")
        event = row["out_of_order_event"]
        self.assertEqual(event["expected"]["instruction"], "saved second")
        self.assertEqual(event["triggered"]["predicates"], [["future"]])
        self.assertEqual(event["previous_logged_step"], 1)
        self.assertIs(event["previous_logged_value"], False)
        self.assertIs(event["current_value"], True)

    def test_timeout_does_not_acquire_an_out_of_order_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.fixture(root, "timeout")
            report = inspect_run(root)
        self.assertEqual(report["inspection_status"], "COMPLETE")
        row = report["episodes"][0]
        self.assertEqual(row["outcome"], "timeout")
        self.assertIsNone(row["out_of_order_event"])
        self.assertEqual(row["next_skill"]["instruction"], "saved second")

    def test_missing_trace_is_reported_without_inventing_a_trigger(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.fixture(root)
            (directory / "skills.jsonl").unlink()
            report = inspect_run(root)
        self.assertEqual(report["inspection_status"], "INCOMPLETE")
        self.assertIsNone(report["episodes"][0]["out_of_order_event"])
        self.assertTrue(report["episodes"][0]["inspection_errors"])


if __name__ == "__main__":
    unittest.main()
