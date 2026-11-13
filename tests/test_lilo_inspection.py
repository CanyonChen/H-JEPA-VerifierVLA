"""Saved-trace inspection must preserve experiment results and frozen skill names."""

import json
import contextlib
import io
from pathlib import Path
import tempfile
import unittest

from scripts.inspect_lilo_run import check_summary_counts, compact_report, inspect_run, main


class LiloInspectionTest(unittest.TestCase):
    def test_error_mode_preserves_exception_and_filters_ordinary_task_failures(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.fixture(root)
            task_dir = directory.parents[3]
            (task_dir / "client.log").write_text("".join("client line {}\n".format(i) for i in range(100)), encoding="utf-8")
            (task_dir / "server.log").write_text("model still alive\n", encoding="utf-8")
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(["--run-dir", str(root), "--errors-only"])
            self.assertEqual(json.loads(output.getvalue())["episodes"], [])
            path = root / "episodes.jsonl"
            failed = json.loads(path.read_text(encoding="utf-8"))
            failed.update(outcome="infrastructure_error", stage="policy_inference", error="TimeoutError: recorded test failure")
            path.write_text(json.dumps(failed) + "\n", encoding="utf-8")
            summary_path = root / "summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary.update(status="INCOMPLETE", run_error="Task 0 incomplete")
            summary_path.write_text(json.dumps(summary), encoding="utf-8")
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                main(["--run-dir", str(root), "--errors-only"])
            report = json.loads(output.getvalue())
            self.assertEqual(before, {p: p.read_bytes() for p in root.rglob("*") if p.is_file()})
        self.assertEqual(report["reported_summary"]["run_error"], "Task 0 incomplete")
        episode, = report["episodes"]
        self.assertEqual(episode["stage"], "policy_inference")
        self.assertEqual(episode["error"], "TimeoutError: recorded test failure")
        self.assertEqual(len(episode["log_tails"]["client.log"]["tail"].splitlines()), 70)
        self.assertTrue(episode["log_tails"]["client.log"]["tail"].startswith("client line 30\n"))
        self.assertEqual(episode["log_tails"]["server.log"]["tail"], "model still alive\n")

    def test_summary_check_detects_swapped_task_counts_and_missing_task_entries(self):
        rows = [dict(task_id=task, policy_steps=steps) for task, steps in ((0, 60), (1, 3200), (8, 102))]
        summary = dict(recorded_episodes=3, policy_steps=3362,
                       per_task=[dict(row, recorded_episodes=1) for row in rows])
        errors = []
        check_summary_counts(dict(task_ids=[0, 1, 8]), summary, rows, errors)
        self.assertEqual(errors, [])
        summary["per_task"][0]["policy_steps"] = 102
        check_summary_counts(dict(task_ids=[0, 1, 8]), summary, rows, errors)
        self.assertIn("Task 0 policy_steps: summary=102, episode rows=60", errors)
        errors = []
        summary["per_task"] = summary["per_task"][:1]
        check_summary_counts(dict(task_ids=[0, 1, 8]), summary, rows, errors)
        self.assertIn("per_task IDs do not match the configured task IDs", errors)
        errors = []
        summary.pop("per_task")
        check_summary_counts(dict(task_ids=[0, 1, 8]), summary, rows, errors)
        self.assertIn("Saved summary has no per_task entries", errors)

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

    def test_live_physics_is_read_at_the_terminal_step_and_missing_steps_are_reported(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory = self.fixture(root)
            path = directory.parent.parent / "run_config.json"
            config = json.loads(path.read_text(encoding="utf-8"))
            config["record_physics"] = True
            config["task"]["predicates"] = [[["pickedup", "a"]], [["pickedup", "b"]], [["pickedup", "c"]]]
            path.write_text(json.dumps(config), encoding="utf-8")
            frames = [dict(source="live_rollout", policy_step=step, objects={
                name: dict(height_delta_m=0.04, both_fingerpads_contact=True,
                           both_fingerpads_contact_steps=step) for name in ("a", "b", "c")}) for step in (1, 2)]
            physics = directory / "physics.jsonl"
            physics.write_text("".join(json.dumps(r) + "\n" for r in frames), encoding="utf-8")
            report = inspect_run(root)
            compact = compact_report(report)
            self.assertEqual(report["inspection_status"], "COMPLETE")
            live = compact["episodes"][0]["live_physics"]
            self.assertEqual(set(live["objects"]), {"b", "c"})
            self.assertEqual(live["policy_step"], 2)
            self.assertEqual(live["objects"]["c"]["both_fingerpads_contact_steps"], 2)
            physics.write_text(json.dumps(frames[0]) + "\n", encoding="utf-8")
            self.assertEqual(inspect_run(root)["inspection_status"], "INCOMPLETE")


if __name__ == "__main__":
    unittest.main()
