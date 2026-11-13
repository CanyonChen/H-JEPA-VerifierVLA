"""CPU checks for Ultra-Long task identity, temporal scoring, AP and recovery."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from scripts import lilo_protocol as protocol
from scripts import run_libero_parallel as parallel


def row(task, trial, count, total, outcome):
    return dict(task_id=task, trial_id=trial, skills_completed=count, total_skills=total,
                outcome=outcome, policy_steps=10, policy_requests=2,
                first_server_request_id=trial * 2 + 1, last_server_request_id=trial * 2 + 2,
                artifacts="episodes/trial{:03d}".format(trial))


class LiloProtocolTest(unittest.TestCase):
    def test_exact_nine_configurations_and_complete_ordered_instructions(self):
        tasks = protocol.task_definitions()
        self.assertEqual([t["definition_id"] for t in tasks], [8, 9, 10, 19, 20, 21, 22, 23, 24])
        self.assertEqual([len(t["skills"]) for t in tasks], [9, 16, 10] * 3)
        for task in tasks:
            self.assertEqual(task["prompt"][:-1].split(", then "), task["skills"])
            self.assertTrue(all(task["predicates"]))
        for scene in range(3):
            self.assertEqual(len({tasks[i]["prompt"] for i in range(scene, 9, 3)}), 3)

    def test_scoring_tracks_ordered_prefix_not_instantaneous_conjunction(self):
        monitor = protocol.OrderedProgress([[['a']], [['b']], [['c']]], [False] * 3)
        monitor.update([True, False, False], 1)
        monitor.update([False, True, False], 2)
        monitor.update([False, False, True], 3)
        self.assertEqual(monitor.completed, 3)
        self.assertIsNone(monitor.out_of_order)

    def test_future_skill_event_fails_instead_of_being_bankable(self):
        monitor = protocol.OrderedProgress([[['a']], [['b']], [['c']]], [False] * 3)
        events = monitor.update([False, False, True], 1)
        self.assertEqual(monitor.completed, 0)
        self.assertEqual(monitor.out_of_order, 2)
        self.assertEqual(events[0]["kind"], "out_of_order")
        monitor.update([True] * 3, 2)
        self.assertEqual(monitor.completed, 0)

    def test_initially_true_future_condition_is_not_a_new_event(self):
        monitor = protocol.OrderedProgress([[['a']], [['b']], [['c']]], [False, False, True])
        monitor.update([True, False, True], 1)
        self.assertEqual(monitor.completed, 1)
        self.assertIsNone(monitor.out_of_order)

    def test_adjacent_cooccurrence_and_repeated_predicates_do_not_false_fail(self):
        for predicates, values in (([[['a']], [['b']], [['c']]], [True, True, False]),
                                   ([[['a']], [['b']], [['a']]], [True, False, True])):
            monitor = protocol.OrderedProgress(predicates, [False] * 3)
            monitor.update(values, 1)
            self.assertEqual(monitor.completed, 1)
            self.assertIsNone(monitor.out_of_order)

    def test_object_resolution_is_exact_or_main_suffix_only(self):
        states = {"bowl_1": object(), "cabinet_region": object()}
        self.assertIs(protocol.resolve_object(states, "bowl_1_main"), states["bowl_1"])
        self.assertIs(protocol.resolve_object(states, "cabinet_region"), states["cabinet_region"])
        with self.assertRaises(KeyError):
            protocol.resolve_object(states, "bowl_2_main")

    def test_ap_is_mean_episode_fraction_and_errors_never_count_as_task_failure(self):
        rows = [row(0, 0, 9, 9, "success"), row(1, 0, 8, 16, "out_of_order")]
        summary = protocol.aggregate_lilo(rows, [0, 1], 1)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["success_rate"], 0.5)
        self.assertEqual(summary["average_progress"], 0.75)
        self.assertEqual(summary["timeouts"], 0)
        self.assertEqual(summary["out_of_order_failures"], 1)
        for outcomes in (rows[:1], [rows[0], row(1, 0, 8, 16, "infrastructure_error")]):
            summary = protocol.aggregate_lilo(outcomes, [0, 1], 1)
            self.assertEqual(summary["status"], "INCOMPLETE")
            self.assertIsNone(summary["success_rate"])
            self.assertIsNone(summary["average_progress"])
        with self.assertRaises(ValueError):
            protocol.aggregate_lilo(rows + rows[:1], [0, 1], 1)
        with self.assertRaises(ValueError):
            protocol.summarize_lilo([row(0, 0, 8, 9, "success")], 1)

    def test_resume_retains_order_failures_and_rejects_protocol_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = dict(suite=protocol.SUITE, seed=None, task_ids=[0], trials_per_task=3,
                          checkpoint=parallel.CHECKPOINT, steps_per_skill=200, lilo_protocol=protocol.PROTOCOL)
            (root / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            client = root / "tasks/task00/client/run"
            client.mkdir(parents=True)
            rows = [row(0, 0, 9, 9, "success"), row(0, 1, 2, 9, "out_of_order"),
                    row(0, 2, 0, 9, "infrastructure_error")]
            (client / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
            (client / "run_config.json").write_text(json.dumps({"server_metadata": {
                "evaluation": {"checkpoint_manifest": {"sha256": "0" * 64}}}}), encoding="utf-8")
            args = SimpleNamespace(**config, resume_from=root)
            retained, starts, digest = parallel.resume_state(args)
            self.assertEqual([r["outcome"] for r in retained], ["success", "out_of_order"])
            self.assertEqual(starts[0], dict(first_init_state=2, request_offset=4))
            self.assertEqual(digest, "0" * 64)
            args.steps_per_skill = 201
            with self.assertRaisesRegex(ValueError, "budget or scoring"):
                parallel.resume_state(args)


if __name__ == "__main__":
    unittest.main()
