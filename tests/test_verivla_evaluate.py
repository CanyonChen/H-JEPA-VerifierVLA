"""Read-only split, paired-outcome and resume checks with synthetic episode records."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from scripts.verivla_evaluate import cached_arm, held_out, summarize_pairs


def episodes():
    return [dict(suite="libero_goal", split="test", episode_id=f"libero_goal_task{t:02d}_init{i:02d}",
                 task_id=t, init_state_id=i) for t in range(10) for i in range(5)]


class HeldOutTests(unittest.TestCase):
    def test_complete_test_partition_and_overlap_with_training(self):
        rows = episodes()
        with mock.patch("scripts.verivla_evaluate.read_rows", return_value=rows):
            self.assertEqual(held_out(Path("."), "libero_goal", set()), rows)
            with self.assertRaises(ValueError):
                held_out(Path("."), "libero_goal", {rows[0]["episode_id"]})
            rows[-1] = rows[-2]
            with self.assertRaises(ValueError):
                held_out(Path("."), "libero_goal", set())

    def test_paired_success_counts(self):
        outcomes = [(False, True), (True, False), (False, True), (True, True), (False, False)]
        pairs = [[dict(episode=i, arm="base", task_success=a), dict(episode=i, arm="verifier", task_success=b)]
                 for i, (a, b) in enumerate(outcomes)]
        report = summarize_pairs(pairs)
        self.assertEqual((report["base_successes"], report["verifier_successes"]), (2, 3))
        self.assertEqual((report["wins"], report["losses"]), (2, 1))
        self.assertEqual(report["difference"], .2)

    def test_resume_checks_both_model_and_trigger_identity(self):
        ep = episodes()[0]
        row = dict(ep, episode=ep["episode_id"], arm="base", outcome="timeout", task_success=False,
                   identity="run", reference="calibration")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "arm.json"
            self.assertIsNone(cached_arm(path, "run", "calibration", ep, "base"))
            path.write_text(json.dumps(row), encoding="utf-8")
            self.assertEqual(cached_arm(path, "run", "calibration", ep, "base"), row)
            for identity, reference in [("other", "calibration"), ("run", "changed")]:
                with self.assertRaises(ValueError):
                    cached_arm(path, identity, reference, ep, "base")
