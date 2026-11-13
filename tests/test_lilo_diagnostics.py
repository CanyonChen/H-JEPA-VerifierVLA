"""Live diagnostic getters must not advance physics or alter predicate baselines."""

from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from scripts.lilo_diagnostics import LiveDiagnostics, object_snapshot


class LiveDiagnosticsTest(unittest.TestCase):
    def test_snapshot_reads_height_and_contacts_without_calling_mutating_predicates(self):
        state = mock.Mock(object_name="object", has_turnon_affordance=True)
        state.get_geom_state.return_value = {"pos": np.array([0.0, 0.0, 0.94])}
        state.get_joint_state.return_value = [0.2]
        heights = {"object": 0.9}
        predicates = SimpleNamespace(states={"object": state}, get_predicate=lambda name: SimpleNamespace(initial_heights=heights))
        gripper = SimpleNamespace(important_sites={"grip_site": "grip"}, contact_geoms=["left", "right", "palm"],
                                  important_geoms={"left_fingerpad": ["left"], "right_fingerpad": ["right"]})
        env = mock.Mock()
        env.env.robots = [SimpleNamespace(gripper=gripper)]
        env.sim.data.get_site_xpos.return_value = np.array([0.0, 0.0, 0.94])
        env.env.check_contact.side_effect = [True, False, True]
        report = object_snapshot(env, predicates, ["object_main"])["object_main"]
        self.assertAlmostEqual(report["height_delta_m"], 0.04)
        self.assertFalse(report["both_fingerpads_contact"])
        self.assertTrue(report["any_gripper_contact"])
        self.assertEqual(report["joint_positions"], [0.2])
        self.assertEqual(heights, {"object": 0.9})
        state.turn_on.assert_not_called()  # Upstream turn_on also writes visualization properties.
        env.step.assert_not_called()

    def test_contact_history_is_per_object_and_per_recorder(self):
        task = dict(predicates=[[["pickedup", "bowl"]], [["pickedup", "bowl"]], [["turnon", "stove"]]])
        recorder = LiveDiagnostics(None, None, task)
        self.assertEqual(recorder.names, ["bowl", "stove"])
        def snapshot(env, predicates, names):
            return {name: {"both_fingerpads_contact": name == "bowl", "any_gripper_contact": True} for name in names}
        with mock.patch("scripts.lilo_diagnostics.object_snapshot", side_effect=snapshot):
            first = recorder.observe(1)
            second = recorder.observe(2)
            other_episode = LiveDiagnostics(None, None, task).observe(1)
        self.assertEqual(first["objects"]["bowl"]["both_fingerpads_contact_steps"], 1)
        self.assertEqual(second["objects"]["bowl"]["both_fingerpads_contact_steps"], 2)
        self.assertEqual(second["objects"]["stove"]["both_fingerpads_contact_steps"], 0)
        self.assertEqual(second["objects"]["stove"]["any_gripper_contact_steps"], 2)
        self.assertEqual(other_episode["objects"]["bowl"]["both_fingerpads_contact_steps"], 1)
        self.assertEqual(second["source"], "live_rollout")


if __name__ == "__main__":
    unittest.main()
