"""Replay evidence is usable only when the saved trajectory can be matched."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np

from scripts import replay_lilo_failures as replay


def observation():
    return {"observation/state": np.zeros(8), "observation/image": np.zeros((2, 2, 3), dtype=np.uint8),
            "observation/wrist_image": np.zeros((2, 2, 3), dtype=np.uint8), "prompt": "saved instruction"}


class ReplayTest(unittest.TestCase):
    def fixture(self, root):
        directory = root / "client/episodes/trial000"
        directory.mkdir(parents=True)
        config = dict(protocol=replay.PROTOCOL, lilo_commit=replay.LILO_COMMIT,
                      task=dict(task_id=0, skills=["pick expected", "pick future"],
                                predicates=[[["pickedup", "expected"]], [["pickedup", "future"]]],
                                prompt="saved instruction"))
        (directory.parent.parent / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
        np.save(directory / "reset_sim_state.npy", np.zeros(3))
        np.save(directory / "initial_sim_state.npy", np.zeros(3))
        np.savez(directory / "observation.npz", **observation())
        actions = np.arange(70, dtype=float).reshape(10, 7)
        decision = dict(policy_step=0, actions=actions.tolist(), executed_steps=2, observation="observation.npz")
        (directory / "decisions.jsonl").write_text(json.dumps(decision) + "\n", encoding="utf-8")
        trace = [dict(policy_step=1, skill_predicates=[False, False], events=[]),
                 dict(policy_step=2, skill_predicates=[False, True],
                      events=[dict(kind="out_of_order", expected_skill_index=0, skill_index=1)])]
        (directory / "skills.jsonl").write_text("".join(json.dumps(r) + "\n" for r in trace), encoding="utf-8")
        return directory, dict(task_id=0, trial_id=0, outcome="out_of_order", policy_steps=2), actions

    def run_replay(self, directory, row, state=None, values=None):
        env = mock.Mock()
        env.get_sim_state.return_value = np.zeros(3) if state is None else state
        env.set_init_state.return_value = {}
        env.step.return_value = ({}, 0, False, {})
        predicates = mock.Mock()
        predicates.values.side_effect = values or [[False, False], [False, True]]
        suite = SimpleNamespace(make_env=lambda task: (env, {}))
        with mock.patch.object(replay, "settle_environment", side_effect=lambda env, obs: obs), \
                mock.patch.object(replay, "SkillPredicates", return_value=predicates), \
                mock.patch.object(replay, "make_policy_observation", side_effect=lambda obs, prompt: observation()), \
                mock.patch.object(replay, "object_snapshot", return_value={"future": {
                    "both_fingerpads_contact": False, "height_delta_m": 0.04}}):
            report = replay.replay_episode(suite, directory, row)
        env.close.assert_called_once()
        return report, env

    def test_replays_only_executed_actions_and_preserves_input_files(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directory, row, actions = self.fixture(root)
            before = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
            report, env = self.run_replay(directory, row)
            after = {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)
        self.assertEqual(report["status"], "MATCHED")
        self.assertEqual(report["replayed_steps"], 2)
        self.assertEqual([call.args[0] for call in env.step.call_args_list], actions[:2].tolist())
        self.assertEqual(report["contact_history"]["future"]["both_fingerpads_contact_steps"], 0)

    def test_initial_state_mismatch_stops_before_any_actions(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, row, _ = self.fixture(Path(temporary))
            report, env = self.run_replay(directory, row, state=np.ones(3))
        self.assertEqual(report["status"], "DIVERGED")
        env.step.assert_not_called()
        self.assertNotIn("contact_history", report)

    def test_trace_mismatch_is_not_reported_as_matched(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory, row, _ = self.fixture(Path(temporary))
            report, env = self.run_replay(directory, row, values=[[True, False]])
        self.assertEqual(report["status"], "DIVERGED")
        self.assertEqual(env.step.call_count, 1)
        self.assertEqual(report["tail"], [])

    def test_observation_check_detects_wrong_prompt_and_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            saved = Path(temporary) / "obs.npz"
            np.savez(saved, **observation())
            for key, replacement in (("prompt", "different"),
                                     ("observation/image", np.ones((2, 2, 3), dtype=np.uint8))):
                current = observation()
                current[key] = replacement
                self.assertFalse(replay.compare_observation(current, saved)["matches"])


if __name__ == "__main__":
    unittest.main()
