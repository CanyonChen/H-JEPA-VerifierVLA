"""Exercise actual B4 rollout with simulated observations and policy responses."""

import unittest

from scripts.eval_libero import episode_progress
from scripts.eval_lilo import rollout
from test_libero_episodes import FakeEnvironment, FakePolicy


class TimedPredicates:
    def __init__(self, env, thresholds):
        self.env, self.thresholds = env, thresholds
        self.started = []

    def values(self):
        return [len(self.env.actions) >= threshold for threshold in self.thresholds]

    def begin_skill(self, index):
        self.started.append(index)


class LiloEpisodeTest(unittest.TestCase):
    def run_episode(self, thresholds, limit=20, policy=None, error_at=None, service_id=None):
        env = FakeEnvironment(success_at=1, error_at=error_at)
        policy = policy or FakePolicy()
        task = dict(prompt="entire task", skills=["skill " + str(i) for i in range(len(thresholds))],
                    predicates=[[["predicate", str(i)]] for i in range(len(thresholds))])
        predicates, result = TimedPredicates(env, thresholds), episode_progress()
        frames, decisions, progress = [], [], []
        try:
            rollout(env, env.observation(), policy, task, predicates, result, frames.append,
                    lambda row, obs: decisions.append(row), progress.append, limit, service_id=service_id)
        finally:
            self.last = (result, env, policy, decisions, progress)
        return self.last

    def test_ignores_env_done_and_replans_after_five_actions(self):
        result, env, policy, decisions, progress = self.run_episode([2, 7])
        self.assertEqual(result["outcome"], "success")
        self.assertEqual(result["skills_completed"], 2)
        self.assertEqual(result["wait_steps"], 0)
        self.assertEqual([a[0] for a in env.actions], [100, 101, 102, 103, 104, 200, 201])
        self.assertEqual([d["executed_steps"] for d in decisions], [5, 2])
        self.assertEqual([state[0] for state in policy.states], [0, 5])
        self.assertTrue(all(p["environment_done"] for p in progress))
        self.assertEqual(env.initial_states, [])
        self.assertEqual(env.seed_calls, [])

    def test_out_of_order_stops_with_partial_progress(self):
        result, env, _, _, _ = self.run_episode([2, 20, 4])
        self.assertEqual(result["outcome"], "out_of_order")
        self.assertEqual(result["skills_completed"], 1)
        self.assertEqual(result["policy_steps"], 4)
        self.assertEqual(result["average_progress"], 1 / 3)

    def test_timeout_is_total_episode_budget_and_records_partial_progress(self):
        result, env, _, decisions, _ = self.run_episode([2, 30], limit=12)
        self.assertEqual(result["outcome"], "timeout")
        self.assertEqual(result["skills_completed"], 1)
        self.assertEqual(len(env.actions), 12)
        self.assertEqual([d["executed_steps"] for d in decisions], [5, 5, 2])

    def test_step_error_keeps_completed_actions_and_skill_progress(self):
        with self.assertRaisesRegex(RuntimeError, "simulator step"):
            self.run_episode([2, 10], error_at=4)
        result, _, _, decisions, _ = self.last
        self.assertEqual(result["policy_steps"], 3)
        self.assertEqual(result["skills_completed"], 1)
        self.assertEqual(decisions[0]["executed_steps"], 3)
        self.assertNotEqual(result["outcome"], "timeout")

    def test_malformed_actions_and_wrong_service_never_reach_simulator(self):
        for policy, service in ((FakePolicy("nan"), None), (FakePolicy("shape"), None),
                                 (FakePolicy(), "required-service")):
            with self.assertRaises(ValueError):
                self.run_episode([2, 7], policy=policy, service_id=service)
            result, env, _, decisions, _ = self.last
            self.assertEqual(result["policy_steps"], 0)
            self.assertEqual(env.actions, [])
            self.assertTrue(decisions[0]["response_received"])


if __name__ == "__main__":
    unittest.main()
