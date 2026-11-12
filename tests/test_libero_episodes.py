"""Validate episode protocol and recording on CPU, without a simulator or model."""

import contextlib
import io
import json
import logging
import os
from pathlib import Path
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import imageio.v2 as imageio
import numpy as np

from scripts.eval_libero import DUMMY_ACTION, MAX_STEPS, episode_progress, main, rollout_episode, summarize


class FakeEnvironment:
    def __init__(self, success_at=None, error_at=None):
        self.success_at = success_at
        self.error_at = error_at
        self.actions = []
        self.wait_count = 0

    def observation(self):
        frame = np.zeros((224, 224, 3), dtype=np.uint8)
        frame[:20, :20, 0] = 255
        return {"agentview_image": frame, "robot0_eye_in_hand_image": frame.copy(),
                "robot0_eef_pos": np.array([len(self.actions), 0.0, 0.0]),
                "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
                "robot0_gripper_qpos": np.zeros(2)}

    def reset(self):
        self.actions = []
        self.wait_count = 0

    def seed(self, value):
        self.seed_value = value

    def close(self):
        pass

    def set_init_state(self, state):
        self.initial_state = state
        return self.observation()

    def step(self, action):
        if action == DUMMY_ACTION:
            self.wait_count += 1
            return self.observation(), 0, False, {}
        if self.error_at == len(self.actions) + 1:
            raise RuntimeError("simulator step failed")
        self.actions.append(action)
        done = self.success_at is not None and len(self.actions) >= self.success_at
        return self.observation(), 0, done, {}


class FakePolicy:
    def __init__(self, invalid=None, error=False):
        self.states = []
        self.invalid = invalid
        self.error = error

    def infer(self, observation):
        self.states.append(observation["observation/state"].copy())
        if self.error:
            raise RuntimeError("server unavailable")
        actions = np.tile((100 * len(self.states) + np.arange(10))[:, None], (1, 7)).astype(float)
        if self.invalid == "shape":
            actions = actions[:, :6]
        elif self.invalid == "nan":
            actions[0, 0] = np.nan
        return {"actions": actions}


class LiberoEpisodeTest(unittest.TestCase):
    def run_rollout(self, env, policy, limit=520, offset=0):
        result = episode_progress()
        frames, decisions = [], []
        rollout_episode(env, policy, [123], "test task", result,
                        lambda obs: frames.append(obs["observation/image"]),
                        lambda row, obs: decisions.append(row), limit, offset)
        return result, frames, decisions

    def test_replans_after_five_discards_tail_and_uses_new_observation(self):
        env, policy = FakeEnvironment(success_at=7), FakePolicy()
        result, frames, decisions = self.run_rollout(env, policy, offset=12)
        self.assertEqual(env.initial_state, [123])
        self.assertEqual(env.wait_count, 10)
        self.assertEqual([a[0] for a in env.actions], [100, 101, 102, 103, 104, 200, 201])
        self.assertEqual([state[0] for state in policy.states], [0, 5])
        self.assertEqual([row["executed_steps"] for row in decisions], [5, 2])
        self.assertEqual([row["client_request_id"] for row in decisions], [13, 14])
        self.assertEqual([row["policy_step"] for row in decisions], [0, 5])
        self.assertEqual(result["outcome"], "success")
        self.assertEqual(result["policy_steps"], 7)
        self.assertEqual(len(frames), 8)  # Final post-action frame is included.

    def test_long_timeout_has_520_policy_steps_plus_ten_wait_steps(self):
        env, policy = FakeEnvironment(), FakePolicy()
        result, _, decisions = self.run_rollout(env, policy, MAX_STEPS["libero_10"])
        self.assertEqual(len(env.actions), 520)
        self.assertEqual(env.wait_count, 10)
        self.assertEqual(len(decisions), 104)
        self.assertEqual(result["outcome"], "timeout")
        self.assertFalse(result["task_success"])

    def test_success_on_first_step_counts_that_action_and_stops(self):
        result, _, decisions = self.run_rollout(FakeEnvironment(success_at=1), FakePolicy())
        self.assertEqual(result["policy_steps"], 1)
        self.assertEqual(result["policy_requests"], 1)
        self.assertEqual(decisions[0]["executed_steps"], 1)

    def test_invalid_actions_never_reach_simulator(self):
        for invalid in ("shape", "nan"):
            with self.subTest(invalid=invalid):
                env, result, decisions = FakeEnvironment(), episode_progress(), []
                with self.assertRaisesRegex(ValueError, "finite policy actions"):
                    rollout_episode(env, FakePolicy(invalid), [], "test", result, lambda obs: None,
                                    lambda row, obs: decisions.append(row), 520)
                self.assertEqual(env.actions, [])
                self.assertEqual(result["policy_steps"], 0)
                self.assertEqual(decisions[0]["executed_steps"], 0)
                self.assertTrue(decisions[0]["response_received"])

    def test_simulator_error_preserves_partial_execution(self):
        env, result, decisions = FakeEnvironment(error_at=4), episode_progress(), []
        with self.assertRaisesRegex(RuntimeError, "simulator step failed"):
            rollout_episode(env, FakePolicy(), [], "test", result, lambda obs: None,
                            lambda row, obs: decisions.append(row), 520)
        self.assertEqual(result["policy_steps"], 3)
        self.assertEqual(result["stage"], "environment_step")
        self.assertEqual(decisions[0]["executed_steps"], 3)

    def test_failed_request_is_recorded_without_executing_actions(self):
        env, result, decisions = FakeEnvironment(), episode_progress(), []
        with self.assertRaisesRegex(RuntimeError, "server unavailable"):
            rollout_episode(env, FakePolicy(error=True), [], "test", result, lambda obs: None,
                            lambda row, obs: decisions.append(row), 520)
        self.assertEqual(env.actions, [])
        self.assertEqual(result["policy_requests"], 1)
        self.assertFalse(decisions[0]["response_received"])

    def test_summary_distinguishes_task_timeout_error_and_unattempted(self):
        def row(outcome):
            return dict(episode_progress(), outcome=outcome)
        valid = [row("success"), row("timeout")]
        summary = summarize(valid, 2)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["success_rate"], 0.5)
        summary = summarize(valid + [row("infrastructure_error")], 10)
        self.assertEqual(summary["status"], "INCOMPLETE")
        self.assertEqual(summary["infrastructure_errors"], 1)
        self.assertEqual(summary["timeouts"], 1)
        self.assertEqual(summary["unattempted_episodes"], 7)
        self.assertEqual(summary["valid_episodes"], 2)
        self.assertIsNone(summary["success_rate"])
        self.assertEqual(summarize(valid, 2, "recording failed")["status"], "INCOMPLETE")

    def test_two_camera_mp4_encode_decode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "video.mp4"
            result = episode_progress()
            with imageio.get_writer(path, fps=10) as writer:
                rollout_episode(FakeEnvironment(success_at=2), FakePolicy(), [], "test", result,
                                lambda obs: writer.append_data(np.concatenate(
                                    (obs["observation/image"], obs["observation/wrist_image"]), axis=1)),
                                lambda row, obs: None, 520)
            with imageio.get_reader(path) as reader:
                self.assertEqual(reader.count_frames(), 3)
                self.assertEqual(reader.get_data(0).shape, (224, 448, 3))

    def test_run_records_unique_episodes_and_stops_on_infrastructure_error(self):
        # Use real installed benchmark assets, JSON/NPZ and FFmpeg recording.
        # Replace only the renderer, simulator and policy to keep this on CPU.
        created = []

        def factory(**kwargs):
            env = FakeEnvironment(success_at=1, error_at=1 if len(created) == 2 else None)
            created.append(env)
            return env

        env_module = ModuleType("libero.libero.envs")
        env_module.OffScreenRenderEnv = factory
        gl_module = ModuleType("OpenGL")
        gl_module.GL = SimpleNamespace(GL_VENDOR=1, GL_RENDERER=2, GL_VERSION=3,
                                       glGetString=lambda flag: b"CPU test stub")
        policy = FakePolicy()
        policy._ws = SimpleNamespace(close=lambda: None)
        policy.get_server_metadata = lambda: {"test_only": True}
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.dict("sys.modules", {"libero.libero.envs": env_module, "OpenGL": gl_module}), \
                    mock.patch.dict(os.environ), \
                    mock.patch("openpi_client.websocket_client_policy.WebsocketClientPolicy", return_value=policy), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    exit_code = main(["--output-root", directory])
                finally:
                    # main normally exits the process; close its log for Windows temp cleanup.
                    for handler in list(logging.getLogger().handlers):
                        logging.getLogger().removeHandler(handler)
                        handler.close()
            run_dir, = Path(directory).iterdir()
            rows = [json.loads(line) for line in (run_dir / "episodes.jsonl").read_text().splitlines()]
            summary = json.loads((run_dir / "summary.json").read_text())
            config = json.loads((run_dir / "run_config.json").read_text())
            self.assertEqual(exit_code, 1)
            self.assertEqual([row["outcome"] for row in rows], ["success", "success", "infrastructure_error"])
            self.assertEqual(len({row["artifacts"] for row in rows}), 3)
            self.assertEqual([row["first_client_request_id"] for row in rows], [1, 2, 3])
            self.assertEqual(summary["unattempted_episodes"], 7)
            self.assertIsNone(summary["success_rate"])
            self.assertEqual(config["server_metadata"], {"test_only": True})
            episode_dir = run_dir / rows[-1]["artifacts"]
            decision = json.loads((episode_dir / "decisions.jsonl").read_text())
            self.assertEqual(decision["executed_steps"], 0)
            with np.load(episode_dir / decision["observation"]) as observation:
                self.assertEqual(observation["observation/state"].shape, (8,))
            self.assertTrue((run_dir / rows[0]["artifacts"] / "video.mp4").is_file())


if __name__ == "__main__":
    unittest.main()
