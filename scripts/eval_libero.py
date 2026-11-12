"""B2: one complete episode per LIBERO task using the pi05 WebSocket policy.

Run with ``python -m scripts.eval_libero`` in the LIBERO environment. The policy
protocol follows the vendored openpi example; recording and error handling are
local additions. A PASS summary means all episodes were recorded without an
infrastructure error, not that every task succeeded.
"""

import argparse
import contextlib
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np

from scripts.check_libero_env import PROJECT_ROOT, configure_libero_paths, make_policy_observation


MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
WAIT_STEPS = 10
REPLAN_STEPS = 5
DUMMY_ACTION = [0.0] * 6 + [-1.0]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def episode_progress():
    return {"outcome": "running", "task_success": False, "wait_steps": 0,
            "policy_steps": 0, "policy_requests": 0, "stage": "initialization"}


def rollout_episode(env, client, initial_state, description, result, on_frame, on_decision,
                    max_steps, request_offset=0):
    """Update result in place so completed steps survive an exception.

    LIBERO's BDDLBaseDomain.step replaces done with _check_success(). The step
    budget is therefore checked here, separately from that success signal.
    """
    result["stage"] = "reset"
    env.reset()
    obs = env.set_init_state(initial_state)
    result["stage"] = "settling"
    for _ in range(WAIT_STEPS):
        obs, _, _, _ = env.step(DUMMY_ACTION)
        result["wait_steps"] += 1

    while result["policy_steps"] < max_steps and not result["task_success"]:
        result["stage"] = "observation"
        policy_obs = make_policy_observation(obs, description)
        if not np.isfinite(policy_obs["observation/state"]).all():
            raise ValueError("Non-finite LIBERO state")
        result["policy_requests"] += 1
        decision = {
            "client_request_id": request_offset + result["policy_requests"],
            "policy_step": result["policy_steps"],
            "executed_steps": 0,
            "response_received": False,
        }
        try:
            result["stage"] = "policy_inference"
            started = time.perf_counter()
            try:
                response = client.infer(policy_obs)
            finally:
                decision["roundtrip_seconds"] = time.perf_counter() - started
            decision["response_received"] = True
            actions = np.asarray(response["actions"])
            if actions.shape != (10, 7) or not np.isfinite(actions).all():
                raise ValueError("Expected finite policy actions of shape (10, 7)")
            decision["actions"] = actions.tolist()
            for action in actions[:REPLAN_STEPS]:
                result["stage"] = "video_frame"
                on_frame(make_policy_observation(obs, description))
                result["stage"] = "environment_step"
                obs, _, done, _ = env.step(action.tolist())
                result["policy_steps"] += 1
                decision["executed_steps"] += 1
                result["task_success"] = bool(done)
                if done or result["policy_steps"] >= max_steps:
                    break
        finally:
            # Keep the requested observation and any partial action execution.
            on_decision(decision, policy_obs)

    result["stage"] = "final_video_frame"
    on_frame(make_policy_observation(obs, description))
    result["outcome"] = "success" if result["task_success"] else "timeout"
    result["stage"] = "complete"


def summarize(episodes, planned_episodes, run_error=None):
    counts = {name: sum(row["outcome"] == name for row in episodes)
              for name in ("success", "timeout", "infrastructure_error", "interrupted")}
    valid = counts["success"] + counts["timeout"]
    complete = len(episodes) == planned_episodes and valid == planned_episodes and not run_error
    return {
        "status": "PASS" if complete else "INCOMPLETE",
        "scope": "complete-episode smoke check; not the 500-episode baseline",
        "planned_episodes": planned_episodes,
        "recorded_episodes": len(episodes),
        "unattempted_episodes": planned_episodes - len(episodes),
        "valid_episodes": valid,
        "successes": counts["success"],
        "timeouts": counts["timeout"],
        "infrastructure_errors": counts["infrastructure_error"],
        "interrupted_episodes": counts["interrupted"],
        # Do not silently drop infrastructure failures from a reported rate.
        "success_rate": counts["success"] / planned_episodes if complete else None,
        "policy_steps": sum(row["policy_steps"] for row in episodes),
        "policy_requests": sum(row["policy_requests"] for row in episodes),
        "run_error": run_error,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--suite", choices=tuple(MAX_STEPS), default="libero_10")
    parser.add_argument("--task-id", type=int, help="Run one task instead of the ten-task smoke check.")
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7, help="Environment seed; does not set the model RNG.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/pi05_libero")
    args = parser.parse_args(argv)
    if args.init_state_id < 0 or (args.task_id is not None and not 0 <= args.task_id < 10):
        parser.error("task-id must be 0..9 and init-state-id must be nonnegative")

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "1")
    task_ids = list(range(10)) if args.task_id is None else [args.task_id]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = args.output_root.resolve() / ("pi05_{}_smoke_seed{}_{}".format(args.suite, args.seed, stamp))
    run_dir.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "run.log", encoding="utf-8")],
        force=True,
    )
    logging.info("Run directory: %s", run_dir)
    logging.info("Use a freshly restarted pi05 server and a single client for this run.")
    config = {
        "experiment": "pi05_libero_smoke", "output_dir": str(run_dir),
        "suite": args.suite, "task_ids": task_ids, "init_state_id": args.init_state_id,
        "environment_seed": args.seed, "python": platform.python_version(),
        "platform": platform.platform(), "egl_device_id": os.environ["MUJOCO_EGL_DEVICE_ID"],
        "host": args.host, "port": args.port, "wait_steps": WAIT_STEPS,
        "max_policy_steps": MAX_STEPS[args.suite], "replan_steps": REPLAN_STEPS,
        "action_shape": [10, 7], "video_fps": 10,
        "expected_policy_config": "pi05_libero",
        "expected_checkpoint": "gs://openpi-assets/checkpoints/pi05_libero",
        "policy_identity_scope": "expected launcher settings; not attested by the WebSocket server",
        "model_rng_protocol": "Restart default JAX server (key 0); single client, no warmup requests.",
        "request_id_scope": "Client attempts; matches server order for an error-free run following the above protocol.",
    }
    config_path = run_dir / "run_config.json"
    write_json(config_path, config)
    episodes_path = run_dir / "episodes.jsonl"
    episodes_path.touch()
    run_error = None
    interrupted = False
    client = None
    request_offset = 0
    try:
        configure_libero_paths()
        import imageio.v2 as imageio
        from OpenGL import GL
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv
        from openpi_client.websocket_client_policy import WebsocketClientPolicy

        config["repo_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        config["repo_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT, text=True).strip())
        config["versions"] = {name: importlib.metadata.version(name) for name in
                              ("libero", "openpi-client", "torch", "numpy", "robosuite", "mujoco",
                               "websockets", "imageio", "imageio-ffmpeg")}
        np.random.seed(args.seed)
        suite = benchmark.get_benchmark_dict()[args.suite]()
        client = WebsocketClientPolicy(args.host, args.port)
        config["server_metadata"] = client.get_server_metadata()
        write_json(config_path, config)

        for task_id in task_ids:
            episode_id = "{}_task{:02d}_init{:02d}".format(args.suite, task_id, args.init_state_id)
            episode_dir = run_dir / "episodes" / episode_id
            episode_dir.mkdir(parents=True)
            result = dict(episode_progress(), episode_id=episode_id, suite=args.suite, task_id=task_id,
                          init_state_id=args.init_state_id, environment_seed=args.seed,
                          artifacts=str(episode_dir.relative_to(run_dir)),
                          first_client_request_id=request_offset + 1)
            started = time.perf_counter()
            try:
                task = suite.get_task(task_id)
                result["task_description"] = task.language
                initial_state = suite.get_task_init_states(task_id)[args.init_state_id]
                logging.info("Starting %s: %s", episode_id, task.language)
                with contextlib.ExitStack() as stack:
                    result["stage"] = "environment_creation"
                    env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(task_id),
                                             camera_heights=256, camera_widths=256)
                    stack.callback(env.close)
                    env.seed(args.seed)
                    writer = stack.enter_context(imageio.get_writer(episode_dir / "video.mp4", fps=10))
                    decisions = stack.enter_context((episode_dir / "decisions.jsonl").open("w", encoding="utf-8"))

                    def on_frame(observation):
                        # Both policy camera views, already rotated and resized.
                        writer.append_data(np.concatenate((observation["observation/image"],
                                                           observation["observation/wrist_image"]), axis=1))

                    def on_decision(decision, observation):
                        filename = "observation_{:06d}.npz".format(decision["client_request_id"])
                        np.savez_compressed(episode_dir / filename, **observation)
                        decision["observation"] = filename
                        decisions.write(json.dumps(decision, allow_nan=False) + "\n")
                        decisions.flush()

                    rollout_episode(env, client, initial_state, task.language, result, on_frame, on_decision,
                                    MAX_STEPS[args.suite], request_offset)
                    result["stage"] = "opengl_info"
                    result["opengl"] = {key: GL.glGetString(flag).decode("utf-8") for key, flag in
                                        (("vendor", GL.GL_VENDOR), ("renderer", GL.GL_RENDERER),
                                         ("version", GL.GL_VERSION))}
                    result["stage"] = "closing_artifacts"
                result["stage"] = "complete"
            except KeyboardInterrupt:
                result.update(outcome="interrupted", error="KeyboardInterrupt")
                interrupted = True
                logging.warning("Interrupted %s", episode_id)
            except Exception as error:
                result.update(outcome="infrastructure_error", error="{}: {}".format(type(error).__name__, error))
                logging.exception("Episode infrastructure error: %s", episode_id)
            finally:
                result["wall_seconds"] = time.perf_counter() - started
                request_offset += result["policy_requests"]
                result["last_client_request_id"] = request_offset if result["policy_requests"] else None
                if not result["policy_requests"]:
                    result["first_client_request_id"] = None
                write_json(episode_dir / "result.json", result)
                with episodes_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
            logging.info("%s: %s, policy_steps=%d, requests=%d", episode_id, result["outcome"],
                         result["policy_steps"], result["policy_requests"])
            if result["outcome"] in ("infrastructure_error", "interrupted"):
                break
    except KeyboardInterrupt:
        interrupted = True
        run_error = "KeyboardInterrupt"
    except Exception as error:
        run_error = "{}: {}".format(type(error).__name__, error)
        logging.exception("Smoke run failed")
    finally:
        if client is not None:
            try:
                client._ws.close()
            except Exception as error:
                run_error = "WebSocket close: {}: {}".format(type(error).__name__, error)
        # Read the committed episode rows, so the summary is reproducible from disk.
        episodes = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()]
        summary = summarize(episodes, len(task_ids), run_error)
        summary.update(output_dir=str(run_dir), suite=args.suite)
        write_json(run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    return 130 if interrupted else (0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    sys.exit(main())
