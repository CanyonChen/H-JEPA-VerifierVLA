"""B2/B3: recorded LIBERO episodes using the pi05 WebSocket policy.

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
import math
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


def build_episode_plan(task_ids, first_init_state, trials_per_task):
    if trials_per_task < 1 or first_init_state < 0 or first_init_state + trials_per_task > 50:
        raise ValueError("Requested initial states must be within 0..49 and trials-per-task must be positive")
    return [(task_id, init_id) for task_id in task_ids
            for init_id in range(first_init_state, first_init_state + trials_per_task)]


def validate_server_provenance(metadata):
    provenance = metadata.get("evaluation", {})
    if (provenance.get("schema_version") != 1 or not provenance.get("service_id")
            or provenance.get("policy_config") != "pi05_libero" or provenance.get("backend") != "jax"
            or provenance.get("action_horizon") != 10 or provenance.get("initial_model_seed") != 0
            or len(provenance.get("checkpoint_manifest", {}).get("sha256", "")) != 64):
        raise ValueError("Baseline requires checkpoint provenance; restart with bash scripts/serve_pi05_libero.sh")
    if provenance.get("request_count") != 0:
        raise ValueError("Baseline requires a freshly restarted server with no previous inference requests")
    return provenance


def rollout_episode(env, client, initial_state, description, result, on_frame, on_decision,
                    max_steps, request_offset=0, expected_service_id=None):
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
            result["stage"] = "response_validation"
            server_request = response.get("evaluation")
            if server_request is not None:
                decision["server_request"] = server_request
            if expected_service_id is not None:
                if (not server_request or server_request.get("service_id") != expected_service_id
                        or server_request.get("request_id") != decision["client_request_id"]):
                    raise ValueError("Server request sequence changed; use one freshly restarted service and one client")
            if server_request:
                result.setdefault("first_server_request_id", server_request["request_id"])
                result["last_server_request_id"] = server_request["request_id"]
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


def wilson_interval(successes, total):
    if total == 0:
        return None
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = z * math.sqrt(proportion * (1 - proportion) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def summarize(episodes, planned_episodes, run_error=None, scope="complete-episode smoke check; not the 500-episode baseline"):
    counts = {name: sum(row["outcome"] == name for row in episodes)
              for name in ("success", "timeout", "infrastructure_error", "interrupted")}
    valid = counts["success"] + counts["timeout"]
    complete = len(episodes) == planned_episodes and valid == planned_episodes and not run_error
    return {
        "status": "PASS" if complete else "INCOMPLETE",
        "scope": scope,
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
        "success_rate_wilson95": wilson_interval(counts["success"], planned_episodes) if complete else None,
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
    parser.add_argument("--init-state-id", type=int, default=0, help="First initial state in the requested range.")
    parser.add_argument("--num-trials-per-task", type=int, default=1, help="Use 50 for the complete suite baseline.")
    parser.add_argument("--seed", type=int, default=7, help="Environment seed; does not set the model RNG.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/pi05_libero")
    args = parser.parse_args(argv)
    if args.task_id is not None and not 0 <= args.task_id < 10:
        parser.error("task-id must be 0..9")
    task_ids = list(range(10)) if args.task_id is None else [args.task_id]
    try:
        plan = build_episode_plan(task_ids, args.init_state_id, args.num_trials_per_task)
    except ValueError as error:
        parser.error(str(error))
    run_kind = "baseline" if len(plan) == 500 else ("smoke" if args.num_trials_per_task == 1 else "subset")
    scope = ("500-episode single-suite baseline" if run_kind == "baseline"
             else "complete-episode {} check; not the 500-episode baseline".format(run_kind))

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "1")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = args.output_root.resolve() / ("pi05_{}_{}_seed{}_{}".format(args.suite, run_kind, args.seed, stamp))
    run_dir.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "run.log", encoding="utf-8")],
        force=True,
    )
    logging.info("Run directory: %s", run_dir)
    logging.info("Use a freshly restarted pi05 server and a single client for this run.")
    config = {
        "experiment": "pi05_libero_" + run_kind, "output_dir": str(run_dir),
        "suite": args.suite, "task_ids": task_ids, "init_state_id": args.init_state_id,
        "num_trials_per_task": args.num_trials_per_task, "planned_episodes": len(plan),
        "episode_order": [{"task_id": task, "init_state_id": init} for task, init in plan],
        "environment_lifetime": "One environment per task, seeded once; reset before each initial state, as upstream.",
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
    environments = contextlib.ExitStack()
    active_task_id = None
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
        provenance = config["server_metadata"].get("evaluation", {})
        expected_service_id = None
        if args.num_trials_per_task > 1:
            provenance = validate_server_provenance(config["server_metadata"])
            expected_service_id = provenance["service_id"]
        if provenance:
            config["policy_identity_scope"] = "Loaded checkpoint identity and startup settings reported by the server"
            write_json(config_path, config)

        for task_id, init_state_id in plan:
            episode_id = "{}_task{:02d}_init{:02d}".format(args.suite, task_id, init_state_id)
            episode_dir = run_dir / "episodes" / episode_id
            episode_dir.mkdir(parents=True)
            result = dict(episode_progress(), episode_id=episode_id, suite=args.suite, task_id=task_id,
                          init_state_id=init_state_id, environment_seed=args.seed,
                          service_id=provenance.get("service_id"),
                          artifacts=str(episode_dir.relative_to(run_dir)),
                          first_client_request_id=request_offset + 1)
            started = time.perf_counter()
            try:
                if task_id != active_task_id:
                    result["stage"] = "environment_creation"
                    environments.close()
                    task = suite.get_task(task_id)
                    initial_states = suite.get_task_init_states(task_id)
                    env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(task_id),
                                             camera_heights=256, camera_widths=256)
                    environments.callback(env.close)
                    env.seed(args.seed)
                    active_task_id = task_id
                result["task_description"] = task.language
                initial_state = initial_states[init_state_id]
                logging.info("Starting %s: %s", episode_id, task.language)
                with contextlib.ExitStack() as stack:
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
                                    MAX_STEPS[args.suite], request_offset, expected_service_id)
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
            episodes = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()]
            progress = summarize(episodes, len(plan), scope=scope)
            progress.update(output_dir=str(run_dir), suite=args.suite)
            write_json(run_dir / "summary.json", progress)
            if result["outcome"] in ("infrastructure_error", "interrupted"):
                break
    except KeyboardInterrupt:
        interrupted = True
        run_error = "KeyboardInterrupt"
    except Exception as error:
        run_error = "{}: {}".format(type(error).__name__, error)
        logging.exception("LIBERO evaluation failed")
    finally:
        try:
            environments.close()
        except Exception as error:
            run_error = "Environment close: {}: {}".format(type(error).__name__, error)
        if client is not None:
            try:
                client._ws.close()
            except Exception as error:
                run_error = "WebSocket close: {}: {}".format(type(error).__name__, error)
        # Read the committed episode rows, so the summary is reproducible from disk.
        episodes = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines()]
        summary = summarize(episodes, len(plan), run_error, scope)
        summary.update(output_dir=str(run_dir), suite=args.suite)
        summary["per_task"] = []
        for task_id in task_ids:
            task_summary = summarize([row for row in episodes if row["task_id"] == task_id],
                                     args.num_trials_per_task, scope=scope)
            summary["per_task"].append(dict(task_id=task_id, **{key: task_summary[key] for key in
                ("planned_episodes", "valid_episodes", "successes", "timeouts", "infrastructure_errors",
                 "success_rate", "success_rate_wilson95")}))
        write_json(run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    return 130 if interrupted else (0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    sys.exit(main())
