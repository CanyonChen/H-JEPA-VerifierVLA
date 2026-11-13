"""Run one Ultra-Long configuration against the existing pi05 WebSocket service."""

import argparse
import contextlib
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from scripts.check_libero_env import PROJECT_ROOT, make_policy_observation
from scripts.eval_libero import episode_progress, validate_server_provenance, write_json
from scripts.lilo_protocol import (LILO_COMMIT, PROTOCOL, STEPS_PER_SKILL, SUITE, WAIT_STEPS, OrderedProgress,
                                   SkillPredicates, initialize_benchmark, settle_environment, summarize_lilo, task_definitions)


def update_result(result, tracker, task):
    result.update(skills_completed=tracker.completed, total_skills=len(task["skills"]),
                  average_progress=tracker.completed / len(task["skills"]),
                  task_success=tracker.completed == len(task["skills"]),
                  out_of_order_skill_index=tracker.out_of_order,
                  next_skill=task["skills"][tracker.completed] if tracker.completed < len(task["skills"]) else None)


def rollout(env, obs, client, task, predicates, result, on_frame, on_decision, on_progress,
            max_steps, request_offset=0, service_id=None):
    tracker = OrderedProgress(task["predicates"], predicates.values())
    update_result(result, tracker, task)
    while result["policy_steps"] < max_steps and not result["task_success"] and tracker.out_of_order is None:
        result["stage"] = "observation"
        policy_obs = make_policy_observation(obs, task["prompt"])
        if not np.isfinite(policy_obs["observation/state"]).all():
            raise ValueError("Non-finite robot state")
        result["policy_requests"] += 1
        decision = {"client_request_id": request_offset + result["policy_requests"],
                    "policy_step": result["policy_steps"], "executed_steps": 0,
                    "response_received": False}
        try:
            result["stage"] = "policy_inference"
            started = time.perf_counter()
            try:
                response = client.infer(policy_obs)
            finally:
                decision["roundtrip_seconds"] = time.perf_counter() - started
            decision["response_received"] = True
            server = response.get("evaluation")
            decision["server_request"] = server
            if service_id is not None and (not server or server.get("service_id") != service_id
                                           or server.get("request_id") != decision["client_request_id"]):
                raise ValueError("Policy service or request sequence changed")
            if server:
                result.setdefault("first_server_request_id", server["request_id"])
                result["last_server_request_id"] = server["request_id"]
            actions = np.asarray(response["actions"])
            if actions.shape != (10, 7) or not np.isfinite(actions).all():
                raise ValueError("Expected finite (10, 7) actions")
            decision["actions"] = actions.tolist()
            for action in actions[:5]:
                result["stage"] = "environment_step"
                on_frame(policy_obs if decision["executed_steps"] == 0 else make_policy_observation(obs, task["prompt"]))
                obs, _, done, _ = env.step(action.tolist())
                result["policy_steps"] += 1
                decision["executed_steps"] += 1
                result["stage"] = "skill_scoring"
                values = predicates.values()
                before = tracker.completed
                events = tracker.update(values, result["policy_steps"])
                update_result(result, tracker, task)
                on_progress({"policy_step": result["policy_steps"], "skill_predicates": values,
                             "skills_completed": tracker.completed, "events": events,
                             "environment_done": bool(done)})
                if tracker.completed != before and not result["task_success"]:
                    predicates.begin_skill(tracker.completed)
                    tracker.previous = predicates.values()
                if result["task_success"] or tracker.out_of_order is not None or result["policy_steps"] >= max_steps:
                    break
        finally:
            on_decision(decision, policy_obs)
    on_frame(make_policy_observation(obs, task["prompt"]))
    result["outcome"] = ("success" if result["task_success"] else
                         "out_of_order" if tracker.out_of_order is not None else "timeout")
    result["stage"] = "complete"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True, choices=range(9))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--trial-start", type=int, default=0)
    parser.add_argument("--num-trials-per-task", type=int, default=1)
    parser.add_argument("--request-offset", type=int, default=0)
    parser.add_argument("--steps-per-skill", type=int, default=STEPS_PER_SKILL)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/pi05_lilo")
    args = parser.parse_args(argv)
    if min(args.trial_start, args.request_offset) < 0 or min(args.num_trials_per_task, args.steps_per_skill) < 1:
        parser.error("Trial count and budget must be positive; offsets must be nonnegative")
    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl")
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "1")
    task = task_definitions()[args.task_id]
    max_steps = args.steps_per_skill * len(task["skills"])
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_dir = args.output_root.resolve() / ("pi05_ultra_long_task{:02d}_{}".format(args.task_id, stamp))
    run_dir.mkdir(parents=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "run.log", encoding="utf-8")])
    config = dict(protocol=PROTOCOL, lilo_commit=LILO_COMMIT, suite=SUITE, task=task,
                  simulation_environment=".venv-libero", simulation_python=sys.executable,
                  egl_device_id=os.environ["MUJOCO_EGL_DEVICE_ID"],
                  environment_seed=None, initialization="unseeded env.reset followed by ten B3 settling steps; new simulator per episode",
                  prompt=task["prompt"], max_policy_steps=max_steps, steps_per_skill=args.steps_per_skill,
                  wait_steps=WAIT_STEPS, action_shape=[10, 7], replan_steps=5,
                  budget_scope="project end-to-end control budget; no LiLo planner or retries",
                  trial_start=args.trial_start, trials=args.num_trials_per_task,
                  restored_request_count=args.request_offset)
    write_json(run_dir / "run_config.json", config)
    rows_path = run_dir / "episodes.jsonl"
    rows_path.touch()
    rows, client, run_error = [], None, None
    offset = args.request_offset
    interrupted = False
    try:
        import imageio.v2 as imageio
        suite = initialize_benchmark()
        from openpi_client.websocket_client_policy import WebsocketClientPolicy
        config["repo_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        config["versions"] = {name: importlib.metadata.version(name) for name in
                              ("torch", "numpy", "scipy", "mujoco", "robosuite", "bddl", "openpi-client")}
        client = WebsocketClientPolicy(args.host, args.port)
        config["server_metadata"] = client.get_server_metadata()
        provenance = validate_server_provenance(config["server_metadata"], offset)
        write_json(run_dir / "run_config.json", config)
        for trial in range(args.trial_start, args.trial_start + args.num_trials_per_task):
            episode_id = "ultra_long_task{:02d}_trial{:03d}".format(args.task_id, trial)
            directory = run_dir / "episodes" / episode_id
            directory.mkdir(parents=True)
            result = dict(episode_progress(), episode_id=episode_id, task_id=args.task_id, trial_id=trial,
                          definition_id=task["definition_id"], task_name=task["name"], suite=SUITE,
                          protocol=PROTOCOL, skills_completed=0, total_skills=len(task["skills"]),
                          artifacts=str(directory.relative_to(run_dir)), environment_seed=None,
                          max_policy_steps=max_steps, service_id=provenance["service_id"])
            started = time.perf_counter()
            try:
                with contextlib.ExitStack() as stack:
                    result["stage"] = "environment_creation"
                    env, obs = suite.make_env(args.task_id)
                    stack.callback(env.close)
                    np.save(directory / "reset_sim_state.npy", env.get_sim_state())
                    result["stage"] = "settling"
                    obs = settle_environment(env, obs, result)
                    np.save(directory / "initial_sim_state.npy", env.get_sim_state())
                    predicates = SkillPredicates(env, task)
                    writer = stack.enter_context(imageio.get_writer(directory / "video.mp4", fps=10))
                    decisions = stack.enter_context((directory / "decisions.jsonl").open("w", encoding="utf-8"))
                    progress = stack.enter_context((directory / "skills.jsonl").open("w", encoding="utf-8"))

                    def on_frame(observation):
                        writer.append_data(np.concatenate((observation["observation/image"], observation["observation/wrist_image"]), axis=1))

                    def on_decision(decision, observation):
                        filename = "observation_{:06d}.npz".format(decision["client_request_id"])
                        np.savez_compressed(directory / filename, **observation)
                        decision["observation"] = filename
                        decisions.write(json.dumps(decision, allow_nan=False) + "\n")
                        decisions.flush()

                    def on_progress(record):
                        progress.write(json.dumps(record, allow_nan=False) + "\n")
                        progress.flush()

                    rollout(env, obs, client, task, predicates, result, on_frame, on_decision, on_progress,
                            max_steps, offset, provenance["service_id"])
            except KeyboardInterrupt:
                result.update(outcome="interrupted", error="KeyboardInterrupt")
                interrupted = True
            except Exception as error:
                result.update(outcome="infrastructure_error", error="{}: {}".format(type(error).__name__, error))
                logging.exception("Episode failed: %s", episode_id)
            finally:
                result["wall_seconds"] = time.perf_counter() - started
                offset += result["policy_requests"]
                write_json(directory / "result.json", result)
                with rows_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(result, allow_nan=False) + "\n")
                rows.append(result)
            logging.info("%s: %s, %d/%d skills, %d policy steps", episode_id, result["outcome"],
                         result["skills_completed"], result["total_skills"], result["policy_steps"])
            write_json(run_dir / "summary.json", summarize_lilo(rows, args.num_trials_per_task))
            if result["outcome"] in ("infrastructure_error", "interrupted"):
                break
    except KeyboardInterrupt:
        interrupted, run_error = True, "KeyboardInterrupt"
    except Exception as error:
        run_error = "{}: {}".format(type(error).__name__, error)
        logging.exception("Ultra-Long evaluation failed")
    finally:
        if client is not None:
            try:
                client._ws.close()
            except Exception as error:
                run_error = "WebSocket close: {}".format(error)
        summary = summarize_lilo(rows, args.num_trials_per_task, run_error)
        summary.update(output_dir=str(run_dir), task_id=args.task_id, suite=SUITE, protocol=PROTOCOL)
        write_json(run_dir / "summary.json", summary)
        print(json.dumps(summary, indent=2))
    return 130 if interrupted else (0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    sys.exit(main())
