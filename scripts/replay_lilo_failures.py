"""Replay recorded B4 actions to inspect contacts; never call or rescore a policy."""

import argparse
from collections import deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path

import numpy as np

from scripts.check_libero_env import PROJECT_ROOT, make_policy_observation
from scripts.inspect_lilo_run import inspect_run, skill_description
from scripts.lilo_diagnostics import object_snapshot
from scripts.lilo_protocol import (PROTOCOL, LILO_COMMIT, SkillPredicates, initialize_benchmark,
                                   settle_environment)


STATE_ATOL = 1e-5
IMAGE_MAE_LIMIT = 0.1  # uint8 pixel units; diagnostic agreement, not a scoring threshold.


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def compare_observation(observation, saved_path):
    with np.load(saved_path, allow_pickle=False) as saved:
        state_error = float(np.max(np.abs(observation["observation/state"] - saved["observation/state"])))
        image_errors = {key: float(np.mean(np.abs(observation[key].astype(float) - saved[key].astype(float))))
                        for key in ("observation/image", "observation/wrist_image")}
        prompt_matches = observation["prompt"] == str(saved["prompt"].item())
    return dict(state_max_abs_error=state_error, image_mae=image_errors, prompt_matches=prompt_matches,
                matches=bool(np.isfinite(state_error) and state_error <= STATE_ATOL and prompt_matches
                             and all(np.isfinite(v) and v <= IMAGE_MAE_LIMIT for v in image_errors.values())))


def replay_episode(suite, directory, row):
    config = read_json(directory.parent.parent / "run_config.json")
    task = config["task"]
    report = dict(task_id=row["task_id"], trial_id=row["trial_id"], status="ERROR", replayed_steps=0,
                  original_outcome=row["outcome"], original_policy_steps=row["policy_steps"],
                  original_repo_commit=config.get("repo_commit"), verification={}, tail=[])
    env = None
    try:
        if config["protocol"] != PROTOCOL or config["lilo_commit"] != LILO_COMMIT:
            raise ValueError("Replay requires the same saved v2 protocol and LiLo revision")
        if task["task_id"] != row["task_id"]:
            raise ValueError("Saved task and episode IDs disagree")
        trace = read_rows(directory / "skills.jsonl")
        decisions = read_rows(directory / "decisions.jsonl")
        if len(trace) != row["policy_steps"] or sum(d["executed_steps"] for d in decisions) != len(trace):
            raise ValueError("Recorded steps, skill trace and executed action counts disagree")
        event = next((event for record in trace for event in record["events"]
                      if event["kind"] == "out_of_order"), None)
        if event is None:
            raise ValueError("This diagnostic expects a recorded out_of_order episode")
        expected = event["expected_skill_index"]
        triggered = event["skill_index"]
        report["expected"] = skill_description(task, expected)
        report["triggered"] = skill_description(task, triggered)
        names = sorted({name for index in (expected, triggered) for atom in task["predicates"][index]
                        for name in atom[1:]})
        env, obs = suite.make_env(row["task_id"])
        obs = env.set_init_state(np.load(directory / "reset_sim_state.npy", allow_pickle=False))
        obs = settle_environment(env, obs)
        initial_error = float(np.max(np.abs(env.get_sim_state() - np.load(directory / "initial_sim_state.npy", allow_pickle=False))))
        report["verification"]["settled_state_max_abs_error"] = initial_error
        if not np.isfinite(initial_error) or initial_error > STATE_ATOL:
            report.update(status="DIVERGED", mismatch="Restored reset plus settling differs from the saved initial state")
            return report
        predicates = SkillPredicates(env, task)
        report["initial_objects"] = object_snapshot(env, predicates, names)
        history = {name: dict(both_fingerpads_contact_steps=0, last_both_fingerpads_contact_step=None,
                              max_height_delta_m=None) for name in names}
        tail = deque(maxlen=3)
        max_state_error, max_image_error, checked = 0.0, 0.0, 0
        for decision in decisions:
            if decision["policy_step"] != report["replayed_steps"]:
                raise ValueError("Decision policy_step is not contiguous")
            actions = np.asarray(decision["actions"])
            if actions.shape != (10, 7) or not 0 <= decision["executed_steps"] <= 5 or not np.isfinite(actions).all():
                raise ValueError("Invalid saved action block or executed step count")
            comparison = compare_observation(make_policy_observation(obs, task["prompt"]),
                                             directory / decision["observation"])
            checked += 1
            max_state_error = max(max_state_error, comparison["state_max_abs_error"])
            max_image_error = max(max_image_error, *comparison["image_mae"].values())
            report["verification"].update(checked_observations=checked, max_state_observation_error=max_state_error,
                                          max_image_mae=max_image_error)
            if not comparison["matches"]:
                report.update(status="DIVERGED", mismatch=dict(policy_step=report["replayed_steps"], observation=comparison))
                return report
            for action in actions[:decision["executed_steps"]]:
                obs, _, _, _ = env.step(action.tolist())
                record = trace[report["replayed_steps"]]
                report["replayed_steps"] += 1
                values = predicates.values()
                if record["policy_step"] != report["replayed_steps"] or values != record["skill_predicates"]:
                    report.update(status="DIVERGED", mismatch=dict(policy_step=report["replayed_steps"],
                                  saved_predicates=record["skill_predicates"], replayed_predicates=values))
                    return report
                objects = object_snapshot(env, predicates, names)
                for name, snapshot in objects.items():
                    item = history[name]
                    if snapshot["both_fingerpads_contact"]:
                        item["both_fingerpads_contact_steps"] += 1
                        item["last_both_fingerpads_contact_step"] = report["replayed_steps"]
                    height = snapshot["height_delta_m"]
                    if height is not None:
                        item["max_height_delta_m"] = max(height, item["max_height_delta_m"]) if item["max_height_delta_m"] is not None else height
                tail.append(dict(policy_step=report["replayed_steps"], objects=objects))
                for skill_event in record["events"]:
                    if skill_event["kind"] == "skill_completed":
                        predicates.begin_skill(skill_event["skill_index"] + 1)
        report.update(status="MATCHED", contact_history=history, tail=list(tail))
        report["verification"]["all_saved_skill_values_match"] = True
    except Exception as error:
        report["error"] = "{}: {}".format(type(error).__name__, error)
    finally:
        if env is not None:
            env.close()
    return report


def console_summary(report):
    """Keep server copy/paste short; the full contact trace stays in report.json."""
    summary = {key: report[key] for key in ("status", "output_dir", "policy_requests", "egl_device")}
    summary["overview"] = []
    for row in report["overview"]:
        event = row["out_of_order_event"] or {}
        summary["overview"].append(dict(task_id=row["task_id"], outcome=row["outcome"], policy_steps=row["policy_steps"],
                                        skills_completed=row["skills_completed"],
                                        expected=(event.get("expected") or row["next_skill"] or {}).get("instruction"),
                                        triggered=(event.get("triggered") or {}).get("instruction")))
    summary["replays"] = []
    for replay in report["replays"]:
        compact = {key: replay[key] for key in ("task_id", "status", "replayed_steps", "verification", "error", "mismatch") if key in replay}
        if replay["status"] == "MATCHED":
            compact["objects_at_trigger"] = {}
            for name, snapshot in replay["tail"][-1]["objects"].items():
                compact["objects_at_trigger"][name] = dict(
                    height_delta_m=snapshot["height_delta_m"], gripper_distance_m=snapshot["gripper_distance_m"],
                    both_fingerpads_contact=snapshot["both_fingerpads_contact"],
                    **replay["contact_history"][name])
                for key in ("turnon", "joint_positions"):
                    if key in snapshot:
                        compact["objects_at_trigger"][name][key] = snapshot[key]
        summary["replays"].append(compact)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--task-ids", type=int, nargs="+", default=[0, 3])
    parser.add_argument("--trial-id", type=int, default=0)
    parser.add_argument("--egl-device", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/lilo_replay")
    args = parser.parse_args(argv)
    if args.egl_device < 0 or args.trial_id < 0 or len(set(args.task_ids)) != len(args.task_ids):
        parser.error("Use nonnegative device/trial indices and unique task IDs")
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    if os.name != "nt":
        os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(args.egl_device))
    run_dir = args.run_dir.resolve()
    inspection = inspect_run(run_dir)
    if inspection["inspection_status"] != "COMPLETE":
        print(json.dumps(inspection, ensure_ascii=False, indent=2))
        return 1
    rows = {(r["task_id"], r["trial_id"]): r for r in inspection["episodes"]}
    if any((task, args.trial_id) not in rows for task in args.task_ids):
        parser.error("Selected task/trial is missing from the saved run")
    output = args.output_root.resolve() / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    output.mkdir(parents=True)
    report = dict(source_run_dir=str(run_dir), output_dir=str(output), egl_device=args.egl_device,
                  policy_requests=0, max_parallel_simulators=1,
                  note="Diagnostic replay only. MATCHED checks saved state, observations and skill flags within tolerances; "
                       "it is not a new episode score. Contact is a physical clue, not proof of intention or a new success rule.",
                  tolerances=dict(state_atol=STATE_ATOL, uint8_image_mae=IMAGE_MAE_LIMIT),
                  overview=[{key: r[key] for key in ("task_id", "trial_id", "outcome", "policy_steps", "skills_completed",
                                                     "next_skill", "out_of_order_event")} for r in rows.values()], replays=[])
    suite = initialize_benchmark()
    for task_id in args.task_ids:
        row = rows[task_id, args.trial_id]
        replay = replay_episode(suite, Path(row["artifacts"]), row)
        (output / "task{:02d}.json".format(task_id)).write_text(json.dumps(replay, indent=2, allow_nan=False), encoding="utf-8")
        report["replays"].append(replay)
    report["status"] = "MATCHED" if all(r["status"] == "MATCHED" for r in report["replays"]) else "CHECK_REQUIRED"
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(console_summary(report), ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == "MATCHED" else 1


if __name__ == "__main__":
    raise SystemExit(main())
