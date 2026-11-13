"""Read saved baseline trajectories for verifier data reuse; CPU only, no writes."""

import argparse
from collections import Counter
import json
from pathlib import Path
import sys

import numpy as np

VALID_OUTCOMES = {"success", "timeout", "out_of_order"}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def video_info(path):
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path))
    try:
        metadata = next(reader)
    finally:
        reader.close()
    frames, _ = imageio_ffmpeg.count_frames_and_secs(str(path))
    return dict(frames=frames, size=metadata["size"], fps=metadata["fps"])


def inspect_episode(directory, row, read_video=video_info):
    saved = json.loads((directory / "result.json").read_text(encoding="utf-8"))
    index_key = "trial_id" if "trial_id" in row else "init_state_id"
    for key in ("task_id", index_key, "outcome", "policy_steps", "policy_requests"):
        require(saved[key] == row[key], "result.json differs from episode index: " + key)
    decisions = read_rows(directory / "decisions.jsonl")
    require(len(decisions) == row["policy_requests"], "Decision count differs from policy_requests")
    require(bool(decisions), "No recorded policy decisions")
    position, prompt = 0, None
    histogram = Counter()
    observations = set()
    for decision in decisions:
        require(decision["policy_step"] == position, "Non-contiguous executed action timeline")
        count = decision["executed_steps"]
        require(isinstance(count, int) and 1 <= count <= 5, "Expected an executed prefix of 1..5 steps")
        require(decision.get("response_received") is True, "Policy response was not received")
        actions = np.asarray(decision["actions"])
        require(actions.shape == (10, 7) and np.isfinite(actions).all(), "Expected finite generated (10,7) actions")
        path = (directory / decision["observation"]).resolve()
        require(path not in observations, "Observation file reused for multiple decisions")
        observations.add(path)
        with np.load(path, allow_pickle=False) as observation:
            state = observation["observation/state"]
            require(state.shape == (8,) and np.isfinite(state).all(), "Expected finite (8,) robot state")
            for key in ("observation/image", "observation/wrist_image"):
                frame = observation[key]
                require(frame.shape == (224, 224, 3) and frame.dtype == np.uint8,
                        "Expected uint8 (224,224,3) " + key)
            current_prompt = observation["prompt"].item()
            require(isinstance(current_prompt, str) and bool(current_prompt.strip()), "Missing task instruction")
            if prompt is None:
                prompt = current_prompt
                if row.get("task_description") is not None:
                    require(prompt == row["task_description"], "Saved instruction differs from indexed task")
            require(current_prompt == prompt, "Task instruction changed inside episode")
        histogram[count] += 1
        position += count
    require(position == row["policy_steps"], "Executed prefixes do not sum to policy_steps")
    video = read_video(directory / "video.mp4")
    require(video["frames"] == position + 1, "Video must have policy_steps+1 frames, including terminal observation")
    require(tuple(video["size"]) == (448, 224), "Expected side-by-side 448x224 policy views")
    require(abs(video["fps"] - 10) < 0.01, "Expected recorded video fps=10")
    return dict(policy_steps=position, policy_requests=len(decisions), video_frames=video["frames"],
                observation_files=len(observations), executed_steps_histogram=histogram,
                full_five_step_prefixes=histogram[5], short_prefixes=sum(v for k, v in histogram.items() if k < 5))


def audit_run(run_dir, read_video=video_info, progress=None):
    run_dir = Path(run_dir).resolve()
    report = dict(run_dir=str(run_dir), audit_status="CHECK_REQUIRED", errors=[], episodes_with_errors=0,
                  checked_episodes=0, retained_artifact_episodes=0, per_task=[])
    totals = Counter()
    histogram = Counter()
    try:
        config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        rows = read_rows(run_dir / "episodes.jsonl")
        suite = config["suite"]
        report.update(suite=suite, reported_status=summary["status"], indexed_episodes=len(rows),
                      planned_episodes=summary["planned_episodes"],
                      reported_outcomes=dict(Counter(row["outcome"] for row in rows)))
        require(suite in ("libero_10", "ultra_long_all"), "Expected the recorded Long or Ultra-Long baseline")
        require(summary["status"] == "PASS", "Use the final completed run, not an interrupted segment")
        require(len(rows) == summary["recorded_episodes"] == summary["planned_episodes"], "Incomplete episode index")
        require(all(row["outcome"] in VALID_OUTCOMES for row in rows), "Episode index contains infrastructure/interrupted records")
        require(len(rows) == summary["valid_episodes"], "Valid episode count differs from summary")
        index_key = "trial_id" if suite == "ultra_long_all" else "init_state_id"
        ids, artifacts = set(), set()
        tasks = {}
        for index, row in enumerate(rows):
            task_id = row["task_id"]
            task = tasks.setdefault(task_id, dict(task_id=task_id, indexed_episodes=0, checked_episodes=0,
                                                 outcomes=Counter(), full_five_step_prefixes=0, short_prefixes=0))
            task["indexed_episodes"] += 1
            task["outcomes"][row["outcome"]] += 1
            try:
                episode_id = (task_id, row[index_key])
                require(episode_id not in ids, "Duplicate episode ID in final index")
                ids.add(episode_id)
                directory = (run_dir / row["artifacts"]).resolve()
                require(directory not in artifacts, "Duplicate artifact directory in final index")
                artifacts.add(directory)
                if run_dir not in directory.parents:
                    report["retained_artifact_episodes"] += 1
                result = inspect_episode(directory, row, read_video)
                report["checked_episodes"] += 1
                task["checked_episodes"] += 1
                for key in ("policy_steps", "policy_requests", "video_frames", "observation_files",
                            "full_five_step_prefixes", "short_prefixes"):
                    totals[key] += result[key]
                for key in ("full_five_step_prefixes", "short_prefixes"):
                    task[key] += result[key]
                histogram.update(result["executed_steps_histogram"])
            except Exception as error:
                report["episodes_with_errors"] += 1
                if len(report["errors"]) < 20:
                    report["errors"].append(dict(task_id=task_id, episode_index=row.get(index_key),
                                                 error="{}: {}".format(type(error).__name__, error)))
            if progress and ((index + 1) % 50 == 0 or index + 1 == len(rows)):
                progress("{}: inspected {}/{} episodes".format(suite, index + 1, len(rows)))
        report["per_task"] = [tasks[key] for key in sorted(tasks)]
        report["split_group_key"] = (["suite", "task_id", "init_state_id"] if suite == "libero_10"
                                    else ["source_episode_identity", "saved_initial_state_identity"])
        report["unique_indexed_episode_ids"] = len(ids)
        report["verified_totals"] = dict(totals, executed_steps_histogram=dict(sorted(histogram.items())))
        require(set(tasks) == set(config["task_ids"]), "Task IDs differ from run_config")
        require(all(t["indexed_episodes"] == config["trials_per_task"] for t in tasks.values()),
                "Per-task episode count differs from run_config")
        if not report["episodes_with_errors"]:
            for key in ("policy_steps", "policy_requests"):
                require(totals[key] == summary[key], "Verified total differs from summary: " + key)
            report["audit_status"] = "COMPLETE"
    except Exception as error:
        report["errors"].append(dict(error="{}: {}".format(type(error).__name__, error)))
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, nargs="+", required=True)
    args = parser.parse_args(argv)
    reports = [audit_run(path, progress=lambda text: print(text, file=sys.stderr, flush=True)) for path in args.run_dir]
    complete = all(report["audit_status"] == "COMPLETE" for report in reports)
    print(json.dumps(dict(
        schema="verifier_data_audit_v1", audit_status="COMPLETE" if complete else "CHECK_REQUIRED",
        scope="Read-only CPU audit of final episode indices, every decision NPZ and decoded video frame counts.",
        training_readiness="Progress labels and independent train/validation/test groups still required.",
        runs=reports,
        data_contract=dict(generated_action_shape=[10, 7], executed_prefix_limit=5, robot_state_shape=[8],
                           state_capture="Decision boundaries only; no per-step or terminal state NPZ.",
                           video_layout="448x224 MP4: left external view, right wrist view, already resized/rotated.",
                           target_frames="For a decision at step t with n executed actions, video frames t+1..t+n.",
                           full_prefix_proposal="Five executed actions split into M=5, K=1; short prefixes need a separate mask policy.",
                           unexecuted_actions="Generated actions after executed_steps have no trajectory targets.",
                           alignment_scope="Checks writer-contract timeline and frame counts; not pixel-level visual alignment.",
                           split_status="No split assigned and no labels or models loaded.")),
        ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
