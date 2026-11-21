"""Create reproducible LIBERO-Long episode splits and executed-action window indices."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np

from scripts.audit_verifier_data import read_rows, require

SPLIT_COUNTS = {"train": 40, "validation": 5, "test": 5}


def json_text(value, indent=None):
    return json.dumps(value, ensure_ascii=False, indent=indent, allow_nan=False) + "\n"


def sha256_file(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assign_splits(rows, split_seed, suite="libero_10"):
    """Assign whole (suite, task, initial-state) groups without consulting outcomes."""
    groups = defaultdict(set)
    for row in rows:
        task, initial = row["task_id"], row["init_state_id"]
        require(isinstance(task, int) and 0 <= task < 10, "Expected a Long task ID in 0..9")
        require(isinstance(initial, int) and 0 <= initial < 50, "Expected an initial-state ID in 0..49")
        require(initial not in groups[task], "Duplicate task/initial-state group in final episode index")
        groups[task].add(initial)
    assignment = {}
    for task in sorted(groups):
        require(groups[task] == set(range(50)), "Each included task must contain initial states 0..49")
        def order_key(initial):
            identity = "{}|{}|{}|{}".format(suite, split_seed, task, initial)
            return hashlib.sha256(identity.encode("utf-8")).digest(), initial
        ordered = sorted(groups[task], key=order_key)
        start = 0
        for split, count in SPLIT_COUNTS.items():
            for initial in ordered[start:start + count]:
                assignment[(task, initial)] = split
            start += count
    return assignment


def index_episode(run_dir, row, split, suite="libero_10"):
    directory = (run_dir / row["artifacts"]).resolve()
    result_path = directory / "result.json"
    saved = json.loads(result_path.read_text(encoding="utf-8"))
    for key in ("task_id", "init_state_id", "outcome", "policy_steps", "policy_requests"):
        require(saved[key] == row[key], "Episode result differs from final index: " + key)
    decision_path = directory / "decisions.jsonl"
    decisions = read_rows(decision_path)
    require(bool(decisions) and len(decisions) == row["policy_requests"], "Decision count differs from episode")
    video_path = directory / "video.mp4"
    require(video_path.is_file(), "Missing video: " + str(video_path))
    require(saved.get("suite", suite) == suite and row.get("suite", suite) == suite, "Episode suite differs")
    episode_id = "{}_task{:02d}_init{:02d}".format(suite, row["task_id"], row["init_state_id"])
    windows, observation_paths = [], set()
    position, prompt = 0, None
    for decision_index, decision in enumerate(decisions):
        require(decision["policy_step"] == position, "Non-contiguous executed action timeline")
        count = decision["executed_steps"]
        require(isinstance(count, int) and 1 <= count <= 5, "Expected an executed prefix of 1..5 steps")
        require(decision.get("response_received") is True, "Policy response was not received")
        actions = np.asarray(decision["actions"])
        require(actions.shape == (10, 7) and np.isfinite(actions).all(), "Expected finite generated (10,7) actions")
        observation_path = (directory / decision["observation"]).resolve()
        require(observation_path.is_file(), "Missing observation: " + str(observation_path))
        require(observation_path not in observation_paths, "Observation file reused inside episode")
        observation_paths.add(observation_path)
        if prompt is None:
            with np.load(observation_path, allow_pickle=False) as observation:
                prompt = observation["prompt"].item()
            require(isinstance(prompt, str) and bool(prompt.strip()), "Missing task instruction")
            if row.get("task_description") is not None:
                require(prompt == row["task_description"], "Task instruction differs from final index")
        windows.append(dict(
            episode_id=episode_id, suite=suite, task_id=row["task_id"],
            init_state_id=row["init_state_id"], split=split, decision_index=decision_index,
            policy_step=position, executed_steps=count, observation=str(observation_path),
            video=str(video_path), context_frame_index=position,
            target_frame_indices=list(range(position + 1, position + count + 1)),
            executed_actions=actions[:count].tolist(), valid_subchunk_mask=[i < count for i in range(5)]))
        position += count
    require(position == row["policy_steps"], "Executed prefixes do not sum to episode policy_steps")
    episode = dict(
        episode_id=episode_id, suite=suite, task_id=row["task_id"],
        init_state_id=row["init_state_id"], split=split, outcome=row["outcome"], prompt=prompt,
        artifacts=str(directory), video=str(video_path), policy_steps=position,
        policy_requests=len(windows), retained_artifacts=run_dir not in directory.parents,
        source_sha256={"result.json": sha256_file(result_path), "decisions.jsonl": sha256_file(decision_path)})
    return episode, windows


def summarize(episodes, windows):
    histogram = Counter(window["executed_steps"] for window in windows)
    return dict(
        episodes=len(episodes), outcomes=dict(sorted(Counter(ep["outcome"] for ep in episodes).items())),
        policy_steps=sum(ep["policy_steps"] for ep in episodes), policy_requests=len(windows),
        full_five_step_prefixes=histogram[5], short_prefixes=sum(histogram[n] for n in range(1, 5)),
        executed_steps_histogram=dict(sorted(histogram.items())))


def publish_dataset(output_dir, files):
    """Publish all indices together and accept byte-identical reruns."""
    def verify_existing():
        for name, content in files.items():
            path = output_dir / name
            require(path.is_file() and path.read_bytes() == content,
                    "Output contains a different or incomplete split; choose another --output-dir: " + str(path))

    if output_dir.exists():
        verify_existing()
        return
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".verifier-long-", dir=str(output_dir.parent)) as temporary:
        staging = Path(temporary) / "dataset"
        staging.mkdir()
        for name, content in files.items():
            (staging / name).write_bytes(content)
        try:
            staging.rename(output_dir)
        except OSError:
            if not output_dir.exists():
                raise
            verify_existing()


def prepare_dataset(run_dir, output_dir, split_seed=7):
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    config = json.loads((run_dir / "run_config.json").read_text(encoding="utf-8"))
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    rows = read_rows(run_dir / "episodes.jsonl")
    suite = config["suite"]
    require(suite in ("libero_10", "libero_spatial", "libero_object", "libero_goal"), "Unknown LIBERO suite")
    require(summary["status"] == "PASS", "Use a completed final run")
    require(len(rows) == summary["planned_episodes"] == summary["recorded_episodes"] == summary["valid_episodes"],
            "Incomplete final episode index")
    require(bool(rows) and all(row["outcome"] in ("success", "timeout") for row in rows),
            "Expected valid Long success/timeout episodes")
    require(config["trials_per_task"] == 50, "Expected 50 initial states per task")
    require(set(row["task_id"] for row in rows) == set(config["task_ids"]), "Task IDs differ from run_config")
    assignment = assign_splits(rows, split_seed, suite)
    episodes = {split: [] for split in SPLIT_COUNTS}
    windows = {split: [] for split in SPLIT_COUNTS}
    artifact_dirs = set()
    for row in sorted(rows, key=lambda item: (item["task_id"], item["init_state_id"])):
        split = assignment[(row["task_id"], row["init_state_id"])]
        episode, episode_windows = index_episode(run_dir, row, split, suite)
        require(episode["artifacts"] not in artifact_dirs, "Duplicate source artifact directory in final index")
        artifact_dirs.add(episode["artifacts"])
        episodes[split].append(episode)
        windows[split].extend(episode_windows)
    all_episodes = [ep for split in SPLIT_COUNTS for ep in episodes[split]]
    all_windows = [window for split in SPLIT_COUNTS for window in windows[split]]
    totals = summarize(all_episodes, all_windows)
    for key in ("policy_steps", "policy_requests"):
        require(totals[key] == summary[key], "Indexed total differs from summary: " + key)
    require(totals["outcomes"].get("success", 0) == summary["successes"], "Success count differs from summary")
    require(totals["outcomes"].get("timeout", 0) == summary["timeouts"], "Timeout count differs from summary")
    files, split_reports = {}, {}
    for split in SPLIT_COUNTS:
        file_names = {"episodes": split + "_episodes.jsonl", "windows": split + "_windows.jsonl"}
        for kind, data in (("episodes", episodes[split]), ("windows", windows[split])):
            files[file_names[kind]] = "".join(json_text(row) for row in data).encode("utf-8")
        per_task = []
        for task in sorted(config["task_ids"]):
            per_task.append(dict(task_id=task, **summarize(
                [ep for ep in episodes[split] if ep["task_id"] == task],
                [window for window in windows[split] if window["task_id"] == task])))
        split_reports[split] = dict(files=file_names, per_task=per_task, **summarize(episodes[split], windows[split]))
    manifest = dict(
        schema="verifier_long_split_v1", status="COMPLETE", suite=suite,
        run_dir=str(run_dir), output_dir=str(output_dir), split_seed=split_seed,
        split_method="Per-task SHA256 order of " + suite + "|split_seed|task_id|init_state_id; 40/5/5 groups",
        split_group_key=["suite", "task_id", "init_state_id"], groups_per_task=SPLIT_COUNTS,
        split_seed_scope="Data partition only", task_count=len(config["task_ids"]),
        retained_artifact_episodes=sum(ep["retained_artifacts"] for ep in all_episodes),
        source_sha256={name: sha256_file(run_dir / name) for name in ("run_config.json", "summary.json", "episodes.jsonl")},
        preparation_script_sha256=sha256_file(Path(__file__)), totals=totals, splits=split_reports,
        index_sha256={name: hashlib.sha256(content).hexdigest() for name, content in files.items()},
        data_contract=dict(
            generated_action_shape=[10, 7], executed_prefix_limit=5, robot_state_shape=[8],
            subchunks=5, actions_per_subchunk=1, frame_index_origin=0,
            short_prefix_policy="Keep n executed actions and n target frames with a five-element validity mask",
            state_policy="Repeat the decision-boundary state across valid subchunks",
            video_layout="448x224: left external view, right wrist view",
            validation_scope="Final index, results, decision timelines, referenced paths and first-observation prompts",
            visual_audit="Use the preceding NPZ/video audit for dense content and decoded frame checks"),
        training_readiness="Episode splits and window indices prepared; progress labels and model training pending")
    files["manifest.json"] = json_text(manifest, indent=2).encode("utf-8")
    publish_dataset(output_dir, files)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split-seed", type=int, default=7)
    args = parser.parse_args(argv)
    try:
        manifest = prepare_dataset(args.run_dir, args.output_dir, args.split_seed)
    except (ValueError, OSError, KeyError) as error:
        print(json_text(dict(status="ERROR", error=str(error))), file=sys.stderr, end="")
        return 1
    report = {key: manifest[key] for key in (
        "schema", "status", "suite", "output_dir", "split_seed", "split_method", "groups_per_task",
        "task_count", "retained_artifact_episodes", "totals", "training_readiness")}
    report["manifest"] = str(args.output_dir.resolve() / "manifest.json")
    report["splits"] = {
        split: {key: value for key, value in details.items() if key != "per_task"}
        for split, details in manifest["splits"].items()}
    print(json_text(report, indent=2), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
