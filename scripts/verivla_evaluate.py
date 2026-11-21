"""Paired pi05 versus pi05 + best verifier on five held-out LIBERO states per task."""
import argparse
from collections import Counter
import contextlib
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from scripts.audit_verifier_data import read_rows
from scripts.b5_closed_loop import PairedClient, calibration, paired_services, receive, save
from scripts.check_libero_env import PROJECT_ROOT, configure_libero_paths
from scripts.eval_libero import MAX_STEPS, episode_progress, rollout_episode
from scripts.verivla_prepare import SUITES, check, digest, inspect_suite, pi05_checkpoint, read, suite_paths

ARMS = ("base", "verifier")


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def held_out(index, suite, seen_in_training):
    episodes = read_rows(index / "test_episodes.jsonl")
    for ep in episodes:
        task, init = ep["task_id"], ep["init_state_id"]
        check(ep["suite"] == suite and ep["split"] == "test" and 0 <= task < 10 and 0 <= init < 50
              and ep["episode_id"] == f"{suite}_task{task:02d}_init{init:02d}"
              and ep["episode_id"] not in seen_in_training, f"Invalid held-out group: {ep['episode_id']}")
    check(Counter(e["task_id"] for e in episodes) == {t: 5 for t in range(10)}
          and len({e["episode_id"] for e in episodes}) == 50, f"Expected five distinct test groups per task: {suite}")
    return episodes


def cached_arm(path, identity, reference, ep, arm):
    if not path.is_file():
        return None
    row = read(path)
    check(row["identity"] == identity and row["reference"] == reference and row["arm"] == arm
          and row["episode"] == ep["episode_id"] and row["suite"] == ep["suite"] and row["split"] == "test"
          and row["task_id"] == ep["task_id"] and row["init_state_id"] == ep["init_state_id"]
          and row["outcome"] in ("success", "timeout") and row["task_success"] == (row["outcome"] == "success"),
          f"Incompatible or incomplete saved arm: {path}")
    return row


def summarize_pairs(pairs):
    wins = losses = both_success = both_timeout = 0
    for first, second in pairs:
        check(first["episode"] == second["episode"] and (first["arm"], second["arm"]) == ARMS,
              "Evaluation arms do not refer to the same held-out state")
        a, b = first["task_success"], second["task_success"]
        wins += int(b and not a)
        losses += int(a and not b)
        both_success += int(a and b)
        both_timeout += int(not a and not b)
    n = len(pairs)
    return dict(pairs=n, base_successes=both_success + losses, verifier_successes=both_success + wins,
                base_success_rate=(both_success + losses) / n, verifier_success_rate=(both_success + wins) / n,
                difference=(wins - losses) / n, wins=wins, losses=losses,
                both_success=both_success, both_timeout=both_timeout)


def run_arm(env_class, suite, ep, arm, remote, worker, reference, query_id):
    task = ep["task_id"]
    initial = suite.get_task_init_states(task)[ep["init_state_id"]]
    result = dict(episode_progress(), episode=ep["episode_id"], suite=ep["suite"], arm=arm, task_id=task,
                  init_state_id=ep["init_state_id"], split="test", query_id=query_id)
    np.random.seed(7)
    env = env_class(bddl_file_name=suite.get_task_bddl_file_path(task), camera_heights=256, camera_widths=256)
    client = PairedClient(remote, worker, ep, arm, reference, query_id=query_id, shadow_first=False)
    try:
        env.seed(7)
        start = time.perf_counter()
        rollout_episode(env, client, initial, suite.get_task(task).language, result,
                        lambda obs: None, lambda decision, obs: None, MAX_STEPS[ep["suite"]])
        result.update(wall_seconds=time.perf_counter() - start, triggered=client.triggered, changed=client.changed)
    finally:
        env.close()
    check(result["outcome"] in ("success", "timeout"), f"Incomplete rollout: {ep['episode_id']} {arm}")
    return result


def evaluate(args):
    directory = args.job_dir.resolve() / "joint40"
    output = directory / "closed_loop"
    training = read(directory / "training.json")
    config = training["config"]
    features_path = directory / "features.json"
    check(training["status"] == "COMPLETE" and config["stage"] == "joint40" and config["tasks"] == 40
          and training["updates"] == config["target_updates"] and tuple(config["suites"]) == SUITES
          and config["features_sha256"] == digest(features_path), "Expected a completed joint 40-task model")
    features = read(features_path)
    seen = {ep["episode"] for part in ("train", "validation") for ep in features[part]}
    paths = suite_paths(args.data_root, args.long_root)
    episodes, splits, policies = {}, {}, set()
    for name, (index, labels) in paths.items():
        info, policy, _ = inspect_suite(name, index, labels, args.recorded_project_root)
        check(info["status"] == "ready", f"Missing completed training data for {name}")
        episodes[name] = held_out(index, name, seen)
        splits[name] = digest(index / "manifest.json")
        policies.add(policy)
    check(len(policies) == 1, "Expected identical pi05 weights for all suites")
    protocol = dict(version=1, policy_sha256=policies.pop(), training_sha256=digest(directory / "training.json"),
                    best_sha256=digest(directory / "best.pt"), manifests=splits, split="test",
                    seed=7, threshold=.8, k=2, K=10, calibration="two train episodes per suite, tasks 0 and 9")
    identity = fingerprint(protocol)
    summary_file = output / "summary.json"
    if summary_file.is_file():
        report = read(summary_file)
        check(report["status"] == "COMPLETE" and report["identity"] == identity
              and report["overall"]["pairs"] == 200, "Existing evaluation belongs to a different run")
        print(json.dumps(report["overall"]), flush=True)
        return

    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv
    output.mkdir(parents=True, exist_ok=True)
    all_pairs, by_suite = [], {}
    with contextlib.ExitStack() as life:
        remote, worker, prov = paired_services(life, output, args.scorer_python,
            ["--job-dir", str(directory.parent), "--stage", "joint40"], pi05_checkpoint())
        ready = receive(worker)
        check(ready["ready"] and ready["best_sha256"] == protocol["best_sha256"]
              and ready["training_sha256"] == protocol["training_sha256"]
              and prov["checkpoint_manifest"]["sha256"] == protocol["policy_sha256"], "Wrong live scoring or policy weights")
        for suite_id, name in enumerate(SUITES):
            suite_dir = output / name
            suite_dir.mkdir(exist_ok=True)
            ref_path = suite_dir / "calibration.json"
            if ref_path.is_file():
                stored = read(ref_path)
                check(stored["identity"] == identity and fingerprint(stored["reference"]) == stored["sha256"],
                      f"Different trigger calibration: {name}")
                ref = stored["reference"]
            else:
                ref = calibration(remote, paths[name][0], args.recorded_project_root)
                save(ref_path, dict(identity=identity, reference=ref, sha256=fingerprint(ref)))
            reference_sha = fingerprint(ref)
            sim_suite = benchmark.get_benchmark_dict()[name]()
            pairs = []
            for ep in episodes[name]:
                arms = []
                for arm in ARMS:
                    path = suite_dir / (ep["episode_id"] + "_" + arm + ".json")
                    row = cached_arm(path, identity, reference_sha, ep, arm)
                    if row is None:
                        row = run_arm(OffScreenRenderEnv, sim_suite, ep, arm, remote, worker, ref,
                                      10 * suite_id + ep["task_id"])
                        row.update(identity=identity, reference=reference_sha)
                        save(path, row)
                        print(ep["episode_id"], arm, row["outcome"], flush=True)
                    arms.append(row)
                pairs.append(arms)
            by_suite[name] = summarize_pairs(pairs)
            all_pairs.extend(pairs)
    report = dict(status="COMPLETE", identity=identity, protocol=protocol, by_suite=by_suite,
                  overall=summarize_pairs(all_pairs))
    save(summary_file, report)
    print(json.dumps(report["overall"]), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--long-root", type=Path, required=True)
    parser.add_argument("--recorded-project-root")
    parser.add_argument("--scorer-python", type=Path, default=PROJECT_ROOT / ".venv-robometer/bin/python")
    options = parser.parse_args()
    configure_libero_paths()
    evaluate(options)
