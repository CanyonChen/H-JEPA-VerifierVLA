"""Prepare the four LIBERO suites for the independent pi05 verifier experiment."""
import argparse
from collections import Counter
from contextlib import nullcontext
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal")
GROUPS = {"train": 40, "validation": 5, "test": 5}
H100_ROOT = "/home/jovyan/canyon43079/H-JEPA-VerifierVLA"


def suite_paths(data_root, long_root):
    paths = {s: (Path(data_root) / s / "split", Path(data_root) / s / "labels") for s in SUITES}
    paths["libero_10"] = (Path(long_root) / "split_seed7", Path(long_root) / "robometer_raw_trainval_seed7_gpu0")
    return paths


def artifact_path(value, recorded_project_root=None, project_root=ROOT):
    if recorded_project_root:
        path = PurePosixPath(value)
        old = PurePosixPath(recorded_project_root)
        try:
            suffix = path.relative_to(old)
        except ValueError:
            pass
        else:
            return Path(project_root).joinpath(*suffix.parts)
    return Path(value)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    with path.open("rb") as stream:
        hashed = hashlib.sha256()
        for block in iter(lambda: stream.read(1 << 20), b""):
            hashed.update(block)
        return hashed.hexdigest()


def check(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_suite(suite, split_dir, labels, recorded_root=None):
    manifest_file = split_dir / "manifest.json"
    if not manifest_file.is_file():
        check(not (labels / "complete/summary.json").exists(), f"{suite}: label summary exists without its split")
        return {"status": "missing split", "split": str(split_dir)}, None, None
    manifest = read(manifest_file)
    check(manifest["schema"] == "verifier_long_split_v1" and manifest["status"] == "COMPLETE"
          and manifest["suite"] == suite and manifest["split_seed"] == 7 and manifest["task_count"] == 10
          and manifest["groups_per_task"] == GROUPS, f"{suite}: incompatible split manifest {manifest_file}")
    contract = manifest["data_contract"]
    check(contract["generated_action_shape"] == [10, 7] and contract["robot_state_shape"] == [8]
          and contract["subchunks"] == 5 and contract["actions_per_subchunk"] == 1
          and contract["executed_prefix_limit"] == 5, f"{suite}: incompatible trajectory contract")
    groups = set()
    clients = set()
    report_path = labels / "complete/summary.json"
    complete = report_path.is_file()
    expected_labels = {}
    for part, per_task in GROUPS.items():
        spec = manifest["splits"][part]
        for fname in spec["files"].values():
            check(digest(split_dir / fname) == manifest["index_sha256"][fname], f"{suite}: changed index {fname}")
        episode_file = split_dir / spec["files"]["episodes"]
        counts = Counter()
        steps = windows = 0
        with episode_file.open(encoding="utf-8") as stream:
            for line in stream:
                ep = json.loads(line)
                task, init = ep["task_id"], ep["init_state_id"]
                key = (task, init)
                eid = f"{suite}_task{task:02d}_init{init:02d}"
                check(ep["episode_id"] == eid and ep["suite"] == suite and ep["split"] == part
                      and 0 <= init < 50 and key not in groups, f"{suite}: invalid episode {ep['episode_id']}")
                groups.add(key)
                counts[task] += 1
                windows += ep["policy_requests"]
                steps += ep["policy_steps"]
                clients.add(artifact_path(ep["artifacts"], recorded_root).parents[1] / "run_config.json")
                if complete and part != "test":
                    path = labels / part / eid
                    summary = read(path / "summary.json")
                    check(summary["episode_id"] == eid and summary["frames"] == ep["policy_steps"] + 1
                          and summary["windows"] == ep["policy_requests"], f"{eid}: invalid label coverage")
                    for fname in ("progress.json", "window_labels.jsonl"):
                        check(digest(path / fname) == summary["artifact_sha256"][fname], f"{eid}: changed {fname}")
        check(counts == {i: per_task for i in range(10)} and steps == spec["policy_steps"]
              and windows == spec["policy_requests"] and spec["episodes"] == 10 * per_task,
              f"{suite}: inconsistent {part} episode counts")
        if part != "test":
            expected_labels[part] = dict(episodes=10 * per_task, frames=steps + 10 * per_task, windows=windows)
    hashes = set()
    for path in clients:
        metadata = read(path)["server_metadata"]["evaluation"]
        check(metadata["policy_config"] == "pi05_libero" and metadata["backend"] == "jax"
              and metadata["action_horizon"] == 10, f"{suite}: incompatible pi05 policy {path}")
        hashes.add(metadata["checkpoint_manifest"]["sha256"])
    check(len(hashes) == 1 and len(next(iter(hashes))) == 64, f"{suite}: inconsistent pi05 weights")
    policy_sha = hashes.pop()
    info = dict(status="ready" if complete else "needs labels", split=str(split_dir), labels=str(labels),
                splits={p: manifest["splits"][p]["episodes"] for p in GROUPS}, policy_sha256=policy_sha)
    signature = None
    if (labels / "run_config.json").is_file() or complete:
        cfg = read(labels / "run_config.json")
        check(cfg["schema"] == "verifier_long_raw_progress_v1" and cfg["splits"] == ["train", "validation"]
              and cfg["manifest_sha256"] == digest(manifest_file), f"{suite}: labels do not match {manifest_file}")
        from scripts.robometer_backend import MODEL_ID, MODEL_REVISION, PROCESSOR_ID, PROCESSOR_REVISION, ROBOMETER_COMMIT

        model = cfg["model"]
        check(model["model_id"] == MODEL_ID and model["model_revision"] == MODEL_REVISION
              and model["processor_id"] == PROCESSOR_ID and model["processor_revision"] == PROCESSOR_REVISION
              and model["source_commit"] == ROBOMETER_COMMIT and cfg["view"] == "external_left_224x224"
              and cfg["context"] == "Four linspace prefix frames 0..t" and cfg["labels"] == "Adjacent differences of raw progress",
              f"{suite}: incompatible Robometer model or label protocol")
        signature = {k: cfg[k] for k in ("model", "view", "labels", "context", "batch_size")}
    if complete:
        summary = read(report_path)
        check(summary["status"] == "COMPLETE" and summary["splits"] == expected_labels,
              f"{suite}: incomplete train/validation labels in {report_path}")
    return info, policy_sha, signature


def pi05_checkpoint():
    explicit = os.environ.get("PI05_CHECKPOINT")
    asset = Path("openpi-assets/checkpoints/pi05_libero")
    if explicit and not explicit.startswith("gs://"):
        candidates = [Path(explicit).expanduser()]
    else:
        check(not explicit or explicit == "gs://openpi-assets/checkpoints/pi05_libero", "Unrecognized pi05 checkpoint URL")
        cache = Path(os.environ.get("OPENPI_DATA_HOME", "~/.cache/openpi")).expanduser()
        candidates = [cache / asset, Path("/data/models/openpi") / asset]
    for path in candidates:
        if (path / "params").is_dir() and (path / "assets/physical-intelligence/libero/norm_stats.json").is_file():
            return path.resolve()
    raise FileNotFoundError(f"Mount the existing pi05_libero checkpoint or set PI05_CHECKPOINT; searched {candidates}")


def collect_and_index(suite, base, sim_python, gpu):
    collection = base / "collection"
    runs = sorted(collection.glob("*/run_config.json"), key=lambda p: p.parent.name.rsplit("_", 2)[-2:])
    for path in runs:
        cfg = read(path)
        check(cfg["suite"] == suite and cfg["seed"] == 7 and cfg["trials_per_task"] == 50
              and cfg["task_ids"] == list(range(10)), f"Different collection protocol at {path}")

    def completed():
        for config in reversed(sorted(collection.glob("*/run_config.json"),
                                      key=lambda p: p.parent.name.rsplit("_", 2)[-2:])):
            run = config.parent
            summary_file = run / "summary.json"
            if summary_file.is_file() and (run / "episodes.jsonl").is_file():
                summary = read(summary_file)
                if summary["status"] == "PASS" and summary["valid_episodes"] == 500:
                    with (run / "episodes.jsonl").open(encoding="utf-8") as stream:
                        if sum(1 for _ in stream) == 500:
                            return run
        return None

    run = completed()
    if run is None:
        checkpoint = pi05_checkpoint()
        args = ["--suite", suite, "--gpu", str(gpu), "--egl-device", str(gpu), "--max-workers", "2",
                "--trials-per-task", "50", "--seed", "7", "--output-root", str(collection)]
        env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=str(gpu))
        env.pop("B5_SCHEDULER_GPU_UUID", None)
        env.pop("B5_SCHEDULER_ALLOCATED", None)
        reference = str(checkpoint)
        if runs:
            latest = runs[-1]
            reference = read(latest)["checkpoint"]
            if reference == "gs://openpi-assets/checkpoints/pi05_libero":
                check(checkpoint.as_posix().endswith("/openpi-assets/checkpoints/pi05_libero"),
                      f"Resuming {latest.parent} requires its original OpenPI cache layout")
                env["OPENPI_DATA_HOME"] = str(checkpoint.parents[2])
            else:
                check(Path(reference).expanduser().resolve() == checkpoint, f"Resume checkpoint differs from {latest}")
            args += ["--resume-from", str(latest.parent)]
        subprocess.run([str(sim_python), "-u", "-m", "scripts.run_libero_parallel", *args,
                        "--checkpoint", reference], cwd=ROOT, env=env, check=True)
        run = completed()
        check(run is not None, f"{suite}: collection did not produce all 500 episodes in {collection}")
    subprocess.run([str(sim_python), "-u", "-m", "scripts.prepare_verifier_long_data",
                    "--run-dir", str(run), "--output-dir", str(base / "split")], cwd=ROOT, check=True)


def prepare(data_root, long_root, gpu, recorded_root, sim_python, plan=False):
    paths = suite_paths(data_root, long_root)
    reference_policy = reference_labels = None
    reports = {}
    for suite, (indices, labels) in paths.items():
        report, policy, signature = inspect_suite(suite, indices, labels, recorded_root)
        if suite == "libero_10":
            reference_policy, reference_labels = policy, signature
            if not plan:
                check(report["status"] == "ready", f"Mount the existing Long splits, labels and rollout artifacts at {long_root}")
        if not plan and report["status"] == "missing split":
            collect_and_index(suite, indices.parent, sim_python, gpu)
            report, policy, signature = inspect_suite(suite, indices, labels, recorded_root)
        if reference_policy and policy:
            check(reference_policy == policy, f"{suite}: pi05 checkpoint differs from Long")
        if reference_labels and signature:
            check(reference_labels == signature, f"{suite}: Robometer configuration differs from Long")
        reports[suite] = report
    pending = [suite for suite in SUITES if reports[suite]["status"] == "needs labels"]
    if pending and not plan:
        # Release all rollout workers before allocating the Robometer GPU memory.
        from scripts.label_verifier_long_splits import run_labels
        from scripts.robometer_backend import RobometerLabeler

        model = RobometerLabeler()
        check(all(model.provenance[k] == v for k, v in reference_labels["model"].items()),
              "Installed Robometer model/processor differs from Long")
        for suite in pending:
            indices, labels = paths[suite]
            run_labels(indices, labels, model, batch_size=reference_labels["batch_size"])
            report, _, signature = inspect_suite(suite, indices, labels, recorded_root)
            check(signature == reference_labels and report["status"] == "ready", f"{suite}: label settings differ")
            reports[suite] = report
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--long-root", required=True, type=Path)
    parser.add_argument("--gpu", type=int, default=0, help="Physical GPU index for pi05, EGL and Robometer")
    parser.add_argument("--sim-python", type=Path, default=ROOT / ".venv-libero/bin/python")
    parser.add_argument("--recorded-project-root", default=H100_ROOT)
    parser.add_argument("--plan", action="store_true", help="Inspect existing inputs without collecting or writing")
    args = parser.parse_args()
    if args.gpu < 0:
        parser.error("--gpu must be nonnegative")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    root = args.data_root.resolve()
    lock = nullcontext()
    if not args.plan:
        from filelock import FileLock
        root.mkdir(parents=True, exist_ok=True)
        lock = FileLock(str(root / ".prepare.lock"), timeout=0)
    with lock:
        reports = prepare(root, args.long_root.resolve(), args.gpu, args.recorded_project_root,
                          args.sim_python.expanduser().absolute(), args.plan)
    print(json.dumps(reports, indent=2), flush=True)


if __name__ == "__main__":
    main()
