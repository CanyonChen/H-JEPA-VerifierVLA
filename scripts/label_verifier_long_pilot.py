"""Label four training trajectories with Robometer and report dense progress diagnostics."""

import argparse
from collections import defaultdict
import hashlib
import io
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
import time

import numpy as np

from scripts.audit_verifier_data import read_rows, require
from scripts.prepare_verifier_long_data import json_text, publish_dataset, sha256_file
from scripts import robometer_backend


def load_pilot(split_dir):
    manifest_path = split_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["schema"] == "verifier_long_split_v1" and manifest["status"] == "COMPLETE",
            "Use the completed Long split manifest")
    train_files = manifest["splits"]["train"]["files"]
    for name in train_files.values():
        require(sha256_file(split_dir / name) == manifest["index_sha256"][name], "Split file changed: " + name)
    groups = defaultdict(lambda: defaultdict(list))
    for episode in read_rows(split_dir / train_files["episodes"]):
        require(episode["suite"] == "libero_10" and episode["split"] == "train", "Expected Long training episodes")
        groups[episode["task_id"]][episode["outcome"]].append(episode)
    tasks = sorted(task for task, outcomes in groups.items() if outcomes["success"] and outcomes["timeout"])[:2]
    require(len(tasks) == 2, "Pilot needs two training tasks with both a success and a timeout")
    episodes = [min(groups[task][outcome], key=lambda episode: episode["init_state_id"])
                for task in tasks for outcome in ("success", "timeout")]
    selected = {episode["episode_id"] for episode in episodes}
    windows = defaultdict(list)
    for window in read_rows(split_dir / train_files["windows"]):
        if window["episode_id"] in selected:
            require(window["split"] == "train", "Expected training windows")
            windows[window["episode_id"]].append(window)
    for episode in episodes:
        require(len(windows[episode["episode_id"]]) == episode["policy_requests"], "Pilot window count differs")
    return episodes, windows


def read_external_video(path, policy_steps):
    import imageio_ffmpeg

    reader = imageio_ffmpeg.read_frames(str(path), pix_fmt="rgb24")
    try:
        metadata = next(reader)
        require(tuple(metadata["size"]) == (448, 224), "Expected the audited 448x224 two-view video")
        frames = [np.frombuffer(frame, dtype=np.uint8).reshape(224, 448, 3)[:, :224].copy() for frame in reader]
    finally:
        reader.close()
    require(len(frames) == policy_steps + 1, "Video must include all control steps and the terminal frame")
    return np.stack(frames)


def prefix_indices(step):
    """Match official frame-step evaluation with four equally spaced prefix frames."""
    return np.linspace(0, step, 4, dtype=int)


def label_episode(episode, windows, labeler, batch_size):
    frames = read_external_video(episode["video"], episode["policy_steps"])
    progress, success = [], []
    for start in range(0, len(frames), batch_size):
        stop = min(start + batch_size, len(frames))
        sequences = np.stack([frames[prefix_indices(step)] for step in range(start, stop)])
        batch_progress, batch_success = labeler.predict(sequences, episode["prompt"])
        for values in (batch_progress, batch_success):
            require(np.asarray(values).shape == (stop - start,), "Prediction length differs from requested frames")
            require(np.isfinite(values).all() and np.all(values >= 0) and np.all(values <= 1),
                    "Expected finite progress and success probabilities in [0,1]")
        progress.extend(np.asarray(batch_progress).tolist())
        success.extend(np.asarray(batch_success).tolist())
        if start == 0 or stop == len(frames) or stop % (batch_size * 10) == 0:
            logging.info("%s: labeled %s/%s frames", episode["episode_id"], stop, len(frames))
    progress = np.asarray(progress, dtype=np.float64)
    deltas = np.diff(progress)
    labels, position = [], 0
    for window in windows:
        count = window["executed_steps"]
        require(window["policy_step"] == window["context_frame_index"] == position, "Window timeline differs")
        require(window["target_frame_indices"] == list(range(position + 1, position + count + 1)),
                "Window target frames differ")
        require(window["valid_subchunk_mask"] == [i < count for i in range(5)], "Window mask differs")
        labels.append(dict(
            episode_id=episode["episode_id"], decision_index=window["decision_index"],
            policy_step=position, executed_steps=count, valid_subchunk_mask=window["valid_subchunk_mask"],
            delta_progress=deltas[position:position + count].tolist(),
            progress_start=float(progress[position]), progress_end=float(progress[position + count])))
        position += count
    require(position == episode["policy_steps"], "Labeled windows do not cover the episode")
    trace = dict(
        episode_id=episode["episode_id"], task_id=episode["task_id"], outcome=episode["outcome"],
        prompt=episode["prompt"], video=episode["video"], view="external_left", frame_index_origin=0,
        frame_indices=list(range(len(frames))),
        context_frame_indices=[prefix_indices(step).tolist() for step in range(len(frames))],
        progress=progress.tolist(), success_probability=success)
    summary = dict(
        episode_id=episode["episode_id"], task_id=episode["task_id"], outcome=episode["outcome"],
        frames=len(frames), windows=len(labels), progress_first=float(progress[0]), progress_last=float(progress[-1]),
        progress_min=float(progress.min()), progress_max=float(progress.max()),
        terminal_success_probability=float(success[-1]),
        delta_min=float(deltas.min()), delta_max=float(deltas.max()),
        negative_delta_fraction=float(np.mean(deltas < 0)),
        near_zero_delta_fraction=float(np.mean(np.abs(deltas) < 0.001)), near_zero_threshold=0.001)
    return trace, labels, summary


def plot_progress(trace):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 10}):
        fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True, layout="constrained")
        steps = np.asarray(trace["frame_indices"])
        axes[0].plot(steps, trace["progress"], color="#0072B2", label="Predicted progress")
        axes[0].plot(steps, trace["success_probability"], color="#D55E00", linestyle="--", label="Success probability")
        axes[0].set(ylim=(-0.03, 1.03), ylabel="Model output [0, 1]",
                    title=trace["episode_id"] + " / " + trace["outcome"])
        axes[0].legend(loc="upper left", frameon=False)
        axes[1].plot(steps[1:], np.diff(trace["progress"]), color="#0072B2")
        axes[1].axhline(0, color="#666666", linewidth=0.7)
        axes[1].set(xlabel="Control step", ylabel="One-step progress change")
        for axis in axes:
            axis.grid(alpha=0.2)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=150, metadata={"Description": "Raw predictions and adjacent differences"})
        plt.close(fig)
    return buffer.getvalue()


def run_pilot(split_dir, output_dir, labeler, batch_size=8, gpu=1):
    require(batch_size > 0, "Batch size must be positive")
    split_dir, output_dir = Path(split_dir).resolve(), Path(output_dir).resolve()
    episodes, windows = load_pilot(split_dir)
    config = dict(
        schema="verifier_long_robometer_pilot_v1", split_manifest=str(split_dir / "manifest.json"),
        split_manifest_sha256=sha256_file(split_dir / "manifest.json"),
        selected_episode_ids=[episode["episode_id"] for episode in episodes],
        selection="First two task IDs with training success and timeout, lowest initial-state ID per outcome",
        video_sha256={episode["episode_id"]: sha256_file(Path(episode["video"])) for episode in episodes},
        labeler=labeler.provenance, batch_size=batch_size, gpu=gpu,
        context="Four linspace indices over frames 0..t, last progress output for each control step t",
        view="external_left_224x224", smoothing="none", labels="Adjacent differences of raw predicted progress",
        scripts_sha256={path.name: sha256_file(path) for path in (Path(__file__), Path(robometer_backend.__file__))})
    publish_dataset(output_dir, {"run_config.json": json_text(config, indent=2).encode("utf-8")})
    started = time.perf_counter()
    reports, retained, new_frames = [], 0, 0
    for episode in episodes:
        directory = output_dir / episode["episode_id"]
        if directory.exists():
            summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
            for name, expected_hash in summary["artifact_sha256"].items():
                require(sha256_file(directory / name) == expected_hash, "Saved label artifact changed: " + name)
            retained += 1
        else:
            episode_started = time.perf_counter()
            trace, labels, summary = label_episode(episode, windows[episode["episode_id"]], labeler, batch_size)
            new_frames += summary["frames"]
            files = {
                "progress.json": json_text(trace).encode("utf-8"),
                "window_labels.jsonl": "".join(json_text(row) for row in labels).encode("utf-8"),
                "progress.png": plot_progress(trace)}
            summary["wall_seconds"] = time.perf_counter() - episode_started
            summary["artifact_sha256"] = {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}
            files["summary.json"] = json_text(summary, indent=2).encode("utf-8")
            publish_dataset(directory, files)
        reports.append(summary)
    elapsed = time.perf_counter() - started
    report = dict(
        schema=config["schema"], status="COMPLETE", label_quality_status="REVIEW_REQUIRED",
        output_dir=str(output_dir), gpu=gpu, episodes=len(reports), retained_episodes=retained,
        labeled_frames=sum(row["frames"] for row in reports), labeled_windows=sum(row["windows"] for row in reports),
        wall_seconds=elapsed, model_load_seconds=getattr(labeler, "load_seconds", None),
        new_labeled_frames=new_frames, new_frames_per_second=new_frames / elapsed if new_frames and elapsed > 0 else None,
        timing_scope="This invocation after model loading, including artifact output. Model loading is reported separately.",
        model=labeler.provenance, runtime=labeler.metrics(),
        per_episode=[{key: value for key, value in row.items() if key != "artifact_sha256"} for row in reports])
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output_dir, suffix=".tmp", delete=False) as stream:
        stream.write(json_text(report, indent=2))
        temporary = Path(stream.name)
    temporary.replace(output_dir / "summary.json")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["MPLBACKEND"] = "Agg"
    try:
        require(args.gpu >= 0 and args.batch_size > 0, "GPU and batch-size must be valid")
        load_pilot(args.split_dir.resolve())
        logging.info("Loading %s on physical GPU %s", robometer_backend.MODEL_ID, args.gpu)
        model_started = time.monotonic()
        labeler = robometer_backend.RobometerLabeler()
        logging.info("Model ready after %.1f seconds", time.monotonic() - model_started)
        report = run_pilot(args.split_dir, args.output_dir, labeler, args.batch_size, args.gpu)
        print(json_text(report, indent=2), end="")
        return 0
    except Exception as error:
        logging.exception("Robometer pilot failed")
        print(json_text(dict(status="ERROR", error=str(error), output_dir=str(args.output_dir))), file=sys.stderr, end="")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
