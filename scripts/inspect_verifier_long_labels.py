"""Review saved Robometer pilot labels against source video on CPU, without inference."""

import argparse
import io
import json
from pathlib import Path
import sys
import textwrap

import numpy as np

from scripts.audit_verifier_data import read_rows, require
from scripts.label_verifier_long_pilot import prefix_indices, read_external_video
from scripts.prepare_verifier_long_data import json_text, publish_dataset, sha256_file


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def delta_statistics(values):
    values = np.asarray(values, dtype=np.float64)
    if not len(values):
        return {"count": 0}
    return dict(count=len(values), minimum=float(values.min()), maximum=float(values.max()),
                mean_absolute=float(np.abs(values).mean()), negative_fraction=float(np.mean(values < 0)),
                near_zero_fraction=float(np.mean(np.abs(values) < 0.001)), near_zero_threshold=0.001)


def inspect_episode(directory, expected, config):
    summary = read_json(directory / "summary.json")
    require({key: value for key, value in summary.items() if key != "artifact_sha256"} == expected,
            "Episode summary differs from final report: " + directory.name)
    for name in ("progress.json", "window_labels.jsonl", "progress.png"):
        require(sha256_file(directory / name) == summary["artifact_sha256"][name],
                "Saved label artifact changed: " + name)
    trace = read_json(directory / "progress.json")
    for key in ("episode_id", "task_id", "outcome"):
        require(trace[key] == summary[key], "Trace identity differs: " + key)
    count = summary["frames"]
    require(count >= 2 and trace["frame_indices"] == list(range(count)), "Trace frame timeline differs")
    require(trace["frame_index_origin"] == 0 and trace["view"] == "external_left", "Unexpected trace view")
    require(trace["context_frame_indices"] == [prefix_indices(step).tolist() for step in range(count)],
            "Trace context frames differ from the four-frame prefix")
    progress = np.asarray(trace["progress"], dtype=np.float64)
    probability = np.asarray(trace["success_probability"], dtype=np.float64)
    for values in (progress, probability):
        require(values.shape == (count,) and np.isfinite(values).all() and
                np.all((values >= 0) & (values <= 1)), "Expected finite frame predictions in [0,1]")
    for key, actual in (("progress_first", progress[0]), ("progress_last", progress[-1]),
                        ("progress_min", progress.min()), ("progress_max", progress.max()),
                        ("terminal_success_probability", probability[-1])):
        require(np.isclose(actual, summary[key], rtol=0, atol=1e-8), "Summary prediction differs: " + key)
    deltas = np.diff(progress)
    windows = read_rows(directory / "window_labels.jsonl")
    require(len(windows) == summary["windows"], "Window count differs")
    position, full_window_deltas = 0, []
    for index, window in enumerate(windows):
        n = window["executed_steps"]
        require(window["episode_id"] == summary["episode_id"] and window["decision_index"] == index,
                "Window identity differs")
        require(1 <= n <= 5 and window["policy_step"] == position and position + n < count,
                "Window timeline differs")
        require(window["valid_subchunk_mask"] == [i < n for i in range(5)], "Window mask differs")
        actual = np.asarray(window["delta_progress"], dtype=np.float64)
        require(actual.shape == (n,) and np.allclose(actual, deltas[position:position + n], rtol=0, atol=1e-8),
                "Window deltas differ from raw progress")
        require(np.allclose([window["progress_start"], window["progress_end"]],
                            progress[[position, position + n]], rtol=0, atol=1e-8), "Window endpoints differ")
        if n == 5:
            full_window_deltas.append(float(progress[position + n] - progress[position]))
        position += n
    require(position == count - 1, "Windows do not cover the full episode")
    video = Path(trace["video"])
    require(sha256_file(video) == config["video_sha256"][summary["episode_id"]], "Source video changed")
    frames = read_external_video(video, count - 1)
    drop = int(np.argmin(deltas)) + 1
    peak = int(np.argmax(progress))
    selected = [("Initial", 0), ("Peak progress", peak), ("Before minimum delta", drop - 1),
                ("After minimum delta", drop), ("Pre-terminal", count - 2), ("Terminal", count - 1)]
    report = dict(
        episode_id=summary["episode_id"], task_id=summary["task_id"], outcome=summary["outcome"],
        prompt=trace["prompt"], frames=count, windows=len(windows),
        progress_first=float(progress[0]), progress_last=float(progress[-1]),
        progress_peak=float(progress[peak]), peak_frame=peak, terminal_success_probability=float(probability[-1]),
        one_step_deltas=delta_statistics(deltas), full_five_step_window_deltas=delta_statistics(full_window_deltas),
        minimum_delta_event=dict(
            before_frame=drop - 1, after_frame=drop, delta=float(deltas[drop - 1]),
            progress_before=float(progress[drop - 1]), progress_after=float(progress[drop]),
            context_before=trace["context_frame_indices"][drop - 1], context_after=trace["context_frame_indices"][drop]),
        selected_frames=[dict(role=role, frame=step) for role, step in selected],
        figure=summary["episode_id"] + "_review.png")
    return report, trace, frames, selected


def plot_review(trace, frames, selected):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with plt.rc_context({"font.size": 10, "figure.facecolor": "white"}):
        fig = plt.figure(figsize=(12, 10), layout="constrained")
        grid = fig.add_gridspec(3, 3, height_ratios=(1, 1, 0.8))
        progress = np.asarray(trace["progress"])
        for index, (role, step) in enumerate(selected):
            axis = fig.add_subplot(grid[index // 3, index % 3])
            axis.imshow(frames[step], interpolation="nearest")
            axis.set_title("{} | frame {}\np={:.4f}, success={:.4f}".format(
                role, step, progress[step], trace["success_probability"][step]), fontsize=10)
            axis.set_axis_off()
        axis = fig.add_subplot(grid[2, :])
        steps = np.asarray(trace["frame_indices"])
        axis.plot(steps, progress, color="#0072B2", label="Raw progress")
        axis.plot(steps, trace["success_probability"], color="#D55E00", linestyle="--", label="Success probability")
        for step in sorted({step for _, step in selected}):
            axis.axvline(step, color="#666666", linestyle=":", linewidth=0.7)
            axis.plot(step, progress[step], "o", color="#0072B2", markersize=4)
        axis.set(xlabel="Control step / video frame index", ylabel="Model output [0, 1]", ylim=(-0.03, 1.03),
                 title="Raw predictions; dotted lines mark the selected frames above")
        axis.grid(alpha=0.2)
        axis.legend(loc="upper left", frameon=False)
        fig.suptitle(trace["episode_id"] + " | recorded outcome: " + trace["outcome"] + "\n" +
                     textwrap.fill(trace["prompt"], 100), fontsize=12)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=130, metadata={
            "Description": "External view from source MP4; raw predictions, no smoothing or rescaling"})
        plt.close(fig)
    return buffer.getvalue()


def run_review(run_dir, output_dir, text_only=False):
    run_dir, output_dir = Path(run_dir).resolve(), Path(output_dir).resolve()
    require(output_dir != run_dir and run_dir not in output_dir.parents, "Use a review directory outside the label run")
    summary = read_json(run_dir / "summary.json")
    config = read_json(run_dir / "run_config.json")
    require(summary["schema"] == config["schema"] == "verifier_long_robometer_pilot_v1" and
            summary["status"] == "COMPLETE", "Use a completed pilot run")
    ids = config["selected_episode_ids"]
    require(ids == [row["episode_id"] for row in summary["per_episode"]] and
            len(ids) == len(set(ids)) == summary["episodes"], "Final episode inventory differs")
    files, reports = {}, []
    for expected in summary["per_episode"]:
        report, trace, frames, selected = inspect_episode(run_dir / expected["episode_id"], expected, config)
        if text_only:
            report.pop("figure")
        else:
            files[report["figure"]] = plot_review(trace, frames, selected)
        reports.append(report)
    require(sum(row["frames"] for row in reports) == summary["labeled_frames"] and
            sum(row["windows"] for row in reports) == summary["labeled_windows"], "Final totals differ")
    pairs = []
    for task in sorted({row["task_id"] for row in reports}):
        rows = {row["outcome"]: row for row in reports if row["task_id"] == task}
        require(set(rows) == {"success", "timeout"}, "Expected one success and timeout per pilot task")
        pairs.append(dict(task_id=task, success_episode=rows["success"]["episode_id"],
                          timeout_episode=rows["timeout"]["episode_id"],
                          terminal_progress_gap=rows["success"]["progress_last"] - rows["timeout"]["progress_last"]))
    report = dict(
        schema="verifier_long_label_review_v1", inspection_status="COMPLETE", label_quality_status="REVIEW_REQUIRED",
        scope="CPU artifact and numerical checks; semantic label quality remains unverified.",
        text_only=text_only,
        run_dir=str(run_dir), output_dir=str(output_dir), model_requests=0, gpu_used=False,
        source_sha256={name: sha256_file(run_dir / name) for name in ("summary.json", "run_config.json")},
        review_script_sha256=sha256_file(Path(__file__)),
        episodes=len(reports), frames=sum(row["frames"] for row in reports), windows=sum(row["windows"] for row in reports),
        note="Negative deltas and partial progress in timeouts are retained. Five-step statistics use full executed windows only.",
        per_episode=reports, paired_terminal_progress=pairs)
    files["review.json"] = json_text(report, indent=2).encode("utf-8")
    publish_dataset(output_dir, files)
    return report


def text_summary(report):
    """Keep the numerical evidence copyable; the full report remains in review.json."""
    lines = [
        "inspection_status: " + report["inspection_status"],
        "label_quality_status: " + report["label_quality_status"],
        "episodes={} frames={} windows={} model_requests=0 gpu_used=false".format(
            report["episodes"], report["frames"], report["windows"]),
        "Semantic label quality remains unverified; negative deltas are preserved."]
    keys = ("progress_first", "progress_last", "progress_peak", "peak_frame", "terminal_success_probability")
    for row in report["per_episode"]:
        lines.extend(["", "{} | task={} | outcome={}".format(row["episode_id"], row["task_id"], row["outcome"]),
                      "prompt: " + row["prompt"],
                      "progress: " + json_text({key: row[key] for key in keys}).strip()])
        for key in ("one_step_deltas", "full_five_step_window_deltas", "minimum_delta_event"):
            lines.append(key + ": " + json_text(row[key]).strip())
    lines.extend(["", "paired_terminal_progress: " + json_text(report["paired_terminal_progress"]).strip(),
                  "full_report: " + str(Path(report["output_dir"]) / "review.json")])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--text-only", action="store_true", help="Print a compact text report and skip new review images")
    args = parser.parse_args(argv)
    try:
        report = run_review(args.run_dir, args.output_dir, text_only=args.text_only)
        print(text_summary(report) if args.text_only else json_text(report, indent=2), end="")
        return 0
    except Exception as error:
        print(json_text(dict(inspection_status="ERROR", error=str(error))), file=sys.stderr, end="")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
