"""Probe history-frame sensitivity at the four pilot minima; preserve all training labels."""

import argparse
import logging
import os
from pathlib import Path
import sys
import time

import numpy as np

from scripts import robometer_backend
from scripts.audit_verifier_data import require
from scripts.inspect_verifier_long_labels import inspect_episode, read_json
from scripts.prepare_verifier_long_data import json_text, publish_dataset, sha256_file


MODEL_FIELDS = ("model_id", "model_revision", "source_commit", "processor_id", "processor_revision",
                "dtype", "attention", "config_sha256", "weight_index_sha256", "package_versions")
VARIANTS = ("A", "B", "C", "D", "A_repeat", "D_repeat")


def crossed_prefixes(before, after):
    """Cross historical frames with the endpoint, plus duplicate endpoint controls."""
    require(len(before) == len(after) == 4 and after[-1] == before[-1] + 1,
            "Expected consecutive four-frame prefixes")
    variants = [list(before), list(before[:3]) + [after[-1]],
                list(after[:3]) + [before[-1]], list(after), list(before), list(after)]
    require(all(indices == sorted(indices) and min(indices) >= 0 for indices in variants),
            "A crossed prefix would contain a future historical frame")
    return variants


def decompose(progress):
    a, b, c, d = progress[:4]
    return dict(
        rerun_delta=float(d - a),
        current_effect_with_old_history=float(b - a), current_effect_with_new_history=float(d - c),
        history_effect_with_old_current=float(c - a), history_effect_with_new_current=float(d - b),
        current_component=float(((b - a) + (d - c)) / 2),
        history_component=float(((c - a) + (d - b)) / 2), interaction=float(d - c - b + a),
        within_batch_repeat_max_abs_error=float(max(abs(a - progress[4]), abs(d - progress[5]))))


def prepare_probe(run_dir):
    run_dir = Path(run_dir).resolve()
    summary = read_json(run_dir / "summary.json")
    config = read_json(run_dir / "run_config.json")
    require(summary["schema"] == config["schema"] == "verifier_long_robometer_pilot_v1" and
            summary["status"] == "COMPLETE", "Use a completed pilot")
    require(config["selected_episode_ids"] == [row["episode_id"] for row in summary["per_episode"]] and
            len(set(config["selected_episode_ids"])) == summary["episodes"] == 4, "Expected all four pilot episodes")
    cases = []
    for expected in summary["per_episode"]:
        row, trace, frames, _ = inspect_episode(run_dir / expected["episode_id"], expected, config)
        event = row["minimum_delta_event"]
        indices = crossed_prefixes(event["context_before"], event["context_after"])
        cases.append(dict(episode_id=row["episode_id"], task_id=row["task_id"], outcome=row["outcome"],
                          prompt=trace["prompt"], event=event, indices=indices,
                          sequences=np.stack([frames[selection] for selection in indices])))
    return dict(run_dir=str(run_dir), model=summary["model"], cases=cases,
                source_sha256={name: sha256_file(run_dir / name) for name in ("summary.json", "run_config.json")})


def run_probe(prepared, output_dir, labeler, gpu=0):
    for name in MODEL_FIELDS:
        require(labeler.provenance[name] == prepared["model"][name], "Pilot model setting changed: " + name)
    output_dir = Path(output_dir).resolve()
    run_dir = Path(prepared["run_dir"])
    require(output_dir != run_dir and run_dir not in output_dir.parents, "Use a separate probe output directory")
    started = time.monotonic()
    reports = []
    for case in prepared["cases"]:
        progress, success = labeler.predict(case["sequences"], case["prompt"])
        progress, success = np.asarray(progress, dtype=np.float64), np.asarray(success, dtype=np.float64)
        for values in (progress, success):
            require(values.shape == (6,) and np.isfinite(values).all() and np.all((values >= 0) & (values <= 1)),
                    "Expected six finite model outputs in [0,1]")
        event = case["event"]
        reports.append(dict(
            episode_id=case["episode_id"], task_id=case["task_id"], outcome=case["outcome"], prompt=case["prompt"],
            source_event=event,
            history_changed=event["context_before"][:3] != event["context_after"][:3],
            frame_indices=dict(zip(VARIANTS, case["indices"])),
            progress=dict(zip(VARIANTS, progress.tolist())), success_probability=dict(zip(VARIANTS, success.tolist())),
            saved_endpoint_max_abs_error=float(np.max(np.abs(
                progress[[0, 3]] - [event["progress_before"], event["progress_after"]]))),
            decomposition=decompose(progress)))
        logging.info("%s: evaluated six prefixes", case["episode_id"])
    report = dict(
        schema="robometer_context_probe_v1", status="COMPLETE", label_quality_status="REVIEW_REQUIRED",
        scope="Model input sensitivity at one minimum adjacent delta per pilot episode; not a label-quality validation.",
        run_dir=prepared["run_dir"], output_dir=str(output_dir), gpu=gpu,
        model_calls=len(reports), evaluated_prefixes=6 * len(reports), frames_per_prefix=4,
        variant_definition=dict(A="old history, old endpoint", B="old history, new endpoint",
                                C="new history, old endpoint", D="new history, new endpoint",
                                A_repeat="repeat A in the same batch", D_repeat="repeat D in the same batch"),
        component_definition="Average each input effect across the other factor; current_component + history_component = D - A.",
        interpretation="Components describe model score sensitivity, not ground-truth physical progress. Original labels are unchanged.",
        model=labeler.provenance, runtime=labeler.metrics(), model_load_seconds=getattr(labeler, "load_seconds", None),
        probe_seconds=time.monotonic() - started, source_sha256=prepared["source_sha256"],
        script_sha256=sha256_file(Path(__file__)), per_episode=reports)
    publish_dataset(output_dir, {"report.json": json_text(report, indent=2).encode("utf-8")})
    return report


def text_summary(report):
    lines = ["status: " + report["status"], "label_quality_status: " + report["label_quality_status"],
             "gpu={} model_calls={} evaluated_prefixes={}".format(
                 report["gpu"], report["model_calls"], report["evaluated_prefixes"]),
             "A=old history/old frame; B=old history/new frame; C=new history/old frame; D=new history/new frame.",
             report["interpretation"]]
    for row in report["per_episode"]:
        lines.extend(["", "{} | outcome={} | history_changed={}".format(
            row["episode_id"], row["outcome"], row["history_changed"]),
            "source_event: " + json_text(row["source_event"]).strip(),
            "progress: " + json_text(row["progress"]).strip(),
            "saved_endpoint_max_abs_error: " + str(row["saved_endpoint_max_abs_error"]),
            "decomposition: " + json_text(row["decomposition"]).strip()])
    lines.append("full_report: " + str(Path(report["output_dir"]) / "report.json"))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    try:
        require(args.gpu >= 0, "GPU must be nonnegative")
        require(not args.output_dir.exists(), "Probe output already exists; read report.json or choose another directory")
        prepared = prepare_probe(args.run_dir)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        logging.info("Loading the pinned Robometer model on physical GPU %s", args.gpu)
        labeler = robometer_backend.RobometerLabeler()
        print(text_summary(run_probe(prepared, args.output_dir, labeler, args.gpu)), end="")
        return 0
    except Exception as error:
        logging.exception("Context probe failed")
        print(json_text(dict(status="ERROR", error=str(error))), file=sys.stderr, end="")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
