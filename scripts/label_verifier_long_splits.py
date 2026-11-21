"""Generate raw Robometer progress labels for the fixed Long train and validation splits."""

from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path

from scripts import robometer_backend as rb
from scripts.audit_verifier_data import read_rows, require
from scripts.inspect_verifier_long_labels import read_json
from scripts.label_verifier_long_pilot import label_episode
from scripts.prepare_verifier_long_data import json_text, publish_dataset, sha256_file
from scripts.probe_robometer_context import MODEL_FIELDS


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def run_labels(indices, output, model, batch_size=8):
    manifest_path = indices / "manifest.json"
    manifest = read_json(manifest_path)
    suite = manifest["suite"]
    require(manifest["schema"] == "verifier_long_split_v1" and manifest["status"] == "COMPLETE"
            and suite in ("libero_10", "libero_spatial", "libero_object", "libero_goal")
            and manifest["split_seed"] == 7, "Expected a completed LIBERO seed-7 split")
    config = dict(schema="verifier_long_raw_progress_v1", splits=["train", "validation"],
                  manifest_sha256=sha256_file(manifest_path), batch_size=batch_size,
                  model={k: model.provenance[k] for k in MODEL_FIELDS}, view="external_left_224x224",
                  context="Four linspace prefix frames 0..t", labels="Adjacent differences of raw progress",
                  labeler_sha256=sha256_file(Path(label_episode.__code__.co_filename)))
    publish_dataset(output, {"run_config.json": encoded(config)})
    totals = {}
    seen = set()
    for split in config["splits"]:
        spec = manifest["splits"][split]
        files = spec["files"]
        for name in files.values():
            require(sha256_file(indices / name) == manifest["index_sha256"][name], "Changed index: " + name)
        episodes = read_rows(indices / files["episodes"])
        windows = defaultdict(list)
        for row in read_rows(indices / files["windows"]):
            require(row["split"] == split, "Window split differs")
            windows[row["episode_id"]].append(row)
        ids = {e["episode_id"] for e in episodes}
        require(len(ids) == len(episodes) == spec["episodes"] and ids == set(windows), "Episode index differs")
        require(ids.isdisjoint(seen), "Episode shared between splits")
        seen.update(ids)
        frames = count = 0
        for i, ep in enumerate(episodes, 1):
            eid = ep["episode_id"]
            require(ep["split"] == split and ep["suite"] == suite and
                    len(windows[eid]) == ep["policy_requests"], "Episode or window index differs: " + eid)
            dest = output / split / eid
            video_hash = sha256_file(Path(ep["video"]))
            if dest.exists():
                summary = read_json(dest / "summary.json")
                require(summary["video_sha256"] == video_hash, "Source video changed: " + eid)
                for name in ("progress.json", "window_labels.jsonl"):
                    require(sha256_file(dest / name) == summary["artifact_sha256"][name], "Changed labels: " + eid)
            else:
                trace, labels, summary = label_episode(ep, windows[eid], model, batch_size)
                data = {"progress.json": encoded(trace),
                        "window_labels.jsonl": b"".join(encoded(row) for row in labels)}
                summary["video_sha256"] = video_hash
                summary["artifact_sha256"] = {name: hashlib.sha256(raw).hexdigest() for name, raw in data.items()}
                data["summary.json"] = encoded(summary)
                publish_dataset(dest, data)
            require(summary["episode_id"] == eid and summary["frames"] == ep["policy_steps"] + 1
                    and summary["windows"] == ep["policy_requests"], "Label counts differ: " + eid)
            frames += summary["frames"]
            count += summary["windows"]
            print(f"{split} {i}/{len(episodes)}  {eid}  frames={frames} windows={count}", flush=True)
        require(frames == spec["policy_steps"] + len(episodes) and count == spec["policy_requests"],
                "Split label totals differ: " + split)
        totals[split] = dict(episodes=len(episodes), frames=frames, windows=count)
    report = dict(status="COMPLETE", label_quality_status="REVIEW_REQUIRED", splits=totals, output_dir=str(output))
    publish_dataset(output / "complete", {"summary.json": encoded(report)})
    return report


if __name__ == "__main__":
    base = Path("outputs/verifier_long").resolve()
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = rb.RobometerLabeler()
    pilot_model = read_json(base / "robometer_pilot_seed7_gpu0" / "summary.json")["model"]
    require(all(model.provenance[k] == pilot_model[k] for k in MODEL_FIELDS), "Model settings differ from pilot")
    print(json_text(run_labels(base / "split_seed7", base / "robometer_raw_trainval_seed7_gpu0", model),
                    indent=2), end="", flush=True)
