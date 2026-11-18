"""Materialize only the frozen visual features used by each train/validation split."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json

import numpy as np
import torch

from scripts.audit_verifier_data import read_rows, require
from scripts.b5_model import JOB, SUITES, VISUAL_VERSION, atomic_write, encode, encoder, locations, read_json, save_json, sha256
from scripts.label_verifier_long_pilot import read_external_video


def selected(episodes, pilot, split):
    if not pilot:
        return episodes
    groups = defaultdict(list)
    for ep in episodes:
        groups[ep["task_id"]].append(ep)
    limit = 2 if split == "train" else 1
    return [ep for task in sorted(groups) for ep in sorted(groups[task], key=lambda e:
            hashlib.sha256(e["episode_id"].encode()).digest())[:limit]]


def read_episode(ep, windows, label_dir):
    from pathlib import Path
    label_file = label_dir / "window_labels.jsonl"
    summary = read_json(label_dir / "summary.json")
    require(sha256(label_file) == summary["artifact_sha256"]["window_labels.jsonl"], "Label checksum differs")
    labels = read_rows(label_file)
    count = len(windows)
    require(count == len(labels) == ep["policy_requests"], "Incomplete episode labels")
    images, states = [], []
    actions = np.zeros((count, 5, 7), dtype=np.float32)
    delta = np.zeros((count, 5), dtype=np.float32)
    mask = np.zeros((count, 5), dtype=bool)
    offsets = []
    pos = 0
    for i, (w, y) in enumerate(zip(windows, labels)):
        n = w["executed_steps"]
        require(1 <= n <= 5 and w["episode_id"] == y["episode_id"] == ep["episode_id"]
                and w["decision_index"] == y["decision_index"] == i and w["split"] == ep["split"]
                and w["policy_step"] == y["policy_step"] == pos and y["executed_steps"] == n
                and w["target_frame_indices"] == list(range(pos + 1, pos + n + 1))
                and w["valid_subchunk_mask"] == y["valid_subchunk_mask"] == [j < n for j in range(5)],
                "Mismatched label/window: " + ep["episode_id"])
        with np.load(w["observation"], allow_pickle=False) as obs:
            rgb, state = obs["observation/image"], obs["observation/state"]
            require(rgb.shape == (224, 224, 3) and rgb.dtype == np.uint8 and state.shape == (8,), "Invalid input")
            images.append(rgb)
            states.append(state)
        actions[i, :n] = np.asarray(w["executed_actions"], dtype=np.float32).reshape(n, 7)
        delta[i, :n] = np.asarray(y["delta_progress"], dtype=np.float32).reshape(n)
        mask[i, :n] = True
        offsets.append(pos)
        pos += n
    states = np.asarray(states, dtype=np.float32)
    require(pos == ep["policy_steps"] and all(np.isfinite(x).all() for x in (actions, delta, states)), "Invalid trajectory")
    frames = read_external_video(Path(ep["video"]), pos)
    return np.stack(images), frames[1:], dict(actions=torch.from_numpy(actions), state=torch.from_numpy(states),
              delta=torch.from_numpy(delta), mask=torch.from_numpy(mask), offset=torch.tensor(offsets))


def encode_frames(net, frames, batch):
    items = []
    for i in range(0, len(frames), batch):
        z = encode(net, frames[i:i + batch]).to(device="cpu", dtype=torch.bfloat16)
        require(z.shape[1:] == (256, 1408) and torch.isfinite(z).all().item(), "Invalid visual features")
        items.append(z)
    return torch.cat(items)


def build(stage, frame_batch):
    pilot = stage == "long100"
    sources = SUITES[:1] if pilot else SUITES
    source = read_json(JOB / "source.json")["sha256"]
    out = {"train": [], "validation": []}
    net = None
    for suite in sources:
        indices, labels = locations(suite)
        manifest = read_json(indices / "manifest.json")
        raw = read_json(labels / "run_config.json")
        done = read_json(labels / "complete/summary.json")
        require(done["status"] == "COMPLETE" and manifest["suite"] == suite and manifest["task_count"] == 10
                and raw["manifest_sha256"] == sha256(indices / "manifest.json"), "Incomplete or mismatched split")
        identity = hashlib.sha256((source + VISUAL_VERSION + raw["manifest_sha256"] + sha256(labels / "run_config.json")).encode()).hexdigest()
        for split in out:
            spec = manifest["splits"][split]
            for name in spec["files"].values():
                require(sha256(indices / name) == manifest["index_sha256"][name], "Split checksum differs")
            episodes = read_rows(indices / spec["files"]["episodes"])
            require(done["splits"][split]["windows"] == spec["policy_requests"] and
                    done["splits"][split]["episodes"] == len(episodes), "Incomplete split labels")
            eps = selected(episodes, pilot, split)
            windows = defaultdict(list)
            for w in read_rows(indices / spec["files"]["windows"]):
                windows[w["episode_id"]].append(w)
            jobs = []
            for ep in eps:
                require(ep["suite"] == suite and ep["split"] == split and 0 <= ep["task_id"] < 10, "Invalid partition")
                key = ep["episode_id"]
                file = JOB / "features" / identity / split / (key + ".pt")
                task = 10 * SUITES.index(suite) + ep["task_id"]
                out[split].append(dict(file=str(file), episode=key, task=task, count=ep["policy_requests"]))
                if not file.exists():
                    jobs.append((ep, windows[key], labels / split / key, file))
            if jobs and net is None:
                net = encoder("cuda:0")
            # Decode the next episode while the GPU encodes the current one.
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = None
                for i, (ep, rows, folder, dest) in enumerate(jobs):
                    if pending is None:
                        pending = pool.submit(read_episode, ep, rows, folder)
                    images, frames, tensors = pending.result()
                    pending = pool.submit(read_episode, *jobs[i + 1][:3]) if i + 1 < len(jobs) else None
                    tensors["context"] = encode_frames(net, images, frame_batch)
                    tensors["target"] = encode_frames(net, frames, frame_batch)
                    atomic_write(dest, lambda p: torch.save(tensors, p))
                    print(f"Encoded {suite}/{split} {i+1}/{len(jobs)} {ep['episode_id']}", flush=True)
    index = dict(version=VISUAL_VERSION, source_sha256=source, suites=list(sources), **out)
    save_json(JOB / stage / "features.json", index)
    print(json.dumps({k:sum(e["count"] for e in v) for k,v in out.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["long100", "joint40"])
    parser.add_argument("--frame-batch", type=int, default=32)
    args = parser.parse_args()
    torch.set_num_threads(2)
    build(args.stage, args.frame_batch)
