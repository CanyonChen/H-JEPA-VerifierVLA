"""Materialize only the frozen visual features used by each train/validation split."""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path, PurePosixPath
import pickle
import shutil

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


def read_episode(ep, windows, label_dir, recorded_root=None):
    from scripts.verivla_prepare import artifact_path
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
        with np.load(artifact_path(w["observation"], recorded_root), allow_pickle=False) as obs:
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
    frames = read_external_video(artifact_path(ep["video"], recorded_root), pos)
    return np.stack(images), frames[1:], dict(actions=torch.from_numpy(actions), state=torch.from_numpy(states),
              delta=torch.from_numpy(delta), mask=torch.from_numpy(mask), offset=torch.tensor(offsets),
              label_sha256=summary["artifact_sha256"]["window_labels.jsonl"])


def encode_frames(net, frames, batch):
    items = []
    for i in range(0, len(frames), batch):
        z = encode(net, frames[i:i + batch]).to(device="cpu", dtype=torch.bfloat16)
        require(z.shape[1:] == (256, 1408) and torch.isfinite(z).all().item(), "Invalid visual features")
        items.append(z)
    return torch.cat(items)


def cache_matches(file, ep, label_dir, label_sha):
    """Check the archive header and small supervision tensors without paging in visual features."""
    if not file.is_file():
        return False
    n, steps = ep["policy_requests"], ep["policy_steps"]
    expected = {
        "context": ((n, 256, 1408), torch.bfloat16),
        "target": ((steps, 256, 1408), torch.bfloat16),
        "actions": ((n, 5, 7), torch.float32),
        "state": ((n, 8), torch.float32),
        "delta": ((n, 5), torch.float32),
        "mask": ((n, 5), torch.bool),
        "offset": ((n,), torch.int64),
    }
    try:
        tensors = torch.load(file, map_location="cpu", weights_only=True, mmap=True)
        if any(tuple(tensors[key].shape) != shape or tensors[key].dtype != dtype
               for key, (shape, dtype) in expected.items()):
            return False
        if "label_sha256" in tensors:
            return tensors["label_sha256"] == label_sha
        # Older B5 caches predate the label digest. Compare their actual
        # offsets, validity masks and deltas before sharing them with this run.
        rows = read_rows(label_dir / "window_labels.jsonl")
        mask = torch.tensor([r["valid_subchunk_mask"] for r in rows], dtype=torch.bool)
        offsets = torch.tensor([r["policy_step"] for r in rows], dtype=torch.int64)
        delta = torch.tensor([d for r in rows for d in r["delta_progress"]], dtype=torch.float32)
        return (torch.equal(tensors["offset"], offsets) and torch.equal(tensors["mask"], mask)
                and torch.equal(tensors["delta"][mask], delta))
    except (EOFError, RuntimeError, KeyError, TypeError, AttributeError, IndexError, pickle.UnpicklingError):
        return False


def build(stage, frame_batch, *, job=JOB, paths=None, feature_cache=None, recorded_root=None):
    pilot = stage == "long100"
    sources = SUITES[:1] if pilot else SUITES
    feature_cache = feature_cache if feature_cache is not None else job / "features"
    source = read_json(job / "source.json")["sha256"]
    out = {"train": [], "validation": []}
    pending_splits = []
    growth = peak_bytes = 0
    for suite in sources:
        indices, labels = paths[suite] if paths is not None else locations(suite)
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
            jobs = []
            for ep in eps:
                require(ep["suite"] == suite and ep["split"] == split and 0 <= ep["task_id"] < 10, "Invalid partition")
                key = ep["episode_id"]
                file = feature_cache / identity / split / (key + ".pt")
                task = 10 * SUITES.index(suite) + ep["task_id"]
                folder = labels / split / key
                label_sha = read_json(folder / "summary.json")["artifact_sha256"]["window_labels.jsonl"]
                require(sha256(folder / "window_labels.jsonl") == label_sha, "Changed labels: " + key)
                item = dict(file=str(file), episode=key, task=task, count=ep["policy_requests"])
                if paths is not None:
                    item["label_sha256"] = label_sha
                out[split].append(item)
                if not cache_matches(file, ep, folder, label_sha):
                    jobs.append((ep, folder, file))
                    # The existing file survives until the replacement is atomically published.
                    payload = (ep["policy_requests"] + ep["policy_steps"]) * 256 * 1408 * 2
                    peak_bytes = max(peak_bytes, growth + payload)
                    growth += payload - (file.stat().st_size if file.exists() else 0)
            if jobs:
                pending_splits.append((suite, split, indices / spec["files"]["windows"], jobs))

    if pending_splits:
        feature_cache.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(feature_cache).free
        if free < peak_bytes:
            raise OSError(f"Pending visual tensors need at least {peak_bytes / 2**30:.2f} GiB; "
                          f"{feature_cache} has {free / 2**30:.2f} GiB free. Select another --feature-cache.")
        net = encoder("cuda:0", job)
    for suite, split, window_file, jobs in pending_splits:
        windows = defaultdict(list)
        for row in read_rows(window_file):
            windows[row["episode_id"]].append(row)
        def decode(item):
            ep, folder, _ = item
            return read_episode(ep, windows[ep["episode_id"]], folder, recorded_root)
        # Decode the next episode while the GPU encodes the current one.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending = pool.submit(decode, jobs[0])
            for i, (ep, _, dest) in enumerate(jobs):
                images, frames, tensors = pending.result()
                if i + 1 < len(jobs):
                    pending = pool.submit(decode, jobs[i + 1])
                tensors["context"] = encode_frames(net, images, frame_batch)
                tensors["target"] = encode_frames(net, frames, frame_batch)
                atomic_write(dest, lambda p: torch.save(tensors, p))
                print(f"Encoded {suite}/{split} {i+1}/{len(jobs)} {ep['episode_id']}", flush=True)
    index = dict(version=VISUAL_VERSION, source_sha256=source, suites=list(sources), **out)
    save_json(job / stage / "features.json", index)
    print(json.dumps({k:sum(e["count"] for e in v) for k,v in out.items()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["long100", "joint40"])
    parser.add_argument("--frame-batch", type=int, default=32)
    parser.add_argument("--job-dir", type=Path, default=JOB)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--long-root", type=Path)
    parser.add_argument("--feature-cache", type=Path)
    parser.add_argument("--recorded-project-root", type=PurePosixPath, help="Original absolute Linux checkout path.")
    args = parser.parse_args()
    if bool(args.data_root) != bool(args.long_root):
        parser.error("--data-root and --long-root must be supplied together")
    if args.frame_batch < 1:
        parser.error("--frame-batch must be positive")
    if args.recorded_project_root and not args.recorded_project_root.is_absolute():
        parser.error("--recorded-project-root must be absolute")
    paths = None
    if args.data_root:
        from scripts.verivla_prepare import suite_paths
        paths = suite_paths(args.data_root.expanduser().resolve(), args.long_root.expanduser().resolve())
    torch.set_num_threads(2)
    build(args.stage, args.frame_batch, job=args.job_dir.expanduser().resolve(), paths=paths,
          feature_cache=args.feature_cache.expanduser().resolve() if args.feature_cache else None,
          recorded_root=args.recorded_project_root)
