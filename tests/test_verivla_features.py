"""CPU-only checks for reusing frozen features in a separate 40-task run."""
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

from scripts import b5_features as features
from scripts import b5_model as model
from scripts import verivla_prepare


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data) + "\n", encoding="utf-8")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class VerifierFeaturesTests(unittest.TestCase):
    def test_explicit_shared_checkpoint_preserves_source_directory(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            shared = base / "shared"
            shared.mkdir()
            ckpt = shared / "weights.pt"
            metadata = b"test metadata"
            with zipfile.ZipFile(ckpt, "w") as archive:
                archive.writestr("latest.pth/data.pkl", metadata)
            with patch.object(model, "WEIGHT_BYTES", ckpt.stat().st_size), \
                    patch.object(model, "META_HASH", hashlib.sha256(metadata).hexdigest()), \
                    patch.object(model, "cached_source", side_effect=AssertionError("unexpected download")):
                model.prepare_source(job=base / "paper", checkpoint=ckpt)
                with self.assertRaises(FileNotFoundError):
                    model.prepare_source(job=base / "paper", checkpoint=shared / "missing")
            self.assertEqual(list(shared.iterdir()), [ckpt])
            self.assertEqual(model.read_json(base / "paper/source.json")["sha256"], digest(ckpt))

    def test_archived_long_reads_relocated_observation_and_video(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            old = "/home/jovyan/archive/project"
            rgb = np.zeros((224, 224, 3), dtype=np.uint8)
            np.savez(root / "observation.npz", **{"observation/image": rgb, "observation/state": np.zeros(8)})
            ep = dict(episode_id="ep", split="train", policy_steps=1, policy_requests=1, video=old + "/video.mp4")
            row = dict(episode_id="ep", policy_step=0, decision_index=0, executed_steps=1,
                       valid_subchunk_mask=[True, False, False, False, False])
            window = dict(row, split="train", observation=old + "/observation.npz",
                          target_frame_indices=[1], executed_actions=[[0.] * 7])
            write_json(root / "window_labels.jsonl", dict(row, delta_progress=[.2]))
            write_json(root / "summary.json",
                       {"artifact_sha256": {"window_labels.jsonl": digest(root / "window_labels.jsonl")}})
            resolve = verivla_prepare.artifact_path
            with patch.object(verivla_prepare, "artifact_path",
                              side_effect=lambda value, prefix: resolve(value, prefix, root)), \
                    patch.object(features, "read_external_video", return_value=np.stack([rgb, rgb])) as video:
                _, targets, tensors = features.read_episode(ep, [window], root, old)
            video.assert_called_once_with(root / "video.mp4", 1)
            self.assertEqual(len(targets), 1)
            self.assertAlmostEqual(tensors["delta"][0, 0].item(), .2)
            self.assertEqual(window["observation"], old + "/observation.npz")

    def test_cached_supervision_and_truncated_file(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            file = folder / "features.pt"
            labels = folder / "window_labels.jsonl"
            write_json(labels, {"policy_step": 0, "delta_progress": [.25],
                                "valid_subchunk_mask": [True, False, False, False, False]})
            ep = {"policy_requests": 1, "policy_steps": 1}
            tensors = dict(context=torch.zeros(1, 256, 1408, dtype=torch.bfloat16),
                           target=torch.zeros(1, 256, 1408, dtype=torch.bfloat16),
                           actions=torch.zeros(1, 5, 7), state=torch.zeros(1, 8), offset=torch.tensor([0]),
                           delta=torch.tensor([[.25, 0, 0, 0, 0]]),
                           mask=torch.tensor([[True, False, False, False, False]]))
            torch.save(tensors, file)
            self.assertTrue(features.cache_matches(file, ep, folder, digest(labels)))
            tensors["delta"][0, 0] = -.25
            torch.save(tensors, file)
            self.assertFalse(features.cache_matches(file, ep, folder, digest(labels)))
            tensors["label_sha256"] = "older-labels"
            torch.save(tensors, file)
            self.assertFalse(features.cache_matches(file, ep, folder, digest(labels)))
            file.write_bytes(b"")
            self.assertFalse(features.cache_matches(file, ep, folder, digest(labels)))

    def test_full_run_reuses_cache_and_publishes_only_its_own_index(self):
        with TemporaryDirectory() as tmp:
            base = Path(tmp)
            job = base / "paper"
            write_json(job / "source.json", {"sha256": "source"})
            paths = {}
            for suite in features.SUITES:
                indices, labels = base / suite / "split", base / suite / "labels"
                paths[suite] = indices, labels
                specs, hashes, totals = {}, {}, {}
                for split in ("train", "validation"):
                    key = suite + "_" + split
                    row = dict(episode_id=key, suite=suite, task_id=9, split=split,
                               policy_steps=1, policy_requests=1)
                    files = {kind: f"{split}_{kind}.jsonl" for kind in ("episodes", "windows")}
                    write_json(indices / files["episodes"], row)
                    write_json(indices / files["windows"], {"episode_id": key})
                    for name in files.values():
                        hashes[name] = digest(indices / name)
                    specs[split] = {"files": files, "policy_requests": 1}
                    totals[split] = {"episodes": 1, "windows": 1}
                    label_file = labels / split / key / "window_labels.jsonl"
                    write_json(label_file, {"episode_id": key})
                    write_json(label_file.parent / "summary.json",
                               {"artifact_sha256": {"window_labels.jsonl": digest(label_file)}})
                manifest = indices / "manifest.json"
                write_json(manifest, {"suite": suite, "task_count": 10, "splits": specs, "index_sha256": hashes})
                write_json(labels / "run_config.json", {"manifest_sha256": digest(manifest)})
                write_json(labels / "complete/summary.json", {"status": "COMPLETE", "splits": totals})
            with patch.object(features, "cache_matches", return_value=True), patch.object(features, "encoder") as encoder:
                features.build("joint40", 32, job=job, paths=paths, feature_cache=base / "shared-cache")
                encoder.assert_not_called()
            report = features.read_json(job / "joint40/features.json")
            for split in ("train", "validation"):
                items = report[split]
                self.assertEqual([row["task"] for row in items], [9, 19, 29, 39])
                self.assertTrue(all(Path(row["file"]).is_relative_to(base / "shared-cache") for row in items))
                self.assertTrue(all(row["label_sha256"] for row in items))
            with patch.object(features.shutil, "disk_usage") as disk, patch.object(features, "encoder") as encoder:
                disk.return_value.free = 1
                with self.assertRaisesRegex(OSError, "feature-cache"):
                    features.build("joint40", 32, job=job, paths=paths, feature_cache=base / "small-cache")
                encoder.assert_not_called()


if __name__ == "__main__":
    unittest.main()
