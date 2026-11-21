"""Batched real-checkpoint verifier worker on private pipes from the LIBERO simulator."""
import argparse
import base64
import json
from pathlib import Path
import sys

import numpy as np
import torch

from scripts.b5_model import JOB, SUITES, encode, encoder, predictor, read_json, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job-dir", type=Path, default=JOB)
    parser.add_argument("--stage", choices=["long100", "joint40"], default="long100")
    args = parser.parse_args()
    wire = sys.stdout
    sys.stdout = sys.stderr
    torch.set_num_threads(2)
    job = args.job_dir.resolve()
    directory = job / args.stage
    joint = args.stage == "joint40"
    tasks = 40 if joint else 10
    record = read_json(directory / "training.json")
    checkpoint = directory / ("best.pt" if joint else "last.pt")
    state = torch.load(checkpoint, map_location="cpu", mmap=True, weights_only=True)
    config = record["config"]
    if (record["status"] != "COMPLETE" or record["updates"] != config["target_updates"]
            or state["config"] != config or config["tasks"] != tasks or config["stage"] != args.stage
            or config["source_sha256"] != read_json(job / "source.json")["sha256"]):
        raise ValueError("Scoring checkpoint and completed training report disagree")
    if joint:
        if (tuple(config["suites"]) != SUITES or not 0 < state["step"] <= record["updates"]
                or state["validation_huber"] != record["best_validation_huber"]
                or sha256(directory / "features.json") != config["features_sha256"]):
            raise ValueError("Expected the best joint 40-task validation checkpoint")
        ready = dict(ready=True, updates=state["step"], best_sha256=sha256(checkpoint),
                     training_sha256=sha256(directory / "training.json"))
    else:
        if state["step"] != 100 or record["updates"] != 100:
            raise ValueError("Expected the completed 100-update pilot checkpoint")
        ready = dict(ready=True, updates=100)
    vision = encoder("cuda:0", job=job)
    net = predictor(tasks, pretrained=False, job=job)
    net.load_state_dict(state["model"])
    net.requires_grad_(False).eval().to("cuda:0")
    del state
    print(json.dumps(ready), file=wire, flush=True)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for line in sys.stdin:
            req = json.loads(line)
            rgb = np.frombuffer(base64.b64decode(req["rgb"]), dtype=np.uint8).reshape(1,224,224,3).copy()
            candidates = torch.tensor(req["actions"], device="cuda:0", dtype=torch.float32)
            state8 = torch.tensor(req["state"], device="cuda:0", dtype=torch.float32)
            z = encode(vision, rgb).expand(len(candidates), -1, -1)
            _, values = net.rollout_kv_cached(z, candidates[:, :5].unsqueeze(2),
                                              state8[None].expand(len(candidates), -1), task_id=req["task"])
            scores = values.squeeze(-1).float().sum(0)
            if scores.shape != (len(candidates),) or not torch.isfinite(scores).all():
                raise ValueError("Invalid candidate scores")
            print(json.dumps(scores.tolist()), file=wire, flush=True)


if __name__ == "__main__":
    main()
