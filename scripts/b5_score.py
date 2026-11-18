"""Batched real-checkpoint verifier worker on private pipes from the LIBERO simulator."""
import base64
import json
import sys

import numpy as np
import torch

from scripts.b5_model import JOB, encode, encoder, predictor, read_json


def main():
    wire = sys.stdout
    sys.stdout = sys.stderr
    torch.set_num_threads(2)
    record = read_json(JOB / "long100/training.json")
    state = torch.load(JOB / "long100/last.pt", map_location="cpu", mmap=True, weights_only=True)
    if state["step"] != 100 or record["updates"] != 100 or state["config"] != record["config"]:
        raise ValueError("Expected the completed 100-update pilot checkpoint")
    if state["config"]["source_sha256"] != read_json(JOB / "source.json")["sha256"]:
        raise ValueError("Encoder checkpoint changed since pilot")
    vision = encoder("cuda:0")
    net = predictor(10, pretrained=False)
    net.load_state_dict(state["model"])
    net.requires_grad_(False).eval().to("cuda:0")
    del state
    print(json.dumps({"ready": True, "updates":100}), file=wire, flush=True)
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
