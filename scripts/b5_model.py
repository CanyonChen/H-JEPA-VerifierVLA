"""Pinned visual encoder and trainable LIBERO predictor used by training and scoring."""
import hashlib
import json
import os
from functools import partial
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "VeriVLA/repository"))
from verifier.ac_predictor import VisionTransformerPredictorAC
from verifier.encoder import vit_giant

SUITES = ("libero_10", "libero_spatial", "libero_object", "libero_goal")
BASE = (ROOT / "outputs/verifier_long").resolve()
JOB = (ROOT / "outputs/verifier_aip_seed7").resolve()
WEIGHT_BYTES = 11760743310
META_HASH = "103682721d3afb275df3402240ff66c5ab818a7cd723a55c70772fae76425fc9"
WEIGHT_URL = "https://dl.fbaipublicfiles.com/vjepa2/vjepa2-ac-vitg.pt"
VISUAL_VERSION = "external-rgb256-imagenet-temporal-sum-bf16-v1"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write(path, writer):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix="." + path.name, delete=False) as stream:
        tmp = Path(stream.name)
    try:
        writer(tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def save_json(path, data):
    atomic_write(path, lambda p: p.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8"))


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_source():
    import fcntl
    cache = Path(torch.hub.get_dir()) / "checkpoints/vjepa2-ac-vitg.pt"
    target = Path(os.environ.get("VJEPA2_AC_CKPT") or cache).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with Path(str(target) + ".b5.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        src = target if target.is_file() else Path(str(target) + ".part")
        if src != target and (not src.exists() or src.stat().st_size != WEIGHT_BYTES):
            subprocess.run(["curl", "-fL", "-C", "-", "--retry", "3", "--connect-timeout", "20",
                            "-H", 'If-Match: "12c945a28cc4f447e728fd2ba810607c-1402"', "-o", str(src), WEIGHT_URL], check=True)
        if src.stat().st_size != WEIGHT_BYTES:
            raise ValueError(f"Unexpected checkpoint size at {src}")
        with zipfile.ZipFile(src) as z:
            if hashlib.sha256(z.read("latest.pth/data.pkl")).hexdigest() != META_HASH:
                raise ValueError("Unexpected V-JEPA 2-AC metadata")
            if z.testzip() is not None:
                raise ValueError("Damaged V-JEPA 2-AC archive")
        digest = sha256(src)
        if src != target:
            src.replace(target)
    info = dict(path=str(target), sha256=digest, source=WEIGHT_URL, bytes=WEIGHT_BYTES)
    save_json(JOB / "source.json", info)
    print(json.dumps(info), flush=True)


def source_group(group):
    info = read_json(JOB / "source.json")
    obj = torch.load(info["path"], map_location="cpu", mmap=True, weights_only=True)[group]
    return {key.removeprefix("module.").removeprefix("backbone."): val for key, val in obj.items()}


def encoder(device):
    from accelerate import init_empty_weights
    weights = source_group("encoder")
    patch = weights["patch_embed.proj.weight"]
    if tuple(patch.shape) != (1408, 3, 2, 16, 16):
        raise ValueError("Unexpected upstream visual patch embedding")
    weights["patch_embed.proj.weight"] = patch.sum(dim=2)
    with init_empty_weights(include_buffers=False):
        net = vit_giant(img_size=256, num_frames=1, use_rope=True)
    net.load_state_dict(weights, strict=True, assign=True)
    return net.requires_grad_(False).eval().to(device)


@torch.inference_mode()
def encode(net, rgb):
    device = next(net.parameters()).device
    x = torch.as_tensor(rgb, device=device).permute(0, 3, 1, 2).float().div_(255)
    x = F.interpolate(x, (256, 256), mode="bilinear", align_corners=False, antialias=True)
    mean = x.new_tensor([.485, .456, .406]).view(1, 3, 1, 1)
    std = x.new_tensor([.229, .224, .225]).view(1, 3, 1, 1)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        return net((x - mean) / std)


class LiberoPredictor(VisionTransformerPredictorAC):
    _STATE_ENCODERS = {**VisionTransformerPredictorAC._STATE_ENCODERS, 8: "libero_state"}

    def __init__(self, tasks, **options):
        super().__init__(num_tasks=tasks, action_chunk_size=1, num_frames=1, max_frames=5,
                         norm_layer=partial(nn.LayerNorm, eps=1e-6), **options)
        self.libero_state = nn.Linear(8, self.predictor_embed.out_features)
        self._init_weights(self.libero_state)
        self._masks = {}
        # LIBERO uses the 7D OSC command and 8D decision-boundary state. DROID
        # coordinate/scale equivalence is unestablished, so learn these projections.
        for attr in ("action_encoder", "state_encoder", "extrinsics_encoder", "action_encoder_16d",
                     "state_encoder_16d", "state_encoder_7d_eef"):
            getattr(self, attr).requires_grad_(False)

    def build_attn_mask(self, action_len, max_frames=10, grid_height=None, grid_width=None, task_token=1):
        dev = self.value_mlp.weight.device
        key = (dev, action_len, max_frames, grid_height, grid_width, task_token)
        if key not in self._masks:
            # A mask first built by deployment-style validation must also remain
            # usable by autograd on the next training epoch.
            with torch.inference_mode(False):
                self._masks[key] = super().build_attn_mask(
                    action_len, max_frames, grid_height, grid_width, task_token).to(dev)
        return self._masks[key]


def predictor(tasks, pretrained=True):
    net = LiberoPredictor(tasks, img_size=256, embed_dim=1408)
    if pretrained:
        upstream = source_group("predictor")
        skip = {"action_encoder", "state_encoder", "extrinsics_encoder"}
        weights = {k: v for k, v in upstream.items() if k.partition(".")[0] not in skip}
        missing, extra = net.load_state_dict(weights, strict=False)
        new = skip | {"action_encoder_7d_eef", "action_encoder_16d", "state_encoder_7d_eef",
                      "state_encoder_16d", "libero_state", "task_query_embedding", "value_mlp"}
        expected_missing = {k for k in net.state_dict() if k.partition(".")[0] in new}
        if set(missing) != expected_missing or extra or len(weights) != 294:
            raise ValueError(f"Unexpected predictor transfer ({len(weights)} shared tensors): {missing} {extra}")
    return net


def locations(suite):
    if suite == "libero_10":
        return BASE / "split_seed7", BASE / "robometer_raw_trainval_seed7_gpu0"
    return JOB / suite / "split", JOB / suite / "labels"


if __name__ == "__main__":
    prepare_source()
