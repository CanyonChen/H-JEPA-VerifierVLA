"""Prepare and evaluate one pi0.5 verifier shared across the 40 LIBERO tasks."""
import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys

from scripts.verivla_prepare import H100_ROOT


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_JOB = ROOT / "outputs/verivla_pi05_libero"


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    inherited = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")[0] or "0"
    parser.add_argument("--gpu", "--cuda", type=int, default=inherited, help="Physical NVIDIA GPU index")
    parser.add_argument("--resume", choices=("auto", "required", "never"), default="auto",
                        help="Resume if available, require a checkpoint, or require a fresh training directory")
    parser.add_argument("--dry-run", "--plan", action="store_true", help="Inspect local inputs and print commands")
    parser.add_argument("--job-dir", type=Path, default=DEFAULT_JOB)
    parser.add_argument("--data-root", type=Path, help="Prepared non-Long suites (default: JOB/data)")
    parser.add_argument("--long-root", type=Path, default=ROOT / "outputs/verifier_long",
                        help="Directory containing split_seed7 and the completed Long labels")
    parser.add_argument("--feature-cache", type=Path, help="Frozen vision features (default: JOB/features)")
    parser.add_argument("--recorded-project-root", default=H100_ROOT,
                        help="Original repository prefix in archived Long observation/video paths")
    parser.add_argument("--checkpoint", type=Path, help="Existing V-JEPA 2-AC checkpoint; also VJEPA2_AC_CKPT")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--microbatch", type=int, choices=(1, 2, 4, 8, 16, 32), default=32)
    args = parser.parse_args(argv)
    if args.gpu < 0 or args.epochs < 1:
        parser.error("--gpu must be nonnegative and --epochs must be positive")
    args.job_dir = args.job_dir.expanduser().resolve()
    args.long_root = args.long_root.expanduser().resolve()
    args.data_root = (args.data_root or args.job_dir / "data").expanduser().resolve()
    args.feature_cache = (args.feature_cache or args.job_dir / "features").expanduser().resolve()
    return args


def source_checkpoint(requested, env):
    if requested or env.get("VJEPA2_AC_CKPT"):
        return Path(requested or env["VJEPA2_AC_CKPT"]).expanduser().resolve()
    cache = Path(env.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    torch_home = Path(env.get("TORCH_HOME", cache / "torch"))
    shared = Path("/data/models/vjepa2-checkpoints/vjepa2-ac-vitg.pt")
    cached = torch_home / "hub/checkpoints/vjepa2-ac-vitg.pt"
    return cached if cached.is_file() else shared


def environment(gpu):
    from scripts.robometer_backend import MODEL_ID, MODEL_REVISION, PROCESSOR_FILES, PROCESSOR_ID, PROCESSOR_REVISION

    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID",
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", PYTHONUNBUFFERED="1",
               MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(gpu),
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    shared_openpi = Path("/data/models/openpi")
    if (shared_openpi / "big_vision/paligemma_tokenizer.model").is_file():
        env.setdefault("OPENPI_DATA_HOME", str(shared_openpi))
    if not any(env.get(k) for k in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "HF_HOME")):
        cache_home = Path(env.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        for hub in (cache_home / "huggingface/hub", Path("/data/models/robometer-hf/hub"),
                    Path("/data/models/huggingface/hub"), Path("/data/models/hf-cache/hub")):
            weights = hub / ("models--" + MODEL_ID.replace("/", "--")) / "snapshots" / MODEL_REVISION
            processor = hub / ("models--" + PROCESSOR_ID.replace("/", "--")) / "snapshots" / PROCESSOR_REVISION
            parts = ("config.json", "config.yaml", "model.safetensors.index.json",
                     "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors")
            if all((weights / f).is_file() for f in parts) and all((processor / f).is_file() for f in PROCESSOR_FILES):
                env["HF_HUB_CACHE"] = str(hub)
                break
    return env


def commands(args, checkpoint):
    python = sys.executable
    sim = str(ROOT / ".venv-libero/bin/python")
    data = ["--data-root", str(args.data_root), "--long-root", str(args.long_root),
            "--recorded-project-root", args.recorded_project_root]
    job = ["--job-dir", str(args.job_dir)]
    return [
        [python, "-u", "-m", "scripts.verivla_prepare", *data, "--gpu", str(args.gpu)],
        [python, "-u", "-m", "scripts.b5_model", *job, "--checkpoint", str(checkpoint)],
        [python, "-u", "-m", "scripts.b5_features", "joint40", *job, *data, "--feature-cache", str(args.feature_cache)],
        [python, "-u", "-m", "scripts.b5_train", "joint40", *job, "--epochs", str(args.epochs),
         "--microbatch", str(args.microbatch)],
        [sim, "-u", "-m", "scripts.verivla_evaluate", *job, *data, "--scorer-python", python],
    ]


def check_resume(mode, directory):
    last = directory / "last.pt"
    if mode == "required" and not last.is_file():
        raise ValueError(f"--resume=required needs {last}")
    if mode == "never" and any((directory / name).exists() for name in ("last.pt", "best.pt", "training.json")):
        raise ValueError(f"--resume=never needs a fresh training directory: {directory}")


def main(argv=None):
    args = arguments(argv)
    env = environment(args.gpu)
    checkpoint = source_checkpoint(args.checkpoint, env)
    stages = commands(args, checkpoint)
    if args.dry_run:
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        print(f"GPU {args.gpu} | resume={args.resume} | V-JEPA 2-AC {'found' if checkpoint.is_file() else 'missing'}")
        for stage in stages:
            print(shlex.join(stage))
        subprocess.run([*stages[0], "--plan"], cwd=ROOT, env=env, check=True)
        return
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Set --checkpoint or VJEPA2_AC_CKPT to an existing V-JEPA 2-AC file: {checkpoint}")

    from filelock import FileLock
    args.job_dir.mkdir(parents=True, exist_ok=True)
    with FileLock(str(args.job_dir / "pipeline.lock"), timeout=0):
        check_resume(args.resume, args.job_dir / "joint40")
        for stage in stages:
            print(shlex.join(stage), flush=True)
            subprocess.run(stage, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
