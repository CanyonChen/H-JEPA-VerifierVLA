"""Load the official pi05_libero checkpoint and infer on a synthetic observation.

This checks checkpoint loading and the action interface, not LIBERO task success.
"""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
CONFIG = "pi05_libero"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CHECKPOINT,
                        help="Official GCS URL or an already downloaded pi05_libero directory.")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/pi05_checkpoint",
                        help="Parent directory for a new run directory and its artifacts.")
    parser.add_argument("--seed", type=int, default=7, help="Seed for the synthetic observation.")
    args = parser.parse_args()

    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_PLATFORMS"] = "cuda"
    run_dir = args.output_root / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "run.log", encoding="utf-8")],
        force=True,
    )
    report = {
        "status": "RUNNING",
        "config": CONFIG,
        "checkpoint": args.checkpoint,
        "observation_seed": args.seed,
        "input_kind": "synthetic LIBERO observation from the upstream example",
        "scope": "official checkpoint inference; no simulator or task success evaluation",
        "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "output_dir": str(run_dir.resolve()),
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logging.info("Run directory: %s", run_dir.resolve())

    try:
        import jax
        import numpy as np

        from openpi.policies import libero_policy, policy_config
        from openpi.shared import download
        from openpi.training import config

        if jax.default_backend() != "gpu":
            raise RuntimeError("The official JAX checkpoint check requires a CUDA backend.")
        report["jax_devices"] = [str(device) for device in jax.devices()]
        report["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("jax", "jaxlib", "flax", "numpy")
        }
        report["repo_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        ).strip()

        logging.info("Downloading or reusing checkpoint: %s", args.checkpoint)
        started = time.perf_counter()
        checkpoint_dir = download.maybe_download(args.checkpoint, gs={"token": "anon"})
        report["download_or_cache_seconds"] = time.perf_counter() - started
        report["checkpoint_local_path"] = str(checkpoint_dir)

        norm_stats = checkpoint_dir / "assets/physical-intelligence/libero/norm_stats.json"
        if not (checkpoint_dir / "params").is_dir() or not norm_stats.is_file():
            raise FileNotFoundError(f"Expected JAX params/ and LIBERO norm_stats.json in {checkpoint_dir}")
        report["norm_stats_path"] = str(norm_stats)

        logging.info("Loading pi05_libero weights and checkpoint normalization statistics")
        started = time.perf_counter()
        train_config = config.get_config(CONFIG)
        policy = policy_config.create_trained_policy(train_config, checkpoint_dir)
        report["load_seconds"] = time.perf_counter() - started

        np.random.seed(args.seed)
        observation = libero_policy.make_libero_example()
        np.savez(run_dir / "observation.npz", **observation)
        actions = []
        call_seconds = []
        for label in ("first call (includes JIT compilation)", "second call"):
            logging.info("Running %s", label)
            started = time.perf_counter()
            result = policy.infer(observation)
            action = np.asarray(result["actions"])
            call_seconds.append(time.perf_counter() - started)
            expected_shape = (train_config.model.action_horizon, 7)
            if action.shape != expected_shape or not np.isfinite(action).all():
                raise ValueError(f"Expected finite actions of shape {expected_shape}, got {action.shape}")
            actions.append(action)

        np.save(run_dir / "actions.npy", np.stack(actions))
        report.update({
            "status": "PASS",
            "action_shape": list(actions[0].shape),
            "all_actions_finite": True,
            "inference_calls": len(actions),
            "call_wall_seconds": call_seconds,
            "timing_scope": "whole policy.infer call including conversion to NumPy; first call includes JIT",
        })
    except Exception as error:
        report.update(status="FAIL", error=f"{type(error).__name__}: {error}")
        logging.exception("Checkpoint inference failed; see %s", report_path)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
