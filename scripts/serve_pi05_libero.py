"""Serve the official JAX pi05_libero policy with recorded checkpoint identity.

The upstream policy loader and WebSocket server do all model and transport work.
This entry point adds provenance and request numbers without changing actions.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import os
from pathlib import Path
import platform
import socket
import subprocess
import time
import uuid

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_libero"
CONFIG = "pi05_libero"


def checkpoint_manifest(directory):
    """Hash relative names, sizes and contents; independent of the cache path."""
    files = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file():
            continue
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        files.append({"path": path.relative_to(directory).as_posix(), "bytes": size,
                      "sha256": digest.hexdigest()})
    if not files:
        raise ValueError("Checkpoint directory is empty: {}".format(directory))
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {"sha256": hashlib.sha256(encoded).hexdigest(), "files": files,
            "total_bytes": sum(item["bytes"] for item in files),
            "algorithm": "sha256 of canonical JSON list of relative paths, byte sizes and file sha256"}


class RecordedPolicy:
    """Attach server request identity, preserving the upstream policy RNG."""

    def __init__(self, policy, provenance):
        self.policy = policy
        self.provenance = provenance

    def infer(self, observation):
        self.provenance["request_count"] += 1
        request_id = self.provenance["request_count"]
        response = dict(self.policy.infer(observation))
        response["evaluation"] = {"service_id": self.provenance["service_id"], "request_id": request_id}
        return response


def warmup_policy(policy, observation):
    """Compile a representative request and restore the JAX sampling key."""
    rng = policy._rng
    try:
        return policy.infer(observation)
    finally:
        policy._rng = rng


def advance_policy_rng(policy, request_count):
    """Restore the next JAX key after a prefix of completed inference requests."""
    import jax

    policy._rng = jax.lax.fori_loop(0, request_count, lambda _, key: jax.random.split(key)[0], policy._rng)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default=CHECKPOINT)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--warmup", action="store_true", help="Compile inference before serving and preserve the initial RNG.")
    parser.add_argument("--request-offset", type=int, default=0)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/pi05_servers")
    args = parser.parse_args(argv)
    if args.request_offset < 0:
        parser.error("request-offset must be nonnegative")
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    os.environ["JAX_PLATFORMS"] = "cuda"
    service_id = str(uuid.uuid4())
    output_dir = args.output_root.resolve() / ("pi05_libero_" + service_id)
    output_dir.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True,
        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "run.log", encoding="utf-8")],
    )
    logging.info("Service records: %s", output_dir)
    provenance = {
        "schema_version": 1, "service_id": service_id, "request_count": args.request_offset, "pid": os.getpid(),
        "restored_request_count": args.request_offset,
        "started_at_utc": datetime.now(timezone.utc).isoformat(), "hostname": socket.gethostname(),
        "policy_config": CONFIG, "backend": "jax", "initial_model_seed": 0,
        "rng_protocol": "Upstream default key(0), split once per inference; no reset between episodes.",
        "checkpoint_source": args.checkpoint, "port": args.port, "python": platform.python_version(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "output_dir": str(output_dir), "startup_status": "LOADING",
    }
    report_path = output_dir / "service.json"

    def save_report():
        report_path.write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    save_report()
    try:
        import jax
        from openpi.policies import policy_config
        from openpi.serving.websocket_policy_server import WebsocketPolicyServer
        from openpi.shared import download
        from openpi.training import config

        if jax.default_backend() != "gpu":
            raise RuntimeError("The pi05_libero service requires the JAX CUDA backend")
        provenance["jax_devices"] = [{"id": str(device), "kind": device.device_kind} for device in jax.devices()]
        provenance["versions"] = {name: importlib.metadata.version(name) for name in
                                  ("jax", "jaxlib", "flax", "numpy", "orbax-checkpoint", "websockets")}
        provenance["repo_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
        provenance["repo_dirty"] = bool(subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=PROJECT_ROOT, text=True).strip())
        logging.info("Downloading or reusing checkpoint: %s", args.checkpoint)
        directory = download.maybe_download(args.checkpoint, gs={"token": "anon"})
        norm_stats = directory / "assets/physical-intelligence/libero/norm_stats.json"
        if not (directory / "params").is_dir() or not norm_stats.is_file():
            raise FileNotFoundError("Expected JAX params and LIBERO normalization statistics in {}".format(directory))
        if (directory / "model.safetensors").exists():
            raise ValueError("Use the official JAX checkpoint directory without a PyTorch model.safetensors")
        provenance["checkpoint_local_path"] = str(directory)
        logging.info("Hashing checkpoint files for reproducibility (reads cached weights; no redownload)")
        started = time.perf_counter()
        provenance["checkpoint_manifest"] = checkpoint_manifest(directory)
        if (args.expected_checkpoint_sha256 is not None
                and provenance["checkpoint_manifest"]["sha256"] != args.expected_checkpoint_sha256):
            raise ValueError("Checkpoint content differs from the evaluation being resumed")
        provenance["hash_seconds"] = time.perf_counter() - started
        provenance["norm_stats_sha256"] = hashlib.sha256(norm_stats.read_bytes()).hexdigest()
        save_report()
        train_config = config.get_config(CONFIG)
        provenance["action_horizon"] = train_config.model.action_horizon
        started = time.perf_counter()
        policy = policy_config.create_trained_policy(train_config, directory)
        provenance["load_seconds"] = time.perf_counter() - started
        if args.request_offset:
            advance_policy_rng(policy, args.request_offset)
            logging.info("Restored model RNG after %d completed requests", args.request_offset)
        if args.warmup:
            from openpi.policies.libero_policy import make_libero_example
            logging.info("Warming up policy inference before accepting clients")
            started = time.perf_counter()
            warmup_policy(policy, make_libero_example())
            provenance["warmup_seconds"] = time.perf_counter() - started
            provenance["warmup_rng_restored"] = True
        provenance["startup_status"] = "LOADED"
        save_report()
        metadata = dict(policy.metadata)
        metadata["evaluation"] = provenance
        server = WebsocketPolicyServer(RecordedPolicy(policy, provenance), host="0.0.0.0", port=args.port,
                                       metadata=metadata)
        logging.info("Serving %s, service_id=%s, checkpoint_sha256=%s", CONFIG, service_id,
                     provenance["checkpoint_manifest"]["sha256"])
        server.serve_forever()
    except KeyboardInterrupt:
        logging.info("Service stopped after %d inference attempts", provenance["request_count"])
    except Exception as error:
        provenance["error"] = "{}: {}".format(type(error).__name__, error)
        logging.exception("Policy service failed")
        raise
    finally:
        provenance["stopped_at_utc"] = datetime.now(timezone.utc).isoformat()
        save_report()


if __name__ == "__main__":
    main()
