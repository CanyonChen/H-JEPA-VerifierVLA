"""Check LIBERO EGL rendering and a real observation over the policy WebSocket.

This is an interface check with at most five policy action steps, not an episode
evaluation or success-rate benchmark. Run in the separate LIBERO environment.
"""

import argparse
import contextlib
from datetime import datetime, timezone
import importlib.metadata
import json
import logging
import math
import os
from pathlib import Path
import platform
import subprocess
import time


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def configure_libero_paths():
    """Write a project-local config before LIBERO can prompt for dataset paths."""
    benchmark_root = PROJECT_ROOT / "openpi/third_party/libero/libero/libero"
    datasets = PROJECT_ROOT / "data/libero/datasets"
    datasets.mkdir(parents=True, exist_ok=True)
    config_dir = PROJECT_ROOT / ".venv-libero/libero-config"
    config_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "benchmark_root": str(benchmark_root),
        "bddl_files": str(benchmark_root / "bddl_files"),
        "init_states": str(benchmark_root / "init_files"),
        "datasets": str(datasets),
        "assets": str(benchmark_root / "assets"),
    }
    # JSON is also valid YAML, so configuration needs no import from LIBERO.
    (config_dir / "config.yaml").write_text(json.dumps(paths, indent=2) + "\n", encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    return paths


def make_policy_observation(obs, task_description):
    """Match openpi/examples/libero/main.py image and state preprocessing."""
    import numpy as np
    from openpi_client import image_tools

    def image(key):
        rotated = np.ascontiguousarray(obs[key][::-1, ::-1])
        return image_tools.convert_to_uint8(image_tools.resize_with_pad(rotated, 224, 224))

    quat = np.asarray(obs["robot0_eef_quat"]).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - quat[3] * quat[3])
    axis_angle = (
        np.zeros(3) if math.isclose(denominator, 0.0)
        else quat[:3] * 2.0 * math.acos(quat[3]) / denominator
    )
    return {
        "observation/image": image("agentview_image"),
        "observation/wrist_image": image("robot0_eye_in_hand_image"),
        "observation/state": np.concatenate((obs["robot0_eef_pos"], axis_angle, obs["robot0_gripper_qpos"])),
        "prompt": str(task_description),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--render-only", action="store_true", help="Check EGL without a policy server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--suite", choices=("libero_10", "libero_spatial", "libero_object", "libero_goal"),
                        default="libero_10")
    parser.add_argument("--task-id", type=int, default=0)
    parser.add_argument("--init-state-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/libero_check")
    args = parser.parse_args()

    os.environ["MUJOCO_GL"] = "egl"
    os.environ["PYOPENGL_PLATFORM"] = "egl"
    os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "1")
    run_dir = args.output_root / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
    run_dir.mkdir(parents=True)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.StreamHandler(), logging.FileHandler(run_dir / "run.log", encoding="utf-8")],
        force=True,
    )
    report = {
        "status": "RUNNING", "python": platform.python_version(),
        "mode": "render_only" if args.render_only else "real_observation_and_actions",
        "suite": args.suite, "task_id": args.task_id, "init_state_id": args.init_state_id,
        "seed": args.seed, "egl_device_id": os.environ["MUJOCO_EGL_DEVICE_ID"],
        "output_dir": str(run_dir.resolve()), "policy_roundtrip": "NOT_RUN", "action_steps": 0,
        "scope": "EGL and interface check; no complete episode or success-rate evaluation",
    }
    report_path = run_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    logging.info("Run directory: %s", run_dir.resolve())

    try:
        configure_libero_paths()
        import numpy as np
        from OpenGL import GL
        from PIL import Image
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv

        report["versions"] = {
            name: importlib.metadata.version(name)
            for name in ("libero", "openpi-client", "torch", "numpy", "robosuite", "mujoco", "websockets")
        }
        report["repo_commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
        ).strip()
        np.random.seed(args.seed)
        suite = benchmark.get_benchmark_dict()[args.suite]()
        task = suite.get_task(args.task_id)
        initial_state = suite.get_task_init_states(args.task_id)[args.init_state_id]
        report["task_description"] = task.language

        with contextlib.ExitStack() as stack:
            logging.info("Creating EGL environment for %s", task.language)
            env = OffScreenRenderEnv(
                bddl_file_name=suite.get_task_bddl_file_path(args.task_id),
                camera_heights=256, camera_widths=256,
            )
            stack.callback(env.close)
            env.seed(args.seed)
            env.reset()
            obs = env.set_init_state(initial_state)
            for _ in range(10):
                obs, _, _, _ = env.step([0.0] * 6 + [-1.0])

            report["opengl"] = {
                "vendor": GL.glGetString(GL.GL_VENDOR).decode("utf-8"),
                "renderer": GL.glGetString(GL.GL_RENDERER).decode("utf-8"),
                "version": GL.glGetString(GL.GL_VERSION).decode("utf-8"),
            }
            policy_obs = make_policy_observation(obs, task.language)
            for key, filename in (("observation/image", "agentview.png"),
                                  ("observation/wrist_image", "wrist.png")):
                frame = policy_obs[key]
                if frame.shape != (224, 224, 3) or frame.dtype != np.uint8 or np.ptp(frame) == 0:
                    raise ValueError("Invalid or constant EGL frame for " + key)
                Image.fromarray(frame).save(run_dir / filename)
            state = policy_obs["observation/state"]
            if state.shape != (8,) or not np.isfinite(state).all():
                raise ValueError("Expected a finite 8-dimensional LIBERO state")
            np.savez(run_dir / "observation.npz", **policy_obs)
            report.update(egl_render="PASS", state_shape=list(state.shape), image_shape=[224, 224, 3])

            if not args.render_only:
                from openpi_client.websocket_client_policy import WebsocketClientPolicy

                client = WebsocketClientPolicy(args.host, args.port)
                stack.callback(client._ws.close)
                report["server_metadata"] = client.get_server_metadata()
                logging.info("Sending a real LIBERO observation to the pi05 policy")
                started = time.perf_counter()
                actions = np.asarray(client.infer(policy_obs)["actions"])
                report["roundtrip_seconds"] = time.perf_counter() - started
                if actions.shape != (10, 7) or not np.isfinite(actions).all():
                    raise ValueError("Expected finite policy actions of shape (10, 7)")
                np.save(run_dir / "actions.npy", actions)
                report.update(policy_roundtrip="PASS", action_shape=list(actions.shape))
                for action in actions[:5]:
                    obs, _, done, _ = env.step(action.tolist())
                    report["action_steps"] += 1
                    if done:
                        break
                report["task_success_observed"] = bool(done)
                after = make_policy_observation(obs, task.language)
                Image.fromarray(after["observation/image"]).save(run_dir / "agentview_after.png")
        report["status"] = "PASS"
    except Exception as error:
        report.update(status="FAIL", error="{}: {}".format(type(error).__name__, error))
        logging.exception("LIBERO check failed; see %s", report_path)
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
