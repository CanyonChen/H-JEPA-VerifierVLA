"""Check all nine Ultra-Long scenes in separate processes, without loading a policy."""

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np

from scripts.check_libero_env import PROJECT_ROOT, make_policy_observation
from scripts.eval_libero import write_json
from scripts.lilo_protocol import LILO_COMMIT, WAIT_STEPS, SkillPredicates, initialize_benchmark, settle_environment, task_definitions


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--egl-device", type=int, default=1)
    devices.add_argument("--egl-devices", type=int, nargs="+", help="Distribute the nine scene checks across these EGL devices.")
    parser.add_argument("--task-id", type=int, choices=range(9))
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/lilo_check")
    args = parser.parse_args(argv)
    egl_devices = args.egl_devices or [args.egl_device]
    if any(device < 0 for device in egl_devices) or len(set(egl_devices)) != len(egl_devices):
        parser.error("EGL device indices must be nonnegative and unique")
    if args.task_id is not None and args.egl_devices:
        parser.error("A single-task check takes --egl-device")
    if args.task_id is None:
        directory = args.output_root.resolve() / datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")
        directory.mkdir(parents=True)
        reports = []
        for task_id in range(9):
            device = egl_devices[task_id % len(egl_devices)]
            completed = subprocess.run([sys.executable, "-B", "-u", "-m", "scripts.check_lilo_env",
                "--task-id", str(task_id), "--egl-device", str(device), "--output-root", str(directory)], cwd=PROJECT_ROOT)
            path = directory / "task{:02d}".format(task_id) / "report.json"
            reports.append(json.loads(path.read_text()) if path.exists() else {"task_id": task_id, "status": "FAIL", "returncode": completed.returncode})
            if completed.returncode:
                break
        report = {"status": "PASS" if len(reports) == 9 and all(r["status"] == "PASS" for r in reports) else "FAIL",
                  "scope": "nine scene renders and skill predicate bindings; no policy episodes",
                  "lilo_commit": LILO_COMMIT, "output_dir": str(directory), "tasks": reports,
                  "egl_devices": egl_devices, "simulation_environment": ".venv-libero"}
        write_json(directory / "report.json", report)
        print(json.dumps(report, indent=2))
        return 0 if report["status"] == "PASS" else 1
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    os.environ.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(args.egl_device))
    directory = args.output_root / "task{:02d}".format(args.task_id)
    directory.mkdir(parents=True, exist_ok=True)
    report = {"status": "FAIL", "task_id": args.task_id, "egl_device": args.egl_device,
              "python_executable": sys.executable, "simulation_environment": ".venv-libero"}
    env = None
    try:
        from PIL import Image
        suite = initialize_benchmark()
        task = task_definitions()[args.task_id]
        env, obs = suite.make_env(args.task_id)
        obs = settle_environment(env, obs)
        predicates = SkillPredicates(env, task)
        values = predicates.values()
        policy_obs = make_policy_observation(obs, task["prompt"])
        for key, filename in (("observation/image", "agentview.png"), ("observation/wrist_image", "wrist.png")):
            frame = policy_obs[key]
            if frame.shape != (224, 224, 3) or frame.dtype != np.uint8 or not np.ptp(frame):
                raise ValueError("Invalid or constant camera image: " + key)
            Image.fromarray(frame).save(directory / filename)
        from OpenGL import GL
        report.update(status="PASS", name=task["name"], skills=len(task["skills"]), wait_steps=WAIT_STEPS,
                      initial_predicates=values, image_shape=[224, 224, 3],
                      opengl_vendor=GL.glGetString(GL.GL_VENDOR).decode(),
                      opengl_renderer=GL.glGetString(GL.GL_RENDERER).decode(),
                      versions={name: importlib.metadata.version(name) for name in ("numpy", "mujoco", "robosuite", "torch")})
    except Exception as error:
        report["error"] = "{}: {}".format(type(error).__name__, error)
        import traceback
        traceback.print_exc()
    finally:
        if env is not None:
            try:
                env.close()
            except Exception as error:
                report.update(status="FAIL", error="Environment close: {}".format(error))
        write_json(directory / "report.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
