"""Run independent LIBERO tasks with fixed concurrency on one or several GPUs."""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import json
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request

from scripts.eval_libero import MAX_STEPS, PROJECT_ROOT, summarize, write_json
from scripts.serve_pi05_libero import CHECKPOINT
from scripts.check_server_resources import GIB, resource_status
from scripts.lilo_protocol import SUITE as LILO_SUITE, PROTOCOL as LILO_PROTOCOL, aggregate_lilo, summarize_lilo


@dataclass
class TaskProcesses:
    task_id: int
    port: int
    directory: Path
    server: object
    server_log: object
    client: object = None
    client_log: object = None
    first_init_state: int = 0
    request_offset: int = 0
    gpu: int = 1
    egl_device: int = 1


def gpu_status(gpu):
    output = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True)
    used, total, utilization = next(csv.reader(output.splitlines()))
    return {"used_mib": int(used.strip()), "total_mib": int(total.strip()),
            "utilization_percent": int(utilization.strip())}


def child_environment(gpu, egl_device, model):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    if model:
        environment.update(CUDA_VISIBLE_DEVICES=str(gpu), JAX_PLATFORMS="cuda",
                           XLA_PYTHON_CLIENT_PREALLOCATE="false")
    else:
        environment.pop("CUDA_VISIBLE_DEVICES", None)
        environment.update(MUJOCO_GL="egl", PYOPENGL_PLATFORM="egl", MUJOCO_EGL_DEVICE_ID=str(egl_device))
    return environment


def available_port(first_port):
    for port in range(first_port, 65536):
        with socket.socket() as probe:
            if os.name == "posix":
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                probe.bind(("0.0.0.0", port))
                probe.listen(1)
            except OSError as error:
                if error.errno == errno.EADDRINUSE:
                    continue
                raise
            return port
    raise OSError("No available policy port in the requested range")


def launch_server(task_id, args, output_dir, first_init_state=0, request_offset=0, gpu=None, egl_device=None):
    gpu = args.gpu if gpu is None else gpu
    egl_device = args.egl_device if egl_device is None else egl_device
    # Each task starts at a distinct port, including tasks replacing completed workers.
    port = available_port(args.base_port + task_id)
    directory = output_dir / "tasks" / "task{:02d}".format(task_id)
    directory.mkdir(parents=True)
    log = (directory / "server.log").open("w", encoding="utf-8")
    command = [str(PROJECT_ROOT / ".venv/bin/python"), "-u", "-m", "scripts.serve_pi05_libero",
               "--port", str(port), "--checkpoint", args.checkpoint, "--warmup",
               "--request-offset", str(request_offset),
               "--output-root", str(directory / "server")]
    if args.checkpoint_sha256:
        command.extend(["--expected-checkpoint-sha256", args.checkpoint_sha256])
    try:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=child_environment(gpu, egl_device, True), start_new_session=True)
    except BaseException:
        log.close()
        raise
    logging.info("Starting task %d model on GPU %d, EGL %d, port %d", task_id, gpu, egl_device, port)
    return TaskProcesses(task_id, port, directory, process, log, first_init_state=first_init_state,
                         request_offset=request_offset, gpu=gpu, egl_device=egl_device)


def server_ready(task):
    try:
        with urllib.request.urlopen("http://127.0.0.1:{}/healthz".format(task.port), timeout=0.5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def launch_client(task, args):
    task.client_log = (task.directory / "client.log").open("w", encoding="utf-8")
    if args.suite == LILO_SUITE:
        command = [str(PROJECT_ROOT / ".venv-libero/bin/python"), "-u", "-m", "scripts.eval_lilo",
                   "--port", str(task.port), "--task-id", str(task.task_id),
                   "--trial-start", str(task.first_init_state),
                   "--num-trials-per-task", str(args.trials_per_task - task.first_init_state),
                   "--request-offset", str(task.request_offset), "--steps-per-skill", str(args.steps_per_skill),
                   "--output-root", str(task.directory / "client")]
    else:
        command = [str(PROJECT_ROOT / ".venv-libero/bin/python"), "-u", "-m", "scripts.eval_libero",
                   "--port", str(task.port), "--suite", args.suite, "--task-id", str(task.task_id),
                   "--init-state-id", str(task.first_init_state),
                   "--num-trials-per-task", str(args.trials_per_task - task.first_init_state),
                   "--request-offset", str(task.request_offset),
                   "--seed", str(args.seed), "--output-root", str(task.directory / "client")]
    task.client = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=task.client_log, stderr=subprocess.STDOUT,
                                   env=child_environment(task.gpu, task.egl_device, False), start_new_session=True)
    logging.info("Task %d started %d episodes from init %d", task.task_id,
                 args.trials_per_task - task.first_init_state, task.first_init_state)


def stop_task(task):
    # Stop the simulator first so it can flush the current episode and video.
    for process in (task.client, task.server):
        if process is not None and process.poll() is None:
            try:
                process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass  # The child exited between poll and signal.
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
    for log in (task.client_log, task.server_log):
        if log is not None:
            log.close()


def episode_rows(output_dir):
    rows = []
    retained = output_dir / "retained_episodes.jsonl"
    if retained.exists():
        rows.extend(json.loads(line) for line in retained.read_text(encoding="utf-8").splitlines())
    for path in sorted((output_dir / "tasks").glob("task*/client/*/episodes.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines(keepends=True):
            # A writer may still be appending the last row during live progress.
            if line.endswith("\n"):
                row = json.loads(line)
                row["artifacts"] = str(path.parent.relative_to(output_dir) / row["artifacts"])
                rows.append(row)
    return rows


def aggregate(rows, task_ids, trials, run_error=None, suite="libero_10"):
    if suite == LILO_SUITE:
        return aggregate_lilo(rows, task_ids, trials, run_error)
    expected = {(task, init) for task in task_ids for init in range(trials)}
    seen = [(row["task_id"], row["init_state_id"]) for row in rows]
    if len(set(seen)) != len(seen) or not set(seen).issubset(expected):
        raise ValueError("Episode IDs overlap or are outside the requested task/init set")
    scope = "task-parallel evaluation with fresh model RNG key 0 for each task"
    result = summarize(rows, len(expected), run_error, scope)
    result["per_task"] = []
    for task in task_ids:
        stats = summarize([row for row in rows if row["task_id"] == task], trials)
        result["per_task"].append(dict(task_id=task, **{key: stats[key] for key in
            ("valid_episodes", "successes", "timeouts", "infrastructure_errors", "success_rate",
             "success_rate_wilson95")}))
    return result


def resume_state(args):
    starts = {task: {"first_init_state": 0, "request_offset": 0} for task in args.task_ids}
    if args.resume_from is None:
        return [], starts, None
    source = args.resume_from.resolve()
    config = json.loads((source / "run_config.json").read_text(encoding="utf-8"))
    for key in ("suite", "seed", "task_ids", "trials_per_task", "checkpoint"):
        if config[key] != getattr(args, key):
            raise ValueError("Resume setting differs from original run: {}".format(key))
    if args.suite == LILO_SUITE and (config.get("steps_per_skill") != args.steps_per_skill
                                    or config.get("lilo_protocol") != LILO_PROTOCOL):
        raise ValueError("Ultra-Long resume budget or scoring protocol differs")
    rows = episode_rows(source)
    aggregate(rows, args.task_ids, args.trials_per_task, suite=args.suite)
    retained = [row for row in rows if row["outcome"] in ("success", "timeout", "out_of_order")]
    hashes = set()
    for row in retained:
        artifact_dir = (source / row["artifacts"]).resolve()
        row["artifacts"] = str(artifact_dir)
        client_config = json.loads((artifact_dir.parents[1] / "run_config.json").read_text(encoding="utf-8"))
        hashes.add(client_config["server_metadata"]["evaluation"]["checkpoint_manifest"]["sha256"])
    if retained and (len(hashes) != 1 or len(next(iter(hashes))) != 64):
        raise ValueError("Retained episodes must identify the same checkpoint content")
    for task in args.task_ids:
        index_key = "trial_id" if args.suite == LILO_SUITE else "init_state_id"
        task_rows = sorted((row for row in retained if row["task_id"] == task), key=lambda row: row[index_key])
        offset = 0
        for init, row in enumerate(task_rows):
            if (row[index_key] != init or row.get("first_server_request_id") != offset + 1
                    or row.get("last_server_request_id") != offset + row["policy_requests"]):
                raise ValueError("Retained task {} must have a contiguous valid episode/request prefix".format(task))
            offset += row["policy_requests"]
        starts[task] = {"first_init_state": len(task_rows), "request_offset": offset}
    return retained, starts, next(iter(hashes), None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    devices = parser.add_mutually_exclusive_group()
    devices.add_argument("--gpu", type=int, default=1, help="Run all workers on this GPU (default: 1).")
    devices.add_argument("--gpus", type=int, nargs="+", help="At most one model/simulator pair per listed GPU; EGL uses the same device index.")
    parser.add_argument("--egl-device", type=int, help="Rendering device for single-GPU mode (default: 1).")
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--base-port", type=int, default=8100)
    parser.add_argument("--suite", choices=tuple(MAX_STEPS) + (LILO_SUITE,), default="libero_10")
    parser.add_argument("--task-ids", type=int, nargs="+")
    parser.add_argument("--trials-per-task", type=int, help="Default: LIBERO 50; Ultra-Long 10. Use 1 for smoke.")
    parser.add_argument("--steps-per-skill", type=int, default=200, help="Ultra-Long total budget = this value times skill count.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default=os.environ.get("PI05_CHECKPOINT", CHECKPOINT))
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    lilo = args.suite == LILO_SUITE
    task_count = 9 if lilo else 10
    args.task_ids = list(range(task_count)) if args.task_ids is None else args.task_ids
    args.trials_per_task = (10 if lilo else 50) if args.trials_per_task is None else args.trials_per_task
    args.output_root = args.output_root or PROJECT_ROOT / ("outputs/pi05_lilo_parallel" if lilo else "outputs/pi05_libero_parallel")
    if lilo:
        args.seed = None  # Native bare reset; never seed or use LIBERO's fixed init-state files.
    if args.trials_per_task < 1 or (not lilo and args.trials_per_task > 50) or args.max_workers < 1:
        parser.error("trials-per-task and max-workers must be positive; standard LIBERO allows at most 50 trials")
    if len(set(args.task_ids)) != len(args.task_ids) or any(task not in range(task_count) for task in args.task_ids):
        parser.error("task-ids must be unique values in 0..{}".format(task_count - 1))
    if args.steps_per_skill < 1:
        parser.error("steps-per-skill must be positive")
    if args.gpus:
        if len(set(args.gpus)) != len(args.gpus) or any(gpu < 0 for gpu in args.gpus):
            parser.error("gpus must be unique nonnegative device indices")
        if args.egl_device is not None:
            parser.error("--gpus assigns EGL to the same devices; omit --egl-device")
        args.gpu = None
    else:
        args.egl_device = 1 if args.egl_device is None else args.egl_device
    gpu_ids = args.gpus or [args.gpu]
    if any(gpu < 0 for gpu in gpu_ids) or (args.egl_device is not None and args.egl_device < 0) or not 0 < args.base_port <= 65526:
        parser.error("GPU/device indices and the port range must be valid")
    retained, task_starts, args.checkpoint_sha256 = resume_state(args)
    prompt_report = None
    if lilo:
        from scripts.check_lilo_prompts import check_prompts
        prompt_report = check_prompts()
    initial_resources = resource_status()
    pending = [task for task in args.task_ids if task_starts[task]["first_init_state"] < args.trials_per_task]
    worker_limit = min(args.max_workers, len(pending))
    if args.gpus:
        worker_limit = min(worker_limit, len(args.gpus))
    initial_gpus = {gpu: gpu_status(gpu) for gpu in gpu_ids}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = args.output_root.resolve() / "pi05_{}_gpu{}_{}_{}".format(
        args.suite, "-".join(map(str, gpu_ids)), "unseeded" if lilo else "seed{}".format(args.seed), stamp)
    output_dir.mkdir(parents=True)
    with (output_dir / "retained_episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in retained:
            row["artifacts"] = os.path.relpath(row["artifacts"], output_dir)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "run.log", encoding="utf-8")])
    config = dict(vars(args), output_root=str(args.output_root), output_dir=str(output_dir),
                  selected_workers=worker_limit,
                  resume_from=str(args.resume_from.resolve()) if args.resume_from else None,
                  retained_episodes=len(retained), task_starts=task_starts, initial_resources=initial_resources,
                  initial_gpu=initial_gpus[args.gpu] if args.gpu is not None else None,
                  initial_gpus=initial_gpus, task_devices={},
                  model_rng_protocol="Each task has a fresh service, warmup restores key 0.",
                  concurrency_protocol="Fixed worker count with concurrent model startup and task completion refill.")
    if args.gpus:
        config["concurrency_protocol"] = "At most one model/simulator pair per GPU; the next task fills the first free GPU."
    if lilo:
        config.update(lilo_protocol=LILO_PROTOCOL, prompt_report=prompt_report, simulation_environment=".venv-libero",
                      initialization="unseeded reset plus ten B3 settling steps; new simulator per trial",
                      budget_scope="project end-to-end policy steps; no LiLo motion planner or retries")
    if args.resume_from:
        config["model_rng_protocol"] = "Key 0 advanced by each task's completed request count, then continuous sampling."
        config["resume_environment_protocol"] = ("New unseeded simulator per missing trial." if lilo
                                                 else "New simulator seeded at each resumed task segment.")
    write_json(output_dir / "run_config.json", config)
    logging.info("Run directory: %s", output_dir)
    logging.info("GPUs %s, fixed concurrency %d", gpu_ids, worker_limit)
    logging.info("Preflight cgroup snapshot: %.2f GiB used, %.2f GiB raw headroom, CPU quota %.1f cores, events %s",
                 initial_resources["memory_current_bytes"] / GIB, initial_resources["memory_available_bytes"] / GIB,
                 initial_resources["cpu_quota_cores"], initial_resources["memory_events"])
    logging.info("Retained %d valid episodes, including all recorded timeouts", len(retained))
    running = []
    peaks_by_gpu = {gpu: status["used_mib"] for gpu, status in initial_gpus.items()}
    peak_memory = sum(peaks_by_gpu.values())
    run_error = None
    started = time.perf_counter()
    next_report = 0.0
    metrics = (output_dir / "gpu.csv").open("w", newline="", encoding="utf-8")
    gpu_writer = csv.writer(metrics)
    gpu_writer.writerow(["elapsed_seconds", "gpu", "used_mib", "total_mib", "utilization_percent", "active_tasks"])
    try:
        while running or (pending and run_error is None):
            statuses = {gpu: gpu_status(gpu) for gpu in gpu_ids}
            peak_memory = max(peak_memory, sum(status["used_mib"] for status in statuses.values()))
            elapsed = time.perf_counter() - started
            for gpu, status in statuses.items():
                peaks_by_gpu[gpu] = max(peaks_by_gpu[gpu], status["used_mib"])
                gpu_writer.writerow([round(elapsed, 2), gpu, status["used_mib"], status["total_mib"],
                                     status["utilization_percent"], sum(task.gpu == gpu for task in running)])
            metrics.flush()
            rows = episode_rows(output_dir)
            for task in list(running):
                if task.client is None:
                    if task.server.poll() is not None:
                        run_error = "Task {} model exited; see {}".format(task.task_id, task.directory / "server.log")
                        stop_task(task)
                        running.remove(task)
                    elif server_ready(task):
                        launch_client(task, args)
                elif task.client.poll() is not None:
                    # Include rows flushed between the live read and process exit.
                    rows = episode_rows(output_dir)
                    task_rows = [row for row in rows if row["task_id"] == task.task_id]
                    task_summary = (summarize_lilo if lilo else summarize)(task_rows, args.trials_per_task)
                    if task.client.returncode != 0 or task_summary["status"] != "PASS":
                        run_error = "Task {} incomplete; see {}".format(task.task_id, task.directory / "client.log")
                    logging.info("Task %d finished with %d successes, %d timeouts, %d valid episodes",
                                 task.task_id, task_summary["successes"], task_summary["timeouts"], task_summary["valid_episodes"])
                    stop_task(task)
                    running.remove(task)
                elif task.server.poll() is not None:
                    run_error = "Task {} model stopped during evaluation".format(task.task_id)
                    stop_task(task)
                    running.remove(task)

            while pending and run_error is None and len(running) < worker_limit:
                task_id = pending[0]
                gpu = (next(gpu for gpu in args.gpus if all(task.gpu != gpu for task in running))
                       if args.gpus else args.gpu)
                egl_device = gpu if args.gpus else args.egl_device
                try:
                    task = launch_server(task_id, args, output_dir, gpu=gpu, egl_device=egl_device, **task_starts[task_id])
                except Exception as error:
                    run_error = "Task {} launch failed: {}: {}".format(task_id, type(error).__name__, error)
                    logging.exception("Task launch failed, allowing active evaluations to finish")
                else:
                    running.append(task)
                    pending.pop(0)
                    config["task_devices"][task_id] = {"gpu": gpu, "egl_device": egl_device}
                    write_json(output_dir / "run_config.json", config)
            if elapsed >= next_report:
                progress = aggregate(rows, args.task_ids, args.trials_per_task, run_error, args.suite)
                write_json(output_dir / "summary.json", progress)
                logging.info("%d/%d valid episodes, GPUs %s, %d active tasks",
                             progress["valid_episodes"], progress["planned_episodes"],
                             {gpu: {"vram_gib": round(status["used_mib"] / 1024, 2),
                                    "utilization_percent": status["utilization_percent"]}
                              for gpu, status in statuses.items()}, len(running))
                next_report = elapsed + 30
            time.sleep(2)
    except KeyboardInterrupt:
        run_error = "Interrupted by user"
        logging.info("Stopping child evaluations and saving their completed records")
    except Exception as error:
        run_error = "{}: {}".format(type(error).__name__, error)
        logging.exception("Parallel evaluation stopped")
    finally:
        for task in running:
            stop_task(task)
        metrics.close()
        rows = episode_rows(output_dir)
        summary = aggregate(rows, args.task_ids, args.trials_per_task, run_error, args.suite)
        if args.resume_from:
            summary["scope"] += " with recorded recovery segments and re-created simulators"
        elapsed = time.perf_counter() - started
        summary.update(output_dir=str(output_dir), suite=args.suite, gpu=args.gpu, egl_device=args.egl_device, gpus=gpu_ids,
                       sampled_peak_vram_gib_by_gpu={gpu: value / 1024 for gpu, value in peaks_by_gpu.items()},
                       vram_scope="Total across selected GPUs; per-device peaks reported separately.",
                       sampled_peak_vram_gib=peak_memory / 1024, selected_workers=worker_limit, wall_seconds=elapsed,
                       retained_episodes=len(retained), new_valid_episodes=summary["valid_episodes"] - len(retained),
                       valid_episodes_per_hour=(summary["valid_episodes"] - len(retained)) * 3600 / elapsed)
        write_json(output_dir / "summary.json", summary)
        with (output_dir / "episodes.jsonl").open("w", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
        print(json.dumps(summary, indent=2))
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
