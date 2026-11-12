"""Run independent LIBERO tasks on one GPU within a measured VRAM budget."""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone
import errno
import json
import logging
import math
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


@dataclass
class TaskProcesses:
    task_id: int
    slot: int
    port: int
    directory: Path
    server: object
    server_log: object
    client: object = None
    client_log: object = None
    first_init_state: int = 0
    request_offset: int = 0
    cpu_ids: tuple = ()


def gpu_status(gpu):
    output = subprocess.check_output([
        "nvidia-smi", "-i", str(gpu),
        "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader,nounits",
    ], text=True)
    used, total, utilization = next(csv.reader(output.splitlines()))
    return {"used_mib": int(used.strip()), "total_mib": int(total.strip()),
            "utilization_percent": int(utilization.strip())}


def worker_capacity(target_mib, baseline_mib, measured_worker_mib, maximum):
    if measured_worker_mib <= 0:
        raise ValueError("A positive warm worker memory measurement is required")
    capacity = int((target_mib - baseline_mib) // measured_worker_mib)
    if capacity < 1:
        raise ValueError("The warmed model and simulator exceed the requested VRAM budget")
    return min(capacity, maximum)


def child_environment(gpu, egl_device, model, cpu_ids=()):
    environment = dict(os.environ)
    environment.pop("PYTHONPATH", None)
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[name] = "1"
    if cpu_ids:
        environment["LIBERO_CPU_IDS"] = ",".join(str(cpu) for cpu in cpu_ids)
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


def launch_server(task_id, slot, args, output_dir, first_init_state=0, request_offset=0):
    # Each task starts at a distinct port, including tasks replacing completed workers.
    port = available_port(args.base_port + task_id)
    directory = output_dir / "tasks" / "task{:02d}".format(task_id)
    directory.mkdir(parents=True)
    cpu_ids = tuple(args.cpu_ids[slot * args.cpus_per_worker:(slot + 1) * args.cpus_per_worker])
    log = (directory / "server.log").open("w", encoding="utf-8")
    command = [str(PROJECT_ROOT / ".venv/bin/python"), "-u", "-m", "scripts.serve_pi05_libero",
               "--port", str(port), "--checkpoint", args.checkpoint, "--warmup",
               "--request-offset", str(request_offset),
               "--output-root", str(directory / "server")]
    if args.checkpoint_sha256:
        command.extend(["--expected-checkpoint-sha256", args.checkpoint_sha256])
    try:
        process = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=log, stderr=subprocess.STDOUT,
                                   env=child_environment(args.gpu, args.egl_device, True, cpu_ids), start_new_session=True)
    except BaseException:
        log.close()
        raise
    logging.info("Starting task %d model on GPU %d, port %d", task_id, args.gpu, port)
    return TaskProcesses(task_id, slot, port, directory, process, log, first_init_state=first_init_state,
                         request_offset=request_offset, cpu_ids=cpu_ids)


def server_ready(task):
    try:
        with urllib.request.urlopen("http://127.0.0.1:{}/healthz".format(task.port), timeout=0.5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def launch_client(task, args):
    task.client_log = (task.directory / "client.log").open("w", encoding="utf-8")
    command = [str(PROJECT_ROOT / ".venv-libero/bin/python"), "-u", "-m", "scripts.eval_libero",
               "--port", str(task.port), "--suite", args.suite, "--task-id", str(task.task_id),
               "--init-state-id", str(task.first_init_state),
               "--num-trials-per-task", str(args.trials_per_task - task.first_init_state),
               "--request-offset", str(task.request_offset),
               "--seed", str(args.seed), "--output-root", str(task.directory / "client")]
    task.client = subprocess.Popen(command, cwd=PROJECT_ROOT, stdout=task.client_log, stderr=subprocess.STDOUT,
                                   env=child_environment(args.gpu, args.egl_device, False, task.cpu_ids), start_new_session=True)
    logging.info("Task %d started %d episodes from init %d on CPUs %s", task.task_id,
                 args.trials_per_task - task.first_init_state, task.first_init_state, task.cpu_ids)


def stop_task(task, terminate=False):
    # Stop the simulator first so it can flush the current episode and video.
    for process in (task.client, task.server):
        if process is not None and process.poll() is None:
            try:
                if terminate:
                    process.terminate()
                else:
                    process.send_signal(signal.SIGINT)
            except ProcessLookupError:
                pass  # The child exited between poll and signal.
            try:
                process.wait(timeout=3 if terminate else 20)
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


def aggregate(rows, task_ids, trials, run_error=None):
    expected = {(task, init) for task in task_ids for init in range(trials)}
    seen = [(row["task_id"], row["init_state_id"]) for row in rows]
    if len(set(seen)) != len(seen) or not set(seen).issubset(expected):
        raise ValueError("Episode IDs overlap or are outside the requested task/init set")
    scope = "single-GPU task-parallel evaluation with fresh model RNG key 0 for each task"
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
    rows = episode_rows(source)
    aggregate(rows, args.task_ids, args.trials_per_task)
    retained = [row for row in rows if row["outcome"] in ("success", "timeout")]
    hashes = set()
    for row in retained:
        artifact_dir = (source / row["artifacts"]).resolve()
        row["artifacts"] = str(artifact_dir)
        client_config = json.loads((artifact_dir.parents[1] / "run_config.json").read_text(encoding="utf-8"))
        hashes.add(client_config["server_metadata"]["evaluation"]["checkpoint_manifest"]["sha256"])
    if retained and (len(hashes) != 1 or len(next(iter(hashes))) != 64):
        raise ValueError("Retained episodes must identify the same checkpoint content")
    for task in args.task_ids:
        task_rows = sorted((row for row in retained if row["task_id"] == task), key=lambda row: row["init_state_id"])
        offset = 0
        for init, row in enumerate(task_rows):
            if (row["init_state_id"] != init or row.get("first_server_request_id") != offset + 1
                    or row.get("last_server_request_id") != offset + row["policy_requests"]):
                raise ValueError("Retained task {} must have a contiguous valid episode/request prefix".format(task))
            offset += row["policy_requests"]
        starts[task] = {"first_init_state": len(task_rows), "request_offset": offset}
    return retained, starts, next(iter(hashes), None)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, default=1)
    parser.add_argument("--egl-device", type=int, default=1)
    parser.add_argument("--target-vram-gib", type=float, default=35)
    parser.add_argument("--max-workers", type=int, default=3)
    parser.add_argument("--cpus-per-worker", type=int, default=2)
    parser.add_argument("--host-memory-reserve-gib", type=float, default=16)
    parser.add_argument("--resume-from", type=Path)
    parser.add_argument("--base-port", type=int, default=8100)
    parser.add_argument("--suite", choices=tuple(MAX_STEPS), default="libero_10")
    parser.add_argument("--task-ids", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--trials-per-task", type=int, default=50)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--checkpoint", default=os.environ.get("PI05_CHECKPOINT", CHECKPOINT))
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "outputs/pi05_libero_parallel")
    args = parser.parse_args(argv)
    if not 1 <= args.trials_per_task <= 50 or not 1 <= args.max_workers <= 10:
        parser.error("trials-per-task must be 1..50 and max-workers must be 1..10")
    if len(set(args.task_ids)) != len(args.task_ids) or any(task not in range(10) for task in args.task_ids):
        parser.error("task-ids must be unique values in 0..9")
    if args.gpu < 0 or args.egl_device < 0 or not 0 < args.base_port <= 65526:
        parser.error("GPU/device indices and the port range must be valid")
    if not math.isfinite(args.target_vram_gib) or args.target_vram_gib <= 0:
        parser.error("target-vram-gib must be a positive finite number")
    if (args.cpus_per_worker < 1 or not math.isfinite(args.host_memory_reserve_gib)
            or args.host_memory_reserve_gib < 0):
        parser.error("cpus-per-worker must be positive and host-memory-reserve-gib must be nonnegative")
    retained, task_starts, args.checkpoint_sha256 = resume_state(args)
    initial_resources = resource_status()
    initial_resource_sample = time.perf_counter()
    args.cpu_ids = initial_resources["cpu_ids"]
    cpu_capacity = int((initial_resources["cpu_quota_cores"] - 2) // args.cpus_per_worker)
    if cpu_capacity < 1:
        parser.error("CPU quota cannot fit a worker while leaving two cores for other processes")
    args.max_workers = min(args.max_workers, cpu_capacity)
    host_reserve_bytes = args.host_memory_reserve_gib * GIB
    host_worker_bytes = 0
    if initial_resources["memory_available_bytes"] <= host_reserve_bytes:
        parser.error("Cgroup has {:.2f} GiB available, at or below the {:.1f} GiB reserve"
                     .format(initial_resources["memory_available_bytes"] / GIB, args.host_memory_reserve_gib))
    initial_gpu = gpu_status(args.gpu)
    target_mib = int(args.target_vram_gib * 1024)
    if not initial_gpu["used_mib"] < target_mib < initial_gpu["total_mib"]:
        parser.error("VRAM target must exceed current usage and be below the GPU's total memory")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_dir = args.output_root.resolve() / "pi05_{}_gpu{}_seed{}_{}".format(args.suite, args.gpu, args.seed, stamp)
    output_dir.mkdir(parents=True)
    with (output_dir / "retained_episodes.jsonl").open("w", encoding="utf-8") as stream:
        for row in retained:
            row["artifacts"] = os.path.relpath(row["artifacts"], output_dir)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True,
                        handlers=[logging.StreamHandler(), logging.FileHandler(output_dir / "run.log", encoding="utf-8")])
    config = dict(vars(args), output_root=str(args.output_root), output_dir=str(output_dir), target_mib=target_mib,
                  resume_from=str(args.resume_from.resolve()) if args.resume_from else None,
                  retained_episodes=len(retained), task_starts=task_starts, initial_resources=initial_resources,
                  initial_gpu=initial_gpu, model_rng_protocol="Each task has a fresh service, warmup restores key 0.",
                  concurrency_protocol="One service and one sequential simulator per active task on the selected GPU.")
    if args.resume_from:
        config["model_rng_protocol"] = "Key 0 advanced by each task's completed request count, then continuous sampling."
        config["resume_environment_protocol"] = "New simulator seeded at each resumed task segment."
    write_json(output_dir / "run_config.json", config)
    logging.info("Run directory: %s", output_dir)
    logging.info("GPU %d target %.1f GiB, calibrating one warmed model and simulator", args.gpu, args.target_vram_gib)
    logging.info("Cgroup %.2f GiB available, %.1f GiB reserved, CPU quota %.1f cores, at most %d workers with %d CPUs each",
                 initial_resources["memory_available_bytes"] / GIB, args.host_memory_reserve_gib,
                 initial_resources["cpu_quota_cores"], args.max_workers, args.cpus_per_worker)
    logging.info("Retained %d valid episodes, including all recorded timeouts", len(retained))
    logging.info("Starting with one monitored worker to measure host memory use")
    running = []
    pending = [task for task in args.task_ids if task_starts[task]["first_init_state"] < args.trials_per_task]
    worker_memory = None
    worker_limit = 1
    peak_memory = initial_gpu["used_mib"]
    calibration_peak = peak_memory
    calibration_host_peak = initial_resources["memory_current_bytes"]
    run_error = None
    started = time.perf_counter()
    previous_resources, previous_sample = initial_resources, initial_resource_sample
    next_report = 0.0
    metrics = (output_dir / "gpu.csv").open("w", newline="", encoding="utf-8")
    gpu_writer = csv.writer(metrics)
    gpu_writer.writerow(["elapsed_seconds", "used_mib", "total_mib", "utilization_percent", "active_tasks",
                         "cgroup_available_gib", "cgroup_cpu_cores_used"])
    try:
        while running or (pending and run_error is None):
            status = gpu_status(args.gpu)
            resources = resource_status()
            sampled_at = time.perf_counter()
            cpu_used = (resources["cpu_usage_usec"] - previous_resources["cpu_usage_usec"]) / (
                max(sampled_at - previous_sample, 0.001) * 1e6)
            previous_resources, previous_sample = resources, sampled_at
            peak_memory = max(peak_memory, status["used_mib"])
            if worker_memory is None:
                calibration_peak = max(calibration_peak, status["used_mib"])
                calibration_host_peak = max(calibration_host_peak, resources["memory_current_bytes"])
            elapsed = time.perf_counter() - started
            gpu_writer.writerow([round(elapsed, 2), status["used_mib"], status["total_mib"],
                                 status["utilization_percent"], len(running),
                                 round(resources["memory_available_bytes"] / GIB, 2), round(cpu_used, 2)])
            metrics.flush()
            rows = episode_rows(output_dir)
            if worker_memory is None and len(rows) > len(retained):
                measured_worker = calibration_peak - initial_gpu["used_mib"]
                worker_memory = measured_worker * 1.25
                host_worker_bytes = max(16 * GIB,
                    (calibration_host_peak - initial_resources["memory_current_bytes"]) * 1.25)
                worker_limit = worker_capacity(target_mib, initial_gpu["used_mib"], worker_memory,
                                               min(args.max_workers, len(args.task_ids)))
                config.update(measured_worker_mib=measured_worker, reserved_worker_mib=worker_memory,
                              reserved_host_worker_bytes=host_worker_bytes, selected_workers=worker_limit)
                write_json(output_dir / "run_config.json", config)
                logging.info("Measured %.2f GiB GPU, reserving %.2f GiB GPU and %.2f GiB host per worker, selected %d tasks",
                             measured_worker / 1024, worker_memory / 1024, host_worker_bytes / GIB, worker_limit)
            if worker_memory is not None and running:
                observed_worker = (status["used_mib"] - initial_gpu["used_mib"]) / len(running)
                if observed_worker > worker_memory:
                    worker_memory = observed_worker
                    worker_limit = min(worker_limit, max(1, int((target_mib - initial_gpu["used_mib"]) // worker_memory)))
                    config.update(reserved_worker_mib=worker_memory, selected_workers=worker_limit)
                    write_json(output_dir / "run_config.json", config)

            for task in list(running):
                if task.client is None:
                    if resources["memory_available_bytes"] < host_reserve_bytes:
                        run_error = "Cgroup memory reserve reached while loading task {}".format(task.task_id)
                        logging.error(run_error)
                        stop_task(task, terminate=True)
                        running.remove(task)
                    elif task.server.poll() is not None:
                        run_error = "Task {} model exited; see {}".format(task.task_id, task.directory / "server.log")
                        stop_task(task)
                        running.remove(task)
                    elif server_ready(task):
                        launch_client(task, args)
                elif task.client.poll() is not None:
                    # Include rows flushed between the live read and process exit.
                    rows = episode_rows(output_dir)
                    task_rows = [row for row in rows if row["task_id"] == task.task_id]
                    task_summary = summarize(task_rows, args.trials_per_task)
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

            loading = any(task.client is None for task in running)
            has_budget = worker_memory is None or status["used_mib"] + worker_memory <= target_mib
            required_host_bytes = host_reserve_bytes
            if running:
                required_host_bytes += host_worker_bytes
            has_host_budget = resources["memory_available_bytes"] > required_host_bytes
            has_cpu_budget = cpu_used + args.cpus_per_worker <= resources["cpu_quota_cores"] - 2
            if (pending and run_error is None and len(running) < worker_limit and not loading
                    and has_budget and has_host_budget and has_cpu_budget):
                used_slots = {task.slot for task in running}
                slot = next(slot for slot in range(args.max_workers) if slot not in used_slots)
                task_id = pending[0]
                try:
                    task = launch_server(task_id, slot, args, output_dir, **task_starts[task_id])
                except Exception as error:
                    run_error = "Task {} launch failed: {}: {}".format(task_id, type(error).__name__, error)
                    logging.exception("Task launch failed, allowing active evaluations to finish")
                else:
                    running.append(task)
                    pending.pop(0)
            if elapsed >= next_report:
                progress = aggregate(rows, args.task_ids, args.trials_per_task, run_error)
                write_json(output_dir / "summary.json", progress)
                logging.info("%d/%d valid episodes, GPU %.2f GiB, utilization %d%%, %d active tasks",
                             progress["valid_episodes"], progress["planned_episodes"], status["used_mib"] / 1024,
                             status["utilization_percent"], len(running))
                logging.info("Cgroup %.2f GiB available, extra worker allowance %.2f GiB, %.1f GiB reserve, CPU %.1f/%.1f cores",
                             resources["memory_available_bytes"] / GIB, host_worker_bytes / GIB,
                             args.host_memory_reserve_gib, cpu_used, resources["cpu_quota_cores"])
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
        summary = aggregate(rows, args.task_ids, args.trials_per_task, run_error)
        if args.resume_from:
            summary["scope"] += " with recorded recovery segments and re-created simulators"
        elapsed = time.perf_counter() - started
        summary.update(output_dir=str(output_dir), suite=args.suite, gpu=args.gpu, egl_device=args.egl_device,
                       target_vram_gib=args.target_vram_gib,
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
