"""CPU checks for GPU budgeting, task scheduling and parallel result aggregation."""

import contextlib
import io
import json
import logging
import os
from pathlib import Path
import socket
import subprocess
import tempfile
import unittest
from unittest import mock
from types import SimpleNamespace

from scripts import run_libero_parallel as parallel


def episode(task, init, outcome="success"):
    return {"task_id": task, "init_state_id": init, "outcome": outcome,
            "policy_steps": 10, "policy_requests": 2, "artifacts": "episodes/init{:02d}".format(init),
            "first_server_request_id": init * 2 + 1, "last_server_request_id": init * 2 + 2}


class SimulatedProcesses:
    """Replace child execution while exercising the controller's normal entry point."""

    def __init__(self, fail_task=None):
        self.tick = 0
        self.processes = []
        self.starts = []
        self.fail_task = fail_task

    def popen(self, command, **kwargs):
        model = command[command.index("-m") + 1] == "scripts.serve_pi05_libero"
        output_root = Path(command[command.index("--output-root") + 1])
        task_id = int(output_root.parent.name[4:])
        port = int(command[command.index("--port") + 1])
        owner = self

        class Process:
            returncode = None

            def __init__(self):
                self.model = model
                self.task_id = task_id
                self.port = port
                self.started = owner.tick
                self.environment = kwargs["env"]
                self.command = command
                if not model:
                    self.first_init = int(command[command.index("--init-state-id") + 1])
                    self.path = output_root / "run" / "episodes.jsonl"
                    self.path.parent.mkdir(parents=True)
                    self.path.write_text(json.dumps(episode(task_id, self.first_init)) + "\n", encoding="utf-8")
                    (self.path.parent / "run_config.json").write_text(json.dumps({"server_metadata": {
                        "evaluation": {"checkpoint_manifest": {"sha256": "0" * 64}}}}), encoding="utf-8")

            def poll(self):
                if self.returncode is None and not model and owner.tick - self.started >= 6:
                    # The final rows arrive after the controller's live progress read.
                    with self.path.open("a", encoding="utf-8") as stream:
                        for init in range(self.first_init + 1, 50):
                            outcome = "infrastructure_error" if task_id == owner.fail_task and init == 49 else "success"
                            stream.write(json.dumps(episode(task_id, init, outcome)) + "\n")
                    self.returncode = 1 if task_id == owner.fail_task else 0
                return self.returncode

            def send_signal(self, value):
                self.returncode = 0

            def wait(self, timeout=None):
                return self.returncode

        process = Process()
        self.processes.append(process)
        self.starts.append((self.tick, model, task_id))
        return process

    def ready(self, task):
        return self.tick - task.server.started >= 2

    def gpu_status(self, gpu):
        active = sum(process.model and process.returncode is None for process in self.processes)
        return {"used_mib": 1024 + active * 10240, "total_mib": 81559, "utilization_percent": 80 if active else 0}

    def sleep(self, seconds):
        self.tick += 1
        if self.tick > 100:
            raise RuntimeError("Controller did not finish")

    def resources(self):
        active = sum(process.model and process.returncode is None for process in self.processes)
        return {"memory_current_bytes": (32 + active * 4) * parallel.GIB,
                "memory_limit_bytes": 200 * parallel.GIB, "memory_available_bytes": (168 - active * 4) * parallel.GIB,
                "cpu_quota_cores": 20, "cpu_ids": list(range(176)), "cpu_usage_usec": 0,
                "cpu_throttled_usec": 0, "memory_events": {"max": 5, "oom": 0, "oom_kill": 0}}


class LiberoParallelTest(unittest.TestCase):
    @unittest.skipUnless(os.name == "posix", "TCP TIME_WAIT reuse is verified on POSIX")
    def test_port_probe_reuses_closed_server_in_time_wait(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            with socket.create_connection(("127.0.0.1", port)) as client:
                connection, address = listener.accept()
                with connection:
                    connection.shutdown(socket.SHUT_WR)
                    self.assertEqual(client.recv(1), b"")
                    client.shutdown(socket.SHUT_WR)
                    self.assertEqual(connection.recv(1), b"")
        self.assertEqual(parallel.available_port(port), port)

    def test_real_occupied_port_is_skipped(self):
        with socket.socket() as listener:
            listener.bind(("0.0.0.0", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            self.assertGreater(parallel.available_port(port), port)

    def test_capacity_uses_measured_worker_memory_and_current_usage(self):
        self.assertEqual(parallel.worker_capacity(70 * 1024, 1024, 9 * 1024, 10), 7)
        self.assertEqual(parallel.worker_capacity(70 * 1024, 1024, 9 * 1024, 3), 3)
        with self.assertRaises(ValueError):
            parallel.worker_capacity(1024, 512, 1024, 10)
        with self.assertRaises(ValueError):
            parallel.worker_capacity(1024, 0, 0, 10)

    def test_gpu_query_and_model_render_device_selection(self):
        with mock.patch.object(parallel.subprocess, "check_output", return_value="8753, 81559, 0\n") as query:
            self.assertEqual(parallel.gpu_status(1)["used_mib"], 8753)
            self.assertEqual(query.call_args[0][0][1:3], ["-i", "1"])
        with mock.patch.dict(os.environ, {"PYTHONPATH": "old", "CUDA_VISIBLE_DEVICES": "0"}):
            model = parallel.child_environment(1, 1, model=True)
            render = parallel.child_environment(1, 1, model=False)
            self.assertEqual(model["CUDA_VISIBLE_DEVICES"], "1")
            self.assertEqual(model["XLA_PYTHON_CLIENT_PREALLOCATE"], "false")
            self.assertEqual(render["MUJOCO_EGL_DEVICE_ID"], "1")
            self.assertNotIn("CUDA_VISIBLE_DEVICES", render)
            self.assertNotIn("PYTHONPATH", model)
            self.assertNotIn("PYTHONPATH", render)
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "0")

    def test_full_aggregation_counts_500_unique_pairs_and_timeouts(self):
        rows = [episode(task, init, "timeout" if init == 49 else "success")
                for task in range(10) for init in range(50)]
        summary = parallel.aggregate(rows, list(range(10)), 50)
        self.assertEqual(summary["status"], "PASS")
        self.assertEqual(summary["valid_episodes"], 500)
        self.assertEqual(summary["timeouts"], 10)
        self.assertEqual(summary["success_rate"], 0.98)
        self.assertEqual(summary["per_task"][9]["successes"], 49)
        with self.assertRaises(ValueError):
            parallel.aggregate(rows + rows[:1], list(range(10)), 50)
        with self.assertRaises(ValueError):
            parallel.aggregate([episode(10, 0)], list(range(10)), 50)
        self.assertIsNone(parallel.aggregate(rows[:-1], list(range(10)), 50)["success_rate"])

    def test_live_reader_ignores_unfinished_row_and_remaps_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "tasks/task00/client/run/episodes.jsonl"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(episode(0, 0)) + '\n{"task_id":', encoding="utf-8")
            rows = parallel.episode_rows(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(root / rows[0]["artifacts"], path.parent / "episodes/init00")

    def test_child_cleanup_handles_signal_race_and_timeout(self):
        client, server = mock.Mock(), mock.Mock()
        client.poll.return_value = server.poll.return_value = None
        client.send_signal.side_effect = ProcessLookupError
        server.wait.side_effect = [subprocess.TimeoutExpired("server", 20), 0]
        task = parallel.TaskProcesses(0, 0, 8100, Path("unused"), server, mock.Mock(), client, mock.Mock())
        parallel.stop_task(task)
        client.wait.assert_called_once_with(timeout=20)
        server.kill.assert_called_once_with()
        task.client_log.close.assert_called_once_with()
        task.server_log.close.assert_called_once_with()

    @contextlib.contextmanager
    def run_controller(self, processes, extra_args=()):
        with tempfile.TemporaryDirectory() as directory:
            with mock.patch.object(parallel.subprocess, "Popen", side_effect=processes.popen), \
                    mock.patch.object(parallel.socket, "socket"), \
                    mock.patch.object(parallel, "server_ready", side_effect=processes.ready), \
                    mock.patch.object(parallel, "gpu_status", side_effect=processes.gpu_status), \
                    mock.patch.object(parallel, "resource_status", side_effect=processes.resources), \
                    mock.patch.object(parallel.time, "sleep", side_effect=processes.sleep), \
                    contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                try:
                    result = parallel.main(["--output-root", directory, *extra_args])
                finally:
                    for handler in list(logging.getLogger().handlers):
                        logging.getLogger().removeHandler(handler)
                        handler.close()
            run_dir, = Path(directory).iterdir()
            yield result, run_dir

    def test_controller_calibrates_then_schedules_and_collects_all_final_rows(self):
        processes = SimulatedProcesses()
        with self.run_controller(processes) as (result, directory):
            summary = json.loads((directory / "summary.json").read_text())
            config = json.loads((directory / "run_config.json").read_text())
            self.assertEqual(result, 0, summary)
            self.assertEqual(summary["valid_episodes"], 500)
            self.assertEqual(summary["selected_workers"], 2)
            self.assertEqual(summary["sampled_peak_vram_gib"], 21)
            self.assertEqual(config["measured_worker_mib"], 10240)
            rows = [json.loads(line) for line in (directory / "episodes.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 500)
            self.assertEqual(len({(row["task_id"], row["init_state_id"]) for row in rows}), 500)
        models = [process for process in processes.processes if process.model]
        self.assertEqual(len(models), 10)
        self.assertEqual({process.port for process in models}, set(range(8100, 8110)))
        for process in models:
            self.assertIn("--warmup", process.command)
            self.assertEqual(process.environment["CUDA_VISIBLE_DEVICES"], "1")
            self.assertIn(process.environment["LIBERO_CPU_IDS"], ("0,1", "2,3"))
            self.assertEqual(process.returncode, 0)
        loading = 0
        for tick, model, task in processes.starts:
            loading += 1 if model else -1
            self.assertIn(loading, (0, 1))
        self.assertGreaterEqual(models[1].started, 3)

    def test_launch_failure_allows_existing_task_to_complete(self):
        processes = SimulatedProcesses()
        launch = processes.popen

        def fail_task_two(command, **kwargs):
            if "task02" in command[-1]:
                raise OSError("launch failed")
            return launch(command, **kwargs)

        processes.popen = fail_task_two
        with self.run_controller(processes) as (result, directory):
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(result, 1)
            self.assertEqual(summary["valid_episodes"], 100)
            self.assertEqual(summary["interrupted_episodes"], 0)
        self.assertTrue(all(process.returncode is not None for process in processes.processes))

    def test_low_cgroup_memory_stops_loading_worker_and_preserves_active_task(self):
        processes = SimulatedProcesses()
        read_resources = processes.resources

        def constrained_resources():
            result = read_resources()
            active = sum(process.model and process.returncode is None for process in processes.processes)
            if active == 2:
                result["memory_available_bytes"] = 15 * parallel.GIB
            return result

        processes.resources = constrained_resources
        with self.run_controller(processes) as (result, directory):
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(result, 1)
            self.assertEqual(summary["valid_episodes"], 50)
            self.assertIn("Cgroup memory reserve", summary["run_error"])
        self.assertTrue(all(process.returncode is not None for process in processes.processes))

    def test_resume_keeps_290_results_and_runs_only_the_remaining_210(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            config = dict(suite="libero_10", seed=7, task_ids=list(range(10)), trials_per_task=50,
                          checkpoint=parallel.CHECKPOINT)
            (source / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
            valid_counts = [50, 48, 45, 46, 34, 41, 26, 0, 0, 0]
            timeouts = [2, 0, 3, 0, 1, 1, 2, 0, 0, 0]
            for task, count in enumerate(valid_counts[:7]):
                client_dir = source / "tasks/task{:02d}/client/run".format(task)
                client_dir.mkdir(parents=True)
                rows = [episode(task, init, "timeout" if init < timeouts[task] else "success") for init in range(count)]
                if task in range(1, 6):
                    rows.append(episode(task, count, "interrupted"))
                (client_dir / "episodes.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                (client_dir / "run_config.json").write_text(json.dumps({"server_metadata": {
                    "evaluation": {"checkpoint_manifest": {"sha256": "0" * 64}}}}), encoding="utf-8")
            original = {path: path.read_bytes() for path in source.rglob("*.json*")}
            processes = SimulatedProcesses()
            with self.run_controller(processes, ["--resume-from", str(source)]) as (result, run_dir):
                summary = json.loads((run_dir / "summary.json").read_text())
                self.assertEqual(result, 0, summary)
                self.assertEqual(summary["valid_episodes"], 500)
                self.assertEqual(summary["retained_episodes"], 290)
                self.assertEqual(summary["new_valid_episodes"], 210)
                self.assertEqual(summary["timeouts"], 9)
                self.assertEqual(summary["interrupted_episodes"], 0)
                args = SimpleNamespace(**config, resume_from=run_dir)
                rows, starts, checkpoint_hash = parallel.resume_state(args)
                self.assertEqual(len(rows), 500)
                self.assertTrue(all(start["first_init_state"] == 50 for start in starts.values()))
                self.assertEqual(checkpoint_hash, "0" * 64)
            clients = [process for process in processes.processes if not process.model]
            self.assertEqual(len(clients), 9)
            for client in clients:
                self.assertEqual(client.first_init, valid_counts[client.task_id])
                self.assertEqual(int(client.command[client.command.index("--request-offset") + 1]),
                                 valid_counts[client.task_id] * 2)
            self.assertEqual(original, {path: path.read_bytes() for path in original})

    def test_failed_task_retains_error_and_stops_scheduling_new_tasks(self):
        processes = SimulatedProcesses(fail_task=0)
        with self.run_controller(processes) as (result, directory):
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(result, 1)
            self.assertEqual(summary["status"], "INCOMPLETE")
            self.assertEqual(summary["infrastructure_errors"], 1)
            self.assertIsNone(summary["success_rate"])
            self.assertEqual(summary["recorded_episodes"], 100)
            self.assertEqual(summary["unattempted_episodes"], 400)
        self.assertTrue(all(process.returncode is not None for process in processes.processes))

    def test_keyboard_interrupt_stops_owned_children_and_keeps_completed_rows(self):
        processes = SimulatedProcesses()
        advance = processes.sleep

        def interrupt_after_calibration(seconds):
            advance(seconds)
            if processes.tick == 4:
                raise KeyboardInterrupt

        processes.sleep = interrupt_after_calibration
        with self.run_controller(processes) as (result, directory):
            summary = json.loads((directory / "summary.json").read_text())
            self.assertEqual(result, 1)
            self.assertEqual(summary["run_error"], "Interrupted by user")
            self.assertEqual(summary["valid_episodes"], 1)
            self.assertIsNone(summary["success_rate"])
        self.assertTrue(all(process.returncode is not None for process in processes.processes))


if __name__ == "__main__":
    unittest.main()
