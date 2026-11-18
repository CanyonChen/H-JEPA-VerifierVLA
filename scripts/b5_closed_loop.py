"""Paired closed-loop Long validation on identical initial states and candidate-noise streams."""
import base64
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

from scripts.audit_verifier_data import read_rows
from scripts.check_libero_env import PROJECT_ROOT, configure_libero_paths
from scripts.eval_libero import MAX_STEPS, episode_progress, rollout_episode, validate_server_provenance
from scripts.run_libero_parallel import available_port
from scripts.serve_pi05_libero import CHECKPOINT

JOB = PROJECT_ROOT / "outputs/verifier_aip_seed7"
INDEX = PROJECT_ROOT / "outputs/verifier_long/split_seed7"
OUTPUT = JOB / "long100/closed_loop"


def save(path, data):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def noise_seeds(key, step, first, count):
    return [int.from_bytes(hashlib.sha256(("7/{}/{}/{}".format(key, step, i)).encode()).digest()[:4], "little")
            for i in range(first, first + count)]


def signals(actions):
    a = np.asarray(actions, dtype=np.float64)[:, :5]
    magnitude = np.maximum(np.linalg.norm(a, axis=-1).mean(axis=1), 1e-8)
    a = a / magnitude[:, None, None]
    return [(np.diff(a, axis=1)**2).sum(-1).mean().item(),
            ((a - a.mean(0))**2).sum(-1).mean().item()]


def candidates(client, obs, key, step, first, count):
    return np.asarray(client.infer(dict(obs, b5_seeds=noise_seeds(key, step, first, count)))["candidate_actions"])


def choose_episodes(episodes, tasks):
    return [min((e for e in episodes if e["task_id"] == t), key=lambda e: e["init_state_id"]) for t in tasks]


def calibration(remote):
    refs = []
    episodes = choose_episodes(read_rows(INDEX / "train_episodes.jsonl"), [0, 9])
    ids = {ep["episode_id"] for ep in episodes}
    for w in read_rows(INDEX / "train_windows.jsonl"):
        if w["episode_id"] in ids:
            with np.load(w["observation"], allow_pickle=False) as raw:
                obs = {k: (raw[k].item() if k == "prompt" else raw[k]) for k in raw.files}
            a = candidates(remote, obs, w["episode_id"], w["decision_index"], 0, 2)
            refs.append(signals(a))
    if not refs:
        raise ValueError("No calibration observations")
    return dict(episodes=sorted(ids), split="train", threshold=.8, signals=refs)


def receive(process):
    line = process.stdout.readline()
    if not line:
        raise RuntimeError("Verifier scorer exited; see scorer.log")
    return json.loads(line)


class PairedClient:
    def __init__(self, remote, scorer, ep, mode, calibration):
        self.remote, self.scorer, self.ep, self.mode = remote, scorer, ep, mode
        self.ref = np.asarray(calibration["signals"])
        self.threshold = calibration["threshold"]
        self.step = self.triggered = self.changed = 0
        self.trace = {}

    def infer(self, obs):
        key, step = self.ep["episode_id"], self.step
        self.step += 1
        verify = self.mode == "verifier"
        a = candidates(self.remote, obs, key, step, 0, 2 if verify else 1)
        self.trace = {"selected": 0, "candidates":len(a), "triggered": False}
        selected = 0
        if verify:
            stat = signals(a)
            percentile = ((self.ref <= stat).mean(0)).tolist()
            self.trace.update(signals=stat, percentiles=percentile)
            triggered = max(percentile) > self.threshold
            # At most one shadow ranking at the first held-out state documents an
            # actual scorer call even if all ECDF scores fall below the threshold.
            shadow = not triggered and step == 0 and self.ep["task_id"] == 0
            if triggered or shadow:
                more = candidates(self.remote, obs, key, step, 2, 8)
                a = np.concatenate([a, more])
                msg = dict(rgb=base64.b64encode(obs["observation/image"].tobytes()).decode("ascii"),
                           state=obs["observation/state"].tolist(), actions=a.tolist(), task=self.ep["task_id"])
                self.scorer.stdin.write(json.dumps(msg) + "\n")
                self.scorer.stdin.flush()
                scores = receive(self.scorer)
                selected = int(np.argmax(scores)) if triggered else 0
                self.triggered += int(triggered)
                self.changed += int(selected != 0)
                self.trace.update(selected=selected, candidates=len(a), triggered=triggered, shadow=shadow, scores=scores)
        return {"actions": a[selected]}


def stop(proc):
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def main():
    import imageio.v2 as imageio
    from openpi_client.websocket_client_policy import WebsocketClientPolicy
    configure_libero_paths()
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    OUTPUT.mkdir(parents=True, exist_ok=True)
    training = (JOB/"long100/training.json").read_bytes()
    identity = hashlib.sha256(training).hexdigest()
    if json.loads(training)["updates"] != 100:
        raise ValueError("Long closed loop requires exactly 100 completed updates")
    report_path = OUTPUT/"summary.json"
    if report_path.is_file():
        old = json.loads(report_path.read_text())
        if old["training_identity"] != identity or old["status"] != "COMPLETE":
            raise ValueError("Closed-loop result belongs to a different pilot")
        return
    held = choose_episodes(read_rows(INDEX/"validation_episodes.jsonl"), [0,3,6,9])
    if any(ep["split"] != "validation" for ep in held):
        raise ValueError("Unexpected evaluation partition")
    suite = benchmark.get_benchmark_dict()["libero_10"]()
    results = []
    with contextlib.ExitStack() as life:
        port = available_port(8160)
        log = life.enter_context((OUTPUT/"policy.log").open("a"))
        env = dict(os.environ, XLA_PYTHON_CLIENT_PREALLOCATE="false", JAX_PLATFORMS="cuda")
        server = subprocess.Popen([str(PROJECT_ROOT/".venv/bin/python"), "-u", "-m", "scripts.serve_pi05_libero",
            "--port", str(port), "--checkpoint", os.environ.get("PI05_CHECKPOINT", CHECKPOINT), "--paired-candidates",
            "--output-root", str(OUTPUT/"service")], cwd=PROJECT_ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        life.callback(stop, server)
        from urllib.request import urlopen
        from urllib.error import URLError
        while True:
            if server.poll() is not None:
                raise RuntimeError("pi0.5 exited; see policy.log")
            try:
                with urlopen("http://127.0.0.1:{}/healthz".format(port), timeout=2) as response:
                    if response.status == 200:
                        break
            except (URLError, TimeoutError):
                pass
            time.sleep(1)
        remote = WebsocketClientPolicy("127.0.0.1", port)
        prov = validate_server_provenance(remote.get_server_metadata())
        worker_log = life.enter_context((OUTPUT/"scorer.log").open("a"))
        worker = subprocess.Popen([str(PROJECT_ROOT/".venv-robometer/bin/python"),"-u","-m","scripts.b5_score"],
                  cwd=PROJECT_ROOT, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=worker_log, text=True)
        life.callback(stop, worker)
        if receive(worker) != {"ready":True,"updates":100}:
            raise RuntimeError("Wrong pilot scorer")
        ref = calibration(remote)
        save(OUTPUT/"calibration.json", ref)
        for ep in held:
            task = suite.get_task(ep["task_id"])
            initial = suite.get_task_init_states(ep["task_id"])[ep["init_state_id"]]
            for arm in ("base", "verifier"):
                name = ep["episode_id"] + "_" + arm
                result = dict(episode_progress(), episode=ep["episode_id"], arm=arm, task_id=ep["task_id"],
                              init_state_id=ep["init_state_id"], split="validation")
                np.random.seed(7)
                env = OffScreenRenderEnv(bddl_file_name=suite.get_task_bddl_file_path(ep["task_id"]),
                                         camera_heights=256, camera_widths=256)
                client = PairedClient(remote, worker, ep, arm, ref)
                try:
                    env.seed(7)
                    begin = time.perf_counter()
                    with imageio.get_writer(OUTPUT/(name+".mp4"), fps=10) as vid, (OUTPUT/(name+".jsonl")).open("w") as trace:
                        def frame(obs):
                            vid.append_data(obs["observation/image"])
                        def decision(d, obs):
                            trace.write(json.dumps(dict(d, selection=client.trace), allow_nan=False) + "\n")
                        rollout_episode(env, client, initial, task.language, result, frame, decision, MAX_STEPS["libero_10"])
                    result.update(wall_seconds=time.perf_counter()-begin, triggered=client.triggered,
                                  changed=client.changed)
                finally:
                    env.close()
                if result["outcome"] not in ("success","timeout"):
                    raise RuntimeError("Incomplete paired rollout")
                results.append(result)
                save(OUTPUT/(name+".json"), result)
                print(json.dumps(result), flush=True)
    report = dict(status="COMPLETE", training_identity=identity, updates=100, pairs=len(held), episodes=results,
                  policy_sha256=prov["checkpoint_manifest"]["sha256"], threshold=ref["threshold"], k=2, K=10,
                  reference="two recorded training rollouts; executed five-step prefixes",
                  success_counts={arm:sum(r["task_success"] for r in results if r["arm"]==arm) for arm in ("base","verifier")})
    save(report_path, report)
    print(json.dumps({"status":"PAIRED_COMPLETE","pairs":len(held),"successes":report["success_counts"]}), flush=True)


if __name__ == "__main__":
    main()
