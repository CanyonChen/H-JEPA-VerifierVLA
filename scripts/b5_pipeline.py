"""Run the Long pilot, paired validation, and four-suite training in one AIP allocation."""
import argparse
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
JOB = (ROOT / "outputs/verifier_aip_seed7").resolve()
LONG = (ROOT / "outputs/verifier_long").resolve()
SUITES = ("libero_spatial", "libero_object", "libero_goal")
TRAIN = ROOT / ".venv-robometer/bin/python"
SIM = ROOT / ".venv-libero/bin/python"


def run(py, module, *args):
    cmd = [str(py), "-u", "-m", "scripts." + module, *map(str,args)]
    print("Running", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def label(suite):
    from scripts.robometer_backend import RobometerLabeler
    from scripts.probe_robometer_context import MODEL_FIELDS
    from scripts.label_verifier_long_splits import run_labels
    model = RobometerLabeler()
    ref = read(LONG / "robometer_raw_trainval_seed7_gpu0/run_config.json")["model"]
    if any(model.provenance[k] != ref[k] for k in MODEL_FIELDS):
        raise ValueError("Robometer provenance differs from the completed Long labels")
    path = JOB / suite
    run_labels(path/"split", path/"labels", model)


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def prepare_suite(suite):
    path = JOB / suite
    splits = path / "split"
    if not (splits/"manifest.json").is_file():
        collection = path / "collection"
        runs = sorted(p.parent for p in collection.glob("*/run_config.json"))
        complete = [r for r in runs if (r/"summary.json").is_file() and read(r/"summary.json")["status"]=="PASS"]
        if not complete:
            args = ["--suite", suite, "--gpu","0", "--egl-device","0", "--max-workers","2",
                    "--trials-per-task","50", "--seed","7", "--output-root", collection]
            if runs:
                args += ["--resume-from", runs[-1]]
            run(SIM, "run_libero_parallel", *args)
            complete = sorted(p.parent for p in collection.glob("*/summary.json") if read(p)["status"]=="PASS")
        if not complete:
            raise RuntimeError("Collection did not complete for " + suite)
        run(SIM, "prepare_verifier_long_data", "--run-dir", complete[-1], "--output-dir", splits)
    if not (path/"labels/complete/summary.json").is_file():
        run(TRAIN, "b5_pipeline", "--label", suite)


def pipeline():
    import fcntl
    JOB.mkdir(parents=True, exist_ok=True)
    with (JOB/"pipeline.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result_path = JOB / "summary.json"
        if result_path.is_file() and read(result_path)["status"] == "COMPLETE":
            print(result_path.read_text(), flush=True)
            return
        run(TRAIN, "b5_model")
        run(TRAIN, "b5_features", "long100")
        run(TRAIN, "b5_train", "long100")
        run(SIM, "b5_closed_loop")
        pilot = read(JOB/"long100/training.json")
        paired = read(JOB/"long100/closed_loop/summary.json")
        if pilot["updates"] != 100 or paired["status"] != "COMPLETE" or paired["pairs"] != 4:
            raise RuntimeError("Pilot has not completed its actual updates and paired rollouts")
        for suite in SUITES:
            prepare_suite(suite)
        run(TRAIN, "b5_features", "joint40")
        run(TRAIN, "b5_train", "joint40", "--epochs", os.environ.get("B5_EPOCHS","10"),
            "--microbatch", pilot["microbatch"])
        joint = read(JOB/"joint40/training.json")
        from scripts.b5_model import save_json
        result = dict(status="COMPLETE", long_updates=pilot["updates"], validation_pairs=paired["pairs"],
                      joint_tasks=joint["config"]["tasks"], joint_updates=joint["updates"],
                      best_checkpoint=str(JOB/"joint40/best.pt"), joint_report=str(JOB/"joint40/training.json"))
        save_json(result_path, result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", choices=SUITES)
    args = parser.parse_args()
    if args.label:
        label(args.label)
    else:
        pipeline()
