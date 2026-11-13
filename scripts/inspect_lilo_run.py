"""Print a read-only audit of saved Ultra-Long skill failures; no GPU is used."""

import argparse
import json
from pathlib import Path


def read_json(path, errors):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        errors.append("{}: {}".format(path, error))
        return {}


def read_rows(path, errors):
    try:
        with path.open(encoding="utf-8") as stream:
            for number, line in enumerate(stream, 1):
                if line.strip():
                    try:
                        yield json.loads(line)
                    except ValueError as error:
                        errors.append("{}:{}: {}".format(path, number, error))
    except OSError as error:
        errors.append("{}: {}".format(path, error))


def skill_description(task, index):
    skills, predicates = task.get("skills", []), task.get("predicates", [])
    if not isinstance(index, int) or not 0 <= index < len(skills):
        return None
    return dict(index=index, instruction=skills[index],
                predicates=predicates[index] if index < len(predicates) else None)


def predicate_value(record, index):
    values = record.get("skill_predicates", []) if record else []
    return values[index] if isinstance(index, int) and 0 <= index < len(values) else None


def inspect_episode(run_dir, row):
    errors = []
    report = {key: row.get(key) for key in (
        "task_id", "trial_id", "task_name", "outcome", "policy_steps", "max_policy_steps",
        "skills_completed", "total_skills", "out_of_order_skill_index")}
    report["inspection_errors"] = errors
    if not row.get("artifacts"):
        errors.append("Episode row has no artifacts path")
        return report
    directory = (run_dir / row["artifacts"]).resolve()
    config = read_json(directory.parent.parent / "run_config.json", errors)
    task = config.get("task", {})
    if not task.get("skills") or not task.get("predicates"):
        errors.append("Saved client config has no skill definitions; live definitions are not substituted")
    evaluation = config.get("server_metadata", {}).get("evaluation", {})
    report["client"] = dict(repo_commit=config.get("repo_commit"), protocol=config.get("protocol"),
                            lilo_commit=config.get("lilo_commit"), egl_device_id=config.get("egl_device_id"),
                            cuda_visible_devices=evaluation.get("cuda_visible_devices"))
    report["prompt"] = task.get("prompt")
    report["next_skill"] = skill_description(task, row.get("skills_completed"))
    report["out_of_order_event"] = None
    previous = None
    trace_count = 0
    for record in read_rows(directory / "skills.jsonl", errors):
        trace_count += 1
        for event in record.get("events", []):
            if event.get("kind") == "out_of_order" and report["out_of_order_event"] is None:
                index = event.get("skill_index")
                report["out_of_order_event"] = dict(
                    policy_step=record.get("policy_step"),
                    expected=skill_description(task, event.get("expected_skill_index")),
                    triggered=skill_description(task, index),
                    previous_logged_step=previous.get("policy_step") if previous else None,
                    previous_logged_value=predicate_value(previous, index),
                    current_value=predicate_value(record, index))
        previous = record
    report["trace_rows"] = trace_count
    report["last_logged_step"] = previous.get("policy_step") if previous else None
    if row.get("outcome") == "out_of_order" and report["out_of_order_event"] is None:
        errors.append("Recorded out_of_order outcome has no saved out_of_order event")
    report["video"] = str(directory / "video.mp4")
    report["video_exists"] = (directory / "video.mp4").is_file()
    report["artifacts"] = str(directory)
    return report


def inspect_run(run_dir):
    run_dir = Path(run_dir).resolve()
    errors = []
    config = read_json(run_dir / "run_config.json", errors)
    summary = read_json(run_dir / "summary.json", errors)
    episodes = [inspect_episode(run_dir, row) for row in read_rows(run_dir / "episodes.jsonl", errors)]
    if len(episodes) != summary.get("recorded_episodes"):
        errors.append("Episode row count does not match the saved summary")
    return dict(
        inspection_status="INCOMPLETE" if errors or any(e["inspection_errors"] for e in episodes) else "COMPLETE",
        note="Read-only trace inspection, not a validation or change of scoring. Skill indices are zero-based. "
             "The previous logged value is from the preceding control step; it may differ from the internal "
             "baseline reset when a new pick skill starts.",
        run_dir=str(run_dir),
        parallel_config={key: config.get(key) for key in ("gpu", "gpus", "egl_device", "max_workers", "selected_workers")},
        reported_summary={key: summary.get(key) for key in (
            "status", "protocol", "valid_episodes", "successes", "timeouts", "out_of_order_failures",
            "infrastructure_errors", "success_rate", "average_progress")},
        inspection_errors=errors, episodes=episodes)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    report = inspect_run(parser.parse_args(argv).run_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["inspection_status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
