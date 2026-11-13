"""Print a read-only audit of saved Ultra-Long skill failures; no GPU is used."""

import argparse
from collections import deque
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
        "skills_completed", "total_skills", "out_of_order_skill_index", "stage", "error")}
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
    report["live_physics"] = None
    if config.get("record_physics"):
        last_physics, physics_count = None, 0
        for frame in read_rows(directory / "physics.jsonl", errors):
            physics_count += 1
            last_physics = frame
        if physics_count != trace_count or not last_physics or last_physics.get("policy_step") != row.get("policy_steps"):
            errors.append("Live physics records do not cover the complete saved skill trace")
        else:
            event = report["out_of_order_event"] or {}
            skills = [event.get("expected") or report["next_skill"], event.get("triggered")]
            names = {name for skill in skills if skill for atom in (skill.get("predicates") or []) for name in atom[1:]}
            report["live_physics"] = dict(last_physics, objects={name: value for name, value in last_physics["objects"].items()
                                                               if not names or name in names})
    report["video"] = str(directory / "video.mp4")
    report["video_exists"] = (directory / "video.mp4").is_file()
    report["artifacts"] = str(directory)
    return report


def check_summary_counts(config, summary, rows, errors):
    """Cross-check saved grouping against episode rows, without changing scores."""
    def compare(label, reported, selected):
        for field in ("recorded_episodes", "policy_steps", "policy_requests"):
            if field not in reported:
                continue
            actual = len(selected) if field == "recorded_episodes" else sum(row[field] for row in selected)
            if reported[field] != actual:
                errors.append("{} {}: summary={}, episode rows={}".format(label, field, reported[field], actual))

    compare("Overall", summary, rows)
    if "per_task" in summary:
        task_ids = config.get("task_ids", sorted({row["task_id"] for row in rows}))
        reported_ids = [item["task_id"] for item in summary["per_task"]]
        if sorted(reported_ids) != sorted(task_ids):
            errors.append("per_task IDs do not match the configured task IDs")
        for item in summary["per_task"]:
            compare("Task {}".format(item["task_id"]), item,
                    [row for row in rows if row["task_id"] == item["task_id"]])
    elif "task_ids" in config:
        errors.append("Saved summary has no per_task entries")


def inspect_run(run_dir):
    run_dir = Path(run_dir).resolve()
    errors = []
    config = read_json(run_dir / "run_config.json", errors)
    summary = read_json(run_dir / "summary.json", errors)
    rows = list(read_rows(run_dir / "episodes.jsonl", errors))
    episodes = [inspect_episode(run_dir, row) for row in rows]
    check_summary_counts(config, summary, rows, errors)
    if "recorded_episodes" not in summary:
        errors.append("Saved summary has no recorded_episodes count")
    return dict(
        inspection_status="INCOMPLETE" if errors or any(e["inspection_errors"] for e in episodes) else "COMPLETE",
        note="Read-only trace inspection, not a validation or change of scoring. Skill indices are zero-based. "
             "The previous logged value is from the preceding control step; it may differ from the internal "
             "baseline reset when a new pick skill starts.",
        run_dir=str(run_dir),
        parallel_config={key: config.get(key) for key in ("gpu", "gpus", "egl_device", "max_workers", "selected_workers")},
        reported_summary={key: summary.get(key) for key in (
            "status", "protocol", "planned_episodes", "recorded_episodes", "unattempted_episodes",
            "valid_episodes", "successes", "timeouts", "out_of_order_failures", "infrastructure_errors",
            "interrupted_episodes", "success_rate", "average_progress", "run_error")},
        inspection_errors=errors, episodes=episodes)


def compact_report(report):
    compact = {key: report[key] for key in ("inspection_status", "run_dir", "parallel_config", "reported_summary", "inspection_errors")}
    compact["episodes"] = []
    for row in report["episodes"]:
        event = row.get("out_of_order_event") or {}
        item = {key: row[key] for key in ("task_id", "trial_id", "outcome", "policy_steps", "skills_completed",
                                         "stage", "error", "inspection_errors")}
        item["expected"] = (event.get("expected") or row.get("next_skill") or {}).get("instruction")
        item["triggered"] = (event.get("triggered") or {}).get("instruction")
        item["live_physics"] = row.get("live_physics")
        if item["live_physics"]:
            fields = ("height_delta_m", "gripper_distance_m", "fingerpad_contacts", "both_fingerpads_contact",
                      "both_fingerpads_contact_steps", "last_both_fingerpads_contact_step", "any_gripper_contact",
                      "any_gripper_contact_steps", "last_any_gripper_contact_step", "turnon", "joint_positions")
            item["live_physics"] = dict(item["live_physics"], objects={name: {key: value[key] for key in fields if key in value}
                                                        for name, value in item["live_physics"]["objects"].items()})
        compact["episodes"].append(item)
    return compact


def error_report(report):
    failed = [row for row in report["episodes"] if row["outcome"] in ("infrastructure_error", "interrupted")]
    result = compact_report(dict(report, episodes=failed))
    for source, item in zip(failed, result["episodes"]):
        item["log_tails"] = {}
        if not source.get("artifacts"):
            continue
        task_dir = Path(source["artifacts"]).parents[3]
        for name, lines in (("client.log", 70), ("server.log", 40)):
            path = task_dir / name
            try:
                with path.open(encoding="utf-8", errors="replace") as stream:
                    tail = "".join(deque(stream, maxlen=lines))
                item["log_tails"][name] = dict(path=str(path), tail=tail)
            except OSError as error:
                item["log_tails"][name] = dict(path=str(path), read_error=str(error))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--compact", action="store_true", help="Print skill names and live physics without full prompts or paths.")
    parser.add_argument("--errors-only", action="store_true", help="Print infrastructure/interruption errors and bounded client/server log tails.")
    args = parser.parse_args(argv)
    report = inspect_run(args.run_dir)
    output = error_report(report) if args.errors_only else compact_report(report) if args.compact else report
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["inspection_status"] == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
