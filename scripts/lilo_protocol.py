"""Ultra-Long task definitions, ordered-prefix scoring and result aggregation.

Only initialize_benchmark imports the simulator. The other functions run on CPU.
"""

import json
import sys

from scripts.check_libero_env import PROJECT_ROOT, configure_libero_paths
from scripts.eval_libero import summarize

LILO_COMMIT = "64aa29d77f5457afb3d1642cfc84534dc1e81fad"
LILO_ROOT = PROJECT_ROOT / "third_party/lilo_vla"
SUITE = "ultra_long_all"
TASK_IDS = (8, 9, 10, 19, 20, 21, 22, 23, 24)
PROTOCOL = "pi05_lilo_ordered_v1"
STEPS_PER_SKILL = 200


def task_definitions():
    config = json.loads((LILO_ROOT / "configs/tasks_and_skills.json").read_text(encoding="utf-8"))
    by_id = {task["task_id"]: task for task in config["long_horizon_tasks"]}
    tasks = []
    for index, original_id in enumerate(TASK_IDS):
        task = dict(by_id[original_id])
        task["definition_id"] = original_id
        task["task_id"] = index
        task["order"] = ("original", "reorder_1", "reorder_2")[index // 3]
        task["scenario"] = index % 3
        task["predicates"] = [config["skill_mappings"][skill]["bddl_predicates"] for skill in task["skills"]]
        # All three orders get their complete instruction, never a live oracle subgoal.
        task["prompt"] = ", then ".join(task["skills"]) + "."
        tasks.append(task)
    return tasks


def initialize_benchmark():
    configure_libero_paths(PROJECT_ROOT / ".venv-lilo/libero-config")
    sys.path.insert(0, str(LILO_ROOT))
    import lilo_vla.benchmark  # noqa: F401; process-local compatibility patches
    from lilo_vla.libero_compat import verify
    from libero.libero.benchmark import get_benchmark
    verify()
    suite = get_benchmark(SUITE)()
    if suite.get_num_tasks() != len(TASK_IDS):
        raise ValueError("Ultra-Long must contain nine configurations")
    return suite


def resolve_object(states, name):
    if name in states:
        return states[name]
    # Upstream skill config uses mesh-body suffixes; LIBERO object keys omit _main.
    if name.endswith("_main") and name[:-5] in states:
        return states[name[:-5]]
    raise KeyError("Skill predicate object missing from the scene: " + name)


class SkillPredicates:
    """Bind the published predicates to one environment; errors remain errors."""

    def __init__(self, env, task):
        from libero.libero.envs.predicates import get_predicate_fn
        self.get_predicate = get_predicate_fn
        self.states = env.env.object_states_dict
        self.predicates = task["predicates"]
        self.bound = [[(get_predicate_fn(p[0]), [resolve_object(self.states, name) for name in p[1:]])
                       for p in skill] for skill in self.predicates]
        # New episodes and repeated picks must not inherit an old object's height.
        self.get_predicate("pickedup").initial_heights.clear()
        for skill in self.bound:
            for predicate, objects in skill:
                predicate(*objects)

    def values(self):
        # Evaluate every atom, including stateful predicates, even after a false atom.
        return [all([bool(fn(*objects)) for fn, objects in skill]) for skill in self.bound]

    def begin_skill(self, index):
        if index == len(self.bound):
            return
        for specification, (predicate, objects) in zip(self.predicates[index], self.bound[index]):
            if specification[0].lower() == "pickedup":
                predicate.initial_heights.pop(objects[0].object_name, None)
                predicate(*objects)


class OrderedProgress:
    """Count the executed prefix; a new future-skill event cannot be credited later.

    Predicates already true at reset are not out-of-order events. Repeated copies
    of the active predicate denote the same event and do not create false failures.
    At most one skill is credited per control step.
    """

    def __init__(self, predicates, initial_values):
        self.keys = [json.dumps(p, sort_keys=True) for p in predicates]
        self.previous = list(initial_values)
        self.completed = 0
        self.out_of_order = None

    def update(self, values, step):
        if self.completed == len(self.keys) or self.out_of_order is not None:
            return []
        active = self.completed
        events = []
        if values[active]:
            self.completed += 1
            events.append({"kind": "skill_completed", "skill_index": active, "policy_step": step})
        for index in range(active + 1, len(values)):
            if self.keys[index] == self.keys[active]:
                continue
            if values[index] and not self.previous[index]:
                # The next skill may complete on the same sampled step as the active skill.
                if self.completed > active and index == self.completed:
                    continue
                self.out_of_order = index
                events.append({"kind": "out_of_order", "skill_index": index,
                               "expected_skill_index": self.completed, "policy_step": step})
                break
        self.previous = list(values)
        return events


def summarize_lilo(rows, planned, run_error=None):
    # Reuse accounting, while treating ordering violations as valid task failures.
    normalized = [dict(row, outcome="timeout" if row["outcome"] == "out_of_order" else row["outcome"])
                  for row in rows]
    result = summarize(normalized, planned, run_error,
                       "Ultra-Long ordered skills; unseeded resets; project end-to-end control budget")
    valid = [r for r in rows if r["outcome"] in ("success", "timeout", "out_of_order")]
    for row in valid:
        count, total = row["skills_completed"], row["total_skills"]
        if not 0 <= count <= total or total <= 0 or (row["outcome"] == "success") != (count == total):
            raise ValueError("Inconsistent ordered skill result")
    result["out_of_order_failures"] = sum(r["outcome"] == "out_of_order" for r in rows)
    result["timeouts"] -= result["out_of_order_failures"]
    result["average_progress"] = (sum(r["skills_completed"] / r["total_skills"] for r in valid) / planned
                                  if result["status"] == "PASS" else None)
    result["completed_skills"] = sum(r["skills_completed"] for r in valid)
    return result


def aggregate_lilo(rows, task_ids, trials, run_error=None):
    expected = {(task, trial) for task in task_ids for trial in range(trials)}
    seen = [(r["task_id"], r["trial_id"]) for r in rows]
    if len(set(seen)) != len(seen) or not set(seen).issubset(expected):
        raise ValueError("Duplicate or out-of-range Ultra-Long trial")
    result = summarize_lilo(rows, len(expected), run_error)
    result.update(suite=SUITE, protocol=PROTOCOL, lilo_commit=LILO_COMMIT)
    definitions = task_definitions()
    result["per_task"] = [dict(task_id=task, name=definitions[task]["name"],
                               **summarize_lilo([r for r in rows if r["task_id"] == task], trials))
                          for task in task_ids]
    result["per_order"] = []
    for order in range(3):
        group = [t for t in task_ids if t // 3 == order]
        if group:
            result["per_order"].append(dict(order=("original", "reorder_1", "reorder_2")[order],
                **summarize_lilo([r for r in rows if r["task_id"] in group], len(group) * trials)))
    return result
