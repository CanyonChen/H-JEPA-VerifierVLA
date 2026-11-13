"""Read physical evidence without changing the policy, predicates or simulator."""

import numpy as np

from scripts.lilo_protocol import resolve_object


PHYSICS_SCHEMA = "lilo_physics_v1"


def object_snapshot(env, predicates, names):
    result = {}
    robot = env.env.robots[0]
    grip_position = env.sim.data.get_site_xpos(robot.gripper.important_sites["grip_site"])
    heights = predicates.get_predicate("pickedup").initial_heights
    for name in names:
        state = resolve_object(predicates.states, name)
        model = env.env.get_object(state.object_name)
        position = np.asarray(state.get_geom_state()["pos"])
        contacts = {side: bool(env.env.check_contact(robot.gripper.important_geoms[side + "_fingerpad"],
                                                    model.contact_geoms)) for side in ("left", "right")}
        baseline = heights.get(state.object_name)
        result[name] = dict(position_m=position.tolist(), height_baseline_m=baseline,
                            height_delta_m=float(position[2] - baseline) if baseline is not None else None,
                            gripper_distance_m=float(np.linalg.norm(position - grip_position)),
                            any_gripper_contact=bool(env.env.check_contact(robot.gripper.contact_geoms, model.contact_geoms)),
                            fingerpad_contacts=contacts, both_fingerpads_contact=all(contacts.values()))
        if state.has_turnon_affordance:
            result[name]["joint_positions"] = [np.asarray(v).tolist() for v in state.get_joint_state()]
    return result


class LiveDiagnostics:
    """Record pickedup/turnon objects at actual policy control steps."""

    def __init__(self, env, predicates, task):
        self.env, self.predicates = env, predicates
        self.names = sorted({atom[1] for skill in task["predicates"] for atom in skill
                             if atom[0].lower() in ("pickedup", "turnon")})
        self.history = {name: dict(both_fingerpads_contact_steps=0, last_both_fingerpads_contact_step=None,
                                  any_gripper_contact_steps=0, last_any_gripper_contact_step=None)
                        for name in self.names}

    def observe(self, policy_step):
        objects = object_snapshot(self.env, self.predicates, self.names)
        for name, snapshot in objects.items():
            history = self.history[name]
            if snapshot["both_fingerpads_contact"]:
                history["both_fingerpads_contact_steps"] += 1
                history["last_both_fingerpads_contact_step"] = policy_step
            if snapshot["any_gripper_contact"]:
                history["any_gripper_contact_steps"] += 1
                history["last_any_gripper_contact_step"] = policy_step
            snapshot.update(history)
        return dict(schema=PHYSICS_SCHEMA, source="live_rollout", policy_step=policy_step, objects=objects)
