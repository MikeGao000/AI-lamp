"""Visually replay the trained pose-imitation policy in MuJoCo.

Run from the repository root:
    .\\.venv-mujoco\\Scripts\\python.exe simulate_mujoco_trained_policy.py
"""

from __future__ import annotations

from time import sleep

import mujoco
import mujoco.viewer

from lamp_core.motion import plan_synchronised_minimum_jerk
from lamp_core.pose_imitation import (
    JOINT_NAMES,
    fit_minimum_jerk_imitation,
    generate_pose_imitation_samples,
    library_poses,
    split_by_transition,
)
from simulate import JOINT_LIMITS


MODEL_PATH = "simulations/mujoco/lamp_5axis.xml"
SAMPLE_PERIOD_S = 0.02


def apply_positions(model: mujoco.MjModel, data: mujoco.MjData, positions: dict[str, float]) -> None:
    for joint_name in JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        data.qpos[model.jnt_qposadr[joint_id]] = positions[joint_name]
    mujoco.mj_forward(model, data)


def play_policy_transition(viewer, model, data, policy, start, target) -> bool:
    # The existing planner supplies duration only.  Every displayed pose comes
    # from the trained imitation policy, not from its reference frame values.
    reference = plan_synchronised_minimum_jerk(start, target, JOINT_LIMITS, SAMPLE_PERIOD_S)
    last_index = len(reference) - 1
    for index in range(len(reference)):
        if not viewer.is_running():
            return False
        predicted = policy.predict(start, target, index / last_index)
        apply_positions(model, data, predicted)
        viewer.sync()
        sleep(SAMPLE_PERIOD_S)
    return True


def main() -> None:
    samples = generate_pose_imitation_samples(JOINT_LIMITS)
    training, _ = split_by_transition(samples)
    policy = fit_minimum_jerk_imitation(training)
    poses = library_poses()
    sequence = ("look_left", "idle", "look_right", "idle", "nod_down", "idle", "reading_pose", "idle")
    current = dict(poses["idle"])
    print("Trained-policy playback: all displayed joint commands come from the imitation model.")
    print("Learned blend: " + ", ".join(f"{value:+.6f}" for value in policy.coefficients))
    with mujoco.viewer.launch_passive(model := mujoco.MjModel.from_xml_path(MODEL_PATH), data := mujoco.MjData(model)) as viewer:
        apply_positions(model, data, current)
        while viewer.is_running():
            for pose_name in sequence:
                if not play_policy_transition(viewer, model, data, policy, current, poses[pose_name]):
                    return
                current = dict(poses[pose_name])
                sleep(0.30)


if __name__ == "__main__":
    main()
