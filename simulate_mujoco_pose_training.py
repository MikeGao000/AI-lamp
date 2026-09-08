"""Train and validate an offline pose-library imitation policy in MuJoCo.

Run from the repository root:
    .\\.venv-mujoco\\Scripts\\python.exe simulate_mujoco_pose_training.py
"""

from __future__ import annotations

import mujoco

from lamp_core.pose_imitation import (
    JOINT_NAMES,
    fit_minimum_jerk_imitation,
    generate_pose_imitation_samples,
    library_poses,
    maximum_angle_error,
    split_by_transition,
)
from simulate import JOINT_LIMITS


MODEL_PATH = "simulations/mujoco/lamp_5axis.xml"


def validate_in_mujoco(policy, samples) -> int:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    for sample in samples:
        predicted = policy.predict(sample.start_rad, sample.target_rad, sample.progress)
        for joint_name in JOINT_NAMES:
            JOINT_LIMITS[joint_name].validate(predicted[joint_name])
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            data.qpos[model.jnt_qposadr[joint_id]] = predicted[joint_name]
        mujoco.mj_forward(model, data)
    return len(samples)


def main() -> None:
    samples = generate_pose_imitation_samples(JOINT_LIMITS)
    training, validation = split_by_transition(samples)
    policy = fit_minimum_jerk_imitation(training)
    validation_error = maximum_angle_error(policy, validation)
    validated_frames = validate_in_mujoco(policy, validation)
    transition_count = len({sample.transition_index for sample in samples})
    print(f"Imported poses: {len(library_poses())}; directed transitions: {transition_count}")
    print(f"Training frames: {len(training)}; held-out frames: {len(validation)}")
    print("Learned blend: " + ", ".join(f"{value:+.9f}" for value in policy.coefficients))
    print(f"Held-out maximum joint error: {validation_error:.12f} rad")
    print(f"PASS MuJoCo pose-imitation validation: {validated_frames} held-out frames")


if __name__ == "__main__":
    main()
