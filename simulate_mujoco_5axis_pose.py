"""Play the project's five-axis pose library on the MuJoCo lamp model.

Run from the repository root:
    .\\.venv-mujoco\\Scripts\\python.exe simulate_mujoco_5axis_pose.py
"""

from __future__ import annotations

from time import sleep

import mujoco
import mujoco.viewer

from lamp_core.motion import TrajectoryPoint, plan_synchronised_minimum_jerk
from lamp_core.pose_library import IDLE_POSE, POSE_LIBRARY
from simulate import JOINT_LIMITS


# Relative path avoids MuJoCo's Windows Unicode absolute-path limitation.
MODEL_PATH = "simulations/mujoco/lamp_5axis.xml"
SAMPLE_PERIOD_S = 0.02
JOINT_NAMES = tuple(JOINT_LIMITS)


def pose_frames(start_rad: dict[str, float], target_rad: dict[str, float]) -> tuple[TrajectoryPoint, ...]:
    return tuple(
        plan_synchronised_minimum_jerk(
            start_rad,
            target_rad,
            JOINT_LIMITS,
            sample_period_s=SAMPLE_PERIOD_S,
        )
    )


def play_trajectory(
    viewer: mujoco.viewer.Handle,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    frames: tuple[TrajectoryPoint, ...],
) -> dict[str, float]:
    current = {name: float(data.qpos[model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]]) for name in JOINT_NAMES}
    for frame in frames:
        if not viewer.is_running():
            return current
        for name in JOINT_NAMES:
            qpos_index = model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)]
            position = frame.positions_rad[name]
            data.qvel[qpos_index] = (position - current[name]) / SAMPLE_PERIOD_S
            data.qpos[qpos_index] = position
            current[name] = position
        mujoco.mj_forward(model, data)
        viewer.sync()
        sleep(SAMPLE_PERIOD_S)
    return current


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    sequence = (
        "look_left", "idle", "look_right", "idle",
        "listening_pose", "idle",
        "nod_up", "nod_down", "idle",
        "curious_left", "curious_right", "idle",
        "head_shake_left", "head_shake_right", "idle",
        "reading_pose", "idle",
    )
    targets = {"idle": IDLE_POSE, **POSE_LIBRARY}
    current = dict(IDLE_POSE)
    print("Five-axis pose loop: production limits + minimum-jerk trajectory; close the window to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for pose_name in sequence:
                current = play_trajectory(viewer, model, data, pose_frames(current, targets[pose_name]))
                if not viewer.is_running():
                    return
                sleep(0.35)


if __name__ == "__main__":
    main()
