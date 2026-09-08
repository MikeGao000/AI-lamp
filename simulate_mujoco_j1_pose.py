"""Show pose-library J1 motions in MuJoCo at the project's planned speed.

Run:
    .\\.venv-mujoco\\Scripts\\python.exe simulate_mujoco_j1_pose.py
"""

from __future__ import annotations

from time import sleep

import mujoco
import mujoco.viewer

from lamp_core.motion import JointLimit, plan_synchronised_minimum_jerk
from lamp_core.pose_library import IDLE_POSE, POSE_LIBRARY


# MuJoCo's Windows XML loader cannot open Unicode absolute paths reliably.
# The documented command runs from the repository root, so keep this relative.
MODEL_PATH = "simulations/mujoco/lamp_j1.xml"
SAMPLE_PERIOD_S = 0.02
# Same existing J1 envelope used by simulate.py and the system tests.
J1_LIMIT = JointLimit(-1.57, 1.57, maximum_speed_rad_s=0.80)


def pose_frames(start_rad: float, target_rad: float):
    """Reuse the production minimum-jerk planner for visual pose playback."""
    return plan_synchronised_minimum_jerk(
        {"j1_base_yaw": start_rad},
        {"j1_base_yaw": target_rad},
        {"j1_base_yaw": J1_LIMIT},
        sample_period_s=SAMPLE_PERIOD_S,
    )


def play_trajectory(viewer: mujoco.viewer.Handle, model: mujoco.MjModel, data: mujoco.MjData, frames) -> float:
    last_position = float(data.qpos[0])
    for frame in frames:
        if not viewer.is_running():
            return last_position
        position = frame.positions_rad["j1_base_yaw"]
        data.qpos[0] = position
        data.qvel[0] = (position - last_position) / SAMPLE_PERIOD_S
        data.time = frame.time_s
        mujoco.mj_forward(model, data)
        viewer.sync()
        sleep(SAMPLE_PERIOD_S)
        last_position = position
    return last_position


def main() -> None:
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    targets = (
        POSE_LIBRARY["look_left"]["j1_base_yaw"],
        IDLE_POSE["j1_base_yaw"],
        POSE_LIBRARY["look_right"]["j1_base_yaw"],
        IDLE_POSE["j1_base_yaw"],
    )
    current = IDLE_POSE["j1_base_yaw"]
    print("J1 pose loop: max 0.80 rad/s; close the MuJoCo window to stop.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for target in targets:
                current = play_trajectory(viewer, model, data, pose_frames(current, target))
                if not viewer.is_running():
                    return
                sleep(0.35)


if __name__ == "__main__":
    main()
